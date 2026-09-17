"""Stage 2 of the agent pipeline (project_brief.md Section 3): grounding retrieval.

Builds a semantic embedding index over grounding_pool.jsonl's customer_message text
using a local sentence-transformers model -- no LLM calls here, which keeps Groq's
chat-completion rate limit free for classify_intent/draft_reply. Plain top-k cosine
similarity search with no intent pre-filter (v1 simplification: intent-based
pre-filtering was dropped to stay within Groq's free-tier rate limits given the
project's time constraints -- see decision log). Below SIMILARITY_THRESHOLD, the
top match is flagged as not strong enough rather than forced -- itself a signal
Stage 4 (routing) can use.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
GROUNDING_POOL_PATH = Path("data/processed/grounding_pool.jsonl")
EMBEDDING_CACHE_PATH = Path("data/interim/grounding_embeddings.npz")
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45

_model = None
_records = None
_embeddings = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _build_or_load_index():
    global _records, _embeddings
    if _records is not None and _embeddings is not None:
        return

    records = _load_records(GROUNDING_POOL_PATH)
    thread_ids = [str(r["thread_id"]) for r in records]

    if EMBEDDING_CACHE_PATH.exists():
        cache = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True)
        if str(cache["model_name"]) == MODEL_NAME and list(cache["thread_ids"]) == thread_ids:
            _records = records
            _embeddings = cache["embeddings"]
            return

    print(f"[retrieve_grounding] Embedding {len(records):,} grounding_pool messages "
          f"with {MODEL_NAME} (one-time, cached after) ...")
    texts = [r["customer_message"] for r in records]
    embeddings = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        EMBEDDING_CACHE_PATH,
        embeddings=embeddings,
        thread_ids=np.array(thread_ids),
        model_name=np.array(MODEL_NAME),
    )
    _records = records
    _embeddings = embeddings


def retrieve_grounding(customer_message: str, k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """Semantic top-k search over grounding_pool for messages similar to customer_message.

    Returns {"grounding_found": bool, "matches": [...]}. `matches` always holds the
    top-k results (even below threshold) so callers/logs can see what almost matched;
    `grounding_found` is the signal to actually treat them as grounding rather than
    forcing a weak match.
    """
    _build_or_load_index()
    query_embedding = _get_model().encode([customer_message], normalize_embeddings=True)[0]
    similarities = _embeddings @ query_embedding
    top_idx = np.argsort(-similarities)[:k]

    matches = [
        {
            "thread_id": _records[i]["thread_id"],
            "customer_message": _records[i]["customer_message"],
            "reference_reply": _records[i]["reference_reply"],
            "similarity": float(similarities[i]),
        }
        for i in top_idx
    ]

    grounding_found = bool(matches) and matches[0]["similarity"] >= threshold
    return {"grounding_found": grounding_found, "matches": matches}


if __name__ == "__main__":
    from sample_rows import load_sample_rows

    for row in load_sample_rows():
        result = retrieve_grounding(row["customer_message"])
        print(f"[{row['thread_id']}] {row['customer_message']!r}")
        flag = "" if result["grounding_found"] else "  (below threshold -- no strong grounding found)"
        print(f"  grounding_found={result['grounding_found']}{flag}")
        for m in result["matches"]:
            print(f"    sim={m['similarity']:.3f} [{m['thread_id']}] cust={m['customer_message']!r}")
            print(f"                     reply={m['reference_reply']!r}")
        print()
