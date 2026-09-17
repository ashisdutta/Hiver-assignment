"""Part 2 of the eval harness: reply-quality scoring on a SUBSET_N-row subset of
the golden set. This is a reduced N from the full 150 -- an explicit, documented
scope decision given time/rate-limit constraints, not a silent shortcut (see the
printed note at runtime and the decision log).

For each sampled row:
  - real system reply: draft_reply() using the pipeline's own suggested_intent
    (already computed when the worksheet was built, so no redundant
    classify_intent call) and a fresh retrieve_grounding() call (local
    embeddings, no LLM cost)
  - trivial baseline reply: one fixed canned reply, identical for every row
  - simple baseline reply: the single nearest-neighbor grounding_pool reply,
    used verbatim (no LLM, no similarity threshold -- always the top-1 match)

All three reply sets are then scored blind and independently by an LLM judge
(llm_judge.judge_reply) against a 4-dimension rubric (groundedness,
correctness/safety, tone, actionability), each 1-5. reference_reply is given
to the judge as CONTEXT only, not an exact-match target (brief Section 5).

Groq calls: SUBSET_N (draft_reply) + SUBSET_N*3 (judge: 3 tiers per row),
paced under the free-tier 30/min limit.

Resumable: draft replies are checkpointed separately from judge scores, and
judge scores are written to disk after every row (not just at the end), so an
interruption (e.g. hitting a daily token cap) loses at most one in-flight row,
not the whole run. Re-running the script picks up where it left off.

Run: python src/eval_replies.py
"""
import json
import random
import time
from pathlib import Path

from draft_reply import draft_reply
from llm_judge import judge_reply
from retrieve_grounding import retrieve_grounding
from run_eval import load_and_validate_golden_set

EVAL_POOL_PATH = Path("data/processed/eval_pool.jsonl")
DRAFTS_PATH = Path("data/processed/eval_replies_drafts.json")
RESULTS_PATH = Path("data/processed/eval_replies_results.json")
SUBSET_N = 30
SEED = 42
SECONDS_BETWEEN_CALLS = 2.2
TIERS = ("trivial", "simple", "real")

TRIVIAL_REPLY = (
    "We're sorry for the trouble! Please reach out to us here so we can look into this further."
)


def _load_eval_pool_by_id() -> dict:
    records = {}
    with open(EVAL_POOL_PATH) as f:
        for line in f:
            rec = json.loads(line)
            records[str(rec["thread_id"])] = rec
    return records


def simple_reply(customer_message: str) -> str:
    result = retrieve_grounding(customer_message, k=1, threshold=0.0)
    return result["matches"][0]["reference_reply"]


def generate_replies(golden_set, eval_pool) -> list:
    all_ids = golden_set["thread_id"].tolist()
    subset_ids = random.Random(SEED).sample(all_ids, SUBSET_N)

    if DRAFTS_PATH.exists():
        with open(DRAFTS_PATH) as f:
            cached = json.load(f)
        if [r["thread_id"] for r in cached] == subset_ids:
            print(f"[Part 2] Reusing cached drafts from {DRAFTS_PATH} (same {SUBSET_N}-row "
                  f"sample, seed={SEED}) -- no re-drafting needed.\n")
            return cached
        print(f"[Part 2] Cached drafts at {DRAFTS_PATH} don't match the current sample; "
              f"redrafting.\n")

    print(f"\n[Part 2] Using a {SUBSET_N}-row random subset of the {len(all_ids)}-row golden "
          f"set (seed={SEED}) -- reduced N given time/Groq rate-limit constraints. "
          f"Documented scope decision, not a silent shortcut.\n")

    rows = []
    for i, tid in enumerate(subset_ids, 1):
        golden_row = golden_set[golden_set["thread_id"] == tid].iloc[0]
        full = eval_pool[tid]

        customer_message = full["customer_message"]
        thread_history = full["thread_history"]
        reference_reply = full["reference_reply"]
        suggested_intent = golden_row["suggested_intent"]

        grounding = retrieve_grounding(customer_message)
        real = draft_reply(customer_message, thread_history, suggested_intent, grounding)
        time.sleep(SECONDS_BETWEEN_CALLS)

        rows.append({
            "thread_id": tid,
            "customer_message": customer_message,
            "thread_history": thread_history,
            "reference_reply": reference_reply,
            "replies": {
                "trivial": TRIVIAL_REPLY,
                "simple": simple_reply(customer_message),
                "real": real,
            },
        })
        print(f"[draft {i}/{SUBSET_N}] [{tid}] real/trivial/simple replies ready", flush=True)

    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRAFTS_PATH, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Checkpointed drafts to {DRAFTS_PATH}")

    return rows


def _load_completed_results() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        completed = json.load(f)
    return {r["thread_id"]: r for r in completed}


def _write_results(completed_by_id: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(list(completed_by_id.values()), f, indent=2, ensure_ascii=False)


def judge_all(rows: list) -> dict:
    completed_by_id = _load_completed_results()
    if completed_by_id:
        print(f"[Part 2] Found {len(completed_by_id)} already-judged rows in {RESULTS_PATH}, "
              f"resuming remaining rows.\n")

    total = len(rows) * len(TIERS)
    n = sum(len(TIERS) for tid in completed_by_id if tid in {r["thread_id"] for r in rows})

    for row in rows:
        tid = row["thread_id"]
        if tid in completed_by_id:
            continue

        row_scores = {}
        for tier in TIERS:
            n += 1
            result = judge_reply(
                customer_message=row["customer_message"],
                thread_history=row["thread_history"],
                reference_reply=row["reference_reply"],
                candidate_reply=row["replies"][tier],
            )
            row_scores[tier] = result
            print(f"[judge {n}/{total}] [{tid}] tier={tier} "
                  f"groundedness={result['groundedness']:.0f} "
                  f"correctness_safety={result['correctness_safety']:.0f} "
                  f"tone={result['tone']:.0f} actionability={result['actionability']:.0f}", flush=True)
            time.sleep(SECONDS_BETWEEN_CALLS)

        completed_by_id[tid] = {
            "thread_id": tid,
            "customer_message": row["customer_message"],
            "thread_history": row["thread_history"],
            "reference_reply": row["reference_reply"],
            "replies": row["replies"],
            "judge_scores": row_scores,
        }
        _write_results(completed_by_id)

    scores = {tier: [completed_by_id[row["thread_id"]]["judge_scores"][tier] for row in rows]
              for tier in TIERS}
    return scores


def print_summary(scores: dict) -> None:
    dims = ["groundedness", "correctness_safety", "tone", "actionability"]
    n = len(scores["real"])
    print("\n" + "=" * 72)
    print(f"REPLY-QUALITY JUDGE SCORES (n={n} rows, 1-5 scale, reference_reply used as "
          f"judge context only -- not an exact-match target)")
    print("=" * 72)
    print(f"{'tier':<10}" + "".join(f"{d:>18}" for d in dims) + f"{'overall':>10}")
    for tier in TIERS:
        tier_rows = scores[tier]
        avgs = {d: sum(r[d] for r in tier_rows) / len(tier_rows) for d in dims}
        overall = sum(avgs.values()) / len(dims)
        print(f"{tier:<10}" + "".join(f"{avgs[d]:>18.2f}" for d in dims) + f"{overall:>10.2f}")
    print("=" * 72 + "\n")


def main():
    golden_set = load_and_validate_golden_set()
    eval_pool = _load_eval_pool_by_id()

    rows = generate_replies(golden_set, eval_pool)
    scores = judge_all(rows)
    print(f"Full replies + judge scores for {len(rows)} rows are in {RESULTS_PATH}")
    print_summary(scores)


if __name__ == "__main__":
    main()
