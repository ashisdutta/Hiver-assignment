"""Build the golden-set labeling worksheet (project_brief.md Section 5): randomly
sample N_ROWS rows from eval_pool, run classify_intent + decide_route to prefill
suggested_intent/suggested_route as a starting point for manual review, and leave
true_intent/true_route/route_reason blank for hand-labeling.

Only classify_intent makes a Groq call here -- decide_route's rules don't use the
draft_reply argument, so draft_reply() is skipped entirely to keep this to N_ROWS
Groq calls (well within the free-tier 30/min, 1000/day limits), paced below.

Run: python src/build_golden_set.py
"""
import json
import random
import time
from pathlib import Path

import pandas as pd

from classify_intent import classify_intent
from decide_route import decide_route
from retrieve_grounding import retrieve_grounding

EVAL_POOL_PATH = Path("data/processed/eval_pool.jsonl")
OUTPUT_PATH = Path("data/processed/golden_set_worksheet.csv")
N_ROWS = 150
SEED = 42
TRUNCATE_LEN = 100
# Groq free tier: 30 requests/minute. Paced safely under that (~27/min).
SECONDS_BETWEEN_CALLS = 2.2


def _truncate(text: str, length: int = TRUNCATE_LEN) -> str:
    text = text or ""
    return text if len(text) <= length else text[:length - 1].rstrip() + "…"


def _load_eval_pool() -> list:
    records = []
    with open(EVAL_POOL_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    records = _load_eval_pool()
    if len(records) < N_ROWS:
        raise ValueError(f"eval_pool has only {len(records)} rows, need {N_ROWS}")

    sample = random.Random(SEED).sample(records, N_ROWS)

    rows = []
    for i, rec in enumerate(sample, 1):
        intent = classify_intent(rec["customer_message"], rec["thread_history"])["intent"]
        grounding = retrieve_grounding(rec["customer_message"])
        route = decide_route(rec["customer_message"], intent, "", grounding)["route"]

        rows.append({
            "thread_id": rec["thread_id"],
            "customer_message": _truncate(rec["customer_message"]),
            "reference_reply": _truncate(rec["reference_reply"]),
            "suggested_intent": intent,
            "suggested_route": route,
            "true_intent": "",
            "true_route": "",
            "route_reason": "",
        })
        print(f"[{i}/{N_ROWS}] [{rec['thread_id']}] intent={intent} route={route}", flush=True)

        if i < N_ROWS:
            time.sleep(SECONDS_BETWEEN_CALLS)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")
    print("Fill in true_intent, true_route, route_reason by hand, then run run_eval.py.")


if __name__ == "__main__":
    main()
