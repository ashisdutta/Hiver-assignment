"""Build a human-scoring worksheet for judge validation: take the first
VALIDATION_N of the 30 real-system rows from eval_replies.py's persisted
results (same order as that run's seeded sample), and output a worksheet with
blank rubric columns for manual 1-5 scoring against the same rubric the LLM
judge used.

The LLM judge's original scores for these rows are saved separately (not in
the worksheet) so manual scoring isn't anchored by seeing them.

Run: python src/build_judge_validation.py
"""
import json
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("data/processed/eval_replies_results.json")
WORKSHEET_PATH = Path("data/processed/judge_validation_worksheet.csv")
LLM_SCORES_PATH = Path("data/processed/judge_validation_llm_scores.json")
VALIDATION_N = 15


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    subset = results[:VALIDATION_N]
    print(f"Using the first {VALIDATION_N} of {len(results)} rows from {RESULTS_PATH} "
          f"(same order as eval_replies.py's seeded sample) for judge validation.")

    worksheet_rows = []
    llm_scores = {}
    for r in subset:
        worksheet_rows.append({
            "thread_id": r["thread_id"],
            "customer_message": r["customer_message"],
            "drafted_reply": r["replies"]["real"],
            "groundedness": "",
            "correctness_safety": "",
            "tone": "",
            "actionability": "",
        })
        llm_scores[r["thread_id"]] = r["judge_scores"]["real"]

    WORKSHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(worksheet_rows).to_csv(WORKSHEET_PATH, index=False)
    with open(LLM_SCORES_PATH, "w") as f:
        json.dump(llm_scores, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(worksheet_rows)}-row worksheet to {WORKSHEET_PATH}")
    print("Fill in groundedness, correctness_safety, tone, actionability (1-5 each) by hand,")
    print("then run: python src/score_judge_validation.py")


if __name__ == "__main__":
    main()
