"""Compute agreement between manual scores (judge_validation_worksheet.csv,
hand-filled) and the LLM judge's original scores on the same rows
(judge_validation_llm_scores.json), per rubric dimension: % exact match and
% within 1 point. This is the judge-validation evidence for the report.

Run: python src/score_judge_validation.py
"""
import json
from pathlib import Path

import pandas as pd

WORKSHEET_PATH = Path("data/processed/judge_validation_worksheet.csv")
LLM_SCORES_PATH = Path("data/processed/judge_validation_llm_scores.json")
DIMENSIONS = ["groundedness", "correctness_safety", "tone", "actionability"]


def load_human_scores() -> pd.DataFrame:
    df = pd.read_csv(WORKSHEET_PATH, dtype=str, keep_default_na=False, na_values=[])
    missing = df[(df[DIMENSIONS] == "").any(axis=1)]
    if len(missing):
        ids = missing["thread_id"].tolist()
        raise ValueError(
            f"{len(missing)}/{len(df)} rows are missing one or more manual scores -- "
            f"scoring isn't complete yet. Unscored thread_ids: {ids}"
        )
    for dim in DIMENSIONS:
        df[dim] = df[dim].astype(float)
        if not df[dim].between(1, 5).all():
            raise ValueError(f"Column {dim!r} has values outside the 1-5 range")
    return df


def main():
    human_df = load_human_scores()
    with open(LLM_SCORES_PATH) as f:
        llm_scores = json.load(f)

    n = len(human_df)
    print("\n" + "=" * 72)
    print(f"JUDGE VALIDATION: human vs LLM-judge agreement (n={n} rows, real-system replies)")
    print("=" * 72)
    print(f"{'dimension':<20}{'% exact match':>16}{'% within 1 point':>20}")

    for dim in DIMENSIONS:
        exact = 0
        within_1 = 0
        for _, row in human_df.iterrows():
            human_val = row[dim]
            llm_val = llm_scores[row["thread_id"]][dim]
            diff = abs(human_val - llm_val)
            if diff == 0:
                exact += 1
            if diff <= 1:
                within_1 += 1
        print(f"{dim:<20}{exact / n:>15.0%}{within_1 / n:>20.0%}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
