"""Stage 5 eval harness (project_brief.md Section 5).

load_and_validate_golden_set() guards entry: every row must have true_intent/
true_route filled in, and true_* must not just mirror suggested_* on an
implausibly high fraction of rows (a sign of unreviewed copying). Both are hard
errors -- this exists specifically to prevent accidentally evaluating against
unreviewed labels if time runs out.

Intent and routing metrics below are computed entirely from columns already in
the worksheet (suggested_intent/suggested_route vs true_intent/true_route) --
no new LLM calls. Reply-quality/judge scoring is a separate, later step: it
needs draft_reply to actually run against the golden set, which wasn't done
when building this worksheet (classify_intent + decide_route only, to save
Groq calls).

Run: python src/run_eval.py
"""
from pathlib import Path

import pandas as pd

GOLDEN_SET_PATH = Path("data/processed/golden_set_worksheet.csv")
REQUIRED_COLUMNS = [
    "thread_id", "customer_message", "reference_reply",
    "suggested_intent", "suggested_route",
    "true_intent", "true_route", "route_reason",
]
AGREEMENT_THRESHOLD = 0.90
INTENT_CLASSES = ["delivery_issue", "billing_dispute", "order_inquiry", "account_technical", "other"]
INTENT_ABBREV = {"delivery_issue": "DI", "billing_dispute": "BD", "order_inquiry": "OI",
                  "account_technical": "AT", "other": "OT"}


def load_and_validate_golden_set(path: Path = GOLDEN_SET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Golden set at {path} is missing required columns: {missing_cols}")

    unlabeled = df[(df["true_intent"].str.strip() == "") | (df["true_route"].str.strip() == "")]
    if len(unlabeled):
        ids = unlabeled["thread_id"].tolist()
        shown = ids[:20]
        more = f" ... and {len(ids) - 20} more" if len(ids) > 20 else ""
        raise ValueError(
            f"{len(unlabeled)}/{len(df)} rows are missing true_intent/true_route -- "
            f"manual labeling isn't complete yet. Unlabeled thread_ids: {shown}{more}"
        )

    intent_agreement = (df["true_intent"] == df["suggested_intent"]).mean()
    route_agreement = (df["true_route"] == df["suggested_route"]).mean()

    if intent_agreement > AGREEMENT_THRESHOLD or route_agreement > AGREEMENT_THRESHOLD:
        raise ValueError(
            f"true_intent matches suggested_intent on {intent_agreement:.0%} of rows and "
            f"true_route matches suggested_route on {route_agreement:.0%} of rows "
            f"(threshold: {AGREEMENT_THRESHOLD:.0%}). This looks like the suggestions may "
            f"have been copied wholesale rather than genuinely reviewed row by row -- "
            f"double check the labeling pass before evaluating against this golden set."
        )

    print(f"Golden set validated: {len(df)} rows, all labeled. "
          f"intent agreement={intent_agreement:.0%}, route agreement={route_agreement:.0%}")
    return df


def compute_intent_metrics(df: pd.DataFrame, pred_col: str = "suggested_intent") -> dict:
    true = df["true_intent"].tolist()
    pred = df[pred_col].tolist()

    accuracy = sum(t == p for t, p in zip(true, pred)) / len(true)

    matrix = {t: {p: 0 for p in INTENT_CLASSES} for t in INTENT_CLASSES}
    for t, p in zip(true, pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    per_class = {}
    for c in INTENT_CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[t][c] for t in INTENT_CLASSES if t != c)
        fn = sum(matrix[c][p] for p in INTENT_CLASSES if p != c)
        support = sum(matrix[c].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(per_class)

    return {"accuracy": accuracy, "macro_f1": macro_f1, "confusion_matrix": matrix, "per_class": per_class}


def compute_routing_metrics(df: pd.DataFrame, pred_col: str = "suggested_route") -> dict:
    true = df["true_route"]
    pred = df[pred_col]

    tp = int(((true == "escalate") & (pred == "escalate")).sum())
    fp = int(((true == "auto_handle") & (pred == "escalate")).sum())
    fn = int(((true == "escalate") & (pred == "auto_handle")).sum())
    tn = int(((true == "auto_handle") & (pred == "auto_handle")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    missed_escalations = df[(true == "escalate") & (pred == "auto_handle")]

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "missed_escalations": missed_escalations,
    }


def _truncate(text: str, length: int = 70) -> str:
    text = text or ""
    return text if len(text) <= length else text[:length - 1].rstrip() + "…"


def print_intent_metrics(intent: dict, title: str = "INTENT METRICS (suggested_intent vs true_intent)") -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Accuracy: {intent['accuracy']:.1%}   Macro F1: {intent['macro_f1']:.3f}")

    print(f"\n{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in INTENT_CLASSES:
        m = intent["per_class"][c]
        print(f"{c:<20}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>10}")

    legend = ", ".join(f"{abbr}={name}" for name, abbr in INTENT_ABBREV.items())
    print(f"\nConfusion matrix (rows=true, columns=predicted) -- {legend}")
    header = " " * 20 + "".join(f"{INTENT_ABBREV[c]:>6}" for c in INTENT_CLASSES)
    print(header)
    matrix = intent["confusion_matrix"]
    for t in INTENT_CLASSES:
        row = "".join(f"{matrix[t][p]:>6}" for p in INTENT_CLASSES)
        print(f"{t:<20}{row}")


def print_routing_metrics(
    routing: dict,
    title: str = "ROUTING METRICS (suggested_route vs true_route, escalate = positive class)",
    show_missed_detail: bool = True,
) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Precision: {routing['precision']:.2f}   Recall: {routing['recall']:.2f}   "
          f"F1: {routing['f1']:.2f}")
    print(f"TP={routing['tp']}  FP={routing['fp']}  FN={routing['fn']}  TN={routing['tn']}")

    missed = routing["missed_escalations"]
    print(f"\nMissed escalations (true_route=escalate, predicted=auto_handle): "
          f"{len(missed)} -- the dangerous error type, an unreliable auto-reply going out live")
    if len(missed) and show_missed_detail:
        for _, row in missed.iterrows():
            print(f"  [{row['thread_id']}] {_truncate(row['customer_message'])}")
            print(f"      reason: {row['route_reason']}")


def print_report(df: pd.DataFrame) -> None:
    intent = compute_intent_metrics(df)
    routing = compute_routing_metrics(df)
    print_intent_metrics(intent)
    print_routing_metrics(routing)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    golden_set = load_and_validate_golden_set()
    print_report(golden_set)
