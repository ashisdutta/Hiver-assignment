"""Trivial and simple (keyword-rule) baselines for the agent pipeline, scored
against the same golden-set worksheet the real system (Stage 5) uses. Exists
to show the real system's numbers (run_eval.py) are actually earning their
complexity -- no new LLM calls, all computed from existing worksheet columns.

Trivial tier:
  - intent: always predict the majority class in true_intent
  - routing: always escalate
Simple tier:
  - intent: keyword rules over customer_message (delivery/billing/account
    vocabulary, in that priority order, else order_inquiry/other)
  - routing: escalate only on obvious billing/financial keywords in the raw
    text -- no dependency on the (simple or real) intent prediction, and no
    grounding lookup, per the brief for this baseline

Run: python src/baselines.py
"""
from collections import Counter

from run_eval import (
    compute_intent_metrics,
    compute_routing_metrics,
    load_and_validate_golden_set,
    print_intent_metrics,
    print_routing_metrics,
)

DELIVERY_KEYWORDS = [
    "deliver", "delivery", "shipped", "shipping", "package", "parcel",
    "arrived", "arrive", "late", "delayed", "delay", "lost", "damaged",
    "damage", "courier", "carrier", "tracking", "track",
]
BILLING_KEYWORDS = [
    "charge", "charged", "refund", "bill", "billing", "cashback",
    "double charged", "overcharge", "deduct", "money back", "cancel my",
]
ACCOUNT_KEYWORDS = [
    "login", "log in", "password", "sign in", "app crash", "crashed",
    "website", "error", "won't load", "not loading", "locked",
    "account locked", "can't access", "can't log",
]
ORDER_INQUIRY_KEYWORDS = [
    "does amazon", "can i", "how do i", "when will", "is there a way",
    "what is the", "available", "does it support", "compatible", "how much",
]


def _contains_any(text: str, keywords: list) -> bool:
    text = text.lower()
    return any(kw in text for kw in keywords)


def simple_intent(customer_message: str) -> str:
    if _contains_any(customer_message, BILLING_KEYWORDS):
        return "billing_dispute"
    if _contains_any(customer_message, DELIVERY_KEYWORDS):
        return "delivery_issue"
    if _contains_any(customer_message, ACCOUNT_KEYWORDS):
        return "account_technical"
    if _contains_any(customer_message, ORDER_INQUIRY_KEYWORDS):
        return "order_inquiry"
    return "other"


def simple_route(customer_message: str) -> str:
    return "escalate" if _contains_any(customer_message, BILLING_KEYWORDS) else "auto_handle"


def main():
    df = load_and_validate_golden_set()

    majority_class = Counter(df["true_intent"]).most_common(1)[0][0]
    df["trivial_intent"] = majority_class
    df["trivial_route"] = "escalate"

    df["simple_intent"] = df["customer_message"].map(simple_intent)
    df["simple_route"] = df["customer_message"].map(simple_route)

    print(f"\nTrivial intent baseline predicts majority class for every row: {majority_class!r}")

    print_intent_metrics(
        compute_intent_metrics(df, pred_col="trivial_intent"),
        title="TRIVIAL INTENT BASELINE (majority class vs true_intent)",
    )
    print_routing_metrics(
        compute_routing_metrics(df, pred_col="trivial_route"),
        title="TRIVIAL ROUTING BASELINE (always escalate vs true_route)",
        show_missed_detail=False,
    )

    print_intent_metrics(
        compute_intent_metrics(df, pred_col="simple_intent"),
        title="SIMPLE INTENT BASELINE (keyword rules vs true_intent)",
    )
    print_routing_metrics(
        compute_routing_metrics(df, pred_col="simple_route"),
        title="SIMPLE ROUTING BASELINE (billing-keyword rule vs true_route)",
        show_missed_detail=False,
    )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
