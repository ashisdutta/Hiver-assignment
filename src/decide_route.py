"""Stage 4 of the agent pipeline (project_brief.md Section 3): routing decision.

Rule-based, no LLM call -- fast and deterministic. Confidence-threshold routing
was dropped (classify_intent's confidence wasn't calibrated enough to be a
useful signal, see decision log); routing instead uses intent-based rules and
grounding strength, the two signals that actually discriminate.
"""


def decide_route(customer_message: str, intent: str, draft_reply: str, grounding: dict) -> dict:
    if intent == "billing_dispute":
        return {
            "route": "escalate",
            "reason": "money-related issue, requires human handling for financial risk",
        }

    if not (grounding or {}).get("grounding_found"):
        return {
            "route": "escalate",
            "reason": "no strong precedent found for this issue, drafted reply may be unreliable",
        }

    return {
        "route": "auto_handle",
        "reason": "routine issue with strong grounding to a known resolution pattern",
    }


if __name__ == "__main__":
    from classify_intent import classify_intent
    from draft_reply import draft_reply as _draft_reply
    from retrieve_grounding import retrieve_grounding
    from sample_rows import load_sample_rows

    for row in load_sample_rows():
        intent_result = classify_intent(row["customer_message"], row["thread_history"])
        grounding_result = retrieve_grounding(row["customer_message"])
        reply = _draft_reply(
            row["customer_message"], row["thread_history"], intent_result["intent"], grounding_result
        )
        route_result = decide_route(row["customer_message"], intent_result["intent"], reply, grounding_result)

        print(f"[{row['thread_id']}] {row['customer_message']!r}")
        print(f"  intent={intent_result['intent']} grounding_found={grounding_result['grounding_found']}")
        print(f"  route={route_result['route']} reason={route_result['reason']!r}")
        print()
