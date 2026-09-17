"""Chains all four agent pipeline stages (project_brief.md Section 3):
classify_intent -> retrieve_grounding -> draft_reply -> decide_route.

This is what the eval harness (Stage 5) calls per golden-set row. Takes a raw
request in the Section 2 agent input schema and returns the full result.
"""
from classify_intent import classify_intent
from decide_route import decide_route
from draft_reply import draft_reply
from retrieve_grounding import retrieve_grounding


def run_pipeline(request: dict) -> dict:
    """request: {"thread_id": ..., "brand": ..., "customer_message": str, "thread_history": list}"""
    customer_message = request["customer_message"]
    thread_history = request.get("thread_history", [])

    intent_result = classify_intent(customer_message, thread_history)
    grounding_result = retrieve_grounding(customer_message)
    reply = draft_reply(customer_message, thread_history, intent_result["intent"], grounding_result)
    route_result = decide_route(customer_message, intent_result["intent"], reply, grounding_result)

    return {
        "thread_id": request.get("thread_id"),
        "brand": request.get("brand"),
        "customer_message": customer_message,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result["confidence"],
        "intent_reasoning": intent_result["reasoning"],
        "grounding_found": grounding_result["grounding_found"],
        "grounding_matches": grounding_result["matches"],
        "draft_reply": reply,
        "route": route_result["route"],
        "route_reason": route_result["reason"],
    }


if __name__ == "__main__":
    from sample_rows import load_sample_rows

    for row in load_sample_rows():
        result = run_pipeline({
            "thread_id": row["thread_id"],
            "brand": "AmazonHelp",
            "customer_message": row["customer_message"],
            "thread_history": row["thread_history"],
        })
        print(f"[{result['thread_id']}] {result['customer_message']!r}")
        print(f"  intent={result['intent']} (confidence={result['intent_confidence']:.2f})")
        print(f"  grounding_found={result['grounding_found']}")
        print(f"  draft_reply={result['draft_reply']!r}")
        print(f"  route={result['route']} reason={result['route_reason']!r}")
        print()
