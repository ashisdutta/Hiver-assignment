"""Stage 1 of the agent pipeline (project_brief.md Section 3): intent classification.

Single LLM call (Groq, OpenAI-compatible API) against a locked 5-way taxonomy.
No fine-tuned classifier, no few-shot examples -- the taxonomy definitions in the
prompt are the only guidance given to the model.
"""
import json

from llm_client import MODEL, get_client

TAXONOMY = {
    "delivery_issue": "package late, lost, damaged, or mishandled in delivery",
    "billing_dispute": "wrong charge, double charge, refund request, or pricing complaint",
    "order_inquiry": "questions about timing, payment schedule, or product/service capability "
                      "— informational, not a complaint",
    "account_technical": "login, app, or website problems",
    "other": "doesn't clearly fit the above",
}

_SYSTEM_PROMPT = "You are an intent classifier for a customer support system. Classify the " \
    "customer's current message into exactly one of these categories:\n\n" + \
    "\n".join(f"- {name}: {desc}" for name, desc in TAXONOMY.items()) + \
    "\n\nUse the prior conversation only as context for what the current message means. " \
    "Respond with JSON only, matching this exact schema:\n" \
    '{"intent": "<one of the five category names above, exactly>", ' \
    '"confidence": <float between 0 and 1>, "reasoning": "<one short sentence>"}'

def _format_history(thread_history: list) -> str:
    if not thread_history:
        return "(no prior context)"
    lines = []
    for turn in thread_history:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def classify_intent(customer_message: str, thread_history: list) -> dict:
    """Classify a customer message into the locked 5-way intent taxonomy.

    Returns {"intent": str, "confidence": float, "reasoning": str}.
    """
    user_prompt = (
        f"Conversation so far:\n{_format_history(thread_history)}\n\n"
        f"Customer's current message:\n{customer_message}"
    )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from e

    intent = result.get("intent")
    confidence = result.get("confidence")
    reasoning = result.get("reasoning")

    if intent not in TAXONOMY:
        raise ValueError(f"Model returned an intent outside the taxonomy: {intent!r}")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        raise ValueError(f"Model returned an invalid confidence: {confidence!r}")

    return {"intent": intent, "confidence": float(confidence), "reasoning": str(reasoning)}


if __name__ == "__main__":
    from sample_rows import load_sample_rows

    for row in load_sample_rows():
        result = classify_intent(row["customer_message"], row["thread_history"])
        print(f"[{row['thread_id']}] {row['customer_message']!r}")
        print(f"  -> intent={result['intent']} confidence={result['confidence']:.2f} "
              f"reasoning={result['reasoning']!r}")
        print()
