"""Stage 3 of the agent pipeline (project_brief.md Section 3): reply drafting.

Single LLM call (Groq, same client/model as classify_intent). Uses
retrieve_grounding's output as reference for tone and resolution pattern when
strong grounding exists; falls back to a generic, honest acknowledgment (no
fabricated resolution) when grounding_found is False.
"""
from llm_client import MODEL, get_client

_SYSTEM_PROMPT = (
    "You are drafting a public Twitter reply for AmazonHelp, Amazon's customer support "
    "handle. Match the brand's real style: concise, an empathetic acknowledgment of the "
    "issue, then a concrete next step. One or two sentences. Do not invent policies, "
    "refunds, timelines, or resolutions -- only offer a specific resolution if it's "
    "supported by the reference examples or conversation given to you. If no reference "
    "examples are given, don't promise a fix; acknowledge the issue honestly and ask for "
    "the detail needed to help. Never state or imply an internal action was already taken "
    "(e.g. \"escalated to our team\", \"forwarded this\", \"flagged this for review\") "
    "unless a reference example or the conversation actually shows that action happening "
    "-- if you want to offer a next step and nothing supports one, ask the customer for "
    "more detail instead of inventing one. Never write out a URL: the reference examples "
    "and conversation history have had all links stripped out, so any URL you produce is "
    "guaranteed to be fabricated. Refer to a link generically instead (e.g. \"the link we "
    "sent\" or \"here\") without writing an actual URL string."
)


def _format_history(thread_history: list) -> str:
    if not thread_history:
        return "(no prior context)"
    return "\n".join(f"{t.get('speaker', 'unknown')}: {t.get('text', '')}" for t in thread_history)


def _format_grounding(grounding: dict) -> str:
    grounding = grounding or {}
    matches = grounding.get("matches") or []
    if not grounding.get("grounding_found") or not matches:
        return "(no strong grounding found -- no reference examples; do not fabricate a resolution)"
    lines = ["Reference examples of how similar past cases were actually resolved by this brand:"]
    for i, m in enumerate(matches, 1):
        lines.append(f"{i}. customer: {m['customer_message']}\n   brand reply: {m['reference_reply']}")
    return "\n".join(lines)


def draft_reply(customer_message: str, thread_history: list, intent: str, grounding: dict) -> str:
    user_prompt = (
        f"Intent: {intent}\n\n"
        f"Conversation so far:\n{_format_history(thread_history)}\n\n"
        f"Customer's current message:\n{customer_message}\n\n"
        f"{_format_grounding(grounding)}\n\n"
        "Draft AmazonHelp's reply now. Match the tone and resolution pattern of the "
        "reference examples if given; otherwise give a brief honest acknowledgment and "
        "ask for the detail needed to help. Reply with the tweet text only -- no quotes, "
        "no explanation."
    )

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    from classify_intent import classify_intent
    from retrieve_grounding import retrieve_grounding
    from sample_rows import load_sample_rows

    for row in load_sample_rows():
        intent_result = classify_intent(row["customer_message"], row["thread_history"])
        grounding_result = retrieve_grounding(row["customer_message"])
        reply = draft_reply(
            row["customer_message"], row["thread_history"], intent_result["intent"], grounding_result
        )

        print(f"[{row['thread_id']}] {row['customer_message']!r}")
        print(f"  intent={intent_result['intent']}")
        if grounding_result["grounding_found"]:
            top = grounding_result["matches"][0]
            print(f"  grounding: sim={top['similarity']:.3f} reply={top['reference_reply']!r}")
        else:
            print("  grounding: none found")
        print(f"  draft_reply: {reply!r}")
        print()
