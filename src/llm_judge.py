"""LLM-as-judge for reply quality (project_brief.md Section 5 rubric): scores a
candidate support reply on groundedness, correctness/safety, tone, and
actionability (1-5 each). reference_reply is given as context only -- NOT an
exact-match target, since multiple different replies can be equally good.
"""
import json

from llm_client import MODEL, get_client

_SYSTEM_PROMPT = (
    "You are evaluating a candidate customer-support reply for AmazonHelp on Twitter. "
    "You'll see the conversation so far, the customer's current message, the brand's "
    "actual historical reply to this situation (context only -- not a target the "
    "candidate must match exactly, since multiple different replies can be equally good), "
    "and a candidate reply to score.\n\n"
    "Score the candidate reply on these 4 dimensions, each 1-5 (5 = best):\n"
    "- groundedness: does it avoid fabricating information (policies, promises, links, "
    "actions already taken) not supported by the conversation?\n"
    "- correctness_safety: is it factually reasonable and safe -- no incorrect "
    "commitments, no risky or misleading claims?\n"
    "- tone: does it read as empathetic, professional, and appropriately concise, "
    "matching a real brand support voice?\n"
    "- actionability: does it give a clear, concrete next step, or an honest, "
    "appropriate request for more detail when no next step is yet knowable?\n\n"
    "Respond with JSON only, matching this exact schema:\n"
    '{"groundedness": <1-5>, "correctness_safety": <1-5>, "tone": <1-5>, '
    '"actionability": <1-5>, "reasoning": "<one short sentence>"}'
)


def _format_history(thread_history: list) -> str:
    if not thread_history:
        return "(no prior context)"
    return "\n".join(f"{t.get('speaker', 'unknown')}: {t.get('text', '')}" for t in thread_history)


def judge_reply(customer_message: str, thread_history: list, reference_reply: str, candidate_reply: str) -> dict:
    user_prompt = (
        f"Conversation so far:\n{_format_history(thread_history)}\n\n"
        f"Customer's current message:\n{customer_message}\n\n"
        f"Brand's actual historical reply (context only, not a target to match):\n{reference_reply}\n\n"
        f"Candidate reply to score:\n{candidate_reply}"
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
        raise ValueError(f"Judge did not return valid JSON: {raw!r}") from e

    scores = {}
    for dim in ("groundedness", "correctness_safety", "tone", "actionability"):
        val = result.get(dim)
        if not isinstance(val, (int, float)) or not (1 <= val <= 5):
            raise ValueError(f"Judge returned invalid {dim}: {val!r}")
        scores[dim] = float(val)
    scores["reasoning"] = str(result.get("reasoning", ""))
    return scores
