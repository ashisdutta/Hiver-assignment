# Hiver SDE Intern Assignment — Project Brief

## Objective
Build an AI support agent for **AmazonHelp** (adjust if a different brand is chosen) using the Kaggle "Customer Support on Twitter" dataset, plus a rigorous evaluation harness that proves it works. The proof matters more than the system — build accordingly.

---

## 1. Data

**Source**: Kaggle `thoughtvector/customer-support-on-twitter`, filtered to `author_id == "AmazonHelp"` plus every thread that brand participates in.

**Raw schema** (one row per tweet):
`tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`

**Processing pipeline**:
1. Filter to the chosen brand's threads.
2. Reconstruct full conversations by chaining `in_response_to_tweet_id` / `response_tweet_id`.
3. Clean text: strip @handle boilerplate and URLs, filter to English, remove exact duplicates, drop threads with no brand reply.
4. Subsample to a few thousand threads (do NOT process the full dataset — a representative subsample is expected).
5. Note: most "resolved" threads are actually public triage replies ("please DM us your order ID") — the real fix happens in a private DM not present in this dataset. Decide explicitly whether to keep these as grounding examples (they still show tone/process) or filter for more substantive replies, and document the choice.
6. **Split into two non-overlapping pools**: a `grounding_pool` (source for the agent's retrieval index) and an `eval_pool` (source for the golden set). These must not overlap — if a golden-set example's exact resolution is also in the retrieval index, reply-quality scores get inflated by leakage instead of testing real generalization.

---

## 2. Agent input schema

Every stage of the agent receives this same object — the newest customer message plus everything before it in the thread:

```json
{
  "thread_id": "th_00482",
  "brand": "AmazonHelp",
  "customer_message": "still nothing, it's been 3 days",
  "thread_history": [
    {"speaker": "customer", "text": "...", "created_at": "..."},
    {"speaker": "brand", "text": "...", "created_at": "..."}
  ]
}
```

This exact schema should also be how golden-set rows are structured, so eval data can be fed straight into the live agent with no translation step.

---

## 3. Agent pipeline — four chained stages

### Stage 1: `classify_intent(input) -> {"intent": str, "confidence": float, "reasoning": str}`
- Taxonomy must be derived from the data, not assumed: sample ~100-150 real messages, read them, hand-cluster into recurring themes.
- Converge on 6-10 intents plus an explicit `other` catch-all. Write a one-line definition for each (used both for consistent hand-labeling and inside the classifier prompt).
- Implementation: LLM prompt-based classification (taxonomy + message + thread history in, structured JSON out). This is what makes it "your system" rather than the simple baseline.
- Carry the confidence score forward — it feeds the routing stage.

### Stage 2: `retrieve_grounding(input, intent) -> list[{customer_message, brand_reply, similarity, intent}]`
- Build a corpus of (customer_message, brand_reply, intent) pairs from the `grounding_pool`.
- Use semantic embedding similarity (not just keyword matching) to find similar past customer messages.
- Pre-filter by the predicted intent before ranking by similarity, to avoid cross-intent false matches on shared vocabulary.
- Return top-k (start with k=3).
- Define a similarity threshold; if nothing clears it, return "no strong grounding found" rather than forcing a weak match — this is itself a signal for routing.

### Stage 3: `draft_reply(input, intent, grounding) -> str`
- LLM prompt using the retrieved examples as grounding context for both tone and resolution pattern — not asking the model to invent a policy from scratch.

### Stage 4: `decide_route(input, intent, draft, confidence) -> {"route": "auto_handle" | "escalate", "reason": str}`
- Criteria to consider: low classifier confidence, no strong grounding found, certain intents that should always escalate (e.g. billing/legal), signs of repeat/unresolved complaints in thread history.
- Always attach a one-line reason, per the assignment's explicit requirement.

---

## 4. Baselines — required for the report, must actually be built and run

| Component | Trivial baseline | Simple baseline | Full system |
|---|---|---|---|
| Intent | Always predict majority class | Keyword rules or TF-IDF + logistic regression | LLM prompt-based classifier |
| Reply | One fixed canned message | Nearest-neighbor retrieval, paste past reply verbatim, no LLM | LLM + retrieval-grounded drafting |
| Route | Always escalate (or always auto-handle) | Hand-written rules (angry keywords, repeat messages) | LLM/rule-based using confidence + grounding signals |

All three tiers run through the **same** eval harness, same golden set, same metrics — that's what makes the comparison valid.

---

## 5. Golden evaluation set

- 150-250 examples, **randomly** sampled from `eval_pool` (not cherry-picked), hand-labeled.
- Fields per row: `id, customer_message, thread_context, true_intent, true_route, route_reason, reference_reply`.
- `reference_reply` = the brand's actual historical reply, used as judge context — NOT an exact-match target, since multiple different replies can be equally good.
- Document the sampling method and labeling approach in the report.

---

## 6. Evaluation harness

`run_eval.py`:
1. Load `golden_set.jsonl`.
2. For each example, run it through each system variant (trivial / simple / full) and capture predicted intent, retrieved grounding, drafted reply, and route decision.
3. Compare to human labels:
   - Intent: accuracy, F1 per class, confusion matrix.
   - Routing: precision/recall — flag false "auto-handle" on cases that should've escalated as the dangerous error type.
   - Reply quality: LLM-judge scored against a fixed rubric (see below).
4. Aggregate and print/save headline metrics per variant.

**Rubric for reply quality** (score 1-5 per dimension, with a one-line justification):
- **Groundedness** — does the approach match the brand's historical resolution pattern (using `reference_reply` as comparison context)?
- **Correctness/safety** — any invented or risky claims?
- **Tone** — matches the brand's real tone?
- **Actionability** — clear next step for the customer?

**Judge validation** (required evidence, not optional):
- Hand-score ~40 examples yourself using the same rubric, independently of the LLM judge.
- Compute agreement (% exact/within-1-point match, or Cohen's kappa) between your scores and the judge's.
- Report this agreement number explicitly — it's what makes the judge's scores on the remaining examples trustworthy.

---

## 7. Report (max 6 pages, or a README section)

Required sections, in order:
1. **Problem framing** — what "good" means for this brand, and what was explicitly chosen not to build.
2. **Results vs. baselines** — trivial and simple, for all three components, with real numbers from the harness.
3. **Failure analysis** — top 5 real failure modes with actual examples and hypotheses for each.
4. **"What is misleading about my headline number?"** — mandatory self-critique (class imbalance vs. a majority-class baseline, sample size / confidence, judge reliability limits, etc.)
5. **What you'd do next with one more week.**

---

## 8. Decision log (10-15 entries, bullets fine)

Entries already justified by design reasoning — write these up as-is once implemented:
- Split grounding pool and eval pool before sampling, to prevent retrieval leakage into evaluation.
- Subsampled instead of using the full ~3M-row dataset (assignment explicitly permits this).
- Used `reference_reply` as judge context, not an exact-match target.
- Used rubric-based LLM-judge scoring instead of string-overlap metrics (BLEU/ROUGE-style), since those reward surface similarity, not groundedness or correctness.
- Validated the judge against hand-scored examples before trusting it at scale.
- Required both a trivial and a simple baseline — each answers a different skeptical question about the headline number.
- Sampled the golden set randomly rather than cherry-picking easy examples.
- (Add more as they come up: taxonomy merge/split decisions, routing threshold choices, model/provider choice and cost tradeoff, multi-turn thread handling, etc.)

---

## 9. Target repo structure

```
repo/
├── README.md              # must reproduce headline results in <15 minutes
├── requirements.txt
├── data/
│   └── sample_subset.csv  # small subsample checked in, or a documented fetch step
├── src/
│   ├── preprocess.py
│   ├── classify_intent.py
│   ├── retrieve_grounding.py
│   ├── draft_reply.py
│   ├── route_decision.py
│   ├── run_pipeline.py
│   └── baselines/
│       ├── trivial.py
│       └── simple.py
├── eval/
│   ├── golden_set.jsonl
│   ├── judge.py
│   └── run_eval.py
└── report.md
```

**README requirement**: a reviewer with no prior context should be able to `pip install -r requirements.txt`, run one or two commands, and reproduce the headline numbers in under 15 minutes, on the checked-in subsample — no full-dataset processing, no manual steps.
