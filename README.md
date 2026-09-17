# AmazonHelp Support Agent

A 4-stage AI customer-support agent (intent classification → grounding retrieval → reply drafting → routing) built on Amazon's real Twitter support history, plus an eval harness that scores it against a hand-labeled golden set. See `project_brief.md` for the spec and `decision_log.md` for the reasoning behind every non-obvious call made while building it.

## Setup

Tested on Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` pulls in `torch`, so a cold install can take a few minutes on a fresh machine — that's the slowest part of setup, not anything below.

Copy the env template and add your own Groq key (used by `classify_intent`, `draft_reply`, and the LLM judge — get one free at [console.groq.com](https://console.groq.com)):

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=<your-key>
```

`.env` is gitignored — your key never gets committed.

## Reproduce the headline results (< 15 minutes)

Everything below runs against data already checked into `data/processed/` — the representative subsample (`sample_subset.csv`), the retrieval corpus (`grounding_pool.jsonl`), the hand-labeled 150-row golden set (`golden_set_worksheet.csv`), and the persisted 30-row reply-quality run (`eval_replies_results.json`). **No full-dataset processing and no manual labeling step is required** — that's all already done and saved.

Run these four commands from the project root, in order:

```bash
# 1. Intent + routing metrics — accuracy, per-class F1, confusion matrix,
#    escalate precision/recall, missed-escalation list.
#    Reads golden_set_worksheet.csv only. NO API calls. Instant.
python src/run_eval.py

# 2. Trivial + simple baselines for the same two metrics, for comparison.
#    Pure computation on the same worksheet. NO API calls. Instant.
python src/baselines.py

# 3. Reply-quality judge scores (trivial vs simple vs real system,
#    4-dimension rubric, 1-5 scale).
#    Reads the persisted eval_replies_results.json cache — reproduces the
#    same numbers with NO new API calls, in a couple of seconds.
#    (If that cache is ever missing/deleted, this script regenerates it
#    live instead: 30 draft_reply calls + 90 judge calls = 120 Groq calls,
#    paced under the free-tier 30/min limit -- budget ~7-9 minutes for that
#    fallback path specifically.)
python src/eval_replies.py

# 4. Judge-validation agreement (human vs LLM-judge scores on a 15-row
#    subset) -- the number that backs the judge's credibility in the report.
#    Reads the already-hand-filled judge_validation_worksheet.csv. NO API
#    calls. Instant.
python src/score_judge_validation.py
```

Total wall-clock time: install (a few minutes) + these four commands (seconds each, cached path) — comfortably under 15 minutes.

## See the live agent work (optional, makes real API calls)

To watch the actual 4-stage pipeline run end-to-end on real examples rather than just reading precomputed metrics:

```bash
python src/run_pipeline.py
```

This runs `classify_intent → retrieve_grounding → draft_reply → decide_route` on 10 real customer messages sampled from `sample_subset.csv` and prints the intent, grounding match, drafted reply, and routing decision for each. **This always makes live Groq calls** (~20 total: one `classify_intent` + one `draft_reply` per row) — typically well under a minute, but not cached, so re-running it costs API calls each time. The individual stage scripts (`classify_intent.py`, `retrieve_grounding.py`, `draft_reply.py`, `decide_route.py`) have the same kind of `__main__` smoke test if you want to see one stage in isolation.

## What's checked in vs. what you'd need to regenerate

| Artifact | Checked in? | To regenerate |
|---|---|---|
| `data/processed/sample_subset.csv`, `grounding_pool.jsonl`, `eval_pool.jsonl` | Yes | `python src/preprocess.py` — **not needed for reproduction**; requires the raw 493MB `data/twcs/twcs.csv` and takes a few minutes even from a cached intermediate stage |
| `data/processed/golden_set_worksheet.csv` (150 rows, fully labeled) | Yes | `python src/build_golden_set.py` (150 live Groq calls, ~9 min) regenerates `suggested_intent`/`suggested_route`, but `true_intent`/`true_route`/`route_reason` were hand-labeled — that step isn't automatable |
| `data/processed/eval_replies_results.json` / `eval_replies_drafts.json` | Yes | `python src/eval_replies.py` (self-healing: reuses whatever's cached, only calls the API for missing rows) |
| `data/processed/judge_validation_worksheet.csv` (LLM scores) / `judge_validation_llm_scores.json` | Yes | `python src/build_judge_validation.py` (no API calls) regenerates the worksheet skeleton, but the `groundedness`/`correctness_safety`/`tone`/`actionability` columns are hand-scored — not automatable |

## Project layout

```
data/
  twcs/twcs.csv              raw Kaggle dataset (not checked in)
  interim/                   regenerable preprocessing checkpoints (gitignored)
  processed/                 checked-in: subsample, pools, golden set, eval results
src/
  preprocess.py              Stages A-E: raw CSV -> grounding_pool / eval_pool
  classify_intent.py         Stage 1 (Groq LLM call)
  retrieve_grounding.py      Stage 2 (local embeddings, no LLM call)
  draft_reply.py             Stage 3 (Groq LLM call)
  decide_route.py            Stage 4 (rule-based, no LLM call)
  run_pipeline.py            chains all four stages
  llm_client.py               shared Groq client config
  sample_rows.py             shared 10-row test fixture used by every stage's smoke test
  build_golden_set.py        samples + prefills the 150-row labeling worksheet
  run_eval.py                golden-set loading/validation + intent/routing metrics
  baselines.py                trivial/simple baselines for comparison
  llm_judge.py                LLM-as-judge rubric scoring
  eval_replies.py            Part 2: real/trivial/simple replies + judge scoring
  build_judge_validation.py  15-row human-scoring worksheet for judge validation
  score_judge_validation.py  human-vs-judge agreement (final report number)
decision_log.md              why behind every non-obvious decision
project_brief.md             the assignment spec
```
