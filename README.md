# AmazonHelp Support Agent

A 4-stage AI support agent (classify → retrieve → draft → route) for AmazonHelp's Twitter support history, with a full eval harness scored against a hand-labeled golden set.

See `project_brief.md` for the spec, `decision_log.md` for the reasoning behind every non-obvious call.

## Setup

Tested on Python 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installing `sentence-transformers` pulls in `torch` — the first install takes a few minutes, that's normal.

```bash
cp .env.example .env
# add your Groq key (free at console.groq.com): GROQ_API_KEY=<your-key>
```

## Reproduce the results (< 15 min)

Everything needed is already in `data/processed/` — no raw dataset processing or manual labeling required.

```bash
python src/run_eval.py               # intent + routing metrics — instant, no API calls
python src/baselines.py              # trivial + simple baselines — instant, no API calls
python src/eval_replies.py           # reply-quality judge scores — cached, a few seconds
python src/score_judge_validation.py # human-vs-judge agreement — instant, no API calls
```

`eval_replies.py` reads from a saved cache. If that cache is ever missing, it falls back to regenerating live — 120 Groq calls, ~7-9 minutes.

**Total time:** a few minutes to install, seconds to run everything else.

## Watch the live agent (optional — makes real API calls)

```bash
python src/run_pipeline.py
```

Runs all 4 stages on 10 real messages, prints each result. Not cached — costs ~20 Groq calls per run.

## What's checked in

| File | Regenerate with | Note |
|---|---|---|
| `sample_subset.csv`, `grounding_pool.jsonl`, `eval_pool.jsonl` | `preprocess.py` | needs the raw dataset; not required for reproduction |
| `golden_set_worksheet.csv` | `build_golden_set.py` | true labels were hand-labeled, not automatable |
| `eval_replies_results.json` | `eval_replies.py` | self-healing, only calls the API for missing rows |
| `judge_validation_worksheet.csv` | `build_judge_validation.py` | human scores aren't automatable |

## Project layout

```
data/
  twcs/twcs.csv              raw dataset (not checked in)
  interim/                   regenerable preprocessing checkpoints
  processed/                 checked in: subsample, pools, golden set, eval results

src/
  preprocess.py              raw CSV -> grounding_pool / eval_pool
  classify_intent.py         Stage 1: intent (Groq)
  retrieve_grounding.py      Stage 2: retrieval (local embeddings)
  draft_reply.py             Stage 3: reply drafting (Groq)
  decide_route.py            Stage 4: routing (rule-based)
  run_pipeline.py            chains all 4 stages
  llm_client.py              shared Groq client
  sample_rows.py             shared test fixture

  build_golden_set.py        builds the 150-row labeling worksheet
  run_eval.py                intent/routing metrics
  baselines.py               trivial/simple baselines
  llm_judge.py                LLM-as-judge scoring
  eval_replies.py            reply drafting + judge scoring
  build_judge_validation.py  15-row human-scoring worksheet
  score_judge_validation.py  human-vs-judge agreement

decision_log.md              reasoning behind every non-obvious decision
project_brief.md             the assignment spec
```