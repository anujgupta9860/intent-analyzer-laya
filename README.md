# Intent Analyzer (per-agent System 1)

Deeper-level companion to [intent-router-laya](https://github.com/anujgupta9860/intent-router-laya).

The intent router's System 1 makes 11 coarse routing decisions per query
(intent, worker_agent, skill_required, …). That's enough to *route* — not
enough to *act*. Each worker agent gets its own fine-tuned **intent
analyzer**: a second, narrower System 1 that answers agent-specific typed
decisions before the worker executes.

## Architecture

```
User query
  │
  ▼
Intent Router (System 1: 11 coarse decisions)
  worker_agent + skill_required + query
  │
  ▼
Agent's Intent Analyzer (System 1: agent-specific decisions)
  billing_action / order_action / support_topic / … + noul flags
  │
  ├── confident ──► refined action + flags ──► worker executes
  └── unsure ──► needs_human ──► escalate (human / router System 2)
```

One ModernBERT encoder per agent (multi-task: one choice head + one
binary head per Noul flag), per-head temperature calibration, same
typed-decision (Choice / Noul) pattern and single-source-of-truth
discipline as the router.

| Agent | Choice question | Noul flags |
|---|---|---|
| billing | billing_action (6) | needs_payment_method, is_dispute, amount_mentioned, is_recurring |
| orders | order_action (6) | has_order_id, is_urgent, needs_address |
| support | support_topic (5) | needs_troubleshooting, is_escalation_worthy |
| account | account_action (5) | is_irreversible, needs_identity_verification |
| sales | sales_intent (5) | is_ready_to_buy, needs_human_sales |

## Quick start

```bash
pip install -r requirements.txt

# 1. build per-agent datasets (synthetic, deterministic heuristics)
python train/build_dataset.py --out train/data --repeat 6

# 2. train an agent's analyzer (CPU-friendly)
python train/finetune.py --agent billing \
    --data train/data/billing.jsonl --out models/billing

# 3. serve
ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app --port 8081

# 4. run the contract tests
python -m pytest tests/ -q
```

## How each agent accesses its model

Three ways, from most to least abstracted:

**1. Via the analyzer service (recommended).** A worker agent never touches
model files — it calls the analyzer service with its agent name:

```bash
curl -s -X POST localhost:8081/analyze \
  -H 'content-type: application/json' \
  -d '{"query": "I want a refund for the $25 overcharge", "worker_agent": "billing"}' \
  | python3 -m json.tool
# -> billing_action: refund_request @ 0.781, amount_mentioned, is_dispute, ...
```

The service lazy-loads each agent's checkpoint on first use and validates
it against the question spec (`src/analyzer_questions.py`) before serving —
a checkpoint trained for the wrong agent is rejected, not silently served.

**2. GCS download (deployments).** `entrypoint.py` pulls every agent's
checkpoint from GCS at container startup (skips agents already present):

```bash
# env knobs
ANALYZER_CHECKPOINT_GCS=gs://laya-checkpoints-anuj/intent-analyzer/
ANALYZER_CHECKPOINT_DIR=/srv/app/models
ANALYZER_AGENTS=billing,orders,support,account,sales

python entrypoint.py   # downloads, then execs uvicorn
```

Trained checkpoints (upload in progress):

| Agent | GCS path |
|---|---|
| billing | `gs://laya-checkpoints-anuj/intent-analyzer/billing/` |
| orders | `gs://laya-checkpoints-anuj/intent-analyzer/orders/` |
| support | `gs://laya-checkpoints-anuj/intent-analyzer/support/` |
| account | `gs://laya-checkpoints-anuj/intent-analyzer/account/` |
| sales | `gs://laya-checkpoints-anuj/intent-analyzer/sales/` |

Each holds: `pytorch_model.bin`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `calibration.json` (~570MB per agent).

**3. Direct local access.** Download once, point at it:

```bash
gsutil -m cp -r gs://laya-checkpoints-anuj/intent-analyzer/billing ./models/billing
ANALYZER_CHECKPOINT_DIR=./models uvicorn src.app:app --port 8081
```

Or load in Python without the service:

```python
from src.analyzers import AnalyzerRegistry
reg = AnalyzerRegistry("models")          # one subdir per agent
out = reg.get("billing").analyze("I want a refund for the $25 overcharge")
print(out["billing_action"], out["confidence"])
```

## Curl examples

```bash
# health: which analyzers have checkpoints
curl -s localhost:8081/health | python3 -m json.tool

# analyze: router hands off to the billing analyzer
curl -s -X POST localhost:8081/analyze \
  -H 'content-type: application/json' \
  -d '{"query": "I want a refund for the $25 overcharge",
       "worker_agent": "billing",
       "router_context": {"intent": "billing_inquiry",
                          "skill_required": "refund_process"}}' \
  | python3 -m json.tool
# -> billing_action: refund_request, amount_mentioned: 1.0, ...

# feedback: low-confidence decisions awaiting human review
curl -s "localhost:8081/feedback/pending?agent=billing" | python3 -m json.tool
curl -s -X POST localhost:8081/feedback/review/<id> \
  -H 'content-type: application/json' -d '{"decision": "approved"}'
```

## Integration demo

```bash
# simulated router output -> real trained analyzers
python demo/router_to_analyzer.py
```

## Layout

```
src/analyzer_questions.py  # single source of truth: all 5 agents' questions
src/analyzer_model.py      # shared multi-task encoder architecture
src/analyzers.py           # per-agent analyzer clients + registry
src/app.py                 # FastAPI: /analyze, /health, /feedback/*
src/feedback.py            # RLCD feedback log + human review gate
src/config.py
train/build_dataset.py     # synthetic per-agent datasets
train/finetune.py          # multi-task fine-tune + per-head calibration
demo/router_to_analyzer.py # router -> analyzer integration demo
tests/test_contract.py     # train<->serve contract tests
docs/SDR.md                # design doc
openapi.yaml               # API spec
```

## Notes

- Mirrors intent-router-laya's typed-decision pattern and the
  train/serve single-source-of-truth rule (the 2026-10-07 lesson).
- `is_irreversible >= 0.8` on the account analyzer forces human
  confirmation regardless of confidence.
- No secrets, no real user data. Synthetic training data only.
