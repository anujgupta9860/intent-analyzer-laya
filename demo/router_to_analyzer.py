"""End-to-end demo: intent router -> per-agent intent analyzer.

Simulates the router's 11-decision output (worker_agent + skill_required,
as produced by intent-router-laya) and feeds it into the matching trained
analyzer for the deeper, domain-specific typed decisions.

Usage:
    python demo/router_to_analyzer.py
    python demo/router_to_analyzer.py --query "I want a refund"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import AGENT_IDS  # noqa: E402
from src.analyzers import AnalyzerRegistry  # noqa: E402

# Simulated router output: query -> (worker_agent, skill_required).
# In production this comes from intent-router-laya's POST /route.
_ROUTER_SIM = [
    ("I want a refund for the $25 overcharge", "billing", "refund_process"),
    ("Where is my order #12345?", "orders", "order_tracking"),
    ("The app keeps crashing on login", "support", "knowledge_search"),
    ("Delete my account", "account", "account_modify"),
    ("How much does the pro plan cost?", "sales", "none"),
    ("I was charged twice this month", "billing", "billing_lookup"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None)
    parser.add_argument("--agent", default=None, choices=list(AGENT_IDS))
    parser.add_argument("--checkpoint-dir", default="models")
    args = parser.parse_args()

    registry = AnalyzerRegistry(args.checkpoint_dir)
    print(f"available analyzers: {registry.available()}\n")

    queries = ([(args.query, args.agent, "n/a")]
               if args.query and args.agent else _ROUTER_SIM)
    for query, worker_agent, skill in queries:
        print(f"query:          {query!r}")
        print(f"router ->       worker_agent={worker_agent} skill_required={skill}")
        if not registry.available().get(worker_agent):
            print(f"analyzer [{worker_agent}]: no checkpoint trained yet\n")
            continue
        d = registry.get(worker_agent).analyze(query)
        print(f"analyzer [{worker_agent}]:")
        print(f"  {d.choice_question} = {d.choice} "
              f"(confidence {d.choice_confidence})")
        for qid, prob in d.noul.items():
            print(f"  {qid} = {prob}")
        print(f"  needs_human = {d.needs_human} "
              f"({d.latency_ms} ms)\n")


if __name__ == "__main__":
    main()
