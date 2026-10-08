"""Analyzer typed-decision spec — SINGLE SOURCE OF TRUTH.

Every question each agent's intent analyzer asks about a query is defined
here, with the EXACT instructions + criteria strings used both when
building training data (train/build_dataset.py) and when serving
(src/analyzers.py).

Prompt/label mismatch between training and serving silently degrades
confidence. This is the direct lesson from the intent-router-laya
2026-10-07 confidence-gap incident: training and serving must import
these strings verbatim. Do NOT rephrase them anywhere else.

Each agent owns one Choice question (its action taxonomy) plus Noul
flags (needs-x probabilities). Agents:

  billing : billing_action (6)  + 4 noul
  orders  : order_action   (6)  + 3 noul
  support : support_topic   (5)  + 2 noul
  account : account_action  (5)  + 2 noul
  sales   : sales_intent    (5)  + 2 noul
"""
from __future__ import annotations

#: Agent id -> spec. Keys are stable and also used as the directory names
#: for per-agent datasets (train/data/<agent>.jsonl) and checkpoints
#: (models/<agent>/).
AGENT_IDS = ("billing", "orders", "support", "account", "sales")

ANALYZERS: dict[str, dict] = {
    "billing": {
        "choice_question": "billing_action",
        "questions": {
            "billing_action": {
                "type": "choice",
                "instructions": "What billing action does this customer need?",
                "criteria": {
                    "check_balance": "asking for current balance, amount due, invoice total",
                    "make_payment": "wants to pay a bill, pay now, set up payment",
                    "dispute_charge": "disputes a charge, says a charge is wrong or unknown",
                    "refund_request": "asks for money back, a refund, reversal of a charge",
                    "update_payment_method": "change card, update payment details, add a payment method",
                    "billing_info": "general billing questions that fit none of the above",
                },
            },
            "needs_payment_method": {
                "type": "noul",
                "instructions": "Does this request require the customer's payment credentials to proceed?",
            },
            "is_dispute": {
                "type": "noul",
                "instructions": "Is the customer disputing a charge?",
            },
            "amount_mentioned": {
                "type": "noul",
                "instructions": "Does the query mention a specific amount of money?",
            },
            "is_recurring": {
                "type": "noul",
                "instructions": "Is this about a subscription or recurring charge?",
            },
        },
    },
    "orders": {
        "choice_question": "order_action",
        "questions": {
            "order_action": {
                "type": "choice",
                "instructions": "What order action does this customer need?",
                "criteria": {
                    "track": "wants to track a shipment or know delivery status",
                    "cancel": "wants to cancel an order",
                    "modify": "wants to change items, quantity, or details of an order",
                    "return": "wants to return an item or start a return",
                    "reorder": "wants to buy the same items again",
                    "order_info": "general order questions that fit none of the above",
                },
            },
            "has_order_id": {
                "type": "noul",
                "instructions": "Does the query include an order identifier?",
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "Is this time-sensitive (needed urgently, deadline, gift)?",
            },
            "needs_address": {
                "type": "noul",
                "instructions": "Does handling this need the customer's shipping address?",
            },
        },
    },
    "support": {
        "choice_question": "support_topic",
        "questions": {
            "support_topic": {
                "type": "choice",
                "instructions": "What support topic is this customer asking about?",
                "criteria": {
                    "technical_issue": "something is broken, an error, a bug, not working",
                    "how_to": "asks how to do something, instructions, guidance",
                    "account_access": "login problems, password reset, locked out",
                    "complaint": "expresses dissatisfaction, wants to complain",
                    "feedback": "shares an opinion, suggestion, or praise",
                },
            },
            "needs_troubleshooting": {
                "type": "noul",
                "instructions": "Does this need step-by-step troubleshooting?",
            },
            "is_escalation_worthy": {
                "type": "noul",
                "instructions": "Should this be escalated to a senior specialist?",
            },
        },
    },
    "account": {
        "choice_question": "account_action",
        "questions": {
            "account_action": {
                "type": "choice",
                "instructions": "What account action does this customer need?",
                "criteria": {
                    "update_info": "change name, email, address, profile details",
                    "delete_account": "delete, close, or deactivate the account",
                    "change_plan": "switch plan, upgrade, downgrade subscription tier",
                    "security": "security settings, two-factor, suspicious activity",
                    "verify_identity": "prove identity, verification code, ID check",
                },
            },
            "is_irreversible": {
                "type": "noul",
                "instructions": "Is this a destructive or irreversible action?",
            },
            "needs_identity_verification": {
                "type": "noul",
                "instructions": "Must the customer's identity be verified before proceeding?",
            },
        },
    },
    "sales": {
        "choice_question": "sales_intent",
        "questions": {
            "sales_intent": {
                "type": "choice",
                "instructions": "What is this customer's sales intent?",
                "criteria": {
                    "pricing": "asks about prices, costs, fees",
                    "compare_plans": "compares plans, tiers, or options",
                    "demo_request": "wants a demo or walkthrough",
                    "trial": "asks about a free trial",
                    "purchase": "ready to buy, asks how to purchase or sign up",
                },
            },
            "is_ready_to_buy": {
                "type": "noul",
                "instructions": "Does this customer look ready to buy now?",
            },
            "needs_human_sales": {
                "type": "noul",
                "instructions": "Should a human salesperson take over this conversation?",
            },
        },
    },
}


def agent_spec(agent: str) -> dict:
    """Return the analyzer spec for an agent id (KeyError if unknown)."""
    return ANALYZERS[agent]


def choice_question(agent: str) -> str:
    return ANALYZERS[agent]["choice_question"]


def choice_options(agent: str) -> list[str]:
    """Choice option names in stable order (label ids for training)."""
    qid = ANALYZERS[agent]["choice_question"]
    return list(ANALYZERS[agent]["questions"][qid]["criteria"].keys())


def noul_questions(agent: str) -> list[str]:
    """Noul question ids in stable order."""
    return [
        qid for qid, q in ANALYZERS[agent]["questions"].items()
        if q["type"] == "noul"
    ]


def question_ids(agent: str) -> list[str]:
    """All question ids for an agent in stable order (choice first)."""
    spec = ANALYZERS[agent]
    cq = spec["choice_question"]
    return [cq] + [qid for qid in spec["questions"] if qid != cq]
