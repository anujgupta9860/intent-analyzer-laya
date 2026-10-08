"""Build per-agent analyzer training datasets.

For each agent in src/analyzer_questions.py, emits JSONL with one record
per seed example, repeated --repeat times with template paraphrasing:

    {"query": "...", "billing_action": "refund_request",
     "needs_payment_method": 0.0, "is_dispute": 0.0,
     "amount_mentioned": 1.0, "is_recurring": 0.0}

Labels are derived deterministically from the SEED example (BEFORE
template paraphrasing) so template noise never changes a label — the
same discipline as intent-router-laya's train/build_dataset.py.

Usage:
    python train/build_dataset.py --out train/data --repeat 6
    python train/build_dataset.py --agent billing --out train/data --repeat 6
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import (  # noqa: E402
    AGENT_IDS,
    choice_options,
    noul_questions,
)

# Deterministic paraphrase templates: {q} is the seed query.
_TEMPLATES = [
    "{q}",
    "{q} Please help.",
    "Hi, {q}",
    "Hello, {q}",
    "Quick question: {q}",
    "Can you help me? {q}",
]

_AMOUNT_RE = re.compile(r"\$\d[\d,]*(\.\d{1,2})?|\b\d+(\.\d{1,2})?\s*(dollars|usd)\b", re.I)
_ORDER_ID_RE = re.compile(r"#\d+|order\s*(number|id|no\.?|#)?\s*\d{3,}", re.I)


def _kw(*words: str):
    pat = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.compile(pat, re.I)


# ---------------------------------------------------------------------------
# Seed examples: (choice_label, query). Noul flags come from _noul_rules.
# ---------------------------------------------------------------------------
_SEEDS: dict[str, list[tuple[str, str]]] = {
    "billing": [
        ("check_balance", "What is my current balance?"),
        ("check_balance", "How much do I owe?"),
        ("check_balance", "Show me my invoice total"),
        ("check_balance", "What is the amount due on my bill?"),
        ("check_balance", "Can I see my current charges?"),
        ("check_balance", "What's my outstanding balance?"),
        ("check_balance", "How much is my bill this month?"),
        ("check_balance", "Tell me what I owe"),
        ("make_payment", "I want to pay my bill now"),
        ("make_payment", "Pay my invoice"),
        ("make_payment", "Can I pay with my card?"),
        ("make_payment", "Please process my payment"),
        ("make_payment", "I'd like to settle my bill"),
        ("make_payment", "Charge my card for the balance"),
        ("make_payment", "Make a payment on my account"),
        ("make_payment", "Pay the full amount due"),
        ("dispute_charge", "I don't recognize this charge"),
        ("dispute_charge", "There's a wrong charge on my bill"),
        ("dispute_charge", "This fee is incorrect"),
        ("dispute_charge", "I was charged $129.99 twice"),
        ("dispute_charge", "Dispute the $50 charge"),
        ("dispute_charge", "That charge is not mine"),
        ("dispute_charge", "I want to contest this fee"),
        ("dispute_charge", "Remove this unauthorized charge"),
        ("refund_request", "I want a refund"),
        ("refund_request", "Can I get my money back?"),
        ("refund_request", "Please refund my last payment"),
        ("refund_request", "Reverse the charge from yesterday"),
        ("refund_request", "I need a refund for the subscription"),
        ("refund_request", "Give me a refund"),
        ("refund_request", "Refund the $25 overcharge"),
        ("refund_request", "Return my payment"),
        ("refund_request", "Why was I charged for the subscription again?"),
        ("billing_info", "Cancel my monthly subscription"),
        ("update_payment_method", "Update my credit card"),
        ("update_payment_method", "Change my payment method"),
        ("update_payment_method", "Add a new card to my account"),
        ("update_payment_method", "My card expired, update it"),
        ("update_payment_method", "Switch to a different payment method"),
        ("update_payment_method", "Remove my old card"),
        ("update_payment_method", "Use my new debit card instead"),
        ("update_payment_method", "Update billing details"),
        ("billing_info", "When is my bill due?"),
        ("billing_info", "What payment methods do you accept?"),
        ("billing_info", "Explain my invoice"),
        ("billing_info", "How does billing work?"),
        ("billing_info", "Where can I download my receipt?"),
        ("billing_info", "What is the billing cycle?"),
        ("billing_info", "Do you offer paperless billing?"),
        ("billing_info", "Who do I contact about billing?"),
    ],
    "orders": [
        ("track", "Where is my order?"),
        ("track", "Track my package"),
        ("track", "What's the delivery status of order #12345?"),
        ("track", "Has my order shipped?"),
        ("track", "When will my package arrive?"),
        ("track", "Track shipment #98765"),
        ("track", "Where is my delivery?"),
        ("track", "Check status of my recent order"),
        ("cancel", "Cancel my order"),
        ("cancel", "I want to cancel order #54321"),
        ("cancel", "Stop my order before it ships"),
        ("cancel", "Please cancel the purchase"),
        ("cancel", "Can I still cancel?"),
        ("cancel", "Cancel my recent order"),
        ("cancel", "Don't ship my order, cancel it"),
        ("cancel", "I changed my mind, cancel"),
        ("modify", "Change the quantity on my order"),
        ("modify", "Update items in my order #111"),
        ("modify", "Can I add an item to my order?"),
        ("modify", "Modify my shipping speed"),
        ("modify", "Change the size I ordered"),
        ("modify", "Edit my order details"),
        ("modify", "Swap the color on my order"),
        ("modify", "Update my order before shipping"),
        ("modify", "Change the delivery address on my order #333"),
        ("modify", "Ship to my office address instead"),
        ("track", "I need this by Friday, it's a gift"),
        ("track", "Rush my order please, it's urgent"),
        ("return", "I want to return this item"),
        ("return", "Start a return for order #222"),
        ("return", "How do I return a product?"),
        ("return", "Return my purchase"),
        ("return", "This arrived damaged, I want to return it"),
        ("return", "Send it back for a refund"),
        ("return", "What's your return policy?"),
        ("return", "I need to return the wrong size"),
        ("reorder", "Order the same items again"),
        ("reorder", "Reorder my last purchase"),
        ("reorder", "Buy the same thing again"),
        ("reorder", "Repeat my previous order"),
        ("reorder", "I want another one of those"),
        ("reorder", "Restock the same products"),
        ("reorder", "Place the same order again"),
        ("reorder", "Reorder from last month"),
        ("order_info", "How long does shipping take?"),
        ("order_info", "What are the delivery options?"),
        ("order_info", "Do you ship internationally?"),
        ("order_info", "Where is your warehouse?"),
        ("order_info", "How do I apply a promo code?"),
        ("order_info", "What is the shipping cost?"),
        ("order_info", "Can I get a gift receipt?"),
        ("order_info", "Do you offer express delivery?"),
    ],
    "support": [
        ("technical_issue", "The app keeps crashing"),
        ("technical_issue", "I get error 500 when I log in"),
        ("technical_issue", "My dashboard won't load"),
        ("technical_issue", "There's a bug in the checkout"),
        ("technical_issue", "The feature is broken"),
        ("technical_issue", "It freezes when I upload"),
        ("technical_issue", "Nothing works after the update"),
        ("technical_issue", "The page shows a blank screen"),
        ("how_to", "How do I export my data?"),
        ("how_to", "How can I change my password?"),
        ("how_to", "Show me how to set up alerts"),
        ("how_to", "How do I invite my team?"),
        ("how_to", "Where do I find the settings?"),
        ("how_to", "How to cancel a subscription?"),
        ("how_to", "Teach me how to use filters"),
        ("how_to", "How do I connect my calendar?"),
        ("account_access", "I can't log in"),
        ("account_access", "I forgot my password"),
        ("account_access", "My account is locked"),
        ("account_access", "Reset my password please"),
        ("account_access", "I'm locked out of my account"),
        ("account_access", "Can't access my account"),
        ("account_access", "Login isn't working"),
        ("account_access", "Help me recover my account"),
        ("complaint", "Your service is terrible"),
        ("complaint", "I'm very disappointed with support"),
        ("complaint", "This is unacceptable"),
        ("complaint", "I want to complain about the outage"),
        ("complaint", "Worst experience ever"),
        ("complaint", "Nobody helped me for days"),
        ("complaint", "I'm frustrated with these bugs"),
        ("complaint", "Your product keeps failing me"),
        ("feedback", "Great job on the new feature!"),
        ("feedback", "I love the new design"),
        ("feedback", "Here's a suggestion: dark mode"),
        ("feedback", "You should add keyboard shortcuts"),
        ("feedback", "Thanks for the quick fix"),
        ("feedback", "The app is much faster now"),
        ("feedback", "I'd like to suggest an integration"),
        ("feedback", "Nice work, team!"),
    ],
    "account": [
        ("update_info", "Change my email address"),
        ("update_info", "Update my phone number"),
        ("update_info", "Change my name on the account"),
        ("update_info", "Update my address"),
        ("update_info", "Edit my profile"),
        ("update_info", "Change my username"),
        ("update_info", "Update my contact details"),
        ("update_info", "Fix the typo in my name"),
        ("delete_account", "Delete my account"),
        ("delete_account", "Close my account permanently"),
        ("delete_account", "Deactivate my profile"),
        ("delete_account", "Remove my account"),
        ("delete_account", "I want to leave, delete everything"),
        ("delete_account", "Erase my account data"),
        ("delete_account", "Cancel and delete my account"),
        ("delete_account", "Shut down my account"),
        ("change_plan", "Upgrade to the pro plan"),
        ("change_plan", "Downgrade my subscription"),
        ("change_plan", "Switch to the annual plan"),
        ("change_plan", "Change my plan tier"),
        ("change_plan", "Move to the team plan"),
        ("change_plan", "I want a cheaper plan"),
        ("change_plan", "Upgrade my account"),
        ("change_plan", "Change from monthly to yearly"),
        ("security", "Enable two-factor authentication"),
        ("security", "I see suspicious activity"),
        ("security", "Someone accessed my account"),
        ("security", "Change my security questions"),
        ("security", "Review my login sessions"),
        ("security", "My account may be hacked"),
        ("security", "Turn on security alerts"),
        ("security", "Check for unauthorized logins"),
        ("verify_identity", "Verify my identity"),
        ("verify_identity", "I need to prove who I am"),
        ("verify_identity", "Send me a verification code"),
        ("verify_identity", "Confirm my identity"),
        ("verify_identity", "Upload my ID for verification"),
        ("verify_identity", "Verify it's really me"),
        ("verify_identity", "Identity check please"),
        ("verify_identity", "Validate my account ownership"),
    ],
    "sales": [
        ("pricing", "How much does it cost?"),
        ("pricing", "What is the price of the pro plan?"),
        ("pricing", "Show me your pricing"),
        ("pricing", "Is there a discount?"),
        ("pricing", "What are the fees?"),
        ("pricing", "How much per user?"),
        ("pricing", "What's the cheapest option?"),
        ("pricing", "Do you have enterprise pricing?"),
        ("compare_plans", "Compare the basic and pro plans"),
        ("compare_plans", "What's the difference between tiers?"),
        ("compare_plans", "Which plan is right for me?"),
        ("compare_plans", "Basic vs premium?"),
        ("compare_plans", "Compare features across plans"),
        ("compare_plans", "How do plans differ?"),
        ("compare_plans", "Should I choose monthly or yearly?"),
        ("compare_plans", "What do I get with each plan?"),
        ("demo_request", "Can I get a demo?"),
        ("demo_request", "Schedule a walkthrough"),
        ("demo_request", "Show me a demo of the product"),
        ("demo_request", "I'd like to see it in action"),
        ("demo_request", "Book a demo call"),
        ("demo_request", "Can someone demo this for me?"),
        ("demo_request", "Demo the dashboard please"),
        ("demo_request", "Walk me through the features"),
        ("trial", "Is there a free trial?"),
        ("trial", "Can I try it for free?"),
        ("trial", "Start a trial"),
        ("trial", "How long is the trial?"),
        ("trial", "Sign me up for the trial"),
        ("trial", "Do you offer a pilot?"),
        ("trial", "Try before I buy?"),
        ("trial", "Free trial available?"),
        ("purchase", "I want to buy now"),
        ("purchase", "How do I sign up?"),
        ("purchase", "Take my money, let's do this"),
        ("purchase", "I'm ready to purchase"),
        ("purchase", "Where do I checkout?"),
        ("purchase", "Sign me up for pro"),
        ("purchase", "Let's get started today"),
        ("purchase", "I want to subscribe"),
    ],
}


def _noul_rules(agent: str, choice_label: str, query: str) -> dict[str, float]:
    """Deterministic noul labels from the SEED (pre-template) query."""
    q = query
    if agent == "billing":
        return {
            "needs_payment_method": 1.0 if choice_label in (
                "make_payment", "update_payment_method") or _kw(
                "pay", "card", "payment method").search(q) else 0.0,
            "is_dispute": 1.0 if choice_label == "dispute_charge" or _kw(
                "dispute", "wrong charge", "not mine", "unauthorized",
                "charged twice", "incorrect", "contest").search(q) else 0.0,
            "amount_mentioned": 1.0 if _AMOUNT_RE.search(q) else 0.0,
            "is_recurring": 1.0 if _kw(
                "subscription", "recurring", "monthly", "plan").search(q) else 0.0,
        }
    if agent == "orders":
        return {
            "has_order_id": 1.0 if _ORDER_ID_RE.search(q) else 0.0,
            "is_urgent": 1.0 if _kw(
                "urgent", "asap", "as soon as possible", "deadline",
                "gift", "needed by", "rush", "emergency").search(q) else 0.0,
            "needs_address": 1.0 if _kw(
                "address", "ship to", "deliver to",
                "shipping address").search(q) else 0.0,
        }
    if agent == "support":
        return {
            "needs_troubleshooting": 1.0 if choice_label == "technical_issue" or _kw(
                "error", "bug", "crash", "not working",
                "broken", "fix").search(q) else 0.0,
            "is_escalation_worthy": 1.0 if choice_label == "complaint" or _kw(
                "unacceptable", "worst", "terrible",
                "for days", "frustrated").search(q) else 0.0,
        }
    if agent == "account":
        return {
            "is_irreversible": 1.0 if choice_label == "delete_account" or _kw(
                "delete", "close", "deactivate", "erase",
                "permanently").search(q) else 0.0,
            "needs_identity_verification": 1.0 if choice_label in (
                "delete_account", "security", "verify_identity") or _kw(
                "verify", "verification", "identity",
                "prove").search(q) else 0.0,
        }
    if agent == "sales":
        return {
            "is_ready_to_buy": 1.0 if choice_label == "purchase" or _kw(
                "buy now", "ready to", "sign me up",
                "let's do this", "checkout",
                "get started today").search(q) else 0.0,
            "needs_human_sales": 1.0 if choice_label == "demo_request" or _kw(
                "enterprise", "custom quote",
                "talk to sales", "call me").search(q) else 0.0,
        }
    raise ValueError(f"unknown agent: {agent}")


def build_agent_records(agent: str, repeat: int, seed: int = 42) -> list[dict]:
    """Emit labeled records for one agent."""
    rng = random.Random(seed)
    choice_qid = {
        "billing": "billing_action", "orders": "order_action",
        "support": "support_topic", "account": "account_action",
        "sales": "sales_intent",
    }[agent]
    expected_noul = set(noul_questions(agent))
    records: list[dict] = []
    for choice_label, seed_query in _SEEDS[agent]:
        # Labels from the SEED — template noise never changes them.
        noul = _noul_rules(agent, choice_label, seed_query)
        assert set(noul) == expected_noul, (
            f"noul mismatch for {agent}: {set(noul)} != {expected_noul}")
        for _ in range(repeat):
            template = rng.choice(_TEMPLATES)
            records.append({
                "query": template.format(q=seed_query),
                choice_qid: choice_label,
                **noul,
            })
    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=None, choices=list(AGENT_IDS) + [None],
                        help="build one agent's dataset (default: all)")
    parser.add_argument("--out", default="train/data",
                        help="output directory (one <agent>.jsonl each)")
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = [args.agent] if args.agent else list(AGENT_IDS)
    total = 0
    for agent in agents:
        recs = build_agent_records(agent, args.repeat, args.seed)
        path = out_dir / f"{agent}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # label balance report
        cq = {"billing": "billing_action", "orders": "order_action",
              "support": "support_topic", "account": "account_action",
              "sales": "sales_intent"}[agent]
        bal = {}
        for r in recs:
            bal[r[cq]] = bal.get(r[cq], 0) + 1
        noul_sums = {n: sum(r[n] for r in recs) for n in noul_questions(agent)}
        print(f"{agent}: {len(recs)} records -> {path}")
        print(f"  choice balance: {bal}")
        print(f"  noul positives: {noul_sums}")
        total += len(recs)
    print(f"total: {total} records")


if __name__ == "__main__":
    main()
