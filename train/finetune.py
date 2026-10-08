"""Fine-tune one agent's intent analyzer: multi-task encoder.

Architecture: a ModernBERT encoder (same family as intent-router-laya's
train/finetune.py) with one classification head per typed decision:
  * choice head -> cross-entropy over the agent's action taxonomy
  * noul heads  -> binary cross-entropy (one logit per Noul question)

Loss = CE(choice) + mean(BCE(noul)). After training, each head gets its
own temperature fitted on the held-out split (calibration — the property
that makes confidence thresholds trustworthy, cf. the router's RLCD
report).

The exported checkpoint plugs into src/analyzers.py, which serves the
same typed-decision contract the dataset was built from
(src/analyzer_questions.py — the single source of truth).

Usage:
    python train/build_dataset.py --agent billing --out train/data
    python train/finetune.py --agent billing --data train/data/billing.jsonl \\
        --out models/billing
    ANALYZER_CHECKPOINT_DIR=models uvicorn src.app:app

Requirements: torch, transformers, scikit-learn, accelerate, pyyaml
A CPU can train this (small data, small model); a GPU is faster.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import (  # noqa: E402
    AGENT_IDS,
    choice_options,
    choice_question,
    noul_questions,
)

# Default: local copy (the sandbox egress proxy breaks huggingface_hub's
# httpx client; the weights were fetched with curl into model-hub/).
# Pass --model answerdotai/ModernBERT-base on a machine with direct
# hub access.
MODEL_ID = str(Path(__file__).resolve().parent.parent
               / "model-hub" / "ModernBERT-base")

_CHOICE_QID = {
    "billing": "billing_action", "orders": "order_action",
    "support": "support_topic", "account": "account_action",
    "sales": "sales_intent",
}


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=list(AGENT_IDS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from src.analyzer_model import MultiTaskAnalyzer

    agent = args.agent
    cqid = _CHOICE_QID[agent]
    choice_labels = choice_options(agent)   # stable order from the spec
    noul_qids = noul_questions(agent)       # stable order from the spec
    label2id = {l: i for i, l in enumerate(choice_labels)}

    records = load_records(Path(args.data))
    for r in records:
        assert r[cqid] in label2id, f"unknown label {r[cqid]!r}"
        for n in noul_qids:
            assert n in r, f"missing noul {n}"
    print(f"{agent}: {len(records)} records, choice={choice_labels}, noul={noul_qids}")

    strat = [r[cqid] for r in records]
    train_recs, val_recs = train_test_split(
        records, test_size=0.2, random_state=args.seed, stratify=strat)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(recs):
        enc = tokenizer([r["query"] for r in recs], truncation=True,
                        padding=True, max_length=128)
        enc["choice_labels"] = [label2id[r[cqid]] for r in recs]
        enc["noul_labels"] = [[float(r[n]) for n in noul_qids] for r in recs]
        return enc

    train_enc, val_enc = encode(train_recs), encode(val_recs)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

    model = MultiTaskAnalyzer(args.model, len(choice_labels), len(noul_qids))

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )

    # Trainer needs a single "logits" key; wrap the multi-output model.
    # NOTE: the forward signature must list argument names explicitly —
    # Trainer filters batch keys against it, and **kwargs makes it drop
    # everything ("The batch received was empty").
    class TrainerWrapper(nn.Module):
        def __init__(self, core, n_choice):
            super().__init__()
            self.core = core
            self.n_choice = n_choice

        def forward(self, input_ids, attention_mask,
                    choice_labels=None, noul_labels=None):
            o = self.core(input_ids=input_ids, attention_mask=attention_mask,
                          choice_labels=choice_labels,
                          noul_labels=noul_labels)
            # concat choice + noul logits; split back for the loss below
            logits = torch.cat([o["choice_logits"], o["noul_logits"]], dim=-1)
            if "loss" in o:
                return {"loss": o["loss"], "logits": logits}
            return {"logits": logits}

    wrapper = TrainerWrapper(model, len(choice_labels))

    # custom compute: Trainer calls model(**inputs) and reads ["loss"]
    trainer = Trainer(model=wrapper, args=training_args,
                      train_dataset=DictDataset(train_enc),
                      eval_dataset=DictDataset(val_enc))
    trainer.train()

    # --- per-head temperature scaling on the validation split ---
    core = wrapper.core
    core.eval()
    choice_logits_list, noul_logits_list = [], []
    choice_t, noul_t = [], []
    with torch.no_grad():
        for i in range(0, len(val_enc["input_ids"]), 32):
            batch = {k: torch.tensor(v[i:i + 32]) for k, v in val_enc.items()
                     if k not in ("choice_labels", "noul_labels")}
            o = core(**batch)
            choice_logits_list.append(o["choice_logits"])
            noul_logits_list.append(o["noul_logits"])
            choice_t.append(torch.tensor(val_enc["choice_labels"][i:i + 32]))
            noul_t.append(torch.tensor(val_enc["noul_labels"][i:i + 32],
                                       dtype=torch.float32))
    cl = torch.cat(choice_logits_list)
    nl = torch.cat(noul_logits_list)
    ct = torch.cat(choice_t)
    nt = torch.cat(noul_t)

    def fit_temp(logits, targets, is_bce=False):
        # Optimize in log-space so the temperature stays positive.
        # (Plain LBFGS on temp can overshoot past 0 into negative
        # territory when the model is very confident, producing a
        # degenerate negative temperature — seen 2026-10-08.)
        log_temp = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            temp = torch.exp(log_temp)
            if is_bce:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits / temp, targets)
            else:
                loss = torch.nn.functional.cross_entropy(logits / temp, targets)
            loss.backward()
            return loss

        opt.step(closure)
        temp = float(torch.exp(log_temp).detach())
        # sanity clamp: a temperature outside [0.05, 10] means the fit
        # found a degenerate solution; fall back to uncalibrated.
        if not (0.05 <= temp <= 10.0):
            log.warning("degenerate temperature %.3f, using 1.0", temp)
            return 1.0
        return temp

    temp_choice = fit_temp(cl, ct)
    # one temperature per noul head (independent binary decisions)
    temp_noul = [fit_temp(nl[:, j], nt[:, j], is_bce=True)
                 for j in range(len(noul_qids))]
    print(f"fitted temperatures: choice={temp_choice:.3f} "
          f"noul={[f'{t:.3f}' for t in temp_noul]}")

    # --- eval report ---
    with torch.no_grad():
        preds = (cl / temp_choice).argmax(dim=-1)
        acc = (preds == ct).float().mean().item()
        conf = torch.softmax(cl / temp_choice, dim=-1).max(dim=-1).values.mean().item()
        noul_probs = torch.sigmoid(nl / torch.tensor(temp_noul))
        noul_acc = (((noul_probs >= 0.5).float() == nt).float().mean().item())
    print(f"val choice accuracy: {acc:.3f} | mean confidence: {conf:.3f} | "
          f"val noul accuracy: {noul_acc:.3f}")

    # --- export (plain HF format: encoder + heads as one state dict) ---
    out.mkdir(parents=True, exist_ok=True)
    torch.save(core.state_dict(), out / "pytorch_model.bin")
    core.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "calibration.json").write_text(json.dumps({
        "agent": agent,
        "choice_question": cqid,
        "choice_labels": choice_labels,
        "noul_questions": noul_qids,
        "temperature_choice": temp_choice,
        "temperatures_noul": temp_noul,
        "val_choice_accuracy": acc,
        "val_noul_accuracy": noul_acc,
        "mean_confidence": conf,
        "base_model": args.model,
    }, indent=2))
    print(f"exported -> {out}")


if __name__ == "__main__":
    main()
