"""Shared multi-task analyzer architecture (train + serve).

One ModernBERT encoder, one classification head per typed decision:
  * choice_head -> logits over the agent's action taxonomy
  * noul_head   -> one logit per Noul question (sigmoid at serve time)

Both train/finetune.py and src/analyzers.py import this class so the
state dict layout is identical in both places. Do not rename attributes
— checkpoint compatibility depends on it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskAnalyzer(nn.Module):
    def __init__(self, base_id: str, n_choice: int, n_noul: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_id)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.choice_head = nn.Linear(h, n_choice)
        self.noul_head = nn.Linear(h, n_noul)

    def forward(self, input_ids, attention_mask,
                choice_labels=None, noul_labels=None):
        pooled = self.dropout(
            self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state[:, 0])
        choice_logits = self.choice_head(pooled)
        noul_logits = self.noul_head(pooled)
        out = {"choice_logits": choice_logits, "noul_logits": noul_logits}
        if choice_labels is not None:
            ce = nn.functional.cross_entropy(choice_logits, choice_labels)
            bce = nn.functional.binary_cross_entropy_with_logits(
                noul_logits, noul_labels.float())
            out["loss"] = ce + bce
        return out

    @classmethod
    def from_local_config(cls, config, n_choice: int, n_noul: int
                          ) -> "MultiTaskAnalyzer":
        """Build the architecture from a local HF config (no hub download);
        caller loads the state dict afterwards."""
        from transformers import AutoModel
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.encoder = AutoModel.from_config(config)
        h = config.hidden_size
        obj.dropout = nn.Dropout(0.1)
        obj.choice_head = nn.Linear(h, n_choice)
        obj.noul_head = nn.Linear(h, n_noul)
        return obj
