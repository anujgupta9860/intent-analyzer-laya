"""Per-agent intent analyzers (serving).

Each analyzer loads a checkpoint exported by train/finetune.py and
answers its agent's typed decisions for a query. Question ids, choice
option order, and instructions all come from src/analyzer_questions.py —
the single source of truth shared with training.

Response contract mirrors the router's SystemOneDecision shape so the
router -> analyzer handoff is trivial: the router's worker_agent picks
the analyzer, the analyzer returns refined action + flags.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .analyzer_model import MultiTaskAnalyzer
from .analyzer_questions import (
    AGENT_IDS,
    agent_spec,
    choice_options,
    choice_question,
    noul_questions,
)

log = logging.getLogger(__name__)


@dataclass
class AnalyzerDecision:
    """Typed decisions for one agent on one query."""
    agent: str
    choice_question: str
    choice: str
    choice_confidence: float
    choice_probabilities: dict[str, float] = field(default_factory=dict)
    noul: dict[str, float] = field(default_factory=dict)
    needs_human: bool = False   # choice confidence below threshold
    model: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "choice_question": self.choice_question,
            self.choice_question: self.choice,
            "choice_confidence": self.choice_confidence,
            "choice_probabilities": self.choice_probabilities,
            **self.noul,
            "needs_human": self.needs_human,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


class AnalyzerError(RuntimeError):
    """Raised when an analyzer checkpoint can't be loaded or run."""


class AnalyzerClient:
    """One agent's fine-tuned analyzer, served from a checkpoint dir."""

    def __init__(self, agent: str, checkpoint: str,
                 confidence_threshold: float = 0.6) -> None:
        if agent not in AGENT_IDS:
            raise AnalyzerError(f"unknown agent: {agent!r}")
        self.agent = agent
        self.checkpoint = checkpoint
        self.confidence_threshold = confidence_threshold
        self._tokenizer = None
        self._model = None
        self._temp_choice = 1.0
        self._temp_noul: list[float] = []
        self._choice_labels: list[str] = []
        self._noul_qids: list[str] = []

    # ------------------------------------------------------------------ API
    def analyze(self, text: str) -> AnalyzerDecision:
        self._ensure_loaded()
        import torch

        started = time.monotonic()
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = self._model(**inputs)
            choice_probs = torch.softmax(
                out["choice_logits"] / self._temp_choice, dim=-1)[0]
            noul_probs = torch.sigmoid(
                out["noul_logits"] / torch.tensor(self._temp_noul))[0]
        latency_ms = (time.monotonic() - started) * 1000.0

        top_idx = int(choice_probs.argmax())
        choice = self._choice_labels[top_idx]
        confidence = round(float(choice_probs[top_idx]), 3)
        probs = {l: round(float(choice_probs[i]), 4)
                 for i, l in enumerate(self._choice_labels)}
        noul = {qid: round(float(noul_probs[i]), 3)
                for i, qid in enumerate(self._noul_qids)}

        return AnalyzerDecision(
            agent=self.agent,
            choice_question=choice_question(self.agent),
            choice=choice,
            choice_confidence=confidence,
            choice_probabilities=probs,
            noul=noul,
            needs_human=confidence < self.confidence_threshold,
            model=f"analyzer:{self.agent}:{self.checkpoint}",
            latency_ms=round(latency_ms, 1),
        )

    # -------------------------------------------------------------- internals
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise AnalyzerError(
                "transformers is required; pip install transformers torch"
            ) from exc
        ckpt = Path(self.checkpoint)
        calib_path = ckpt / "calibration.json"
        if not calib_path.exists():
            raise AnalyzerError(
                f"analyzer checkpoint not found: {ckpt} "
                "(train one with train/finetune.py --agent "
                f"{self.agent})")
        calib = json.loads(calib_path.read_text())

        # Contract check: checkpoint labels must match the question spec
        # byte-for-byte in order. This is the 2026-10-07 lesson: a silent
        # label-order mismatch degrades everything downstream.
        spec_choice = choice_options(self.agent)
        spec_noul = noul_questions(self.agent)
        if list(calib.get("choice_labels", [])) != spec_choice:
            raise AnalyzerError(
                f"checkpoint choice labels {calib.get('choice_labels')} "
                f"!= spec {spec_choice}")
        if list(calib.get("noul_questions", [])) != spec_noul:
            raise AnalyzerError(
                f"checkpoint noul questions {calib.get('noul_questions')} "
                f"!= spec {spec_noul}")
        if calib.get("agent") != self.agent:
            raise AnalyzerError(
                f"checkpoint agent {calib.get('agent')!r} != {self.agent!r}")

        self._choice_labels = spec_choice
        self._noul_qids = spec_noul
        self._temp_choice = float(calib.get("temperature_choice", 1.0))
        self._temp_noul = [float(t) for t in calib.get("temperatures_noul", [])]
        if len(self._temp_noul) != len(self._noul_qids):
            raise AnalyzerError("temperatures_noul length mismatch")

        log.info("loading analyzer checkpoint %s ...", ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)
        import torch
        from transformers import AutoConfig
        # Build from the LOCAL config (no hub download); weights come
        # from the state dict below.
        enc_config = AutoConfig.from_pretrained(ckpt)
        self._model = MultiTaskAnalyzer.from_local_config(
            enc_config, len(self._choice_labels), len(self._noul_qids))
        state = torch.load(ckpt / "pytorch_model.bin", map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()
        log.info("analyzer %s loaded (temp_choice=%.3f)",
                 self.agent, self._temp_choice)


class AnalyzerRegistry:
    """Lazy per-agent analyzer pool for the service."""

    def __init__(self, checkpoint_dir: str,
                 confidence_threshold: float = 0.6) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.confidence_threshold = confidence_threshold
        self._clients: dict[str, AnalyzerClient] = {}

    def get(self, agent: str) -> AnalyzerClient:
        if agent not in self._clients:
            self._clients[agent] = AnalyzerClient(
                agent, str(self.checkpoint_dir / agent),
                confidence_threshold=self.confidence_threshold)
        return self._clients[agent]

    def available(self) -> dict[str, bool]:
        """Which agents have a trained checkpoint on disk."""
        return {
            agent: (self.checkpoint_dir / agent / "calibration.json").exists()
            for agent in AGENT_IDS
        }
