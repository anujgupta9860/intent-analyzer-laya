"""Intent Analyzer service: second-tier System 1 for worker agents.

The intent router (intent-router-laya) makes 11 coarse routing decisions
and picks a worker_agent. This service takes over from there: given the
query + the router's worker_agent (+ optional skill), the agent's own
fine-tuned analyzer answers deeper, domain-specific typed decisions the
worker needs before it can act.

Endpoints:
  GET  /health    - liveness + which analyzers have checkpoints
  POST /analyze   - {"query": ..., "worker_agent": "billing",
                     "router_context": {...}} -> agent-specific decisions
  GET  /feedback/stats   - low-confidence log counters
  GET  /feedback/pending - records awaiting human review
  POST /feedback/review/{id} - approve | correct | reject
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analyzer_questions import AGENT_IDS, choice_question
from .analyzers import AnalyzerError, AnalyzerRegistry
from .config import Settings
from .feedback import AnalyzerFeedback

log = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    registry = AnalyzerRegistry(
        settings.checkpoint_dir,
        confidence_threshold=settings.confidence_threshold)
    feedback = AnalyzerFeedback(enabled=settings.feedback_enabled)
    _state.update(settings=settings, registry=registry, feedback=feedback)
    log.info("intent-analyzer ready: checkpoint_dir=%s available=%s",
             settings.checkpoint_dir, registry.available())
    yield
    _state.clear()


app = FastAPI(title="Intent Analyzer (per-agent System 1)",
              version="0.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    worker_agent: str = Field(
        description="router's worker_agent decision: one of "
                    "billing|orders|support|account|sales")
    router_context: dict = Field(
        default_factory=dict,
        description="optional router output echoed for the feedback log "
                    "(intent, skill_required, confidence, ...)")


def _registry() -> AnalyzerRegistry:
    reg = _state.get("registry")
    if reg is None:
        raise HTTPException(status_code=503, detail="analyzers not initialized")
    return reg


def _feedback() -> AnalyzerFeedback:
    return _state.get("feedback")


@app.get("/health")
def health():
    reg = _state.get("registry")
    settings = _state.get("settings")
    return {
        "status": "ok",
        "agents": list(AGENT_IDS),
        "available": reg.available() if reg else {},
        "checkpoint_dir": settings.checkpoint_dir if settings else None,
        "confidence_threshold": settings.confidence_threshold if settings else None,
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run the worker agent's analyzer on the query.

    Returns the agent's Choice decision + Noul flags. When choice
    confidence is below threshold, needs_human=true and the decision is
    logged for human review (RLCD feedback loop).
    """
    agent = req.worker_agent.strip().lower()
    if agent not in AGENT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown worker_agent {agent!r}; expected one of "
                   f"{list(AGENT_IDS)}")
    registry = _registry()
    if not registry.available().get(agent):
        raise HTTPException(
            status_code=503,
            detail=f"no trained checkpoint for agent {agent!r} "
                   f"(train one: python train/finetune.py --agent {agent})")
    try:
        decision = registry.get(agent).analyze(req.query.strip())
    except AnalyzerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    out = decision.to_dict()
    # RLCD feedback: log low-confidence analyzer decisions.
    if decision.needs_human:
        fb = _feedback()
        if fb is not None:
            fb.log(agent, req.query.strip(), out,
                   router_context=req.router_context)
    return out


@app.get("/feedback/stats")
def feedback_stats():
    return _feedback().stats()


@app.get("/feedback/pending")
def feedback_pending(agent: str | None = None, limit: int = 50):
    return {"pending": _feedback().list_pending(agent=agent, limit=limit)}


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(approved|corrected|rejected)$")
    reviewer: str = "human"
    corrections: dict | None = None


@app.post("/feedback/review/{record_id}")
def feedback_review(record_id: str, req: ReviewDecision):
    ok = _feedback().review(record_id, req.decision,
                            reviewer=req.reviewer,
                            corrections=req.corrections)
    if not ok:
        raise HTTPException(status_code=404, detail="record not found")
    return {"record_id": record_id, "decision": req.decision}
