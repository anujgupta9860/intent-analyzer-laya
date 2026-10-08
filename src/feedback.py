"""Analyzer RLCD feedback: log low-confidence decisions for human review.

Mirrors intent-router-laya's src/feedback.py pattern, scoped to one
analyzer tier: when an analyzer's choice confidence falls below
threshold, the record is appended as JSONL with the query, the analyzer's
typed decisions, and a review gate (pending -> approved / corrected /
rejected). Only approved/corrected records may be used for the next
retraining cycle.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)


class AnalyzerFeedback:
    """Thread-safe JSONL appender for low-confidence analyzer decisions."""

    def __init__(self, log_dir: str | Path = "feedback", enabled: bool = True):
        self.dir = Path(log_dir)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.path = self.dir / "analyzer_fallbacks.jsonl"
        else:
            self.path = None

    def log(self, agent: str, query: str, decision: dict,
            router_context: dict | None = None) -> bool:
        """Append one feedback record. Returns True if written."""
        if not self.enabled or self.path is None:
            return False
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "agent": agent,
            "query": query,
            "decision": decision,
            "router_context": router_context or {},
            "review_status": "pending",   # pending | approved | corrected | rejected
            "reviewed_by": None,
            "reviewed_ts": None,
            "corrections": None,
        }
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            log.warning("analyzer feedback write failed: %s", exc)
            return False

    def stats(self) -> dict:
        total = pending = approved = 0
        per_agent: dict[str, int] = {}
        if self.path and self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        total += 1
                        per_agent[rec.get("agent", "?")] = \
                            per_agent.get(rec.get("agent", "?"), 0) + 1
                        status = rec.get("review_status", "pending")
                        if status == "pending":
                            pending += 1
                        elif status in ("approved", "corrected"):
                            approved += 1
            except OSError as exc:
                log.warning("analyzer feedback stats failed: %s", exc)
        return {
            "feedback_log": str(self.path) if self.path else None,
            "total_low_confidence": total,
            "pending_review": pending,
            "approved_for_training": approved,
            "per_agent": per_agent,
        }

    def list_pending(self, agent: str | None = None,
                     limit: int = 50) -> list[dict]:
        out: list[dict] = []
        if not (self.path and self.path.exists()):
            return out
        try:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("review_status", "pending") != "pending":
                        continue
                    if agent and rec.get("agent") != agent:
                        continue
                    out.append(rec)
        except OSError as exc:
            log.warning("analyzer feedback pending read failed: %s", exc)
        out.sort(key=lambda r: r.get("ts", 0), reverse=True)
        return out[:limit]

    def review(self, record_id: str, decision: str,
               reviewer: str = "human",
               corrections: dict | None = None) -> bool:
        if decision not in ("approved", "corrected", "rejected"):
            raise ValueError(f"bad review decision: {decision!r}")
        if not (self.path and self.path.exists()):
            return False
        found = False
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.warning("analyzer feedback review read failed: %s", exc)
                return False
            new_lines: list[str] = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    new_lines.append(line)
                    continue
                if rec.get("id") == record_id:
                    rec["review_status"] = (
                        "corrected" if decision == "corrected" else decision)
                    rec["reviewed_by"] = reviewer
                    rec["reviewed_ts"] = time.time()
                    rec["corrections"] = corrections or None
                    found = True
                    line = json.dumps(rec, ensure_ascii=False)
                new_lines.append(line)
            if found:
                try:
                    self.path.write_text("\n".join(new_lines) + "\n",
                                         encoding="utf-8")
                except OSError as exc:
                    log.warning("analyzer feedback review write failed: %s", exc)
                    return False
        return found
