"""Contract tests: training <-> serving consistency.

The 2026-10-07 lesson from intent-router-laya: training and serving must
use the question spec byte-identically, or confidence silently degrades.
These tests pin that contract:

1. Every dataset record's label keys match the spec for its agent.
2. Choice option order in the dataset builder == spec order (the order
   becomes label2id at train time; the checkpoint must agree at serve).
3. Serving (analyzers.AnalyzerClient) validates checkpoint labels against
   the spec and raises on mismatch (tested with a forged calibration).
4. The OpenAPI spec lists every route the FastAPI app serves.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analyzer_questions import (  # noqa: E402
    AGENT_IDS,
    ANALYZERS,
    choice_options,
    choice_question,
    noul_questions,
    question_ids,
)
from src.analyzers import AnalyzerClient, AnalyzerError  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_question_inventory_complete():
    for agent in AGENT_IDS:
        spec = ANALYZERS[agent]
        cq = choice_question(agent)
        assert spec["questions"][cq]["type"] == "choice"
        assert len(choice_options(agent)) >= 5
        assert len(noul_questions(agent)) >= 2
        # every question has instructions; choice has criteria
        for qid, q in spec["questions"].items():
            assert q["instructions"].strip()
            if q["type"] == "choice":
                assert len(q["criteria"]) >= 5


def test_dataset_records_match_spec():
    """Label keys in built datasets == the spec's question ids."""
    sys.path.insert(0, str(REPO / "train"))
    from build_dataset import build_agent_records
    for agent in AGENT_IDS:
        recs = build_agent_records(agent, repeat=2, seed=1)
        expected = set(question_ids(agent)) | {"query"}
        for r in recs:
            assert set(r.keys()) == expected, (agent, set(r.keys()))
            assert r[choice_question(agent)] in choice_options(agent)
            for n in noul_questions(agent):
                assert r[n] in (0.0, 1.0)


def test_choice_order_stable():
    """Option order is insertion order — the same order train and serve use."""
    for agent in AGENT_IDS:
        opts = choice_options(agent)
        assert opts == list(
            ANALYZERS[agent]["questions"][choice_question(agent)]["criteria"].keys())


def test_serving_rejects_label_mismatch():
    """A checkpoint whose labels disagree with the spec must fail loudly."""
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d)
        (ckpt / "calibration.json").write_text(json.dumps({
            "agent": "billing",
            "choice_question": "billing_action",
            # WRONG order vs the spec -> must raise
            "choice_labels": list(reversed(choice_options("billing"))),
            "noul_questions": noul_questions("billing"),
            "temperature_choice": 1.0,
            "temperatures_noul": [1.0] * len(noul_questions("billing")),
        }))
        client = AnalyzerClient("billing", str(ckpt))
        with pytest.raises(AnalyzerError, match="choice labels"):
            client._ensure_loaded()


def test_serving_rejects_agent_mismatch():
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d)
        (ckpt / "calibration.json").write_text(json.dumps({
            "agent": "orders",  # wrong agent
            "choice_question": "billing_action",
            "choice_labels": choice_options("billing"),
            "noul_questions": noul_questions("billing"),
            "temperature_choice": 1.0,
            "temperatures_noul": [1.0] * len(noul_questions("billing")),
        }))
        client = AnalyzerClient("billing", str(ckpt))
        with pytest.raises(AnalyzerError, match="checkpoint agent"):
            client._ensure_loaded()


def test_openapi_covers_app_routes():
    try:
        from src.app import app
    except ImportError:
        pytest.skip("fastapi not installed")
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text())
    app_paths = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        # skip FastAPI's auto-generated docs routes
        if path and not path.startswith(
                ("/openapi.json", "/docs", "/redoc")):
            app_paths.add(path)
    for p in app_paths:
        assert p in spec["paths"], f"app route {p} missing from openapi.yaml"
