"""Retention artifacts stay candidates until cross-scene held-out validation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent.contracts import RetentionDecision, RetentionKind
from agent.retention import RetentionManager


def test_single_run_playbook_is_persisted_as_candidate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.retention.REPO_ROOT", tmp_path)
    manager = RetentionManager.__new__(RetentionManager)
    manager.backend = None
    manager.root = tmp_path / "experience"
    manager.index_path = manager.root / "index.yaml"
    decision = RetentionDecision(
        kind=RetentionKind.PLAYBOOK,
        name="Move blocker before pick",
        category="layout",
        summary="A measured scene-layout repair succeeded once.",
        reusable_scope="Candidate only until validation across scenes and seeds.",
        evidence_refs=["run/variants/v1/attempts/00/evidence.json"],
    )

    path = manager.materialize(decision)

    assert path == (
        manager.root
        / "playbooks"
        / "candidates"
        / "move_blocker_before_pick"
        / "playbook.md"
    )
    payload = json.loads(path.with_name("candidate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "candidate"
    assert payload["promotion_gate"]["cross_scene_validation"] == "pending"
    assert payload["promotion_gate"]["heldout_seeds"] == "0/20"
    index = yaml.safe_load(manager.index_path.read_text(encoding="utf-8"))
    assert index["experiences"][0]["status"] == "candidate"
