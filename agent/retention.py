"""Record reusable lessons without activating unvalidated robot behavior."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .contracts import RetentionDecision, RetentionKind
from .resolver import CodexBackend, OpenAIBackend


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = Path(__file__).resolve().parent


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.strip().lower()).strip("_")
    return text[:80] or "unnamed_experience"


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


class RetentionManager:
    def __init__(self, backend: CodexBackend|OpenAIBackend):
        self.backend = backend
        self.root = AGENT_DIR / "experience"
        self.index_path = self.root / "index.yaml"

    def decide(self, run_summary: dict[str, Any], artifact_dir: Path) -> RetentionDecision:
        template = (AGENT_DIR / "workflow" / "templates" / "retain.md").read_text(
            encoding="utf-8"
        )
        prompt = template.replace(
            "{{RUN_SUMMARY}}", json.dumps(run_summary, ensure_ascii=False, indent=2)
        )
        return self.backend.generate(RetentionDecision, prompt, artifact_dir, "retention")

    def materialize(self, decision: RetentionDecision) -> Path | None:
        if decision.kind == RetentionKind.NONE:
            return None
        name = _slug(decision.name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if decision.kind == RetentionKind.PLAYBOOK:
            candidate_root = self.root / "playbooks" / "candidates" / name
            if candidate_root.exists():
                raise FileExistsError(
                    f"candidate already exists and will not be overwritten: {candidate_root}"
                )
            path = candidate_root / "playbook.md"
            content = (
                f"# {decision.name}\n\n"
                f"{decision.summary}\n\n"
                f"## 适用范围\n\n{decision.reusable_scope}\n\n"
                "## 证据\n\n"
                + "\n".join(f"- `{item}`" for item in decision.evidence_refs)
                + "\n"
            )
            path.parent.mkdir(parents=True, exist_ok=False)
            path.write_text(content, encoding="utf-8")
        else:
            category = _slug(decision.category or _enum_text(decision.kind))
            if decision.kind == RetentionKind.DEBUG_TOOL:
                candidate_root = self.root / "debug_tools" / "candidates" / name
            else:
                candidate_root = (
                    REPO_ROOT
                    / "workflows"
                    / "simbox"
                    / "core"
                    / "skills"
                    / "generated"
                    / category
                    / "candidates"
                    / name
                )
            if candidate_root.exists():
                raise FileExistsError(f"candidate already exists and will not be overwritten: {candidate_root}")
            generated_targets: list[tuple[Path, str]] = []
            for generated in decision.files:
                relative = Path(generated.relative_path)
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError(f"unsafe generated candidate path: {generated.relative_path}")
                target = (candidate_root / relative).resolve()
                try:
                    target.relative_to(candidate_root.resolve())
                except ValueError as exc:
                    raise ValueError(f"generated candidate escapes whitelist: {target}") from exc
                if target.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}:
                    raise ValueError(f"unsupported generated candidate file type: {target.suffix}")
                generated_targets.append((target, generated.content))
            candidate_root.mkdir(parents=True, exist_ok=False)
            for target, content in generated_targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        payload = decision.to_dict()
        payload.update(
            {
                "status": "candidate",
                "created_at": timestamp,
                "promotion_gate": {
                    "unit_and_contract_tests": "pending",
                    "cross_scene_validation": "pending",
                    "debug_seeds": "0/5",
                    "heldout_seeds": "0/20",
                    "baseline_regression": "pending",
                },
            }
        )
        candidate_path = path.parent / "candidate.json"
        candidate_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._append_index(decision, path, timestamp)
        return path

    def _append_index(self, decision: RetentionDecision, path: Path, timestamp: str) -> None:
        if self.index_path.is_file():
            payload = yaml.safe_load(self.index_path.read_text(encoding="utf-8")) or {}
        else:
            payload = {"version": 1, "experiences": []}
        payload.setdefault("experiences", []).append(
            {
                "name": decision.name,
                "kind": _enum_text(decision.kind),
                "category": decision.category,
                "status": "candidate",
                "path": str(path.relative_to(REPO_ROOT)),
                "created_at": timestamp,
            }
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
