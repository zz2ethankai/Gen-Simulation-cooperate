from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.specs import EvalSpec


class JsonlResultStore:
    def __init__(self, spec: EvalSpec):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = spec.output_dir / spec.name / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self.config_path.write_text(json.dumps(_to_jsonable(spec), indent=2), encoding="utf-8")

    def write_episode(self, episode: dict[str, Any]) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_to_jsonable(episode), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value

