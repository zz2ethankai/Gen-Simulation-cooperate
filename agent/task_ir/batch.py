"""Batch helpers used by TaskIR knowledge aggregation scripts."""

from __future__ import annotations

import json
from pathlib import Path

from .parser import parse_tasks_yaml_to_ir


def discover_task_yamls(tasks_root: str | Path) -> list[Path]:
    root = Path(tasks_root).resolve()
    return sorted(path for path in root.rglob("*.yaml") if path.is_file())


def load_or_parse_task_irs(
    tasks_root: str | Path,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    cache_root = Path(cache_dir).resolve() if cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    result = []
    for path in discover_task_yamls(tasks_root):
        key = str(path).replace("/", "__").replace(".", "_")
        cache_path = cache_root / f"{key}.json" if cache_root else None
        if cache_path is not None and cache_path.is_file():
            values = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                values = parse_tasks_yaml_to_ir(path)
            except Exception:
                continue
            if cache_path is not None:
                cache_path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.extend(values)
    return result
