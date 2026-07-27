#!/usr/bin/env python3
"""Run the pure geometry planner over every Bench2.1 scene_4 task."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.workspace_planner import WorkspacePlanningError, generate_manifest_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Bench2.1 target-annulus workspace candidates.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "InternDataAssets/Bench_2.1_isaacsim/scene_4")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.root.resolve().glob("*/assets/basic/*/simbox_task.yaml"))
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in paths:
        case_dir = args.output / path.parent.name
        try:
            manifest = generate_manifest_file(path, case_dir)
            candidates = manifest.geometry_candidates
            feasible = [item for item in candidates if item["geometry_feasible"]]
            row = {
                "task": path.parent.name,
                "target": manifest.target["name"],
                "status": manifest.status,
                "failure_code": manifest.failure_code or "",
                "candidate_count": len(candidates),
                "geometry_feasible_count": len(feasible),
                "best_candidate": feasible[0]["candidate_id"] if feasible else "",
                "manifest": str(case_dir / "candidates.json"),
            }
        except WorkspacePlanningError as exc:
            row = {
                "task": path.parent.name,
                "target": "",
                "status": "blocked",
                "failure_code": exc.code,
                "candidate_count": 0,
                "geometry_feasible_count": 0,
                "best_candidate": "",
                "manifest": str(case_dir / "candidates.json"),
            }
        rows.append(row)

    (args.output / "summary.json").write_text(
        json.dumps({"task_count": len(rows), "tasks": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Bench2.1 Target-annulus Workspace Audit",
        "",
        "| Task | Target | Status | Failure | Feasible/Total | Best |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['target']} | {row['status']} | {row['failure_code']} | "
            f"{row['geometry_feasible_count']}/{row['candidate_count']} | {row['best_candidate']} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audited {len(rows)} tasks: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
