#!/usr/bin/env python3
"""Generate tabletop robot base candidates along accessible table edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.workspace.planner import (  # noqa: E402
    WorkspacePlanningError,
    generate_tabletop_manifest_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate collision-free tabletop robot poses along table edges."
    )
    parser.add_argument("--task", required=True, type=Path, help="Input simbox_task.yaml")
    parser.add_argument(
        "--target",
        help="Target object; defaults to first Pick object, then delivery_active_objects[0]",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right"),
        help="Bind the manifest to a preselected manipulation arm; required by the Agent workflow.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory that receives candidates.json"
    )
    parser.add_argument("--min-radius-m", type=float)
    parser.add_argument("--max-radius-m", type=float)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--preferred-radius-m", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_path = args.task.resolve()
    output_dir = args.output_dir.resolve()
    overrides = {
        "min_radius_m": args.min_radius_m,
        "max_radius_m": args.max_radius_m,
        "candidate_count": args.candidate_count,
        "preferred_radius_m": args.preferred_radius_m,
    }
    try:
        manifest = generate_tabletop_manifest_file(
            task_path, output_dir, args.target, overrides, required_arm=args.arm
        )
    except WorkspacePlanningError as exc:
        print(f"tabletop workspace planning failed [{exc.code}]: {exc}", file=sys.stderr)
        print(f"manifest: {output_dir / 'candidates.json'}", file=sys.stderr)
        return 2
    feasible_count = sum(item["geometry_feasible"] for item in manifest.geometry_candidates)
    print(f"workspace status: {manifest.status}")
    print(f"geometry candidates: {feasible_count}/{len(manifest.geometry_candidates)}")
    print(f"required arm: {manifest.required_arm or 'legacy_unspecified'}")
    print(f"manifest: {output_dir / 'candidates.json'}")
    return 0 if feasible_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
