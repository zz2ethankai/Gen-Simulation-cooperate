"""Build coverage matrices from TaskIR corpus."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.task_ir.batch import load_or_parse_task_irs


def extract_task_features(task_ir: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized per-task features used by coverage matrix."""
    task_family = str(task_ir.get("task_family") or "unknown")
    scene_type = str(task_ir.get("scene_type") or "unknown_scene")

    robots = task_ir.get("robots", [])
    embodiments = [
        str(robot.get("embodiment") or "unknown")
        for robot in robots
        if isinstance(robot, dict)
    ]
    if not embodiments:
        embodiment = "unknown"
    elif len(set(embodiments)) == 1:
        embodiment = embodiments[0]
    else:
        embodiment = "multi:" + "+".join(sorted(set(embodiments)))

    steps = task_ir.get("skill_steps", [])
    skill_names = [
        str(step.get("skill_name") or "unknown")
        for step in steps
        if isinstance(step, dict)
    ]
    skill_set = "+".join(sorted(set(skill_names))) if skill_names else "none"
    arm_usage = _infer_arm_usage(steps)

    object_count = len(task_ir.get("objects", [])) if isinstance(task_ir.get("objects"), list) else 0
    step_count = len(steps) if isinstance(steps, list) else 0

    return {
        "task_family": task_family,
        "scene_type": scene_type,
        "embodiment": embodiment,
        "skill_set": skill_set,
        "arm_usage": arm_usage,
        "object_count": object_count,
        "step_count": step_count,
        "object_count_bucket": _bucket_object_count(object_count),
        "step_count_bucket": _bucket_step_count(step_count),
    }


def build_cross_table(
    features: list[dict[str, Any]],
    row_key: str,
    col_key: str,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
) -> dict[str, Any]:
    """Build a 2D cross table for feature keys."""
    rows = sorted({str(item.get(row_key) or "unknown") for item in features})
    cols = sorted({str(item.get(col_key) or "unknown") for item in features})

    if row_order is not None:
        ordered_rows = [item for item in row_order if item in rows]
        remaining_rows = [item for item in rows if item not in ordered_rows]
        rows = ordered_rows + remaining_rows
    if col_order is not None:
        ordered_cols = [item for item in col_order if item in cols]
        remaining_cols = [item for item in cols if item not in ordered_cols]
        cols = ordered_cols + remaining_cols

    data = [[0 for _ in cols] for _ in rows]
    row_index = {name: idx for idx, name in enumerate(rows)}
    col_index = {name: idx for idx, name in enumerate(cols)}

    for item in features:
        row_val = str(item.get(row_key) or "unknown")
        col_val = str(item.get(col_key) or "unknown")
        if row_val not in row_index or col_val not in col_index:
            continue
        data[row_index[row_val]][col_index[col_val]] += 1

    return {"rows": rows, "cols": cols, "data": data}


def identify_gaps(matrices: dict[str, dict[str, Any]], threshold: int = 5) -> list[dict[str, Any]]:
    """Identify sparse/empty cells from 2D matrices."""
    gaps: list[dict[str, Any]] = []
    for matrix_name, matrix in matrices.items():
        rows = matrix.get("rows", [])
        cols = matrix.get("cols", [])
        data = matrix.get("data", [])
        for row_idx, row_name in enumerate(rows):
            for col_idx, col_name in enumerate(cols):
                count = int(data[row_idx][col_idx])
                if count >= threshold:
                    continue
                gaps.append(
                    {
                        "matrix": matrix_name,
                        "row": row_name,
                        "col": col_name,
                        "count": count,
                        "coverage": _coverage_status(count),
                        "description": (
                            f"{matrix_name}: row={row_name}, col={col_name}, "
                            f"count={count}, status={_coverage_status(count)}"
                        ),
                    }
                )
    gaps.sort(key=lambda item: (int(item["count"]), str(item["matrix"]), str(item["row"]), str(item["col"])))
    return gaps[:20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TaskIR coverage matrices.")
    parser.add_argument(
        "--tasks-root",
        default="workflows/simbox/core/configs/tasks",
        help="Root directory of task YAML files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="output/task_ir_cache",
        help="TaskIR cache directory used by shared batch loader.",
    )
    parser.add_argument(
        "--output",
        default="output/knowledge/coverage_matrix.yaml",
        help="Output path for coverage matrix YAML.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional filter by task_family.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of TaskIR entries.",
    )
    parser.add_argument(
        "--sparse-threshold",
        type=int,
        default=5,
        help="Threshold below which a matrix cell is marked as sparse/empty.",
    )
    parser.add_argument(
        "--skill-combo-top-n",
        type=int,
        default=40,
        help="Keep top-N skill_set combos in skill_combo_x_embodiment matrix.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_irs = load_or_parse_task_irs(tasks_root=tasks_root, cache_dir=cache_dir)
    if args.category:
        category_text = str(args.category).strip().lower()
        task_irs = [item for item in task_irs if str(item.get("task_family", "")).lower() == category_text]
    if args.limit is not None:
        task_irs = task_irs[: max(args.limit, 0)]

    features = [extract_task_features(task_ir) for task_ir in task_irs]

    family_x_embodiment = build_cross_table(
        features=features,
        row_key="task_family",
        col_key="embodiment",
        row_order=["basic", "art", "pick_and_place", "long_horizon", "navigation", "example"],
    )

    skill_set_counter = Counter(item["skill_set"] for item in features)
    top_skill_sets = [name for name, _ in skill_set_counter.most_common(max(1, args.skill_combo_top_n))]
    filtered_for_skill_combo = [item for item in features if item["skill_set"] in top_skill_sets]
    skill_combo_x_embodiment = build_cross_table(
        features=filtered_for_skill_combo,
        row_key="skill_set",
        col_key="embodiment",
        row_order=top_skill_sets,
    )

    family_x_scene_type = build_cross_table(
        features=features,
        row_key="task_family",
        col_key="scene_type",
        row_order=["basic", "art", "pick_and_place", "long_horizon", "navigation", "example"],
    )

    complexity_cube = _build_complexity_cube(features)

    flat_matrices = {
        "family_x_embodiment": family_x_embodiment,
        "skill_combo_x_embodiment": skill_combo_x_embodiment,
        "family_x_scene_type": family_x_scene_type,
        "embodiment_x_object_count_bucket": complexity_cube["embodiment_x_object_count_bucket"],
        "embodiment_x_step_count_bucket": complexity_cube["embodiment_x_step_count_bucket"],
    }
    gaps = identify_gaps(flat_matrices, threshold=max(0, args.sparse_threshold))

    payload = {
        "summary": {
            "total_task_irs": len(task_irs),
            "sparse_threshold": max(0, args.sparse_threshold),
            "skill_combo_top_n": max(1, args.skill_combo_top_n),
        },
        "matrices": {
            "family_x_embodiment": _with_coverage_labels(family_x_embodiment),
            "skill_combo_x_embodiment": _with_coverage_labels(skill_combo_x_embodiment),
            "family_x_scene_type": _with_coverage_labels(family_x_scene_type),
            "embodiment_x_object_count_x_step_count": complexity_cube["embodiment_x_object_count_x_step_count"],
            "embodiment_x_object_count_bucket": _with_coverage_labels(complexity_cube["embodiment_x_object_count_bucket"]),
            "embodiment_x_step_count_bucket": _with_coverage_labels(complexity_cube["embodiment_x_step_count_bucket"]),
        },
        "gaps": gaps,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    print(f"TaskIR entries: {len(task_irs)}")
    print(f"Generated matrices: {len(payload['matrices'])}")
    print(f"Gap entries: {len(gaps)}")
    print(f"Coverage matrix written to: {output_path}")


def _infer_arm_usage(steps: Any) -> str:
    if not isinstance(steps, list) or not steps:
        return "unknown"
    arms = {
        str(step.get("arm") or "").lower()
        for step in steps
        if isinstance(step, dict)
    }
    if not arms:
        return "unknown"
    if any(arm in {"both", "dual", "bimanual"} for arm in arms):
        return "dual_arm"
    has_left = any("left" in arm for arm in arms)
    has_right = any("right" in arm for arm in arms)
    if has_left and has_right:
        return "dual_arm"
    if has_left:
        return "single_left"
    if has_right:
        return "single_right"
    return "other"


def _bucket_object_count(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    return "4+"


def _bucket_step_count(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def _coverage_status(count: int) -> str:
    if count <= 0:
        return "empty"
    if count <= 5:
        return "sparse"
    if count <= 20:
        return "moderate"
    return "rich"


def _with_coverage_labels(matrix: dict[str, Any]) -> dict[str, Any]:
    rows = matrix.get("rows", [])
    cols = matrix.get("cols", [])
    data = matrix.get("data", [])
    labels = []
    for row_idx, _ in enumerate(rows):
        row_labels = []
        for col_idx, _ in enumerate(cols):
            row_labels.append(_coverage_status(int(data[row_idx][col_idx])))
        labels.append(row_labels)
    return {
        "rows": rows,
        "cols": cols,
        "data": data,
        "coverage": labels,
    }


def _build_complexity_cube(features: list[dict[str, Any]]) -> dict[str, Any]:
    embodiments = sorted({str(item.get("embodiment") or "unknown") for item in features})
    object_bins = ["1", "2-3", "4+"]
    step_bins = ["1-2", "3-5", "6+"]

    cube: dict[str, dict[str, dict[str, int]]] = {
        emb: {obj_bin: {step_bin: 0 for step_bin in step_bins} for obj_bin in object_bins}
        for emb in embodiments
    }
    for item in features:
        emb = str(item.get("embodiment") or "unknown")
        obj_bin = str(item.get("object_count_bucket") or "1")
        step_bin = str(item.get("step_count_bucket") or "1-2")
        cube.setdefault(emb, {bin_name: {step: 0 for step in step_bins} for bin_name in object_bins})
        cube[emb].setdefault(obj_bin, {step: 0 for step in step_bins})
        cube[emb][obj_bin][step_bin] = int(cube[emb][obj_bin].get(step_bin, 0)) + 1

    object_bucket_table = build_cross_table(
        features=features,
        row_key="embodiment",
        col_key="object_count_bucket",
        col_order=object_bins,
    )
    step_bucket_table = build_cross_table(
        features=features,
        row_key="embodiment",
        col_key="step_count_bucket",
        col_order=step_bins,
    )

    data = {}
    for emb in embodiments:
        data[emb] = [[int(cube[emb][obj_bin][step_bin]) for step_bin in step_bins] for obj_bin in object_bins]

    return {
        "embodiment_x_object_count_x_step_count": {
            "embodiments": embodiments,
            "object_count_bins": object_bins,
            "step_count_bins": step_bins,
            "data": data,
        },
        "embodiment_x_object_count_bucket": object_bucket_table,
        "embodiment_x_step_count_bucket": step_bucket_table,
    }


if __name__ == "__main__":
    main()
