#!/usr/bin/env python3
"""Audit Bench2.1 rigid roots and CuRobo attach-collision prim contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pxr import Usd, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.utils.attach_collision_utils import resolve_attach_collision_prims  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect USD collision prims without modifying source assets or task YAML."
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        metavar="TASK_YAML::OBJECT",
        help="Audit one object in one task; may be repeated.",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=REPO_ROOT / "InternDataAssets/Bench_2.1_isaacsim/scene_4",
        help="Used only when --selection is omitted; scans delivery_active_objects in all tasks.",
    )
    parser.add_argument(
        "--runtime-root",
        action="append",
        default=[],
        type=Path,
        help="Optional Probe output root whose result JSON files are correlated by target name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output/bench21_attach_audit",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("tasks"):
        raise ValueError(f"task YAML has no tasks: {path}")
    return value


def selections(args: argparse.Namespace) -> list[tuple[Path, str]]:
    values: list[tuple[Path, str]] = []
    if args.selection:
        for raw in args.selection:
            task_value, separator, object_name = raw.partition("::")
            if not separator or not object_name:
                raise ValueError(f"invalid --selection, expected TASK_YAML::OBJECT: {raw}")
            task_path = Path(task_value)
            task_path = task_path if task_path.is_absolute() else REPO_ROOT / task_path
            values.append((task_path.resolve(), object_name))
        return values
    for task_path in sorted(args.bench_root.resolve().glob("*/assets/basic/*/simbox_task.yaml")):
        task = load_yaml(task_path)["tasks"][0]
        for object_name in task.get("delivery_active_objects", []):
            values.append((task_path, str(object_name)))
    return values


def resolve_usd_path(task: dict[str, Any], object_cfg: dict[str, Any]) -> Path:
    configured = Path(str(object_cfg["path"])).expanduser()
    if configured.is_absolute() and configured.exists():
        return configured.resolve()
    asset_root = Path(str(task["asset_root"])).expanduser()
    if not asset_root.is_absolute():
        asset_root = REPO_ROOT / asset_root
    return (asset_root / str(configured).lstrip("/")).resolve()


def relative_to_default(default_path: str, full_path: str) -> str:
    prefix = default_path.rstrip("/") + "/"
    return full_path[len(prefix) :] if full_path.startswith(prefix) else full_path


def runtime_results(roots: list[Path]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for root in roots:
        for path in root.resolve().glob("**/results/*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            target = value.get("target")
            if not target:
                continue
            indexed.setdefault(str(target), []).append(
                {
                    "result_path": str(path),
                    "candidate_id": value.get("candidate_id"),
                    "arm": value.get("arm"),
                    "feasible": value.get("feasible"),
                    "attach_prim_valid": value.get("attach_prim_valid"),
                    "attach_prim_paths": value.get("attach_prim_paths")
                    or ([value["attach_prim_path"]] if value.get("attach_prim_path") else []),
                    "missing_attach_prim_paths": value.get("missing_attach_prim_paths", []),
                    "joint_success_count": value.get("joint_success_count", 0),
                    "failure_code": value.get("failure_code"),
                }
            )
    return indexed


def audit_one(task_path: Path, object_name: str, runtime: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    document = load_yaml(task_path)
    task = document["tasks"][0]
    object_cfg = next(
        (item for item in task.get("objects", []) if item.get("name") == object_name),
        None,
    )
    if object_cfg is None:
        return {
            "task": str(task_path),
            "object": object_name,
            "status": "object_not_found",
            "failure_code": "OBJECT_NOT_FOUND",
        }
    usd_path = resolve_usd_path(task, object_cfg)
    if not usd_path.is_file():
        return {
            "task": str(task_path),
            "object": object_name,
            "usd": str(usd_path),
            "status": "usd_not_found",
            "failure_code": "USD_NOT_FOUND",
        }
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim():
        return {
            "task": str(task_path),
            "object": object_name,
            "usd": str(usd_path),
            "status": "usd_open_failed",
            "failure_code": "USD_DEFAULT_PRIM_MISSING",
        }
    default_path = str(stage.GetDefaultPrim().GetPath())
    base_path = default_path
    rigid_path = f"{default_path.rstrip('/')}/{str(object_cfg['prim_path_child']).strip('/')}"
    resolution = resolve_attach_collision_prims(base_path, rigid_path, object_cfg, stage.GetPrimAtPath)
    collision_prims = []
    rigid_prim = stage.GetPrimAtPath(rigid_path)
    if rigid_prim and rigid_prim.IsValid():
        for prim in Usd.PrimRange(rigid_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            collision_prims.append(
                {
                    "path": relative_to_default(default_path, str(prim.GetPath())),
                    "type": prim.GetTypeName(),
                    "enabled": enabled is not False,
                }
            )
    paths = [relative_to_default(default_path, path) for path in resolution.prim_paths]
    candidates = [relative_to_default(default_path, path) for path in resolution.candidates]
    evidence = runtime.get(object_name, [])
    runtime_attach_valid = any(item.get("attach_prim_valid") for item in evidence)
    runtime_plan_feasible = any(item.get("feasible") for item in evidence)
    status = "offline_valid" if resolution.failure_code is None else "offline_invalid"
    if runtime_plan_feasible:
        status = "curobo_plan_feasible"
    elif runtime_attach_valid:
        status = "curobo_attach_valid"
    return {
        "task": str(task_path),
        "task_name": task.get("name", task_path.parent.name),
        "object": object_name,
        "usd": str(usd_path),
        "rigid_prim_path": str(object_cfg.get("prim_path_child")),
        "configured_attach_prim_path_children": object_cfg.get("attach_prim_path_children"),
        "configured_attach_prim_path_child": object_cfg.get("attach_prim_path_child"),
        "resolved_attach_prim_paths": paths,
        "resolution_source": resolution.source,
        "failure_code": resolution.failure_code,
        "failure_message": resolution.message,
        "collision_coverage_mode": "single_prim" if len(paths) == 1 else "multi_prim" if paths else "unresolved",
        "collision_candidate_count": len(candidates),
        "collision_candidates": collision_prims,
        "runtime_evidence": evidence,
        "runtime_attach_valid": runtime_attach_valid,
        "runtime_plan_feasible": runtime_plan_feasible,
        "status": status,
    }


def write_summary(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# Bench2.1 attach collision audit",
        "",
        "This audit separates offline USD/config validity from runtime CuRobo and Pick feasibility.",
        "",
        "| Task | Object | Attach paths | Offline | CuRobo attach | CuRobo plan | Failure |",
        "|---|---|---:|---|---|---|---|",
    ]
    for item in manifest["items"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | `{}` |".format(
                item.get("task_name", Path(item["task"]).parent.name),
                item["object"],
                len(item.get("resolved_attach_prim_paths", [])),
                "pass" if item.get("failure_code") is None else "fail",
                "pass" if item.get("runtime_attach_valid") else "not tested",
                (
                    "pass"
                    if item.get("runtime_plan_feasible")
                    else "fail"
                    if item.get("runtime_evidence")
                    else "not tested"
                ),
                item.get("failure_code") or "-",
            )
        )
    lines.extend(["", f"Manifest: `{path.with_name('manifest.json')}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runtime = runtime_results(args.runtime_root)
    items = [audit_one(task_path, object_name, runtime) for task_path, object_name in selections(args)]
    counts = Counter(item["status"] for item in items)
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_assets_modified": False,
        "summary": {"object_count": len(items), "status_counts": dict(sorted(counts.items()))},
        "items": items,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(manifest, output_dir / "summary.md")
    print(f"audited objects: {len(items)}")
    print(f"status counts: {dict(sorted(counts.items()))}")
    print(f"manifest: {manifest_path}")
    return 0 if all(item.get("failure_code") is None for item in items) else 2


if __name__ == "__main__":
    raise SystemExit(main())
