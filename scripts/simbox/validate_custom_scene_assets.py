#!/usr/bin/env python3
"""Validate generated custom-scene SimBox task assets.

The checker is intentionally offline: it validates YAML references and asset
paths without importing Isaac modules. The inner basic-task `simbox_task.yaml`
is the only readable task-structure entry. Directory roots are used only to
enumerate entries matching `*/assets/basic/<task>/simbox_task.yaml`; outer
`task.yaml` / `arena.yaml` files and directory names are not structure sources.
Arena files, HDR libraries, USDs, and texture libraries are opened only after
fields inside the entry YAML reference them. It targets layouts like:

    InternDataAssets/assets/custom/scene_8/<room>/assets/basic/<task>/simbox_task.yaml
"""

from __future__ import annotations

import argparse
import ast
import glob
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"

SIMBOX_OBJECT_CLASSES = {
    "ArticulatedObject",
    "BoxObject",
    "ConveyorObject",
    "GeometryObject",
    "PlaneObject",
    "RigidObject",
    "ShapeObject",
    "XFormObject",
}

REFERENCE_KEYS = ("object", "target", "container", "target2", "A", "B")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_task_files(root: Path) -> list[Path]:
    """Find inner task entries; directories are enumeration roots only."""
    if root.is_file():
        return [root] if is_basic_task_entry(root) else []
    tasks = sorted(root.rglob("simbox_task.yaml"))
    return [path for path in tasks if is_basic_task_entry(path)]


def is_basic_task_entry(path: Path) -> bool:
    """Return true only for */assets/basic/<task>/simbox_task.yaml entries."""
    if path.name != "simbox_task.yaml":
        return False
    parts = path.parts
    for idx in range(len(parts) - 3):
        if parts[idx] == "assets" and parts[idx + 1] == "basic" and parts[idx + 3] == "simbox_task.yaml":
            return True
    return False


def scene_root_for_task(task_path: Path) -> Path:
    parts = task_path.parts
    for idx in range(len(parts) - 1):
        if parts[idx] == "assets" and idx + 1 < len(parts) and parts[idx + 1] == "basic":
            return Path(*parts[:idx])
    raise ValueError(f"cannot infer scene root from {task_path}")


def task_relative_path(task_path: Path, path_value: str | Path | None, default: str) -> Path:
    raw_path = Path(str(path_value or default))
    return raw_path if raw_path.is_absolute() else task_path.parent / raw_path


def iter_skill_cfgs(skills: list[Any]):
    for phase in skills or []:
        if not isinstance(phase, dict):
            continue
        for queues in phase.values():
            if not isinstance(queues, list):
                continue
            for queue_dict in queues:
                if not isinstance(queue_dict, dict):
                    continue
                for skill_list in queue_dict.values():
                    if not isinstance(skill_list, list):
                        continue
                    for skill in skill_list:
                        if isinstance(skill, dict):
                            yield skill


def validate_skill_structure(skills: list[Any]) -> tuple[list[str], int, bool]:
    """Validate executable skill shape used by SimBoxDualWorkFlow.

    Legacy skills cannot contain empty controller queues because
    `plan_first_skill()` indexes the first item of every queue. DAG skills may
    leave inactive controllers empty, but they still need at least one node.
    """

    errors: list[str] = []
    skill_count = 0
    uses_dag = False

    if not isinstance(skills, list) or not skills:
        return ["skills must contain at least one phase"], 0, False

    for phase_idx, phase in enumerate(skills):
        if not isinstance(phase, dict) or not phase:
            errors.append(f"skills[{phase_idx}] must be a non-empty mapping")
            continue
        for robot_name, sequences in phase.items():
            if not isinstance(sequences, list) or not sequences:
                errors.append(f"skills[{phase_idx}].{robot_name} must contain at least one sequence")
                continue
            for sequence_idx, queue_dict in enumerate(sequences):
                if not isinstance(queue_dict, dict) or not queue_dict:
                    errors.append(
                        f"skills[{phase_idx}].{robot_name}[{sequence_idx}] must be a non-empty controller mapping"
                    )
                    continue
                for controller_name, skill_list in queue_dict.items():
                    if not isinstance(skill_list, list):
                        errors.append(
                            f"skills[{phase_idx}].{robot_name}[{sequence_idx}].{controller_name} must be a list"
                        )
                        continue
                    if not skill_list:
                        continue
                    for skill_idx, skill in enumerate(skill_list):
                        if not isinstance(skill, dict):
                            errors.append(
                                f"skills[{phase_idx}].{robot_name}[{sequence_idx}]."
                                f"{controller_name}[{skill_idx}] must be a mapping"
                            )
                            continue
                        skill_count += 1
                        if "id" in skill or "depends_on" in skill:
                            uses_dag = True

    if skill_count == 0:
        errors.append("skills contain no executable skill entries")
    if not uses_dag:
        for phase_idx, phase in enumerate(skills):
            if not isinstance(phase, dict):
                continue
            for robot_name, sequences in phase.items():
                if not isinstance(sequences, list):
                    continue
                for sequence_idx, queue_dict in enumerate(sequences):
                    if not isinstance(queue_dict, dict):
                        continue
                    for controller_name, skill_list in queue_dict.items():
                        if isinstance(skill_list, list) and not skill_list:
                            errors.append(
                                f"legacy skills cannot leave empty controller queue: "
                                f"skills[{phase_idx}].{robot_name}[{sequence_idx}].{controller_name}"
                            )
    return errors, skill_count, uses_dag


def has_sampler_signature_filter() -> bool:
    banana_path = SIMBOX_ROOT / "core" / "tasks" / "banana.py"
    if not banana_path.exists():
        return False
    text = banana_path.read_text(encoding="utf-8")
    return (
        "def _filter_sampler_random_config" in text
        and text.count("_filter_sampler_random_config(") >= 4
    )


def load_sampler_signatures() -> dict[str, set[str] | None]:
    """Return RandomRegionSampler method parameters without importing the module.

    `core.utils.region_sampler` imports geometry dependencies that are not
    always installed in lightweight validation environments, so parse the source
    file directly. A value of None means the sampler accepts **kwargs.
    """

    sampler_path = SIMBOX_ROOT / "core" / "utils" / "region_sampler.py"
    if not sampler_path.exists():
        return {}
    tree = ast.parse(sampler_path.read_text(encoding="utf-8"), filename=str(sampler_path))
    signatures: dict[str, set[str] | None] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "RandomRegionSampler":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.args.kwarg is not None:
                signatures[item.name] = None
                continue
            names = {arg.arg for arg in item.args.args}
            names.update(arg.arg for arg in item.args.kwonlyargs)
            names.discard("self")
            signatures[item.name] = names
    return signatures


def validate_texture_ref(asset_root: Path, owner: str, cfg: dict[str, Any]) -> list[str]:
    texture = cfg.get("texture")
    if not isinstance(texture, dict):
        return []

    texture_lib = texture.get("texture_lib")
    if not texture_lib:
        return []
    candidates = sorted(glob.glob(str(asset_root / texture_lib / "*")))
    errors: list[str] = []
    if not candidates:
        errors.append(f"{owner} texture_lib has no files: {texture_lib}")
        return errors
    if not texture.get("apply_randomization", False):
        texture_id = texture.get("texture_id", 0)
        if not isinstance(texture_id, int) or texture_id < 0 or texture_id >= len(candidates):
            errors.append(
                f"{owner} texture_id {texture_id!r} out of range for {texture_lib} "
                f"(count={len(candidates)})"
            )
    return errors


def validate_navigate_approach_fields(skill: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    approach_arm = skill.get("approach_arm")
    if approach_arm is not None and str(approach_arm).strip().lower() not in {"left", "right"}:
        errors.append(
            f"navigate skill {skill.get('id')!r} has invalid approach_arm {approach_arm!r}; "
            "expected 'left' or 'right'"
        )
    object_armbase_xy = skill.get("approach_object_armbase_xy")
    if object_armbase_xy is None:
        return errors
    if not isinstance(object_armbase_xy, (list, tuple)) or len(object_armbase_xy) != 2:
        errors.append(
            f"navigate skill {skill.get('id')!r} approach_object_armbase_xy must be a two-element list"
        )
        return errors
    for value in object_armbase_xy:
        if not isinstance(value, (int, float)):
            errors.append(
                f"navigate skill {skill.get('id')!r} approach_object_armbase_xy values must be numeric"
            )
            break
    if approach_arm is None:
        errors.append(
            f"navigate skill {skill.get('id')!r} approach_object_armbase_xy requires approach_arm"
        )
    return errors


def validate_task(
    task_path: Path,
    sampler_filter_present: bool,
    sampler_signatures: dict[str, set[str] | None],
):
    errors: list[str] = []
    warnings: list[str] = []
    task_payload = load_yaml(task_path)
    if not isinstance(task_payload, dict) or not task_payload.get("tasks"):
        return errors + [f"{display_path(task_path)} missing tasks list"], warnings

    task = task_payload["tasks"][0]
    scene_root = scene_root_for_task(task_path)
    expected_asset_root = display_path(scene_root)
    asset_root_value = task.get("asset_root")
    if asset_root_value != expected_asset_root:
        errors.append(
            f"asset_root mismatch: got {asset_root_value!r}, expected {expected_asset_root!r}"
        )
    asset_root = repo_path(asset_root_value or expected_asset_root)

    arena_path = task_relative_path(task_path, task.get("arena_file"), "simbox_arena.yaml")
    if not arena_path.exists():
        errors.append(f"arena missing: {display_path(arena_path)}")
        arena = {"fixtures": []}
    else:
        arena = load_yaml(arena_path) or {"fixtures": []}

    envmap_lib = (task.get("env_map") or {}).get("envmap_lib")
    hdrs = sorted(glob.glob(str(asset_root / str(envmap_lib or "") / "*.hdr")))
    if not hdrs:
        errors.append(f"envmap has no HDR files: {envmap_lib!r}")

    robots = task.get("robots", []) or []
    objects = task.get("objects", []) or []
    fixtures = arena.get("fixtures", []) or []
    for idx, robot in enumerate(robots):
        rel_path = robot.get("path")
        if rel_path and not (asset_root / rel_path).exists():
            errors.append(f"robots[{idx}] path not found: {rel_path}")
    for idx, obj in enumerate(objects):
        cls = obj.get("target_class")
        if cls not in SIMBOX_OBJECT_CLASSES:
            errors.append(f"objects[{idx}] unsupported target_class {cls!r}")
        rel_path = obj.get("path")
        if rel_path and not (asset_root / rel_path).exists():
            errors.append(f"objects[{idx}] path not found: {rel_path}")
        errors.extend(validate_texture_ref(asset_root, f"objects[{idx}]", obj))
    for idx, fixture in enumerate(fixtures):
        cls = fixture.get("target_class")
        if cls not in SIMBOX_OBJECT_CLASSES:
            errors.append(f"arena.fixtures[{idx}] unsupported target_class {cls!r}")
        rel_path = fixture.get("path")
        if rel_path and not (asset_root / rel_path).exists():
            errors.append(f"arena.fixtures[{idx}] path not found: {rel_path}")
        errors.extend(validate_texture_ref(asset_root, f"arena.fixtures[{idx}]", fixture))

    names = (
        {obj["name"] for obj in objects if isinstance(obj, dict) and obj.get("name")}
        | {fixture["name"] for fixture in fixtures if isinstance(fixture, dict) and fixture.get("name")}
        | {robot["name"] for robot in robots if isinstance(robot, dict) and robot.get("name")}
    )
    region_names = {
        region.get("name")
        for region in task.get("regions", []) or []
        if isinstance(region, dict) and region.get("name")
    }

    sampler_extra_count = 0
    for idx, region in enumerate(task.get("regions", []) or []):
        if not isinstance(region, dict):
            continue
        for key in REFERENCE_KEYS:
            value = region.get(key)
            if isinstance(value, str) and value not in names:
                errors.append(f"regions[{idx}].{key} unknown object: {value!r}")
        random_config = region.get("random_config")
        sampler_name = region.get("random_type")
        if isinstance(random_config, dict) and sampler_name in sampler_signatures:
            accepted = sampler_signatures[sampler_name]
            if accepted is None:
                continue
            extras = sorted(set(random_config) - accepted)
            if extras:
                sampler_extra_count += 1
    if sampler_extra_count and not sampler_filter_present:
        errors.append(
            f"regions have {sampler_extra_count} sampler random_config entries with extra keys, "
            "but banana.py does not appear to filter sampler kwargs"
        )

    spawn_missing = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for value in (
            obj.get("spawn_region"),
            (obj.get("placement") or {}).get("spawn_region"),
        ):
            if isinstance(value, str) and value not in region_names:
                spawn_missing += 1
    if spawn_missing:
        errors.append(f"objects have {spawn_missing} spawn_region values not present in regions")

    object_names = {obj["name"] for obj in objects if isinstance(obj, dict) and obj.get("name")}
    fixture_names = {fixture["name"] for fixture in fixtures if isinstance(fixture, dict) and fixture.get("name")}
    all_placeable_names = object_names | fixture_names
    approach_target_names = all_placeable_names | {
        robot["name"] for robot in robots if isinstance(robot, dict) and robot.get("name")
    }
    position_names = set((task.get("positions") or {}).keys())
    skill_structure_errors, skill_count, uses_dag = validate_skill_structure(task.get("skills", []))
    errors.extend(skill_structure_errors)
    skill_ids: set[str] = set()
    for skill in iter_skill_cfgs(task.get("skills", [])):
        skill_id = skill.get("id")
        if skill_id:
            skill_id = str(skill_id)
            if skill_id in skill_ids:
                errors.append(f"duplicate skill id: {skill_id!r}")
            skill_ids.add(skill_id)
        for obj_name in skill.get("objects", []) or []:
            if obj_name not in all_placeable_names:
                errors.append(f"skill {skill.get('name')!r} references unknown object {obj_name!r}")
        if skill.get("name") == "navigate":
            approach_target = str(skill.get("approach", "") or "").strip()
            if approach_target:
                if approach_target not in approach_target_names:
                    errors.append(f"navigate skill references unknown approach target {approach_target!r}")
                errors.extend(validate_navigate_approach_fields(skill))
            elif skill.get("goal") and skill.get("goal") not in position_names:
                errors.append(f"navigate skill references unknown goal {skill.get('goal')!r}")
    for skill in iter_skill_cfgs(task.get("skills", [])):
        for dep_id in skill.get("depends_on", []) or []:
            if str(dep_id) not in skill_ids:
                errors.append(f"skill {skill.get('id')!r} depends on unknown id {dep_id!r}")

    warnings.append(
        " ".join(
            [
                f"hdrs={len(hdrs)}",
                f"robots={len(robots)}",
                f"objects={len(objects)}",
                f"fixtures={len(fixtures)}",
                f"regions={len(task.get('regions', []) or [])}",
                f"skills={skill_count}",
                f"skill_dag={str(uses_dag).lower()}",
                f"sampler_extra={sampler_extra_count}",
            ]
        )
    )
    return errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=["InternDataAssets/assets/custom/scene_8"],
        help=(
            "Scene family root, room root, or inner */assets/basic/<task>/simbox_task.yaml "
            "entry file. Directories are only enumerated for those entries; outer "
            "task.yaml/arena.yaml files are intentionally ignored."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sampler_filter_present = has_sampler_signature_filter()
    sampler_signatures = load_sampler_signatures()
    task_files: list[Path] = []
    for root_arg in args.roots:
        task_files.extend(find_task_files(repo_path(root_arg)))
    task_files = sorted(set(task_files))
    print(f"task_count {len(task_files)}")
    if not task_files:
        print("no simbox_task.yaml files found", file=sys.stderr)
        return 1

    failed = False
    for task_path in task_files:
        errors, warnings = validate_task(task_path, sampler_filter_present, sampler_signatures)
        rel = display_path(task_path)
        status = "FAIL" if errors else "OK"
        print(f"{status} {rel}")
        for warning in warnings:
            print(f"  {warning}")
        for error in errors:
            print(f"  error: {error}")
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
