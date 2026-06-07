#!/usr/bin/env python3
"""Install scene_4 assets under workflows/simbox/assets/custom and fix paths."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - depends on caller env
    raise SystemExit(
        "PyYAML is required. Run with an environment that provides yaml, for example "
        "/home/dyf/miniconda3/envs/anygrasp/bin/python."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("download/new/scene_4")
DEFAULT_DEST_ROOT = Path("workflows/simbox/assets/custom/scene_4")
DEFAULT_ENVMAP_LIB = Path("workflows/simbox/example_assets/envmap_lib")
DEFAULT_ROBOT_USD = Path("workflows/simbox/example_assets/split_aloha_mid_360/robot.usd")

ABSOLUTE_PATH_RE = re.compile(r"/[A-Za-z0-9_./@:+,=-]+")
TEXT_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


@dataclass
class Report:
    copied_scenes: list[str] = field(default_factory=list)
    task_files: int = 0
    task_path_updates: int = 0
    yaml_files: int = 0
    yaml_path_updates: int = 0
    text_files_scanned: int = 0
    text_files_rewritten: int = 0
    text_path_updates: int = 0
    skipped_binary_files: list[Path] = field(default_factory=list)
    missing_mapped_targets: dict[Path, set[str]] = field(default_factory=dict)
    unresolved_absolute_paths: dict[Path, set[str]] = field(default_factory=dict)

    def add_missing_target(self, path: Path, value: str) -> None:
        self.missing_mapped_targets.setdefault(path, set()).add(value)

    def add_unresolved(self, path: Path, value: str) -> None:
        self.unresolved_absolute_paths.setdefault(path, set()).add(value)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def repo_relative(path: Path) -> str:
    return os.path.relpath(os.path.abspath(path), REPO_ROOT).replace(os.sep, "/")


def relpath_between(target: Path, start: Path) -> str:
    return os.path.relpath(os.path.abspath(target), os.path.abspath(start)).replace(os.sep, "/")


def filesystem_relpath_between(target: Path, start: Path) -> str:
    return os.path.relpath(os.path.realpath(target), os.path.realpath(start)).replace(os.sep, "/")


def display_path(path: Path) -> str:
    return repo_relative(path)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: Any, dry_run: bool) -> None:
    if dry_run:
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False, width=120)


def scene_dirs(source_root: Path) -> list[Path]:
    return [path for path in sorted(source_root.iterdir()) if path.is_dir()]


def copy_scene_dirs(source_root: Path, dest_root: Path, overwrite: bool, dry_run: bool, report: Report) -> None:
    for source_scene in scene_dirs(source_root):
        dest_scene = dest_root / source_scene.name
        if dest_scene.exists() and not overwrite:
            raise FileExistsError(f"{dest_scene} already exists; pass --overwrite to replace/update it")
        report.copied_scenes.append(source_scene.name)
        if dry_run:
            continue
        shutil.copytree(source_scene, dest_scene, dirs_exist_ok=overwrite)


def rewrite_task_yaml(
    task_path: Path,
    dest_scene_root: Path,
    envmap_lib: Path,
    robot_usd: Path,
    dry_run: bool,
    report: Report,
) -> None:
    payload = load_yaml(task_path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return

    changed = False
    asset_root_cfg = repo_relative(dest_scene_root)
    envmap_cfg = filesystem_relpath_between(envmap_lib, dest_scene_root)
    robot_cfg = filesystem_relpath_between(robot_usd, dest_scene_root)

    for task in tasks:
        if not isinstance(task, dict):
            continue

        if task.get("asset_root") != asset_root_cfg:
            task["asset_root"] = asset_root_cfg
            changed = True
            report.task_path_updates += 1

        arena_file = task_path.with_name("simbox_arena.yaml")
        arena_cfg = repo_relative(arena_file)
        if task.get("arena_file") != arena_cfg:
            task["arena_file"] = arena_cfg
            changed = True
            report.task_path_updates += 1

        env_map = task.get("env_map")
        if isinstance(env_map, dict) and env_map.get("envmap_lib") != envmap_cfg:
            env_map["envmap_lib"] = envmap_cfg
            changed = True
            report.task_path_updates += 1

        robots = task.get("robots")
        if isinstance(robots, list):
            for robot in robots:
                if isinstance(robot, dict) and robot.get("path") != robot_cfg:
                    robot["path"] = robot_cfg
                    changed = True
                    report.task_path_updates += 1

    if changed:
        write_yaml(task_path, payload, dry_run=dry_run)
        report.task_files += 1


def mapped_absolute_path(abs_path: str, scene_name: str, dest_scene_root: Path) -> Path | None:
    marker = f"/scene_4/{scene_name}/"
    if marker in abs_path:
        suffix = abs_path.split(marker, 1)[1]
        return dest_scene_root / suffix

    dataset_marker = f"/{scene_name}/"
    if "Bench_2.0_isaac" in abs_path and dataset_marker in abs_path:
        suffix = abs_path.split(dataset_marker, 1)[1]
        return dest_scene_root / suffix

    return None


def replacement_for_absolute_path(
    abs_path: str,
    text_file: Path,
    scene_name: str,
    dest_scene_root: Path,
    report: Report,
) -> str | None:
    mapped = mapped_absolute_path(abs_path, scene_name, dest_scene_root)
    if mapped is None:
        return None
    if not mapped.exists():
        report.add_missing_target(text_file, relpath_between(mapped, text_file.parent))
    return relpath_between(mapped, text_file.parent)


def rewrite_text_asset_paths(text_file: Path, scene_name: str, dest_scene_root: Path, dry_run: bool, report: Report) -> None:
    try:
        original = text_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        report.skipped_binary_files.append(text_file)
        return

    report.text_files_scanned += 1
    replacements: dict[str, str] = {}
    for match in ABSOLUTE_PATH_RE.finditer(original):
        abs_path = match.group(0).rstrip("@,)]}\"'")
        replacement = replacement_for_absolute_path(abs_path, text_file, scene_name, dest_scene_root, report)
        if replacement is None:
            if any(token in abs_path for token in ("/root/autodl-tmp", "/home/mz", "scene_4_deliverable", "Bench_2.0_isaac")):
                report.add_unresolved(text_file, abs_path)
            continue
        replacements[abs_path] = replacement

    if not replacements:
        return

    rewritten = original
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        rewritten = rewritten.replace(old, new)

    if rewritten != original:
        report.text_files_rewritten += 1
        report.text_path_updates += len(replacements)
        if not dry_run:
            text_file.write_text(rewritten, encoding="utf-8")


def rewrite_yaml_absolute_strings(yaml_file: Path, scene_name: str, dest_scene_root: Path, dry_run: bool, report: Report) -> None:
    payload = load_yaml(yaml_file)
    updates = 0

    def visit(value: Any) -> Any:
        nonlocal updates
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, str) or not value.startswith("/"):
            return value

        mapped = mapped_absolute_path(value, scene_name, dest_scene_root)
        if mapped is None:
            if any(token in value for token in ("/root/autodl-tmp", "/home/mz", "scene_4_deliverable", "Bench_2.0_isaac")):
                report.add_unresolved(yaml_file, value)
            return value

        if not mapped.exists():
            report.add_missing_target(yaml_file, relpath_between(mapped, yaml_file.parent))
        updates += 1
        return relpath_between(mapped, yaml_file.parent)

    rewritten = visit(payload)
    if updates:
        report.yaml_files += 1
        report.yaml_path_updates += updates
        write_yaml(yaml_file, rewritten, dry_run=dry_run)


def rewrite_copied_paths(dest_root: Path, envmap_lib: Path, robot_usd: Path, dry_run: bool, report: Report) -> None:
    for dest_scene_root in scene_dirs(dest_root):
        for task_path in sorted(dest_scene_root.glob("assets/basic/*/simbox_task.yaml")):
            rewrite_task_yaml(
                task_path=task_path,
                dest_scene_root=dest_scene_root,
                envmap_lib=envmap_lib,
                robot_usd=robot_usd,
                dry_run=dry_run,
                report=report,
            )

        for yaml_file in sorted(dest_scene_root.rglob("*.yaml")) + sorted(dest_scene_root.rglob("*.yml")):
            rewrite_yaml_absolute_strings(
                yaml_file=yaml_file,
                scene_name=dest_scene_root.name,
                dest_scene_root=dest_scene_root,
                dry_run=dry_run,
                report=report,
            )

        for text_file in sorted(dest_scene_root.rglob("*")):
            if not text_file.is_file():
                continue
            if text_file.suffix.lower() not in TEXT_USD_SUFFIXES:
                continue
            rewrite_text_asset_paths(
                text_file=text_file,
                scene_name=dest_scene_root.name,
                dest_scene_root=dest_scene_root,
                dry_run=dry_run,
                report=report,
            )


def validate_inputs(source_root: Path, dest_root: Path, envmap_lib: Path, robot_usd: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not envmap_lib.is_dir():
        raise FileNotFoundError(f"envmap lib does not exist: {envmap_lib}")
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot USD does not exist: {robot_usd}")
    dest_root.parent.mkdir(parents=True, exist_ok=True)


def print_report(report: Report, dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "DONE"
    print(f"[{mode}] copied/updated scenes: {len(report.copied_scenes)}")
    for scene in report.copied_scenes:
        print(f"  - {scene}")
    print(f"[{mode}] rewritten simbox_task.yaml files: {report.task_files}")
    print(f"[{mode}] task path field updates: {report.task_path_updates}")
    print(f"[{mode}] rewritten YAML files with extra absolute paths: {report.yaml_files}")
    print(f"[{mode}] extra YAML path updates: {report.yaml_path_updates}")
    print(f"[{mode}] scanned text USD files: {report.text_files_scanned}")
    print(f"[{mode}] rewritten text USD files: {report.text_files_rewritten}")
    print(f"[{mode}] text asset path updates: {report.text_path_updates}")
    print(f"[{mode}] skipped binary USD files: {len(report.skipped_binary_files)}")
    missing_total = sum(len(values) for values in report.missing_mapped_targets.values())
    print(f"[{mode}] relative paths with missing targets: {missing_total}")
    for path, values in sorted(report.missing_mapped_targets.items()):
        print(f"  {display_path(path)}")
        for value in sorted(values):
            print(f"    - {value}")
    unresolved_total = sum(len(values) for values in report.unresolved_absolute_paths.values())
    print(f"[{mode}] unresolved absolute paths: {unresolved_total}")
    for path, values in sorted(report.unresolved_absolute_paths.items()):
        print(f"  {display_path(path)}")
        for value in sorted(values):
            print(f"    - {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--envmap-lib", type=Path, default=DEFAULT_ENVMAP_LIB)
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--overwrite", action="store_true", help="Allow updating an existing destination tree.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = repo_path(args.source_root)
    dest_root = repo_path(args.dest_root)
    envmap_lib = repo_path(args.envmap_lib)
    robot_usd = repo_path(args.robot_usd)

    validate_inputs(source_root, dest_root, envmap_lib, robot_usd)
    report = Report()
    copy_scene_dirs(source_root, dest_root, overwrite=args.overwrite, dry_run=args.dry_run, report=report)

    if not args.dry_run:
        rewrite_copied_paths(dest_root, envmap_lib, robot_usd, dry_run=False, report=report)
    else:
        rewrite_root = dest_root if dest_root.exists() else source_root
        rewrite_copied_paths(rewrite_root, envmap_lib, robot_usd, dry_run=True, report=report)

    print_report(report, dry_run=args.dry_run)
    return 1 if report.unresolved_absolute_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
