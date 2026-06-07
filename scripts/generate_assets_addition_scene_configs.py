#!/usr/bin/env python3
"""Generate SimBox arena/task YAMLs from assets_addition scene packages."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("workflows/simbox/assets")
DEFAULT_ADDITION_ROOT = DEFAULT_ASSET_ROOT / "assets_addition"
DEFAULT_ARENA_OUT_DIR = Path("workflows/simbox/core/configs/arenas/addition")
DEFAULT_TASK_OUT_DIR = Path("workflows/simbox/core/configs/tasks/addition")


class SimBoxYamlDumper(yaml.SafeDumper):
    pass


def represent_bool(dumper: yaml.Dumper, value: bool) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:bool", "True" if value else "False")


SimBoxYamlDumper.add_representer(bool, represent_bool)


def sanitize_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    name = re.sub(r"_+", "_", name)
    if not name:
        name = "asset"
    if name[0].isdigit():
        name = f"asset_{name}"
    return name


def unique_name(base: str, used: set[str]) -> str:
    name = base
    idx = 1
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def posix_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is not under asset_root {root}") from exc


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    manifest_path = scene_dir / "package_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sorted_usds(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.exists():
        return []
    pattern = "**/*.usd" if recursive else "*.usd"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def sorted_aligned_object_usds(directory: Path) -> list[Path]:
    """Return one exported object USD per asset directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*/Aligned_obj.usd") if p.is_file())


def object_category_from_stem(stem: str, scene_name: str) -> str:
    prefix = f"{scene_name}__"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    stem = re.sub(r"_\d+_id\d+$", "", stem)
    stem = re.sub(r"_id\d+$", "", stem)
    stem = re.sub(r"^mesh_small_", "", stem)
    stem = re.sub(r"_mesh_\d+_\d+$", "", stem)
    return sanitize_name(stem)


def geometry_fixture(name: str, rel_path: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": rel_path,
        "target_class": "GeometryObject",
        "translation": [0.0, 0.0, 0.0],
        "euler": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
    }


def geometry_object(name: str, rel_path: str, category: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": rel_path,
        "target_class": "GeometryObject",
        "dataset": "assets_addition",
        "category": category,
        "translation": [0.0, 0.0, 0.0],
        "euler": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "apply_randomization": False,
    }


def require_usd_modules() -> tuple[Any, Any]:
    try:
        from pxr import Gf, Sdf
    except ImportError as exc:
        raise RuntimeError(
            "USD parsing requires pxr. Run this script in an environment with usd-core, "
            "for example /home/dyf/miniconda3/envs/anygrasp/bin/python."
        ) from exc
    return Gf, Sdf


def list_references(prim_spec: Any) -> list[str]:
    references: list[str] = []
    for attr in ("prependedItems", "addedItems", "explicitItems"):
        references.extend(ref.assetPath for ref in getattr(prim_spec.referenceList, attr))
    return references


def strip_scene_prefix(asset_stem: str, scene_name: str) -> str:
    prefix = f"{scene_name}__"
    if asset_stem.startswith(prefix):
        return asset_stem[len(prefix) :]
    return asset_stem


def aligned_task_asset_from_scene_ref(scene_dir: Path, ref_path: str, scene_name: str) -> Path | None:
    ref = Path(ref_path)
    parts = ref.parts
    if len(parts) < 3 or parts[0] != "assets" or parts[1] != "task_obj":
        return None

    stem = ref.stem
    if len(parts) >= 4 and parts[2] == "small_usd":
        return scene_dir / "assets" / "task_obj" / "small_usd" / stem / "Aligned_obj.usd"

    return scene_dir / "assets" / "task_obj" / strip_scene_prefix(stem, scene_name) / "Aligned_obj.usd"


def xform_from_prim_spec(prim_spec: Any, gf_module: Any) -> tuple[list[float], list[float], list[float]]:
    props = {prop.name: prop.default for prop in prim_spec.properties}
    translation = props.get("xformOp:translate", gf_module.Vec3d(0.0, 0.0, 0.0))
    scale = props.get("xformOp:scale", gf_module.Vec3d(1.0, 1.0, 1.0))
    yaw_matrix = props.get("xformOp:transform:yaw")
    yaw = 0.0
    if yaw_matrix is not None:
        yaw = math.degrees(math.atan2(yaw_matrix[1][0], yaw_matrix[0][0]))

    return (
        [round(float(translation[0]), 6), round(float(translation[1]), 6), round(float(translation[2]), 6)],
        [0.0, 0.0, round(float(yaw), 6)],
        [round(float(scale[0]), 6), round(float(scale[1]), 6), round(float(scale[2]), 6)],
    )


def iter_prim_specs(prim_spec: Any):
    yield prim_spec
    for child in prim_spec.nameChildren:
        yield from iter_prim_specs(child)


def build_task_objects_from_scene_usd(
    scene_dir: Path,
    asset_root: Path,
    include_small_task_objects: bool,
) -> list[dict[str, Any]]:
    Gf, Sdf = require_usd_modules()
    scene_name = scene_dir.name
    scene_file = scene_dir / load_manifest(scene_dir).get("layout", {}).get("full_scene", "scene.usd")
    if not scene_file.exists():
        raise FileNotFoundError(scene_file)

    root_layer = Sdf.Layer.FindOrOpen(str(scene_file))
    if root_layer is None:
        raise RuntimeError(f"Failed to open USD layer: {scene_file}")

    used: set[str] = set()
    objects: list[dict[str, Any]] = []
    missing_assets: list[str] = []

    for root_prim in root_layer.rootPrims:
        for spec in iter_prim_specs(root_prim):
            refs = [ref for ref in list_references(spec) if ref.startswith("assets/task_obj/")]
            if not refs:
                continue

            ref_path = refs[0]
            is_small = ref_path.startswith("assets/task_obj/small_usd/")
            if is_small and not include_small_task_objects:
                continue

            aligned_usd = aligned_task_asset_from_scene_ref(scene_dir, ref_path, scene_name)
            if aligned_usd is None:
                continue
            if not aligned_usd.exists():
                missing_assets.append(f"{spec.path}: {ref_path} -> {aligned_usd}")
                continue

            translation, euler, scale = xform_from_prim_spec(spec, Gf)
            asset_stem = Path(ref_path).stem
            object_name = unique_name(sanitize_name(spec.name), used)

            obj = geometry_object(
                object_name,
                posix_rel(aligned_usd, asset_root),
                object_category_from_stem(strip_scene_prefix(asset_stem, scene_name), scene_name),
            )
            obj["translation"] = translation
            obj["euler"] = euler
            obj["scale"] = scale
            obj["scene_prim_path"] = spec.path.pathString
            obj["scene_reference"] = ref_path
            objects.append(obj)

    if missing_assets:
        preview = "\n".join(missing_assets[:20])
        extra = "" if len(missing_assets) <= 20 else f"\n... and {len(missing_assets) - 20} more"
        raise FileNotFoundError(f"Missing converted Aligned_obj.usd assets:\n{preview}{extra}")

    if not objects:
        raise RuntimeError(f"No task_obj references found in {scene_file}")

    return objects


def build_arena(
    scene_dir: Path,
    asset_root: Path,
    mode: str,
    include_floor_plane: bool,
) -> dict[str, Any]:
    scene_name = scene_dir.name
    manifest = load_manifest(scene_dir)
    layout = manifest.get("layout", {})
    used: set[str] = set()
    fixtures: list[dict[str, Any]] = []

    if mode == "full-scene":
        scene_file = scene_dir / layout.get("full_scene", "scene.usd")
        fixtures.append(geometry_fixture("scene", posix_rel(scene_file, asset_root)))
    else:
        empty_room = scene_dir / layout.get("empty_room", "empty_room.usd")
        if empty_room.exists():
            fixtures.append(geometry_fixture("empty_room", posix_rel(empty_room, asset_root)))

        arena_dir = scene_dir / layout.get("arena_assets", "assets/arena")
        for usd_path in sorted_usds(arena_dir, recursive=False):
            name = unique_name(sanitize_name(usd_path.stem), used)
            fixtures.append(geometry_fixture(name, posix_rel(usd_path, asset_root)))

    if include_floor_plane:
        fixtures.append(
            {
                "name": "floor",
                "target_class": "PlaneObject",
                "size": [8.0, 8.0],
                "translation": [0.0, 0.0, 0.0],
                "collision_enabled": True,
                "collision_thickness": 0.02,
            }
        )

    return {
        "name": f"assets_addition_{scene_name}_arena",
        "fixtures": fixtures,
    }


def build_task_objects(
    scene_dir: Path,
    asset_root: Path,
    include_small_task_objects: bool,
) -> list[dict[str, Any]]:
    return build_task_objects_from_scene_usd(
        scene_dir=scene_dir,
        asset_root=asset_root,
        include_small_task_objects=include_small_task_objects,
    )


def build_task_objects_from_exported_assets(
    scene_dir: Path,
    asset_root: Path,
    include_small_task_objects: bool,
) -> list[dict[str, Any]]:
    scene_name = scene_dir.name
    layout = load_manifest(scene_dir).get("layout", {})
    task_dir = scene_dir / layout.get("task_assets", "assets/task_obj")
    usd_paths = sorted_aligned_object_usds(task_dir)
    if not usd_paths:
        usd_paths = sorted_usds(task_dir, recursive=False)
    if include_small_task_objects:
        small_usd_paths = sorted_aligned_object_usds(task_dir / "small_usd")
        if not small_usd_paths:
            small_usd_paths = sorted_usds(task_dir / "small_usd", recursive=False)
        usd_paths.extend(small_usd_paths)

    used: set[str] = set()
    objects: list[dict[str, Any]] = []
    for usd_path in usd_paths:
        asset_stem = usd_path.parent.name if usd_path.name == "Aligned_obj.usd" else usd_path.stem
        name = unique_name(sanitize_name(asset_stem), used)
        category = object_category_from_stem(asset_stem, scene_name)
        objects.append(geometry_object(name, posix_rel(usd_path, asset_root), category))
    return objects


def build_task(
    scene_dir: Path,
    asset_root_cfg: str,
    arena_file: Path,
    objects: list[dict[str, Any]],
    max_episode_length: int,
) -> dict[str, Any]:
    scene_name = scene_dir.name
    task_name = f"assets_addition_{scene_name}_scene_task"
    arena_file_rel = arena_file.relative_to(REPO_ROOT).as_posix() if arena_file.is_absolute() else arena_file.as_posix()
    return {
        "tasks": [
            {
                "name": task_name,
                "asset_root": asset_root_cfg,
                "task": "BananaBaseTask",
                "task_id": 0,
                "offset": None,
                "render": True,
                "arena_file": arena_file_rel,
                "env_map": {
                    "envmap_lib": "envmap_lib",
                    "apply_randomization": False,
                    "intensity_range": [5000, 5000],
                    "rotation_range": [0, 0],
                },
                "robots": [],
                "objects": objects,
                "regions": [],
                "cameras": [],
                "data": {
                    "task_dir": f"assets_addition/{scene_name}",
                    "language_instruction": f"Load assets_addition scene {scene_name}.",
                    "detailed_language_instruction": (
                        f"Load assets_addition scene {scene_name} without robots or action skills."
                    ),
                    "collect_info": f"assets_addition_{scene_name}",
                    "version": "v1.0",
                    "update": True,
                    "max_episode_length": max_episode_length,
                },
                "skills": [],
            }
        ]
    }


def write_yaml(path: Path, payload: dict[str, Any], overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    text = yaml.dump(
        payload,
        Dumper=SimBoxYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if dry_run:
        print(f"[DRY-RUN] Would write {path}")
        print(text)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[WROTE] {path.relative_to(REPO_ROOT) if path.is_absolute() else path}")


def discover_scene_dirs(args: argparse.Namespace) -> list[Path]:
    if args.scene_dir:
        return [repo_path(Path(p)) for p in args.scene_dir]

    addition_root = repo_path(args.addition_root)
    if args.all:
        return sorted(p for p in addition_root.iterdir() if p.is_dir() and (p / "package_manifest.json").exists())

    raise SystemExit("Pass --scene-dir PATH or --all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scene-only SimBox arena/task YAMLs from assets_addition scene packages."
    )
    parser.add_argument("--scene-dir", action="append", help="Path to an assets_addition/file_* scene directory.")
    parser.add_argument("--all", action="store_true", help="Generate configs for all file_* scenes under addition root.")
    parser.add_argument("--addition-root", type=Path, default=DEFAULT_ADDITION_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-arena-dir", type=Path, default=DEFAULT_ARENA_OUT_DIR)
    parser.add_argument("--output-task-dir", type=Path, default=DEFAULT_TASK_OUT_DIR)
    parser.add_argument("--mode", choices=["split", "full-scene"], default="split")
    parser.add_argument(
        "--exclude-small-task-objects",
        action="store_true",
        help="Skip assets/task_obj/small_usd/*/Aligned_obj.usd entries in the generated task yaml.",
    )
    parser.add_argument(
        "--include-floor-plane",
        action="store_true",
        help="Add a synthetic PlaneObject named floor to the arena yaml.",
    )
    parser.add_argument("--max-episode-length", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset_root = repo_path(args.asset_root)
    arena_out_dir = repo_path(args.output_arena_dir)
    task_out_dir = repo_path(args.output_task_dir)

    scene_dirs = discover_scene_dirs(args)
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            raise FileNotFoundError(scene_dir)
        if not (scene_dir / "package_manifest.json").exists():
            raise FileNotFoundError(f"Missing package_manifest.json in {scene_dir}")

        scene_name = scene_dir.name
        arena_path = arena_out_dir / f"{scene_name}_arena.yaml"
        task_path = task_out_dir / f"{scene_name}_scene_task.yaml"

        arena_payload = build_arena(
            scene_dir=scene_dir,
            asset_root=asset_root,
            mode=args.mode,
            include_floor_plane=args.include_floor_plane,
        )
        task_objects = [] if args.mode == "full-scene" else build_task_objects(
            scene_dir=scene_dir,
            asset_root=asset_root,
            include_small_task_objects=not args.exclude_small_task_objects,
        )
        task_payload = build_task(
            scene_dir=scene_dir,
            asset_root_cfg=args.asset_root.as_posix(),
            arena_file=arena_path,
            objects=task_objects,
            max_episode_length=args.max_episode_length,
        )

        write_yaml(arena_path, arena_payload, overwrite=args.overwrite, dry_run=args.dry_run)
        write_yaml(task_path, task_payload, overwrite=args.overwrite, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
