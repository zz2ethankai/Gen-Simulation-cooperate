#!/usr/bin/env python3
"""Copy original OBJ files from download/ into normalized assets_addition folders."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


SMALL_PREFIX = "mesh_small_"
SMALL_SUFFIX = "_mesh_0_00"


def normalize_name(value: str) -> str:
    value = value.lower().replace("-", "_")
    value = re.sub(r"_obj$", "", value)
    value = re.sub(r"__+", "_", value)
    return value.strip("_")


def scene_name(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"file_\d+", part):
            return part
    return None


def target_name(asset_dir: Path) -> str:
    return normalize_name(asset_dir.name)


def small_candidates(name: str) -> list[str]:
    if not (name.startswith(SMALL_PREFIX) and name.endswith(SMALL_SUFFIX)):
        return []
    core = name[len(SMALL_PREFIX) : -len(SMALL_SUFFIX)]
    candidates: list[str] = []

    def add(value: str) -> None:
        value = normalize_name(value)
        if value and value not in candidates:
            candidates.append(value)

    add(core)
    parts = core.split("_")

    # The USD names often append placement/instance ids, e.g.
    # bedside_lamp_0_1_0054 -> bedside_lamp_0_1 -> bedside_lamp_0.
    current = parts[:]
    while current and current[-1].isdigit():
        current = current[:-1]
        add("_".join(current))

    # Also keep a class-only fallback.
    add("_".join([part for part in parts if not part.isdigit()]))
    return candidates


def index_download_objs(download_root: Path) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], list[Path]]]:
    hub: dict[tuple[str, str], Path] = {}
    mesh_out: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for obj in sorted(download_root.glob("scenes/file_*/assets_hub/*/model.obj")):
        scene = scene_name(obj)
        if scene is None:
            continue
        hub[(scene, normalize_name(obj.parent.name))] = obj

    for obj in sorted(download_root.glob("scenes/file_*/mesh_out/**/*.obj")):
        scene = scene_name(obj)
        if scene is None:
            continue
        mesh_out[(scene, normalize_name(obj.stem))].append(obj)
        mesh_out[(scene, normalize_name(obj.parent.name))].append(obj)
    return hub, mesh_out


def iter_asset_dirs(asset_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for usd in sorted(asset_root.glob("file_*/assets/task_obj/*/Aligned_obj.usd")):
        dirs.append(usd.parent)
    for usd in sorted(asset_root.glob("file_*/assets/task_obj/small_usd/*/Aligned_obj.usd")):
        dirs.append(usd.parent)
    return dirs


def choose_match(asset_dir: Path, hub: dict[tuple[str, str], Path], mesh_out: dict[tuple[str, str], list[Path]]) -> Path | None:
    scene = scene_name(asset_dir)
    if scene is None:
        return None
    name = target_name(asset_dir)

    hub_match = hub.get((scene, name))
    if hub_match is not None:
        return hub_match

    for candidate in small_candidates(name):
        matches = mesh_out.get((scene, candidate), [])
        exact = [p for p in matches if p.name == f"{candidate}.obj"]
        if exact:
            return exact[0]
        if matches:
            return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=Path("InternDataAssets/assets/assets_addition"))
    parser.add_argument("--download-root", type=Path, default=Path("download"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    hub, mesh_out = index_download_objs(args.download_root)
    asset_dirs = iter_asset_dirs(args.asset_root)

    matched: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    ambiguous: list[tuple[Path, list[Path]]] = []

    for asset_dir in asset_dirs:
        source = choose_match(asset_dir, hub, mesh_out)
        if source is None:
            missing.append(asset_dir)
            continue
        matched.append((asset_dir, source))

    print(f"asset_dirs={len(asset_dirs)} matched={len(matched)} missing={len(missing)}")
    print(f"download_assets_hub={len(hub)} download_mesh_out_keys={len(mesh_out)}")
    for asset_dir, source in matched[:20]:
        print(f"MATCH {asset_dir} <- {source}")
    if len(matched) > 20:
        print(f"... {len(matched) - 20} more matches")

    if missing:
        print("\nMissing matches:")
        for path in missing[:100]:
            print(path)
        if len(missing) > 100:
            print(f"... {len(missing) - 100} more missing")
        return 1

    if ambiguous:
        print("\nAmbiguous matches:")
        for path, options in ambiguous[:100]:
            print(path)
            for option in options:
                print(f"  {option}")
        return 1

    if args.execute:
        copied = 0
        skipped = 0
        for asset_dir, source in matched:
            target = asset_dir / "Aligned_obj.obj"
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            shutil.copy2(source, target)
            copied += 1
        print(f"copied={copied} skipped={skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
