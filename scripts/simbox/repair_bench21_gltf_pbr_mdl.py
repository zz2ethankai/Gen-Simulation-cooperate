#!/usr/bin/env python3
"""Clean up dangling 'gltf/pbr.mdl' MDL material references in Bench 2.1 assets.

Omniverse's gltf->usd conversion can leave shader material references pointing
at '@gltf/pbr.mdl@' (Shader ``subIdentifier`` metadata and/or asset-typed
``info:mdl:sourceAsset`` attributes). The library path does not exist on disk,
so usdchecker's MissingReferenceChecker flags it. Isaac Sim generally resolves
the real look through the declared YAML texture override, but the dangling
reference is dead weight and trips the checker.

This tool finds every gltf/*.mdl reference on a USD stage and remaps it to the
resolvable ``OmniPBR.mdl``. It never touches geometry, physics, or texture
bindings. Run with ``--check`` first; it reports only. Original files are
backed up before any modification.

NOTE: the reference is NOT present in the local (Mac) copy of
Bench_2.1_isaacsim — it lives only on the server. Run this script on the
server where the usdchecker warning was produced.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Sdf, Usd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_ROOT = REPO_ROOT / "InternDataAssets/Bench_2.1_isaacsim"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "output/.bench21_gltf_mdl_backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument(
        "--target",
        type=Path,
        action="append",
        default=[],
        help="Explicit USD file(s) to scan. Default: Aligned_obj.usd + source_package/*.usda under --bench-root.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Report only; do not modify any USD (default).",
    )
    action.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite references (must pass --apply to modify).",
    )
    return parser.parse_args()


def default_targets(bench_root: Path) -> list[Path]:
    targets: set[Path] = set()
    for asset_root in bench_root.rglob("Aligned_obj.usd"):
        targets.add(asset_root)
        targets.update(asset_root.parent.rglob("source_package/*.usda"))
    return sorted(targets)


def collect_targets(args) -> list[Path]:
    if args.target:
        return [p.resolve() for p in args.target]
    return default_targets(args.bench_root.resolve())


def gltf_mdl_refs(stage: Usd.Stage) -> list[tuple[str, object]]:
    """Return (location_label, current_value) for every gltf/*.mdl reference."""
    refs: list[tuple[str, object]] = []
    for prim in stage.Traverse():
        sub_id = prim.GetMetadata("subIdentifier")
        if isinstance(sub_id, str) and sub_id.startswith("gltf/"):
            refs.append((f"{prim.GetPath()} subIdentifier", sub_id))
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and str(value.path).startswith("gltf/"):
                refs.append((f"{attr.GetPath()}", str(value.path)))
    return refs


def repair_refs(stage: Usd.Stage) -> int:
    n = 0
    for prim in stage.Traverse():
        sub_id = prim.GetMetadata("subIdentifier")
        if isinstance(sub_id, str) and sub_id.startswith("gltf/"):
            prim.SetMetadata("subIdentifier", "OmniPBR.mdl")
            n += 1
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and str(value.path).startswith("gltf/"):
                attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
                n += 1
    return n


def main() -> None:
    args = parse_args()
    bench_root = args.bench_root.resolve()
    backup_root = args.backup_root.resolve()
    targets = collect_targets(args)
    affected = 0
    refs_total = 0

    for usd_path in targets:
        if not usd_path.is_file():
            print(f"skip (missing): {usd_path}")
            continue
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            print(f"skip (cannot open): {usd_path}")
            continue
        refs = gltf_mdl_refs(stage)
        if not refs:
            continue
        relative_path = usd_path.relative_to(bench_root)
        print(f"found {len(refs)} gltf/*.mdl reference(s) in {relative_path}:")
        for label, value in refs:
            print(f"  {label} = {value}")
        affected += 1
        refs_total += len(refs)
        if not args.apply:
            continue
        backup_path = backup_root / relative_path
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(usd_path, backup_path)
        n = repair_refs(stage)
        stage.GetRootLayer().Save()
        print(f"  repaired {n} reference(s); backup: {backup_path}")

    mode = "repaired" if args.apply else "would repair (--apply)"
    print(f"gltf/pbr.mdl cleanup complete: {mode} {refs_total} reference(s) in {affected} file(s)")
    if affected == 0:
        print("No gltf/*.mdl references found in scanned files.")
    elif args.apply:
        print(f"original USD backups: {backup_root}")


if __name__ == "__main__":
    main()


