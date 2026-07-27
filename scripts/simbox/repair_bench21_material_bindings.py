#!/usr/bin/env python3
"""Restore Bench 2.1 canonical USD assets to their declared RGB textures.

SimBox follows the official rule that a YAML ``texture`` block explicitly
overrides an object's material, while objects without that block retain the
material authored in their USD. Some canonical Bench 2.1 ``Aligned_obj.usd``
files bind ``semantic_*.png`` even though the room asset manifest declares
``texture.png`` as the primary RGB texture. This tool repairs only that
inconsistent binding and leaves geometry, physics and YAML overrides intact.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Sdf, Usd, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_ROOT = REPO_ROOT / "InternDataAssets/Bench_2.1_isaacsim/scene_4"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "output/.bench21_material_backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--check", action="store_true", help="Report only; do not modify USD files")
    return parser.parse_args()


def semantic_texture_inputs(stage: Usd.Stage) -> list[UsdShade.Input]:
    inputs = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        texture_input = UsdShade.Shader(prim).GetInput("file")
        value = texture_input.Get() if texture_input else None
        if value and Path(str(value)).name.startswith("semantic_"):
            inputs.append(texture_input)
    return inputs


def canonical_asset_paths(bench_root: Path):
    for room_root in sorted(path for path in bench_root.iterdir() if (path / "asset_manifest.json").is_file()):
        canonical_root = room_root / "assets/basic" / room_root.name
        yield from sorted(canonical_root.rglob("Aligned_obj.usd"))


def main() -> None:
    args = parse_args()
    bench_root = args.bench_root.resolve()
    backup_root = args.backup_root.resolve()
    repaired = 0
    already_correct = 0
    skipped_missing_texture = 0

    for usd_path in canonical_asset_paths(bench_root):
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"cannot open USD: {usd_path}")

        texture_inputs = semantic_texture_inputs(stage)
        if not texture_inputs:
            already_correct += 1
            continue

        rgb_texture = usd_path.with_name("texture.png")
        if not rgb_texture.is_file():
            skipped_missing_texture += 1
            print(f"skip (no declared RGB texture): {usd_path.relative_to(bench_root)}")
            continue

        relative_path = usd_path.relative_to(bench_root)
        if args.check:
            print(f"would repair: {relative_path} -> texture.png")
            repaired += 1
            continue

        backup_path = backup_root / relative_path
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(usd_path, backup_path)

        for texture_input in texture_inputs:
            texture_input.Set(Sdf.AssetPath("texture.png"))
        stage.GetRootLayer().Save()
        repaired += 1
        print(f"repaired: {relative_path} -> texture.png")

    mode = "would repair" if args.check else "repaired"
    print(
        f"material audit complete: {mode} {repaired}, "
        f"already correct {already_correct}, skipped without texture.png {skipped_missing_texture}"
    )
    if not args.check and repaired:
        print(f"original USD backups: {backup_root}")


if __name__ == "__main__":
    main()
