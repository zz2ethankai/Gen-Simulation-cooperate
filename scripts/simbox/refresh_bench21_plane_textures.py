#!/usr/bin/env python3
"""Losslessly refresh Bench 2.1 floor/wall PNGs for Isaac texture caches."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_ROOT = REPO_ROOT / "InternDataAssets/Bench_2.1_isaacsim/scene_4"
DEFAULT_BACKUP_ROOT = REPO_ROOT / "output/.bench21_texture_backups"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_root = args.bench_root.resolve()
    backup_root = args.backup_root.resolve()
    refreshed = 0

    for room_root in sorted(path for path in bench_root.iterdir() if path.is_dir()):
        for library_name in ("floor_textures", "wall_textures"):
            library = room_root / "texture_libs" / library_name
            if not library.is_dir():
                continue
            for texture_path in sorted(library.glob("*.png")):
                relative_path = texture_path.relative_to(bench_root)
                backup_path = backup_root / relative_path
                if not backup_path.exists():
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(texture_path, backup_path)

                with Image.open(texture_path) as source:
                    rgb = source.convert("RGB")
                    temporary_path = texture_path.with_suffix(".refresh.png")
                    rgb.save(temporary_path, format="PNG")
                temporary_path.replace(texture_path)
                refreshed += 1
                print(f"refreshed: {relative_path}")

    print(f"refreshed {refreshed} RGB plane textures")
    print(f"original texture backups: {backup_root}")


if __name__ == "__main__":
    main()
