#!/usr/bin/env python3
"""Download split 7z InternDataAssets archive from ModelScope and extract it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "MinMaxMex/InterndataAssets"
REPO_TYPE = "dataset"
PATH_IN_REPO = "InternDataAssets_7z"
ARCHIVE_NAME = "InternDataAssets.7z"
DOWNLOAD_DIR = REPO_ROOT / "output" / "modelscope_interndata_assets_download"
TARGET_DIR = REPO_ROOT / "InternDataAssets"
SIMBOX_DIR = REPO_ROOT / "workflows" / "simbox"
SIMBOX_LINKS = {
    "assets": Path("../../InternDataAssets/assets"),
    "panda_drake": Path("../../InternDataAssets/panda_drake"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        required=True,
        help="ModelScope access token.",
    )
    return parser.parse_args()


def run_7z_extract(first_part: Path) -> None:
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z is required but was not found in PATH")

    if TARGET_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {TARGET_DIR}")

    cmd = [seven_zip, "x", str(first_part), f"-o{REPO_ROOT}", "-y"]
    print("Extracting:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def ensure_simbox_symlinks() -> None:
    if not SIMBOX_DIR.is_dir():
        raise FileNotFoundError(f"Missing SimBox directory: {SIMBOX_DIR}")

    for link_name, relative_target in SIMBOX_LINKS.items():
        link_path = SIMBOX_DIR / link_name
        source_path = SIMBOX_DIR / relative_target
        if not source_path.is_dir():
            raise FileNotFoundError(f"Missing symlink target: {relative_target}")

        if link_path.is_symlink():
            current_target = os.readlink(link_path)
            if current_target == str(relative_target):
                print(f"Symlink already exists: {link_path} -> {current_target}")
                continue
            link_path.unlink()
        elif link_path.exists():
            raise FileExistsError(f"Refusing to replace non-symlink path: {link_path}")

        link_path.symlink_to(relative_target, target_is_directory=True)
        print(f"Created symlink: {link_path} -> {relative_target}")


def main() -> int:
    args = parse_args()

    from modelscope.hub.snapshot_download import snapshot_download

    print(f"Downloading {REPO_TYPE} {REPO_ID}/{PATH_IN_REPO}/{ARCHIVE_NAME}.*")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=f"{PATH_IN_REPO}/{ARCHIVE_NAME}.*",
        local_dir=str(DOWNLOAD_DIR),
        token=args.token,
    )

    first_part = DOWNLOAD_DIR / PATH_IN_REPO / f"{ARCHIVE_NAME}.001"
    if not first_part.is_file():
        first_part = DOWNLOAD_DIR / f"{ARCHIVE_NAME}.001"
    if not first_part.is_file():
        raise FileNotFoundError(f"Missing first archive part: {first_part}")

    run_7z_extract(first_part)
    ensure_simbox_symlinks()
    print(f"Done. Extracted assets to {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
