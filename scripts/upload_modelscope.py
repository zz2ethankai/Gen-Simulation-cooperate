#!/usr/bin/env python3
"""Compress InternDataAssets into split 7z volumes and upload to ModelScope."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "InternDataAssets"
ARCHIVE_DIR = REPO_ROOT / "output" / "modelscope_interndata_assets"
ARCHIVE_NAME = "InternDataAssets.7z"
VOLUME_SIZE = "7g"
REPO_ID = "MinMaxMex/InterndataAssets"
REPO_TYPE = "dataset"
PATH_IN_REPO = "InternDataAssets_7z"
COMMIT_MESSAGE = "Upload split 7z InternDataAssets archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        required=True,
        help="ModelScope access token.",
    )
    return parser.parse_args()


def run_7z_compress() -> None:
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z is required but was not found in PATH")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for old_part in ARCHIVE_DIR.glob(f"{ARCHIVE_NAME}.*"):
        old_part.unlink()

    archive_path = ARCHIVE_DIR / ARCHIVE_NAME
    cmd = [
        seven_zip,
        "a",
        "-t7z",
        "-mx=3",
        "-mmt=on",
        f"-v{VOLUME_SIZE}",
        str(archive_path),
        ASSET_DIR.name,
    ]
    print("Compressing:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    parts = sorted(ARCHIVE_DIR.glob(f"{ARCHIVE_NAME}.*"))
    if not parts:
        raise RuntimeError(f"No archive parts were generated under {ARCHIVE_DIR}")
    print(f"Generated {len(parts)} archive part(s) under {ARCHIVE_DIR}")


def main() -> int:
    args = parse_args()
    asset_dir = ASSET_DIR.resolve()

    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory does not exist: {asset_dir}")

    run_7z_compress()

    print(f"Uploading {ARCHIVE_DIR} to {REPO_TYPE} {REPO_ID}/{PATH_IN_REPO}")

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(args.token)

    commit_info = api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=ARCHIVE_DIR,
        path_in_repo=PATH_IN_REPO,
        commit_message=COMMIT_MESSAGE,
        token=args.token,
    )
    print(f"upload result: {commit_info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
