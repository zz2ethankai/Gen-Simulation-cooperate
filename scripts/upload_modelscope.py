#!/usr/bin/env python3
"""Compress InternDataAssets into split 7z volumes and upload to ModelScope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "InternDataAssets"
ARCHIVE_DIR = REPO_ROOT / "output" / "modelscope_interndata_assets"
ARCHIVE_NAME = "InternDataAssets.7z"
MANIFEST_NAME = "InternDataAssets.7z.manifest.json"
VOLUME_SIZE = "7g"
REPO_ID = "MinMaxMex/InterndataAssets"
REPO_TYPE = "dataset"
PATH_IN_REPO = "InternDataAssets_7z"
COMMIT_MESSAGE = "Upload split 7z InternDataAssets archive"
EXCLUDED_SCENE_ASSETS = [
    "assets_addition",
    "background_textures",
    "pick_and_place",
    "dark_table_textures",
    # "floor_textures",
    "home_scenes",
    "light_table_textures",
    "table0",
    "table_textures",
    "instance.usd",
    "table_info.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token",
        required=True,
        help="ModelScope access token.",
    )
    return parser.parse_args()


def manifest_payload() -> dict[str, object]:
    return {
        "archive_name": ARCHIVE_NAME,
        "source_dir": ASSET_DIR.name,
        "volume_size": VOLUME_SIZE,
        "excluded_scene_assets": EXCLUDED_SCENE_ASSETS,
        "kept_assets_note": (
            "Keeps envmap_lib, robot directories, and task/object asset "
            "directories under InternDataAssets/assets."
        ),
    }


def write_manifest() -> None:
    manifest_path = ARCHIVE_DIR / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_existing_archive() -> bool:
    existing_parts = sorted(ARCHIVE_DIR.glob(f"{ARCHIVE_NAME}.[0-9][0-9][0-9]"))
    existing_first_part = ARCHIVE_DIR / f"{ARCHIVE_NAME}.001"
    if not existing_parts or not existing_first_part.is_file():
        return False

    manifest_path = ARCHIVE_DIR / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Existing archive parts were found under {ARCHIVE_DIR}, but "
            f"{MANIFEST_NAME} is missing. Refusing to upload because the archive "
            "may still contain excluded scene assets. Remove the old archive parts "
            "and rerun this script to recompress with the current exclude list."
        )

    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing_manifest != manifest_payload():
        raise RuntimeError(
            f"Existing archive manifest does not match the current exclude list: "
            f"{manifest_path}. Remove the old archive parts and rerun this script."
        )

    print(f"Found existing archive part(s) under {ARCHIVE_DIR}; skip compression.")
    print(f"Existing archive part count: {len(existing_parts)}")
    return True


def run_7z_compress() -> None:
    if validate_existing_archive():
        return

    seven_zip = shutil.which("7z")
    if seven_zip is None:
        raise RuntimeError("7z is required but was not found in PATH")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for old_part in ARCHIVE_DIR.glob(f"{ARCHIVE_NAME}.*"):
        old_part.unlink()
    manifest_path = ARCHIVE_DIR / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()

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
    for scene_asset in EXCLUDED_SCENE_ASSETS:
        relative_path = Path(ASSET_DIR.name) / "assets" / scene_asset
        cmd.append(f"-xr!{relative_path.as_posix()}")
        cmd.append(f"-xr!{relative_path.as_posix()}/*")

    print("Compressing:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    parts = sorted(ARCHIVE_DIR.glob(f"{ARCHIVE_NAME}.[0-9][0-9][0-9]"))
    if not parts:
        raise RuntimeError(f"No archive parts were generated under {ARCHIVE_DIR}")
    write_manifest()
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
