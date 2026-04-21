"""Download InternDataAssets from HuggingFace, with progress monitoring."""

import os
import sys
import time
import signal
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "InternRobotics/InternData-A1"
REPO_TYPE = "dataset"
ASSET_ROOT = "InternDataAssets/assets"
LOCAL_DIR = "."
SCAN_TIMEOUT = 120  # seconds per directory scan

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

def get_token():
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    print("ERROR: No HuggingFace token found. Run: hf auth login")
    sys.exit(1)

def list_subdirs(token):
    items = list(list_repo_tree(REPO_ID, path_in_repo=ASSET_ROOT, repo_type=REPO_TYPE, token=token))
    dirs = [item.path for item in items if item.path != ASSET_ROOT and not hasattr(item, 'size')]
    files = [item for item in items if hasattr(item, 'size') and item.size is not None]
    return dirs, files

def list_files_in_dir(path, token):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(SCAN_TIMEOUT)
    try:
        items = list(list_repo_tree(REPO_ID, path_in_repo=path, repo_type=REPO_TYPE, token=token, recursive=True))
        signal.alarm(0)
        return [item for item in items if hasattr(item, 'size') and item.size is not None]
    except TimeoutError:
        print(f"TIMEOUT after {SCAN_TIMEOUT}s, retrying non-recursively...")
        signal.alarm(0)
        return list_files_shallow(path, token)

def list_files_shallow(path, token):
    """List files by scanning one level at a time to avoid timeout."""
    all_files = []
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(SCAN_TIMEOUT)
    try:
        items = list(list_repo_tree(REPO_ID, path_in_repo=path, repo_type=REPO_TYPE, token=token, recursive=False))
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0)
        print(f"  TIMEOUT even non-recursive, skipping {path}")
        return []

    for item in items:
        if hasattr(item, 'size') and item.size is not None:
            all_files.append(item)
        elif item.path != path:
            sub_files = list_files_in_dir(item.path, token)
            all_files.extend(sub_files)
    return all_files

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def main():
    token = get_token()

    print(f"Connecting to {REPO_ID} ...")
    print(f"Listing {ASSET_ROOT}/ ...")
    subdirs, root_files = list_subdirs(token)

    print(f"\nFound {len(subdirs)} subdirectories:")
    for d in sorted(subdirs):
        print(f"  {d}")
    if root_files:
        print(f"Found {len(root_files)} root files:")
        for f in root_files:
            print(f"  {f.path} ({format_size(f.size)})")
    print()

    all_tasks = []
    skipped = []
    for subdir in sorted(subdirs):
        dir_name = subdir.split("/")[-1]
        print(f"Scanning {dir_name}/ ...", end=" ", flush=True)
        files = list_files_in_dir(subdir, token)
        if not files:
            print("SKIPPED (timeout)")
            skipped.append(subdir)
            continue
        total_size = sum(f.size for f in files if f.size)
        print(f"{len(files)} files, {format_size(total_size)}")
        all_tasks.append((subdir, files, total_size))

    grand_total_files = sum(len(t[1]) for t in all_tasks) + len(root_files)
    grand_total_size = sum(t[2] for t in all_tasks) + sum(f.size for f in root_files if f.size)
    print(f"\nTotal: {grand_total_files} files, {format_size(grand_total_size)}")
    if skipped:
        print(f"Skipped dirs (will retry at end): {skipped}")
    print("=" * 60)

    downloaded_files = 0
    downloaded_bytes = 0
    failed_files = []
    start_time = time.time()

    for root_file in root_files:
        downloaded_files += 1
        print(f"[{downloaded_files}/{grand_total_files}] {root_file.path} ({format_size(root_file.size)})")
        local_path = Path(LOCAL_DIR) / root_file.path
        if local_path.exists() and local_path.stat().st_size == root_file.size:
            print(f"  Already exists, skipping")
            downloaded_bytes += root_file.size
            continue
        try:
            hf_hub_download(REPO_ID, root_file.path, repo_type=REPO_TYPE, token=token, local_dir=LOCAL_DIR)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed_files.append(root_file.path)
            continue
        downloaded_bytes += root_file.size

    for subdir, files, dir_size in all_tasks:
        dir_name = subdir.split("/")[-1]
        print(f"\n{'=' * 60}")
        print(f"Downloading {dir_name}/ ({len(files)} files, {format_size(dir_size)})")
        print(f"{'=' * 60}")

        for f in files:
            downloaded_files += 1
            elapsed = time.time() - start_time
            speed = downloaded_bytes / elapsed if elapsed > 0 else 0
            eta = (grand_total_size - downloaded_bytes) / speed if speed > 0 else 0

            progress_pct = downloaded_bytes / grand_total_size * 100 if grand_total_size > 0 else 0
            print(
                f"[{downloaded_files}/{grand_total_files}] "
                f"[{progress_pct:.1f}%] "
                f"[{format_size(speed)}/s] "
                f"[ETA {int(eta//3600)}h{int(eta%3600//60)}m] "
                f"{f.path.split('/')[-1]} ({format_size(f.size)})"
            )

            local_path = Path(LOCAL_DIR) / f.path
            if local_path.exists() and local_path.stat().st_size == f.size:
                downloaded_bytes += f.size
                continue
            try:
                hf_hub_download(REPO_ID, f.path, repo_type=REPO_TYPE, token=token, local_dir=LOCAL_DIR)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed_files.append(f.path)
                continue
            downloaded_bytes += f.size

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE! {downloaded_files} files processed in {int(elapsed//3600)}h{int(elapsed%3600//60)}m{int(elapsed%60)}s")
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for fp in failed_files:
            print(f"  {fp}")
    if skipped:
        print(f"\nSkipped directories ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
        print("Re-run this script to retry skipped directories.")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
