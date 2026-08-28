#!/usr/bin/env python3
"""Move robot asset references to ``InternDataAssets/robots`` in YAML files.

The scene asset root remains unchanged.  Only scalar values on ``path`` and
``usd_path`` keys that point into a known robot asset directory, plus legacy
CuRobo ``robot_file`` entries, are rewritten.  The script edits YAML as text so
comments, ordering, and formatting are kept.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROBOT_ASSET_DIRS = (
    "G1_120s",
    "fr3",
    "franka",
    "frankarobotiq",
    "lift2",
    "panda_omron",
    "panda_omron_virtual",
    "split_aloha_mid_360",
    "split_aloha_mid_360_virtual",
    "tracer2_franka",
)

ROBOT_DIR_PATTERN = "|".join(re.escape(name) for name in ROBOT_ASSET_DIRS)
PATH_LINE = re.compile(
    rf"^(?P<prefix>\s*(?:path|usd_path)\s*:\s*)"
    rf"(?P<quote>[\"']?)(?P<value>[^\"'#\s]+)(?P=quote)"
    rf"(?P<suffix>\s*(?:#.*)?)(?P<newline>\n?)$"
)
ROBOT_CONFIG_LINE = re.compile(
    r"^(?P<prefix>\s*-\s*)(?P<quote>[\"']?)"
    r"(?P<value>[^\"'#\s]+)(?P=quote)"
    r"(?P<suffix>\s*(?:#.*)?)(?P<newline>\n?)$"
)
OLD_ROBOT_CONFIG_PREFIXES = (
    "workflows/simbox/curobo/src/curobo/content/configs/robot/",
    "InternDataAssets/curobo/src/curobo/content/configs/robot/",
)
NEW_ROBOT_CONFIG_PREFIX = (
    "InternDataAssets/curobov2/curobo/content/custom/configs/robot/"
)


def iter_yaml_files(repo_root: Path):
    """Yield YAML files in the repository without following directory links."""
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if ".git" in path.parts:
            continue
        # macOS resource forks have YAML-looking names but are binary files.
        if path.name.startswith("._"):
            continue
        yield path


def migrate_path(value: str) -> str:
    """Return a repository-relative robot path, or the original value."""
    value = value.replace("\\", "/")
    if value.startswith("InternDataAssets/robots/"):
        return value

    # Only accept paths rooted at one of the old asset roots.  In particular,
    # CuRobo's own paths such as ``robot/non_shipping/franka/...`` must stay
    # untouched even though they contain a robot name.
    old_root_match = re.match(
        rf"^(?:(?:InternDataAssets/assets|workflows/simbox/assets)/)?"
        rf"({ROBOT_DIR_PATTERN})/(.+)$",
        value,
    )
    relative_match = re.match(
        rf"^(?:(?:\.\./)+)(?:({ROBOT_DIR_PATTERN}))/(.+)$",
        value,
    )
    match = old_root_match or relative_match
    if match is None:
        return value

    robot_dir = match.group(1)
    asset_suffix = match.group(2)
    return f"InternDataAssets/robots/{robot_dir}/{asset_suffix}"


def migrate_robot_config_path(value: str) -> str:
    """Move a legacy CuRobo robot config reference into the v2 checkout."""
    value = value.replace("\\", "/")
    if value.startswith(NEW_ROBOT_CONFIG_PREFIX):
        return value
    for prefix in OLD_ROBOT_CONFIG_PREFIXES:
        if value.startswith(prefix):
            return NEW_ROBOT_CONFIG_PREFIX + value[len(prefix):]
    return value


def migrate_file(path: Path, check: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    changed_lines: list[str] = []
    changed = False

    for line in original.splitlines(keepends=True):
        match = PATH_LINE.match(line)
        if match is not None:
            old_value = match.group("value")
            new_value = migrate_path(old_value)
        else:
            match = ROBOT_CONFIG_LINE.match(line)
            if match is None:
                changed_lines.append(line)
                continue
            old_value = match.group("value")
            new_value = migrate_robot_config_path(old_value)
        if new_value == old_value:
            changed_lines.append(line)
            continue

        changed = True
        changed_lines.append(
            f"{match.group('prefix')}{match.group('quote')}{new_value}"
            f"{match.group('quote')}{match.group('suffix')}{match.group('newline')}"
        )

    if changed and not check:
        path.write_text("".join(changed_lines), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the current project root).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report files that would change without modifying them.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    changed_files = [
        path
        for path in iter_yaml_files(repo_root)
        if migrate_file(path, check=args.check)
    ]

    mode = "would update" if args.check else "updated"
    for path in sorted(changed_files):
        print(f"{mode}: {path.relative_to(repo_root)}")
    print(f"{mode} {len(changed_files)} YAML files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
