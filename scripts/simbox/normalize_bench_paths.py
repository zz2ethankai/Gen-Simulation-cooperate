#!/usr/bin/env python3
"""Normalize Bench 2.1 SimBox task configs for execution from the repo root."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "InternDataAssets" / "Bench_2.1_isaacsim"
EXPECTED_TASK_COUNT = 20
# Bench 2.1 ships its scene-level robot and HDR assets under scene_4/shared_assets.
# Keep task paths inside that delivery instead of redirecting HDR lookup to the
# engine-wide workflows/simbox/assets/envmap_lib collection.
ENVMAP_LIB = "../shared_assets/envmap_lib"
MOBILE_ROBOT_PATH = "../shared_assets/split_aloha_mid_360/robot.usd"
UNSUPPORTED_REGION_ARGS = ("support_surface_z", "support_surface_source")
WALL_EULERS = {
    # UsdGeom.Plane's local normal is +Z. Point every wall normal into the room;
    # otherwise the collision volume is offset to the room side instead of
    # behind the visible wall.
    "wall_north": (90.0, 0.0, 0.0),
    "wall_south": (-90.0, 0.0, 0.0),
    "wall_west": (90.0, 0.0, 90.0),
    "wall_east": (90.0, 0.0, -90.0),
}


def _replace_scalar(line: str, key: str, value: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    indent = line[: len(line) - len(line.lstrip())]
    return f"{indent}{key}: {value}{newline}"


def normalize_task(task_path: Path) -> bool:
    """Normalize one task file and return whether its contents changed."""
    scene_dir = task_path.parents[3]
    asset_root = scene_dir.relative_to(REPO_ROOT).as_posix()
    arena_file = task_path.with_name("simbox_arena.yaml").relative_to(REPO_ROOT).as_posix()

    lines = task_path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = [
        line
        for line in lines
        if not any(line.startswith(f"      {key}:") for key in UNSUPPORTED_REGION_ARGS)
    ]
    replacements = {"asset_root": 0, "arena_file": 0, "envmap_lib": 0, "robot_path": 0}
    in_robots = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if line.startswith("  asset_root:"):
            lines[index] = _replace_scalar(line, "asset_root", asset_root)
            replacements["asset_root"] += 1
        elif line.startswith("  arena_file:"):
            lines[index] = _replace_scalar(line, "arena_file", arena_file)
            replacements["arena_file"] += 1
        elif line.startswith("    envmap_lib:"):
            lines[index] = _replace_scalar(line, "envmap_lib", ENVMAP_LIB)
            replacements["envmap_lib"] += 1

        if line.startswith("  robots:"):
            in_robots = True
            continue
        if in_robots and line.startswith("  objects:"):
            in_robots = False
        if in_robots and stripped.startswith("path:") and replacements["robot_path"] == 0:
            lines[index] = _replace_scalar(line, "path", MOBILE_ROBOT_PATH)
            replacements["robot_path"] += 1

    missing_or_duplicate = {key: count for key, count in replacements.items() if count != 1}
    if missing_or_duplicate:
        raise ValueError(f"Unexpected task structure in {task_path}: {missing_or_duplicate}")

    original = task_path.read_text(encoding="utf-8")
    normalized = "".join(lines)
    if normalized == original:
        return False

    task_path.write_text(normalized, encoding="utf-8")
    return True


def normalize_arena(arena_path: Path) -> bool:
    """Point the four generated room-wall planes toward the room interior."""
    original = arena_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    current_fixture: str | None = None
    seen: set[str] = set()

    for index, line in enumerate(lines):
        if line.startswith("- name:"):
            current_fixture = line.split(":", 1)[1].strip()
            continue
        if current_fixture not in WALL_EULERS or not line.startswith("  euler:"):
            continue
        if index + 3 >= len(lines):
            raise ValueError(f"Incomplete Euler block for {current_fixture} in {arena_path}")

        for offset, value in enumerate(WALL_EULERS[current_fixture], start=1):
            newline = "\n" if lines[index + offset].endswith("\n") else ""
            lines[index + offset] = f"  - {value}{newline}"
        seen.add(current_fixture)

    missing = set(WALL_EULERS) - seen
    if missing:
        raise ValueError(f"Missing generated walls in {arena_path}: {sorted(missing)}")

    normalized = "".join(lines)
    if normalized == original:
        return False
    arena_path.write_text(normalized, encoding="utf-8")
    return True


def main() -> None:
    task_paths = sorted(
        path
        for path in BENCH_ROOT.rglob("simbox_task.yaml")
        if "._____temp" not in path.parts
    )
    if len(task_paths) != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TASK_COUNT} task files under {BENCH_ROOT}, found {len(task_paths)}"
        )

    changed = 0
    changed_arenas = 0
    for task_path in task_paths:
        was_changed = normalize_task(task_path)
        changed += int(was_changed)
        arena_changed = normalize_arena(task_path.with_name("simbox_arena.yaml"))
        changed_arenas += int(arena_changed)
        status = "updated" if was_changed else "unchanged"
        arena_status = "updated" if arena_changed else "unchanged"
        print(
            f"{status}: {task_path.relative_to(REPO_ROOT)}; "
            f"arena {arena_status}"
        )

    print(
        f"normalized {len(task_paths)} task files; changed {changed}; "
        f"changed arenas {changed_arenas}"
    )


if __name__ == "__main__":
    main()
