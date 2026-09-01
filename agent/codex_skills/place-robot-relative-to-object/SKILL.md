---
name: place-robot-relative-to-object
description: Place a floor-mobile robot at a qualitative world-axis side of a named SimBox fixture and orient it toward a target or explicit axis; use for requests such as placing a robot to the left (-X), right (+X), top (+Y), or bottom (-Y) of a table while filling robots, start regions, and waypoints consistently. Do not use for arm IK, navigation planning, vertical placement, or task skill sequencing.
---

# Place Robot Relative to Object

Translate qualitative spatial language into a validated robot base pose in the scene's declared world frame. Treat image directions only as visual evidence; never infer world axes from pixels when the arena already declares a coordinate frame.

## Required inputs

- SimBox task YAML and its referenced arena YAML.
- Named target fixture, such as `wide_oak_work_table`.
- Position relation: `-X`, `+X`, `-Y`, or `+Y` side.
- Facing direction: explicit world axis, `toward_target`, or `away_from_target`.
- Robot runtime profile. Default to `config/robots/split_aloha_runtime.yaml` only when the task requests Split ALOHA.
- The profile's `usd_asset` is the canonical repository-relative runtime USD
  path. For virtual-base Split ALOHA it is
  `InternDataAssets/robots/split_aloha_mid_360_virtual/robot.usd`;
  do not emit the retired `example_assets/...` path.

If the user gives natural-language directions, normalize them with [the qualitative direction contract](references/qualitative-directions.md). Explicit annotations such as `左边(-x)` and `面对桌子(+x)` override unstated linguistic conventions.

## Workflow

1. Read the task, arena, robot profile, and any named reference image. Confirm `+Z` is up and `XY` is the floor plane.
2. Resolve the target by fixture name. Report ambiguity instead of choosing between multiple fuzzy matches.
3. Normalize the requested side and facing direction to world axes. For `toward_target`, use the axis opposite the requested side.
4. Compute the target's world XY bounds from `translation` and `asset_world_extents` (or `size` as a fallback).
5. Put the robot base center beyond the requested target edge using the profile's `approach_offset_m`; align the orthogonal coordinate with the target center unless collision avoidance requires a documented lateral shift.
6. Convert the robot forward axis to yaw. For an `isaac_z_up_x_front` robot: `+X=0 deg`, `+Y=90 deg`, `-X=180 deg`, and `-Y=-90 deg`.
7. Preview and validate the floor bounds, robot footprint, target clearance, other fixture overlaps, and manipulator approach reach.
8. Update all three bindings atomically: keep canonical identity and allowed heading overrides in `robots[]`, put the fixed world pose in the robot's `regions[]` entry, and mirror it in `positions.wp_robot_start`. Preserve objects, object regions, skills, and unrelated metadata.
9. Mark the task metadata as `robot_placement_only` when task skills remain empty. Re-read the output and repeat the geometric checks.
10. Preserve the profile-selected USD path in `robots[].path`; if it is missing,
    report the asset path instead of silently substituting another robot.

Use the bundled helper for deterministic geometry and YAML synchronization. Preview first; add `--execute` only after the preview passes:

```bash
python3 agent/codex_skills/place-robot-relative-to-object/scripts/place_robot_relative.py \
  --task runs/basic/s04_map04/simbox_task.yaml \
  --target wide_oak_work_table \
  --relation=-x \
  --facing=+x

python3 agent/codex_skills/place-robot-relative-to-object/scripts/place_robot_relative.py \
  --task runs/basic/s04_map04/simbox_task.yaml \
  --target wide_oak_work_table \
  --relation=-x \
  --facing=+x \
  --execute
```

## Output contract

- `robots[]` contains only profile assertions and supported instance overrides: `name`, `robot_config_file`, canonical `path`, `target_class`, `euler`, and runtime tuning fields. It must not emit `translation`, `initial_pose`, `spawn_region`, or `placement`.
- `regions[]` owns the fixed robot pose through `placement_mode: fixed_from_region_pose`, `world_translation`, and `world_euler`. Its `pos_range` remains floor-center-relative for schema compatibility.
- `positions.wp_robot_start` contains the same world X, Y, and yaw.
- `metadata.robot_placement` records the target, normalized relation/facing axes, profile, pose, clearance, and optional reference image.
- No task skills, navigation waypoints, or arm trajectories are invented.

## Current scene example

For `s04_map04`, the reference image is:

`runs/memory/s04_map04/visual/interdata/south_interior/rgb_0000.png`

It is a south-side interior view looking approximately toward world `+Y`. The requested instruction, “将机器人放在桌子的左边(-x)面对桌子(+x),” resolves to the `-X` side of `wide_oak_work_table`, facing `+X`. With the current table bounds and virtual-base Split ALOHA profile, the base pose is `[-1.03, -0.055, 0.0]` with Euler yaw `0 deg`.

## Stop conditions

Stop without writing if the target is unresolved, coordinate frame is missing or non-planar, requested language is ambiguous, robot profile lacks footprint/approach data, the candidate is outside the floor, any fixture collision remains, or the target is beyond the declared manipulator reach. A request for “上方/+Z” is vertical placement and is outside this Skill.
