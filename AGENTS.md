# InterndataEngine Working Notes

## Scene 4 Mobile Manipulation

- Scene-4 task `positions` are floor-center relative. Convert to world/layout XY with `world_x = floor_center_x + x` and `world_y = floor_center_y + y`; do not add another reference frame field.
- Keep generated scene-4 skill graphs short. The expected basic-task shape is usually five skills: `nav_to_pick`, `pick_*`, `nav_to_place`, `place_*`, `home_*`.
- Navigation uses the ROS-free local A* and waypoint controller; task YAML must not reintroduce external Nav2/ROS control.
- Do not fix base navigation by editing dummy/mobile_support nodes; they are not on the effective mobile-base control path.
- Use generated nav overlays and reports as design evidence only. A collision-free overlay does not prove 4WIS stability or arm reachability.

## Validation Workflow

- Use `/home/dyf/miniconda3/envs/anygrasp/bin/python` for scene-4 helper scripts and compile checks.
- Start Isaac through `scripts/docker/up_simbox_isaac.sh` or the validation wrapper that calls it; avoid ad-hoc container startup.
- For real validation, prefer the Scene-4 validation wrapper when present, or `scripts/docker/run_simbox_task.sh` for one task. Judge success from the validation summary, per-task logs, and skill snapshots.
- A successful run needs `Task is successful, mode=plan_with_render` and no `[LmdbLogger] Episode failed`; a video or missing traceback is not enough.
- Keep `emit_obs_on_failure` disabled for strict validation. Placeholder observations can hide retry/reset behavior.
- Stop and inspect the first failure when using `--stop-on-failure`; use `output/local_navigation/skills/*` snapshots before changing logic.

## Isaac Bash Development Environment

- For code development, shell access, imports, and lightweight checks, use `scripts/docker/isaac_dev.sh`; do not write an agent implementation or start an ad-hoc Isaac container.
- The developer wrapper reuses the current Isaac 6.0.1 + CuRobo v2 image, mounts the repository at `/workspace`, forces `INTERNDATA_AUTOSTART_LAUNCHER=0`, and does not start `launcher.py`.
- Start and enter the isolated developer container with `scripts/docker/isaac_dev.sh shell --gpu 0`; add `--build` only when the image needs rebuilding. For a background container use `scripts/docker/isaac_dev.sh start --gpu 0`.
- Run non-interactive checks through the wrapper, for example `scripts/docker/isaac_dev.sh exec -- bash -lc 'pwd; python -V'`; use `scripts/docker/isaac_dev.sh stop` to release the GPU.
- The default developer container is `isaac-dev-dev`; custom `--stack-id` values isolate the container name and cache. Developer caches live under `output/isaac-dev/`, not the root-owned `.docker/isaac-sim/` tree.
- Verify the developer environment before relying on it: `scripts/docker/isaac_dev.sh status`, container state `running`, `/workspace` as the working directory, `INTERNDATA_AUTOSTART_LAUNCHER=0`, and a Bash process as the container main process. CuRobo/Isaac Torch verification from the entrypoint must pass.
- A running Bash developer container is not task validation. For task success, stop it if it competes for GPU resources and use the normal validation wrapper or `scripts/docker/run_simbox_task.sh`; require `Task is successful, mode=plan_with_render` and no `[LmdbLogger] Episode failed`.

## Reset And Randomization

- Fixed rigid objects should normally use `apply_randomization: false` unless their reload/reset path has been verified.
- Retry reset should prefer restoring existing rigid-object pose, scale, visibility, and velocity when the USD path is unchanged. Repeated delete/recreate can race USD loading and produce invalid null prims.
- Fixed-object reset must run in the normal `randomization()` layout reset paths, not only in `reset_after_failed_generation()`, otherwise later retries can plan against stale object states.
- Randomized rigid-object USDs may use a different rigid-body child than the source asset. Validate the configured child and fall back to the first rigid body under the loaded object root.
- Grasp annotation paths should be resolved from the selected USD directory plus `npy_name`; do not derive them only by replacing `Aligned_obj.usd`.

## Navigation Debugging

- Navigation points must balance obstacle clearance, 4WIS dynamic stability, and arm reachability. A point that is valid on the 2D map can still be too close to counters or force a bad lateral approach.
- If navigation reports `bridge_aborted`, compare `world_xy`, `nav_xy`, `world_dist`, `nav_dist`, yaw error, and the bridge command history before changing task points.
- If the base state becomes invalid, inspect roll/pitch, wheel/steering commands, and restore-after-navigation traces. Do not assume local A* path planning is the root cause.

## Pick Skill Debugging

- Separate candidate generation from execution. Use `pick_plan_snapshot.json` to check candidates and `pick_execution_trace.json` to check whether the object actually moved.
- If later retries have `success_found: true` in `pick_plan_snapshot.json`, the old "no grasp candidates after first attempt" issue is not the active failure.
- To prove pick success, check that object z increases during close/post-grasp and that no `pick_runtime_failure_snapshot.json` is produced. Do not mislabel downstream place failure as pick failure.
- Command transitions in `pick_execution_trace.json` are the best timing evidence: pre-grasp/open should finish before `close_gripper`; `attach_obj` should occur after close.
- For apple-like top grasps, prefer physical ranking from actual candidate geometry and execution traces over loosening YAML filters blindly.
- `post_grasp_offset_min` and `post_grasp_offset_max` alone control planned post-grasp lift height. Do not add hidden caps that override these values.
- `lift_th` is only a pick success threshold. If absent, it defaults to `0.0` and disables the lift-height success check; it should not affect the post-grasp motion target.

## Place Skill Debugging

- Place failures should be diagnosed from `place_success_check_snapshot.json`; inspect `success_mode`, target object, bbox limits, margin, and final object XY/Z.
- If a task name says "tray" but the place skill targets `sink`, trust the skill objects in YAML/runtime snapshots when diagnosing behavior.
- Debug artifact writing must never break an episode. Convert NumPy values and USD/Gf vector types such as `Vec3d` into JSON-safe scalars/lists before `json.dump`.
- When `success_mode: xybbox` fails, compare `pick_xy` against `valid_xy_min/max`; a small outside-bbox error is a placement target/settling issue, not a pick failure.

## Scene-4 Task Skills and Positions Reference

- `kitchen_apple_to_tray` is the verified reference task; its pick/place parameters are treated as the scene-4 baseline. Do not modify this task unless a new validation run explicitly requires it.
- All scene-4 basic tasks use the **5-skill graph** pattern (≤ 8 skills):
  1. `base: navigate` (`id: nav_to_pick`, `depends_on: []`)
  2. `<arm>: pick` (`id: pick_<pick_object>`, `depends_on: [nav_to_pick]`)
  3. `base: navigate` (`id: nav_to_place`, `depends_on: [pick_<pick_object>]`)
  4. `<arm>: place` (`id: place_<pick_object>`, `depends_on: [nav_to_place]`)
  5. `<arm>: heuristic__skill` (`id: home_<arm>`, `mode: home`, `depends_on: [place_<pick_object>]`)
- Navigation skills do not expose a heading-controller enable/disable option.
- Navigation tolerances: `xy_goal_tolerance: 0.1`, `yaw_goal_tolerance: 0.1`.
- Positions are **floor-center relative** (`floor_center_layout_xy` varies per room). Convert to world/layout XY with `world_x = floor_center_x + x` and `world_y = floor_center_y + y`.
- Positions and object-to-arm mappings for all 20 tasks are canonically stored in `output/scene4_nav_skill_generation/scene4_nav_skill_generation_summary.json`. When updating a task, read from that summary rather than recomputing from the obstacle map.

### Task inventory (generated from nav-skill summary)

| Task | Pick object | Place object | Arm | Floor center |
|------|-------------|--------------|-----|--------------|
| kitchen_apple_to_tray | apple_0_id9008 | sink | left | (2.0, 1.5) |
| kitchen_breakfast_setup | apple_0_id9008 | metal_tray_0_id9016 | left | (2.0, 1.5) |
| kitchen_cup_transfer | white_mug_a_0_id9000 | metal_tray_0_id9016 | right | (2.0, 1.5) |
| kitchen_prep_assembly | fruit_knife_0_id9007 | cutting_board_0_id9006 | right | (2.0, 1.5) |
| kitchen_salt_bottle_placement | salt_bottle_0_id9011 | metal_tray_0_id9016 | right | (2.0, 1.5) |
| bookroom_book_retrieval | hardcover_book_a_0_id9000 | main_desk_0_id1 | right | (2.1, 1.6) |
| bookroom_cross_zone_filing | metal_file_folder_0_id9009 | storage_box_0_id9016 | right | (2.1, 1.6) |
| bookroom_cup_relocation | coffee_mug_0_id9013 | open_bookshelf_0_id2 | right | (2.1, 1.6) |
| bookroom_device_zone | tablet_0_id9011 | open_bookshelf_0_id2 | right | (2.1, 1.6) |
| bookroom_pen_to_holder | black_pen_a_0_id9006 | pen_holder_0_id9005 | right | (2.1, 1.6) |
| livingroom_coffee_table_cleanup | magazine_a_0_id9003 | storage_basket_0_id9008 | right | (2.5, 2.0) |
| livingroom_mug_to_coaster | livingroom_mug_0_id9005 | round_coaster_a_0_id9006 | right | (2.5, 2.0) |
| livingroom_phone_to_cabinet | phone_0_id9001 | side_cabinet_0_id4 | right | (2.5, 2.0) |
| livingroom_remote_to_basket | remote_control_0_id9000 | storage_basket_0_id9008 | right | (2.5, 2.0) |
| livingroom_toy_blocks_cleanup | toy_block_0_id9014 | storage_basket_0_id9008 | right | (2.5, 2.0) |
| bedroom_bedside_clothing | folded_towel_0_id9001 | clothing_storage_box_0_id9013 | right | (2.25, 1.8) |
| bedroom_bedtime_items | bedside_book_0_id9007 | right_nightstand_0_id3 | right | (2.25, 1.8) |
| bedroom_hand_cream_to_organizer | hand_cream_0_id9009 | small_organizer_0_id9012 | right | (2.25, 1.8) |
| bedroom_phone_placement | bedroom_phone_0_id9003 | left_nightstand_0_id2 | right | (2.25, 1.8) |
| bedroom_tshirt_to_storage | folded_tshirt_0_id9000 | clothing_storage_box_0_id9013 | right | (2.25, 1.8) |

### Pick/place parameter templates

- **Left-arm pick** (copied from `kitchen_apple_to_tray`):
  - `filter_x_dir: [forward, 90]`, `filter_z_dir: [downward, 140]`
  - `pre_grasp_offset: 0.12`, `post_grasp_offset_min: 0.26`, `post_grasp_offset_max: 0.28`
  - `lift_th: 0.02`, `gripper_change_steps: 20`, `t_eps: 0.025`, `o_eps: 1`, `process_valid: true`
- **Right-arm pick** (room-level baseline with tuned offsets):
  - `filter_y_dir: [forward, 60]`, `filter_z_dir: [downward, 150]`
  - `pre_grasp_offset: 0.12`, `post_grasp_offset_min: 0.26`, `post_grasp_offset_max: 0.28`
  - `lift_th: 0.02`, `gripper_change_steps: 20`, `t_eps: 0.025`, `o_eps: 1`, `process_valid: true`
- **Place** (generic, used for both arms):
  - `position_constraint: object`, `success_mode: xybbox`
  - `filter_x_dir: [backward, 110]`, `filter_y_dir: [downward, 120]`, `filter_z_dir: [forward, 70]`
  - `x_ratio_range: [0.35, 0.65]`, `y_ratio_range: [0.35, 0.65]`
  - `pre_place_z_offset: 0.1`, `place_z_offset: 0.1`, `gripper_change_steps: 20`

## Code And Git Hygiene

- Prefer existing SimBox helpers and local patterns over new abstractions.
- Keep fixes scoped: do not change YAML to mask a code bug, and do not change code when the user explicitly asks for YAML-only repair.
- Before risky rollback or checkpoint work, verify `git status --short`, branch, and `git log -1 --oneline`.
- When saving a successful state, create a clear checkpoint commit and verify the final status. Include generated assets only when the user explicitly asks.

## PnP/CuRobo Closed-Loop Debugging

- Fix the first runtime error that prevents task generation; do not treat a later traceback, placeholder observation, video, or renderer startup warning as the root cause.
- Keep each fix minimal and local to the proven regression. Do not add a large new control path, retries, or fallback layer merely to make a failed episode appear to progress.
- For PnP/CuRobo planning regressions, inspect the earliest working implementation and the commit diff before changing current behavior; preserve the original project contract unless runtime evidence requires a targeted correction.
- After every suspected bug-point fix, run the project-prescribed real single-episode wrapper and inspect the fresh first-failure log before making the next change.
- Static compilation and unit/contract checks are useful gates, but completion requires a fresh runtime result with `Task is successful, mode=plan_with_render` and no `[LmdbLogger] Episode failed`.
