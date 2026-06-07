# New Asset Import Fix Log

## 2026-06-07

### Issue 1: robot USD path produced a duplicated `workflows` segment

- Symptom:
  - Isaac reported it could not open `/workspace/workflows/workflows/simbox/example_assets/split_aloha_mid_360/robot.usd`.
  - The follow-up `Empty typeName` error on `physxArticulation:solverPositionIterationCount` was a secondary failure after the robot prim failed to load.
- Cause:
  - Imported `simbox_task.yaml` files used `../../../../../workflows/simbox/example_assets/split_aloha_mid_360/robot.usd`.
  - USD reference resolution normalized the string from `asset_root` to `/workspace/workflows/workflows/simbox/...`.
- Fix:
  - Updated all 20 imported `assets/basic/*/simbox_task.yaml` files to use `../../../../example_assets/split_aloha_mid_360/robot.usd`.
- Verification:
  - All 20 task YAML files now contain the corrected robot path.
  - Static path validation for the 20 final task entries passed with `task_files=20 errors=0`.

### Issue 2: empty skill lists crashed sequence generation

- Symptom:
  - `simbox_dual_workflow.py::plan_first_skill` raised `IndexError: list index out of range` at `lr_skill_list[0].simple_generate_manip_cmds()`.
- Cause:
  - All 20 imported `assets/basic/*/simbox_task.yaml` files had the first skill group as `base: []`, `left: []`, `right: []`.
  - The old non-DAG skill planner expects each listed left/right skill list to contain at least one skill.
- Planned fix:
  - Replace the empty skill group in each imported task YAML with a minimal right-arm pick/place/home sequence using objects and fixtures already present in that task.
- Fix:
  - Updated all 20 imported `assets/basic/*/simbox_task.yaml` files.
  - Each task now has one non-empty `right` skill list: `pick`, `place`, then `heuristic__skill` home.
  - Pick targets were selected from task objects that have `Aligned_grasp_sparse.npy`; place targets were selected from non-wall/non-floor arena fixtures already present in the same task.
- Verification:
  - Skill structure validation passed with `skill_lists=20 empty=0`.
  - Static task validation passed with `task_files=20 errors=0`.

### Issue 3: generated pick started before the mobile base moved into reach

- Symptom:
  - Runtime progressed past scene loading and skill initialization, then failed with:
    `Episode failed: robot=split_aloha skill=pick reason=skill_not_feasible`.
  - `pick-debug` snapshots for `white_mug_a_0_id9000` reported no candidate passed pregrasp+grasp screening.
- Cause:
  - The temporary generated skill sequence only contained arm skills.
  - The object was about 1.5 m in front of and 1.2 m lateral to the arm base in the pick snapshot, outside direct arm reach.
  - Switching the same pick from right arm to left arm did not solve reachability; the missing mobile-base navigation step was the real blocker.
- Fix in progress:
  - For the current entry task `01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml`, added `positions.nav_to_pick` and `positions.nav_to_place`.
  - Converted the current entry task's skill group to DAG form:
    `base.navigate(nav_to_pick) -> left.pick -> base.navigate(nav_to_place) -> left.place -> left.home`.
  - First tried `nav_to_pick=(1.70, 0.75, 0.0)`, which reached the `navigate` skill but Nav2 aborted with status code 6.
  - Nav2 logs reported `GridBased: failed to create plan` and later `Starting point in lethal space`; the bridge snapshots showed `received_cmd_vel_count=0`.
  - Based on the generated debug `map.pgm`, added connected free-space candidates:
    `nav_to_pick=(1.18, 0.98, pi)`, `nav_to_pick_candidate_a=(1.26, 0.90, pi)`,
    `nav_to_pick_candidate_b=(1.40, 0.84, pi)`, and `nav_to_pick_candidate_c=(1.50, 1.00, pi)`.
- Verification:
  - `ruamel.yaml` parses the edited YAML.
  - Parsed `positions` keys now include `wp_center`, `nav_to_pick`, `nav_to_place`, and three `nav_to_pick_candidate_*` entries.
  - Parsed skill arms are `base` and `left`; base nodes are `nav_to_pick` and `nav_to_place`, left nodes are `pick_white_mug_a`, `place_white_mug_a`, and `home_left`.

### Issue 4: current entry task envmap path still resolved through a duplicated `workflows` segment

- Symptom:
  - Runtime logged:
    `Failed to read texture file /workspace/workflows/simbox/assets/custom/scene_4/01_kitchen/../../../../../workflows/simbox/example_assets/envmap_lib/abandoned_factory_canteen_01_1k.hdr`.
  - RTX then failed to upload the DomeLight texture.
- Cause:
  - The current entry task still had `env_map.envmap_lib: ../../../../../workflows/simbox/example_assets/envmap_lib`.
  - From `asset_root=workflows/simbox/assets/custom/scene_4/01_kitchen`, that relative string walks above `workflows/simbox` and re-enters `workflows/simbox`, producing a bad path inside the container.
- Fix:
  - Updated current entry task `01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml` to use:
    `assets/envmap_lib`.
  - Copied the required HDR into the current scene asset tree:
    `workflows/simbox/assets/custom/scene_4/01_kitchen/assets/envmap_lib/abandoned_factory_canteen_01_1k.hdr`.
  - Reason for not using the previous relative path: `workflows/simbox/assets` is a symlink. The `glob` in `_set_envmap`
    follows the physical symlink target, while USD texture loading later receives the raw path string. A path that worked
    for `glob` produced `/workspace/workflows/workflows/...` for USD; a path that was lexically correct for USD produced
    an empty `glob`. Keeping the HDR under the scene's own asset tree makes both sides resolve the same file.
- Verification:
  - Host and container path checks both find one HDR under `asset_root/assets/envmap_lib`.

### Issue 5: first navigation goal still left the mug too far from the left arm

- Symptom:
  - The earlier arm-only run failed at `pick` with `skill_not_feasible`.
  - The pick debug snapshot showed `white_mug_a_0_id9000` at roughly `(x=1.54, y=1.20)` in the left arm-base frame.
- Cause:
  - The first navigation candidates around `y=0.98` would still leave the object with a large lateral offset from the left arm.
  - The latest Nav2 debug map marks the lower approach corridor as free, so the active goal should move closer to the object side.
- Fix:
  - In the current entry task, first set active `positions.nav_to_pick` and `positions.nav_to_place` to `(1.00, 0.72, pi)`.
  - Added backup candidates:
    `(1.10, 0.72, pi)`, `(0.86, 0.52, pi)`, `(0.80, 0.42, pi)`, `(0.74, 0.30, pi)`,
    `(0.70, 0.20, pi)`, and `(1.20, 0.72, pi)`.
  - Runtime then proved `(1.00, 0.72, pi)` reaches Nav2 but aborts with status code 6 before any `cmd_vel`;
    Nav2 logs alternate between `Starting point in lethal space` and `no valid path found`.
  - Switched the active point to the wider corridor at `(1.20, 1.10, pi)` and added higher corridor candidates:
    `(1.40, 1.20, pi)`, `(1.60, 1.30, pi)`, `(1.80, 1.40, pi)`, `(2.20, 1.70, pi)`,
    and `(2.50, 2.10, pi)`.
  - Runtime also proved `(1.20, 1.10, pi)` reaches Nav2 but aborts before any `cmd_vel`, again primarily with
    `Starting point in lethal space`.
  - Switched the active point to the near-start high candidate `(2.50, 2.10, pi)` to isolate whether the blocker is
    goal placement or the initial robot footprint/costmap.
  - Runtime proved `(2.50, 2.10, pi)` also reaches Nav2 but aborts before any `cmd_vel`; the planner still reports
    `Starting point in lethal space` from several randomized starts.
- Verification:
  - Static YAML parsing passed.
  - The listed candidates are in free cells on the latest Nav2 debug `map.pgm`.
  - Runtime verification completed for `(1.00, 0.72, pi)`, `(1.20, 1.10, pi)`, and `(2.50, 2.10, pi)`.
  - Conclusion: the remaining navigation blocker is not solved by changing `positions` alone; the robot initial
    footprint/costmap setup is already lethal or disconnected before the target point matters.

### Issue 6: Nav2 costmap footprint/inflation makes otherwise free start cells lethal

- Symptom:
  - Runtime continued to fail during `navigate` before any `cmd_vel` was received.
  - Nav2 repeatedly logged `GridBased: failed to create plan, invalid use: Starting point in lethal space`.
  - This happened even when the active goal was moved near the robot at `(2.50, 2.10, pi)`.
- Cause:
  - Latest debug maps show the robot start cell itself is free in `map.pgm`, but nearest static obstacles are only
    about `0.30-0.36m` away.
  - The default Ranger Nav2 model uses an approximately `0.29m` inscribed footprint radius, `0.03m` footprint padding,
    and `0.34m` inflation radius. After Nav2 applies footprint and inflation, these start cells are treated as lethal.
  - The previous `positions` changes reached Nav2 correctly; the current failure is a costmap model issue.
- Fix:
  - Added task-level `nav2_skill` override support in `nav2/runtime/config.py`.
  - Added `navigate` skill passthrough for `nav2_skill` overrides from YAML.
  - In the current entry task's two `navigate` nodes, set a reduced planning footprint:
    `[[0.18, 0.12], [0.18, -0.12], [-0.18, -0.12], [-0.18, 0.12]]`,
    `inflation_radius_m: 0.12`, and zero footprint padding for local/global costmaps.
- Verification:
  - `python -m py_compile nav2/runtime/config.py workflows/simbox/core/skills/navigate.py` passed.
  - `ruamel.yaml` parses the current entry task and both `navigate` nodes contain the new `nav2_skill` override.
  - Runtime verification with Docker session `882b4435-8486-460a-b399-95ebde4fe71c` showed the generated
    per-skill params are not enough by themselves: the resident Nav2 bootstrap file still used the default Ranger
    footprint, `footprint_padding: 0.03`, and `inflation_radius: 0.34`.
  - Nav2 still aborted `navigate` with status code 6.

### Issue 7: trying closer `positions` does not remove the Nav2 blocker

- Symptom:
  - After changing the active `nav_to_pick/nav_to_place` goal from `(2.50, 2.10, pi)` to `(2.82, 1.82, pi)`,
    runtime still failed during `navigate` with `reason=bridge_aborted message=goal finished with status_code=6`.
  - No episode reached the later `pick` skill.
- Trial:
  - Added these position candidates to the current entry task:
    `(2.82, 1.82, pi)`, `(2.72, 1.86, pi)`, `(2.62, 1.92, pi)`,
    `(2.90, 1.62, pi)`, and `(2.70, 1.62, pi)`.
  - Static map check on the latest debug map showed `(2.82, 1.82)`, `(2.72, 1.86)`, and `(2.62, 1.92)` were
    free cells with nearest obstacle distance about `0.32-0.39m`.
  - Restarted with `docker compose -f docker/docker-compose.yml down && scripts/docker/up_nav2_stack.sh isaac nav2`.
- Runtime evidence:
  - `split_aloha_nav2_goal_20260607_071704`: goal `(2.82, 1.82, -pi)`, `cmd_vel=0`,
    final distance about `0.48m`; Nav2 logged `Starting point in lethal space`.
  - `split_aloha_nav2_goal_20260607_071717`: goal `(2.82, 1.82, -pi)`, `cmd_vel=7`,
    final xy distance about `0.062m`, but yaw error about `0.273rad`; controller aborted.
  - `split_aloha_nav2_goal_20260607_071729`: goal `(2.82, 1.82, -pi)`, `cmd_vel=46`,
    final xy distance about `0.140m`, yaw error about `0.088rad`; planner still retried with
    `Starting point in lethal space`.
- Conclusion:
  - Positions alone are not sufficient. Some randomized starts can get close to the new point, but the resident Nav2
    process is still running with the default bootstrap footprint/inflation and tight Nav2 goal tolerances.
  - The next real fix should make the running Nav2 stack use the intended reduced footprint/inflation parameters,
    or otherwise reconfigure/restart Nav2 with those params before sending goals.

### Issue 8: resident Nav2 bootstrap ignored the per-skill footprint/inflation overrides

- Symptom:
  - The task-level `nav2_skill` override generated the intended per-skill debug params, but the already-running Nav2
    stack still used the default Ranger footprint, default `footprint_padding: 0.03`, and default
    `inflation_radius: 0.34`.
  - Runtime still failed in `navigate` before stable motion.
- Cause:
  - `nav2/mapgen/prepare_stack.py` only built resident bootstrap params from the robot/base config.
  - The Docker `nav2` service also did not receive any per-skill override environment variable, so even adding JSON
    parsing to `prepare_stack.py` was not enough until Compose passed the value through.
- Fix:
  - Added `INTERNDATA_NAV2_SKILL_OVERRIDES_JSON` / `--nav2-skill-overrides-json` support to
    `nav2/mapgen/prepare_stack.py`.
  - Updated `scripts/docker/up_nav2_stack.sh` to provide the reduced footprint/inflation JSON by default when the env
    var is unset.
  - Updated `docker/docker-compose.yml` so the `nav2` container receives
    `INTERNDATA_NAV2_SKILL_OVERRIDES_JSON`.
- Verification:
  - `python -m py_compile nav2/mapgen/prepare_stack.py` passed.
  - Runtime session `d41e3ffe-103c-486c-a352-4ae81e2f408d` generated bootstrap params with:
    footprint `[[0.180, 0.120], [0.180, -0.120], [-0.180, -0.120], [-0.180, 0.120]]`,
    `footprint_padding: 0.0`, and `inflation_radius: 0.12`.
  - New session `75fe1841-8db5-4e81-a136-9f8cba9ed0e8` re-verified the same resident bootstrap params.

### Issue 9: after the footprint fix, navigation commands move but do not reliably converge to the goal

- Symptom:
  - Nav2 no longer fails immediately with `Starting point in lethal space`.
  - The robot receives many `/cmd_vel` commands, but the workflow often terminates while `navigate` is still active.
- Runtime evidence:
  - `split_aloha_nav2_goal_20260607_072827`: goal `(2.82, 1.82, -pi)`, start about `(3.306, 1.834)`,
    end about `(3.377, 1.868)`, final distance about `0.56m`; the robot drifted away from the goal.
  - `split_aloha_nav2_goal_20260607_073044`: goal `(2.82, 1.82, -pi)`, start about `(2.764, 1.842)`,
    end about `(2.837, 1.761)`; the robot did not satisfy the goal before step-limit reset.
  - `split_aloha_nav2_goal_20260607_073252`: goal `(2.82, 1.82, -pi)`, start about `(2.961, 1.788)`,
    end about `(2.672, 1.978)`; still not within Nav2 tolerance before reset.
- Conclusion:
  - The remaining blocker is no longer only a lethal costmap. Nav2 is producing commands, but the physical base
    response is not reliably following those commands to the goal.

### Issue 10: additional `positions` around the observed reachable area still do not solve navigation

- Trial:
  - Added more current-entry candidates:
    `(2.67, 1.98, pi)`, `(2.76, 1.76, pi)`, `(2.84, 1.74, pi)`, and `(2.95, 1.78, pi)`.
  - Set active `nav_to_pick` and `nav_to_place` to `(2.67, 1.98, pi)`.
  - Restarted with:
    `docker compose -f docker/docker-compose.yml down && scripts/docker/up_nav2_stack.sh isaac nav2`.
- Runtime evidence from session `75fe1841-8db5-4e81-a136-9f8cba9ed0e8`:
  - `split_aloha_nav2_goal_20260607_074050`: goal `(2.67, 1.98, -pi)`, start about `(3.306, 1.834)`,
    end about `(3.374, 1.877)`, final distance about `0.712m`; step-limit reset.
  - `split_aloha_nav2_goal_20260607_074307`: start about `(2.765, 1.843)`, end about `(2.860, 1.760)`,
    final distance about `0.291m`, yaw error about `0.355rad`; step-limit reset.
  - `split_aloha_nav2_goal_20260607_074516`: start about `(2.961, 1.788)`, end about `(2.868, 1.633)`,
    final distance about `0.399m`; Nav2 aborted with status code `6`.
  - `split_aloha_nav2_goal_20260607_074620`: start about `(2.751, 1.930)`, end about `(2.790, 1.949)`,
    final distance about `0.124m`, yaw error about `0.274rad`; step-limit reset.
- Conclusion:
  - Adding closer positions improves some starts but does not produce a legal completed navigation.
  - The active blocker is the base command/actuation behavior, not missing `positions` entries.
