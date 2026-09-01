---
name: scene-format-conversion
description: Convert arbitrary InterData and SimBox scene YAML pairs in either direction while preserving assets, layouts, regions, textures, and physics; use when a scene format must be translated or round-trip validated. Do not use for USD editing, robot placement solving, or skill planning.
---

# Scene Format Conversion

Convert scene semantics through a format-neutral intermediate representation. Never copy YAML fields blindly and never overwrite the source scene.

## Inputs and outputs

- InterData pair: `interdata/task.yaml` + `interdata/arena.yaml`.
- SimBox pair: `simbox_task.yaml` + `simbox_arena.yaml`.
- `engine.yaml` is an optional launcher configuration, not part of semantic conversion; update it separately when requested.
- The final SimBox runtime pair is exactly `simbox_task.yaml` and `simbox_arena.yaml`.
- For generated scene suites, promote that pair to `runs/<suite>/base_scenes/<scene_id>/`; do not treat the generation-job root or `conversion_staging` as the final runtime location.

Before editing, detect the format from the root schema and resolve the arena reference. Stage outputs in a new directory, validate them, then promote them. Keep source files read-only.

## Required workflow

1. Load both files and record source paths, format/version, asset root, and coordinate frame.
2. Normalize into `SceneIR` with `scene_id`, canonical coordinate frame, fixtures, objects, regions, textures, environment, cameras, and provenance.
3. Resolve every asset, texture, camera, and arena reference once; rewrite paths relative to the target contract. Fail on missing files or unresolved references.
4. Normalize aliases and physics: `usd_path` ↔ `path`, `arena` ↔ `arena_file`, `mass_kg` ↔ `mass`, and rigid/collision/friction fields through one physics block. Preserve unmapped data under `source_*` or a conversion manifest.
5. Canonicalize transforms as world XYZ meters, +Z up, XY floor, Euler XYZ degrees. InterData `isaac_sim_world_xyz` is already canonical. Apply SimBox's legacy `layout [x,y,z] -> usd [x,z,y]` only when the field is explicitly in layout coordinates; if already world/Isaac coordinates, do not apply it again. Convert rotations with a matrix/quaternion frame transform and extract Euler—never swap Euler components by hand.
6. Preserve fixture/object geometry, scale, transforms, parent/support relations, regions, textures, cameras, and physics. Regions are one semantic collection even if formats store them in different files; merge by name and reject conflicting duplicates.
7. Clear execution-specific data unless the user explicitly asks to compile it: robot pose, robot waypoints, navigation waypoints, derived positions, skills, and task substeps. Preserve object placement regions, task-grounding roles, and cameras. Clearing robot execution does not clear scene acquisition: fixed and robot-mounted cameras remain unless the user explicitly requests a camera-free scene.
8. Emit the requested target pair, validate schema and references, and run a round-trip check (`InterData → SimBox → InterData` or the reverse) before publishing.

## Direction-specific mapping

### InterData → SimBox

- `fixtures` map to SimBox arena fixtures; normalize `usd_path` to `path` if present.
- Put object/support/container regions in the SimBox task because that is where the runtime task schema consumes them; keep fixture/navigation/texture metadata in the arena.
- Map `objects[].usd_path` to `objects[].path` and retain object asset, transform, placement, role, and physics metadata.
- For a fixed HDR, preserve the direct file as `env_map.path` and also emit the legacy-compatible parent directory as `env_map.envmap_lib`; set `apply_randomization: false`. Keep normalized `intensity`/`rotation_deg` alongside any required legacy ranges so both runtime schemas resolve the same lighting asset.
- Preserve non-empty source cameras even when `skills`, waypoints, and robot poses are cleared. Normalize each `camera_file` to a target-repository-relative runtime path and preserve its calibrated translation, orientation, axes, parent prim, and randomization flag. For the SplitAloha SimBox runtime, use `workflows/simbox/core/configs/cameras/*.yaml` (for example `astra.yaml` and `realsense_d455_v3.yaml`), not a scene-local alias.
- Set the execution placeholders to `robots: []`, `skills: []`, `source_tasks: []`, `waypoints: []`, `positions: {}`, and `robot_waypoints: []` in the arena. If a downstream schema requires a robot profile, keep only its name/config with `translation/euler: null` and mark the output `scene-only`.

### SimBox → InterData

- Read the task's `arena_file`, then map arena fixtures back to InterData fixtures; normalize `path` to `usd_path`.
- Collect task `regions`, `source_regions`, and `container_regions` into the InterData task after deduplication and conflict checks.
- Map `objects[].path` to `objects[].usd_path`; preserve environment, cameras, physics, asset roots, and provenance.
- Set `robot: null` (or `{}` only when required by a strict schema), `waypoints: []`, `positions: {}`, and `tasks: []`. Do not infer a robot pose or reconstruct skills from names. Store discarded execution data only in the manifest.

## Blank execution contract

The following fields must be empty in a scene-only conversion:

```yaml
# SimBox task
task: BananaBaseTask
task_id: 0
offset: null
robots: []
skills: []
source_tasks: []
waypoints: []
positions: {}
cameras: <preserved normalized scene cameras>

# SimBox arena
robot_waypoints: []

# InterData task
robot: null
waypoints: []
positions: {}
tasks: []
```

The SimBox `task`, `task_id`, and `offset` fields are the runtime identity scaffold, not an execution plan. Keep them in scene-only output so the file can later be promoted by robot-placement and planning skills without failing during workflow reset.

`cameras` is intentionally not an empty execution placeholder. It is scene/acquisition configuration. Preserve a non-empty camera rig across `clear_execution_plan`; remove it only when the source has no cameras or the user explicitly requests no cameras. Robot-mounted camera parents must still match the selected canonical robot profile even when there are no executable skills.

Do not delete object translations, spawn regions, support surfaces, fixture transforms, or physics just because execution fields are empty. If source robot/task data is retained, place it under `conversion_manifest.source`, never in executable fields.

## Agent behavior

When this skill is invoked, the Agent should use typed stages such as `detect_format`, `load_pair`, `normalize_to_scene_ir`, `convert_direction`, `clear_execution_plan`, `validate_scene`, `round_trip_check`, and `promote_outputs`. Report field mappings, warnings, and intentional losses. On any failed check, stop and roll back staging; do not guess missing robot, coordinate, asset, or skill information.

## Validation checklist

- Root schema, format/version, and arena reference are valid.
- Object, fixture, and region names are unique; `parent_fixture`, `A/B/target`, and spawn references resolve.
- Asset, texture, camera, and arena files exist; paths are target-relative and not accidentally absolute.
- A non-empty source camera list remains non-empty after execution clearing; camera names are unique, `camera_file` paths resolve from the target repository root, and robot-mounted `parent` prims match the selected robot profile.
- Transforms are finite, scales are positive, units are meters, and exactly one coordinate transform was applied.
- Physics flags and support relationships are internally consistent.
- Blank execution contract is satisfied.
- Round-trip preserves IDs, asset paths, dimensions, scale, transforms, support relations, material/physics fields, and coordinate declarations within tolerance. Ignore only field aliases, layout/caches, and explicitly cleared execution data.

## Non-goals

This skill does not generate or edit USD geometry, collision meshes, grasp annotations, robot placements, reachability, navigation, skills, substeps, or final Isaac Sim physics results. Those belong to later embodiment, planning, and simulation-validation stages.

For the project-specific field inventory and examples, read [the source workflow](../../../docs/Set_New_Task/scene_format_conversion_workflow.md).
