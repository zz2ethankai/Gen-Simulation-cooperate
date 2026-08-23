# Arm Controllers

CuRobo arm controllers use one typed command path and four responsibility
layers:

| Layer | Responsibility |
|---|---|
| `TemplateController` | Isaac lifecycle and component assembly |
| `MotionPlannerRuntime` | CuRobo planning, scene revisions, and attachments |
| `ControllerExecution` | trajectory consumption, status, gripper, hold, and direct actions |
| `SkillRuntimePort` | the only controller API visible to Skills |

`ArmSpec` in `controller_registry.py` contains robot differences such as
planner/control joint names, gripper range, collision cache, and grasp axis.
Robot-specific controllers should normally only register an `ArmSpec`.

## Module layout

The active CuRobo implementation is intentionally small:

- `curobo/controller.py` — `TemplateController` and Isaac lifecycle bridge.
- `curobo/components.py` — shared mutable execution state, planning config,
  and setup/execution wiring.
- `curobo/scene_setup.py` — robot/frame resolution, world construction, reset,
  dynamic scene synchronization, and plan diagnostics.
- `curobo/runtime.py` — state conversion, pose/c-space planning, FK/path
  metrics, and native attachment synchronization.
- `curobo/execution.py` — typed phase execution and direct articulation actions.
- `curobo/skill_runtime.py` — the narrow Skill-facing port.
- `curobo/phase_execution.py` and `curobo/trajectory.py` — trajectory cursor
  and named trajectory boundaries.

Pick and Place own candidate generation, ranking, and manipulation phase
construction. They call the shared `plan_pose*` methods and attachment sync;
the controller does not construct Pick/Place business phases.

## Typed commands and direct actions

Normal motion is represented by `MotionPhaseCommand` and is consumed through
`execute(command)`. A planner-backed command may contain an EE target or a
`joint_target`.

An execution-only command uses the explicit field
`direct_joint_action: np.ndarray | None`. It is valid only without an EE or
joint planner target, is automatically assigned
`CollisionPolicy.PASSTHROUGH`, and may carry `gripper_action` or
`gripper_state` (`1.0` open, `-1.0` closed). It never creates a Physics-schema
planning request. Home and heuristic home use this typed form.

Any Skill may also call:

```python
action = skill_runtime.dummy_forward(arm_action, gripper_state)
```

This sends one direct arm action through `ControllerExecution`; the caller
owns interpolation and completion. The old tuple command and
legacy tuple-parameter parser are removed.

## SkillRuntimePort API

The port exposes robot/configuration values (`robot`, `name`, `arm_name`,
joint indices, robot paths, `robot_config`, `batch_capability`,
`interpolation_dt`, and `num_plan_failed`), FK/pose queries, the unified
`plan_pose*` methods, `measure_cartesian_path`, typed execution/status/hold
operations, collision-world transitions, attachment ownership/synchronizing
operations, and workflow timing scopes. It does not expose a controller
façade, native CuRobo planner, legacy `lr_name`/`robot_cfg` aliases, or raw
scene/collision diagnostics.

The public planning calls are:

```text
plan_pose()
plan_pose_batch()
plan_pose_result()
plan_pose_from_path()
plan_pose_from_joint_positions()
measure_cartesian_path()
```

Use `arm_name` for the selected arm and `compute_fk()` for forward
kinematics. Scene ownership is accessed through `transition_target()`,
`restore_world()`, `source_support()`, `assert_attached_owner()`,
`sync_native_batch_attachment()`, and `complete_contact_phase()`.
