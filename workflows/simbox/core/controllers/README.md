# Arm Controllers

CuRobo arm controllers keep two intentionally separate execution lanes. The
legacy lane preserves the original controller contract for non-Pick/Place
Skills; the Physics-schema lane is reserved for Pick and Place.

The concrete owners are:

| Layer | Responsibility |
|---|---|
| `TemplateController` | Isaac lifecycle and the legacy tuple dispatcher |
| `MotionPlannerRuntime` | CuRobo planning, scene revisions, and attachments |
| `ControllerSetup` | frame/joint setup, scene synchronization, reset, and compact diagnostics |
| `ControllerExecution` | trajectory consumption, legacy `ee_forward`, typed phases, status, gripper, and hold |
| `SkillRuntimePort` | the only controller API visible to Skills |

`ArmSpec` in `controller_registry.py` contains robot differences such as
planner/control joint names, gripper range, collision cache, and grasp axis.
Robot-specific controllers should normally only register an `ArmSpec`.

## Module layout

The active CuRobo implementation is intentionally small:

- `curobo/controller.py` — `TemplateController` and Isaac lifecycle bridge.
- `curobo/components.py` — shared mutable execution state and planning config.
- `curobo/scene_setup.py` — robot/frame resolution, world construction, reset,
  dynamic scene synchronization, and plan diagnostics.
- `curobo/runtime.py` — state conversion, pose/c-space planning, FK/path
  metrics, and native attachment synchronization.
- `curobo/execution.py` — typed phase execution and direct articulation actions.
- `curobo/skill_runtime.py` — the narrow Skill-facing port.
- `curobo/phase_execution.py` and `curobo/trajectory.py` — trajectory cursor
  and named trajectory boundaries.

Pick and Place own candidate generation, ranking, command construction, and
attachment synchronization. They call the shared `plan_pose*` methods and
typed phase executor; the controller does not construct Pick/Place business
phases.

## Typed commands and direct actions

Pick/Place motion is represented by `MotionPhaseCommand` and is consumed
through the Physics-schema `execute(command)` path. A planner-backed command
may contain an EE target or a `joint_target`.

Non-Pick/Place legacy motion continues to use the original public boundary:

```python
action = controller.forward((ee_position, ee_orientation, method, params))
```

Tuple decoding exists only in `TemplateController.forward`. The dispatcher
routes `ee_forward`, `pre_forward`, scene update hooks, mobile/wrist actions,
and `dummy_forward` to the single `ControllerExecution` owner.

An execution-only command uses the explicit field
`direct_joint_action: np.ndarray | None`. It is valid only without an EE or
joint planner target, is automatically assigned
`CollisionPolicy.PASSTHROUGH`, and may carry `gripper_action` or
`gripper_state` (`1.0` open, `-1.0` closed). It never creates a Physics-schema
planning request. Home and heuristic home use this typed form.

Any Skill may also call the direct primitive:

```python
action = skill_runtime.dummy_forward(arm_action, gripper_state)
```

This sends one direct arm action through `ControllerExecution`; the caller
owns interpolation and completion. Home, heuristic joint interpolation, and
joint-control commands use this lane. Typed direct commands preserve a
completed execution status for Skills that consume the shared status API.

## SkillRuntimePort API

`SkillRuntimePort` is the only controller-facing object bound into a Skill.
Its public contract is intentionally explicit:

```text
robot, name, arm_name, arm_indices, gripper_indices
robot_file, robot_config, robot_base_path, robot_ee_path, reference_prim_path
batch_capability, interpolation_dt, num_plan_failed

ee_pose(), arm_base_pose(), initial_ee_pose(), compute_fk(), arm_base_transform()
plan_pose(), plan_pose_batch(), plan_pose_result()
plan_pose_from_path(), plan_pose_from_joint_positions(), measure_cartesian_path()
execute(command), dummy_forward(arm_action, gripper_state)
phase_complete(command), execution_status(command=None)
hold(reason=None), clear_plan_and_hold()
transition_target(), restore_world(), source_support()
assert_attached_owner(), sync_native_batch_attachment(), complete_contact_phase()
push_timing_scope(), restore_timing_scope(), clear_timing_scope()
```

The port does not expose a controller object, native CuRobo planner, raw
scene/collision diagnostics, or old compatibility aliases. Use `arm_name` for
the selected arm and `compute_fk()` for forward kinematics.
