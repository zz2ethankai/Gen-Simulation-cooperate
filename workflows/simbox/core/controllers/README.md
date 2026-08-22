# Arm Controllers

Template-based arm controllers for CuRobo motion planning. All arm controllers inherit from `TemplateController` and customize behavior via overrides.

## Directory layout

The controller package separates shared CuRobo implementation from robot
adapters:

- `core/controllers/curobo/` contains `TemplateController` and the shared
  execution, scene setup, planning-query, attachment, phase, trajectory, and
  runtime modules.
- `core/controllers/*_controller.py` contains the registered robot-specific
  subclasses (`FR3Controller`, `PandaOmronController`, `Lift2Controller`, and
  so on).
- `core/controllers/controller_registry.py` contains `ArmSpec` and controller
  registration. Skill-facing ports remain in the CuRobo package because they
  are composed by the shared controller runtime.

The shared modules are named by responsibility rather than by repeating the
`controller_` prefix: for example, `curobo/execution.py`,
`curobo/scene_setup.py`, `curobo/motion_phases.py`, and
`curobo/phase_execution.py`. Import `TemplateController` from
`core.controllers.curobo.controller`; the parent package keeps its lazy
`core.controllers.TemplateController` export for callers that use the public
registry API.

| Former module | Responsibility module |
|---|---|
| `controller_component.py` | `curobo/components.py` |
| `controller_execution.py` | `curobo/execution.py` |
| `controller_setup.py` | `curobo/scene_setup.py` |
| `controller_phases.py` | `curobo/motion_phases.py` |
| `controller_state_planning.py` | `curobo/state_planning.py` |
| `controller_planning_queries.py` | `curobo/planning_queries.py` |
| `controller_attachment.py` | `curobo/attachments.py` |
| `phase_executor.py` | `curobo/phase_execution.py` |
| `trajectory_boundary.py` | `curobo/trajectory.py` |
| `template_controller.py` | `curobo/controller.py` |
| `runtime.py` | `curobo/runtime.py` |
| `skill_runtime.py` | `curobo/skill_runtime.py` |
| `base_controller.py` | `controller_registry.py` |

## Direct joint execution

Structured manipulation uses `execute(MotionPhaseCommand)` and the Physics-schema
planner. Skills that own a direct joint interpolation may instead call
`dummy_forward(arm_action, gripper_state, *args, **kwargs)` on the controller
(or the corresponding `SkillRuntimePort` method). It returns the normal
articulation action and does not invoke c-space planning. The Skill is
responsible for generating interpolation points and deciding when each point is
complete. The workflow accepts the legacy command form
`(ee_position, ee_orientation, "dummy_forward", params)` as well, where
`params` contains `arm_action` and `gripper_state`.

This interface is available to any Skill; it is not selected by a skill-name
allowlist. It bypasses Physics-schema collision planning, automatic replanning,
and the Physics-schema execution-safety precheck, so callers must only use it
for paths whose safety has already been established.

## Pick/Place responsibility boundary

Pick and Place keep the task semantics: they sample and rank grasp/place
candidate poses, apply task-specific geometry constraints, and emit the
pre-grasp, post-grasp, pre-place, post-place, and gripper phases.  Both use the
same `SkillRuntimePort` for the generic `plan_pose*` queries, FK/path metrics,
and attachment synchronization; the runtime is the sole owner of collision-
world transitions.  There is no
controller façade split by operation type: the controller owns the generic
CuRobo/scene primitives, while the Skills own operation-specific candidate and
phase semantics.

## Available controllers

| Controller | Robot | Notes |
|------------|--------|------|
| `FR3Controller` | Franka FR3 (Panda arm, 7+1 gripper) | Single arm only; larger collision cache (1000). |
| `FrankaRobotiq85Controller` | Franka + Robotiq 85 (7+2 gripper) | Single arm only; custom gripper action (inverted, clip 0–5). |
| `Genie1Controller` | Genie1 dual arm (7 DoF per arm) | Left/right via `robot_file`; path-selection weights. |
| `Lift2Controller` | ARX-Lift2 dual arm (6 DoF arm) | Left/right; custom world (cuboid offset 0.02), grasp axis 0 (x), in-plane rotation index 5. |
| `SplitAlohaController` | Agilex Split Aloha dual arm (6 DoF per arm) | Left/right; grasp axis 2 (z); optional `joint_ctrl`. |

Register a controller by importing it (see `__init__.py`) so it is added to `CONTROLLER_DICT`.

---

## Customizing a robot arm controller

Subclass `TemplateController` and implement or override the following.

### 1. Required: `_configure_joint_indices(self, robot_file: str)`

Set joint names and indices for the planner and the simulation articulation.

You must set:

- **`self.raw_js_names`** – Joint names in the **planner / CuRobo** order (arm only, no gripper). Used to reorder the public `JointTrajectory` before execution.
- **`self.cmd_js_names`** – Same as `raw_js_names` use the **scene/articulation** names in the robot usd (e.g. `fl_joint1`… or `idx21_arm_l_joint1`…).
- **`self.arm_indices`** – Indices of arm joints in the **simulation** `dof_names` (e.g. `np.array([0,1,2,3,4,5,6])`).
- **`self.gripper_indices`** – Indices of gripper joints in the simulation (e.g. `np.array([7])` or `np.array([7,8])`).
- **`self.reference_prim_path`** – Prim path used for collision reference (e.g. `self.task.robots[self.name].fl_base_path` for left arm).
- **`self.lr_name`** – `"left"` or `"right"` (for dual-arm).
- **`self._gripper_state`** – Initial gripper state from robot (e.g. `1.0 if self.robot.left_gripper_state == 1.0 else -1.0`). By convention, `1.0` means **open**, `-1.0` means **closed**.
- **`self._gripper_joint_position`** – Initial gripper joint position(s), shape matching `gripper_indices` (e.g. `np.array([1.0])` or `np.array([5.0, 5.0])`).

For dual-arm, branch on `"left"` / `"right"` in `robot_file` and set the above per arm. For single-arm, only implement the arm you support and we set it as left arm.

### 2. Required: `_get_default_ignore_substring(self) -> List[str]`

Return the default list of name substrings for collision filtering (e.g. `["material", "Plane", "conveyor", "scene", "table"]`). The controller name is appended automatically. Override to add or remove terms (e.g. `"fluid"` for some setups).

### 3. Required: `get_gripper_action(self)`

Map the logical gripper state to gripper joint targets.

- **Input**: uses `self._gripper_state` (1.0 = open, -1.0 = closed) and `self._gripper_joint_position` as the magnitude / joint-space template.
- **Default mapping**: for simple parallel grippers, a good starting point is:

  ```python
  def get_gripper_action(self):
      return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)
  ```

- **Robot-specific variants**: some robots change the range or sign (e.g. Robotiq85 uses two joints, inverted sign, and clips to `[0, 5]`). Adjust the formula and clip range, but keep the convention that `self._gripper_state` is `1.0` for **open** and `-1.0` for **closed**.

### 4. Optional overrides

Override only when the default template behavior is wrong for your robot.

- **`_load_world(self, use_default: bool = True)`**
  Default uses native `SceneCfg()` when `use_default=True`, and when `False` uses a table with cuboid z offset `10.5`. Override if your table height or world is different (e.g. Genie1 uses `5.02`, Lift2 uses `0.02`).

- **`_get_native_collision_cache(self)`**
  Default: `{"obb": 700, "mesh": 700}`. Override to change cache size (e.g. FR3 uses `1000`).

- **`_get_grasp_approach_linear_axis(self) -> int`**
  Default: `2` (z-axis). Override if your grasp approach constraint uses another axis (e.g. Lift2 uses `0` for x).

- **`_get_sort_path_weights(self) -> Optional[List[float]]`**
  Default: `None` (equal weights). Override to pass per-joint weights for batch path selection (e.g. Genie1 uses `[1,1,1,1,3,3,1]` for 7 joints).

- **`get_gripper_action(self)`**
  Default: `np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)`. Override if your gripper mapping or range differs (e.g. Robotiq85: inverted sign and clip to `[0, 5]`).

### 5. Registration

- Use the `@register_controller` decorator on your class.
- Import the new controller in `__init__.py` so it is registered in `CONTROLLER_DICT`.

### 6. Robot config (YAML)

Your robot must have a CuRobo config YAML (passed as `robot_file`) with at least:

- `robot_cfg` with `kinematics` (e.g. `urdf_path`, `base_link`, `tool_frames`).

Template uses this for `_load_robot`, `_load_kin_model`, and `_init_native_planners`; no code change needed if the YAML is correct.

---

## Summary checklist for a new arm

1. Add a new file, e.g. `myrobot_controller.py`.
2. Subclass `TemplateController` and apply `@register_controller`.
3. Implement **`_configure_joint_indices(robot_file)`** (joint names and indices for planner and sim).
4. Implement **`_get_default_ignore_substring()`** (collision ignore list).
5. Implement **`get_gripper_action(self)`** to map `self._gripper_state` (1.0 = open, -1.0 = closed) and `self._gripper_joint_position` to gripper joint targets (clip to a sensible range for your hardware).
6. Override **`_load_world`** only if table/world differs from default.
7. Override **`_get_native_collision_cache`** / **`_get_grasp_approach_linear_axis`** / **`_get_sort_path_weights`** only if needed.
8. Import the new controller in `__init__.py`.
