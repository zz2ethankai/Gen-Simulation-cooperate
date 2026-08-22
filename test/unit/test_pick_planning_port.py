"""Narrow host-side contract tests for the typed Pick planning entry."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"

import sys

if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.curobo.pick_planning import PickPlanningPort  # noqa: E402
from core.controllers.curobo.runtime import MotionPlannerRuntime  # noqa: E402
from core.planning.collision_scene_manager import PlannerScenePort  # noqa: E402
from core.planning.domain_types import (  # noqa: E402
    BatchPosePlanRequest,
    CollisionPolicy,
    PosePlanRequest,
)
from core.planning.planner_runtime import PlannerRuntime  # noqa: E402


class _TensorArgs:
    """Device adapter required by the formal PlannerScenePort contract."""

    @staticmethod
    def to_device(value):
        return value


def _runtime(scene_revision=4):
    runtime = PlannerRuntime(scene_revision=scene_revision, name="pick-port-test")
    runtime.robot_port = SimpleNamespace(
        tensor_args=_TensorArgs(),
        interpolation_dt=0.01,
    )
    return runtime


class _Manager:
    def __init__(self):
        self.calls = []

    def refresh_controller_reference_world(self, port, *, force=False):
        self.calls.append(("refresh", port.name, force))
        return True

    def sync_dynamic_poses(self, step_id, *, interval_steps, force=False):
        self.calls.append(("sync", step_id, interval_steps, force))
        return ["target"]

    def begin_target_transit(self, entity, robot, arm):
        self.calls.append(("transit", entity, robot, arm))

    def begin_target_approach(self, entity, robot, arm):
        self.calls.append(("approach", entity, robot, arm))

    def restore_world(self, entity):
        self.calls.append(("restore", entity))

    def has_native_obstacle(self, _port, path):
        return path == "/target/mesh"


def _port(manager):
    scene_port = PlannerScenePort(
        name="robot",
        lr_name="right",
        reference_prim_path="/World/robot/base",
        robot_ee_path="/World/robot/ee",
        tensor_args=_TensorArgs(),
        robot=SimpleNamespace(),
        runtime=_runtime(),
    )
    return PickPlanningPort(
        scene_port=scene_port,
        collision_scene_manager=manager,
        update_pose_cost_metric=lambda value: manager.calls.append(("criteria", value)),
        build_commands=lambda **kwargs: kwargs,
        arm_base_transform=lambda: "base",
        frame_debug=lambda: {},
        capture_reference=lambda _name: None,
        retarget_commands=lambda _name, commands: commands,
        replan_after_safety=lambda _name, _command, _commands: True,
        execution_ee_pose=lambda: ("position", "orientation"),
        phase_complete=lambda _command: True,
    )


def test_pick_entry_uses_formal_scene_port_and_typed_collision_policy():
    manager = _Manager()
    planning = _port(manager)

    assert planning.prepare_world("target") == 4
    assert manager.calls == [
        ("criteria", None),
        ("refresh", "robot", True),
        ("sync", 0, 1, True),
        ("transit", "target", "robot", "right"),
    ]

    assert planning.transition_target(
        "target", collision_policy=CollisionPolicy.TARGET_APPROACH
    ) == 4
    assert manager.calls[-1] == ("approach", "target", "robot", "right")
    planning.restore_world("target")
    assert manager.calls[-1] == ("restore", "target")

    try:
        planning.transition_target("target", collision_policy="target_approach")
    except TypeError as exc:
        assert "CollisionPolicy" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("string collision policies must be rejected")


def test_pick_entry_no_longer_calls_removed_controller_method():
    path = ROOT / "workflows" / "simbox" / "core" / "skills" / "pick.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_physics_schema_generate_manip_cmds"
    )
    source = ast.unparse(method)
    assert "planning" in source
    assert "prepare_pick_planning_world" not in source


def test_standard_pick_uses_only_narrow_planning_and_runtime_ports():
    path = ROOT / "workflows" / "simbox" / "core" / "skills" / "pick.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "self.controller" not in source

    pick_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Pick"
    )
    init = next(
        node
        for node in pick_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    init_source = ast.unparse(init)
    assert "self.bind_skill_runtime(skill_runtime, pick_planning=pick_planning)" in init_source
    assert "self.planning = self._require_pick_planning()" in init_source

    for forbidden in (
        "controller.robot_file",
        "controller.reference_prim_path",
        "controller.lr_name",
        "controller.arm_indices",
        "controller.gripper_indices",
        "controller.num_last_cmd",
        "controller.num_plan_failed",
    ):
        assert forbidden not in source
    assert "self.skill_runtime.arm_indices" in source
    assert "self.skill_runtime.gripper_indices" in source
    assert "self.planning.last_command_count" in source
    assert "self.planning.plan_failure_count" in source


def test_query_port_forwards_typed_metadata_and_strict_presence():
    manager = _Manager()
    requests = []

    def batch(_positions, _orientations, **kwargs):
        requests.append(("batch", kwargs))
        return "batch-result"

    def single(_position, _orientation, **kwargs):
        requests.append(("single", kwargs))
        return "single-result"

    planning = PickPlanningPort(
        scene_port=_port(manager).scene_port,
        collision_scene_manager=manager,
        update_pose_cost_metric=lambda _value: None,
        build_commands=lambda **kwargs: kwargs,
        arm_base_transform=lambda: "base",
        frame_debug=lambda: {},
        capture_reference=lambda _name: None,
        retarget_commands=lambda _name, commands: commands,
        replan_after_safety=lambda _name, _command, _commands: True,
        execution_ee_pose=lambda: ("position", "orientation"),
        phase_complete=lambda _command: True,
        robot_file="panda_right.yml",
        batch_capability=True,
        plan_pose_batch=batch,
        plan_pose_result=single,
        plan_pose_from_path=lambda *_args, **kwargs: requests.append(
            ("path", kwargs)
        ),
        measure_cartesian_path=lambda *_args: (1.0, 0.0),
    )
    planning.prepare_world("target")
    planning.plan_pose_batch([], [])
    assert requests[-1][1]["request_metadata"] == {
        "phase_id": "pick_pregrasp_batch",
        "collision_policy": CollisionPolicy.WORLD_TRANSIT,
        "active_target": "target",
    }
    assert planning.has_native_obstacle("/target/mesh")
    assert not planning.has_native_obstacle("/target/missing")

    planning.transition_target(
        "target", collision_policy=CollisionPolicy.TARGET_APPROACH
    )
    planning.plan_pose_result([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert requests[-1][1]["request_metadata"]["collision_policy"] is CollisionPolicy.TARGET_APPROACH
    assert requests[-1][1]["request_metadata"]["phase_id"] == "pick_terminal_grasp"
    terminal_options = requests[-1][1]["request_metadata"]["collision_options"]
    assert terminal_options.policy is CollisionPolicy.TARGET_APPROACH
    assert terminal_options.allow_target_contact is True
    assert terminal_options.allow_target_finger_contact is True


def test_pick_port_typed_runtime_callbacks_preserve_world_revision():
    """The formal Pick port stamps one scene revision on both typed requests."""

    class _NativeRuntime:
        scene_revision = 17

        def __init__(self):
            self.batch = SimpleNamespace(batch_size=4)
            self.requests = []

        def ensure_batch_planner(self):
            return self.batch

        def plan_pose(self, request):
            self.requests.append(request)
            return request

        def plan_pose_batch(self, request):
            self.requests.append(request)
            return request

    native_runtime = _NativeRuntime()
    motion_runtime = object.__new__(MotionPlannerRuntime)
    motion_runtime.max_plan_attempts = 3
    motion_runtime.batch_max_attempts = 2
    motion_runtime.single_graph_attempt = 1
    motion_runtime.batch_graph_attempt = 1
    motion_runtime.planner_runtime = native_runtime
    motion_runtime.robot_port = SimpleNamespace(
        robot=SimpleNamespace(get_joints_state=lambda: "sim-state"),
        tensor_args=_TensorArgs(),
    )
    motion_runtime._goal_tool_pose = lambda position, orientation, batch_size=1: (
        position,
        orientation,
        batch_size,
    )
    motion_runtime.arm_joint_state = lambda state, repeat=1: (state, repeat)

    manager = _Manager()
    scene_port = PlannerScenePort(
        name="robot",
        lr_name="right",
        reference_prim_path="/World/robot/base",
        robot_ee_path="/World/robot/ee",
        tensor_args=_TensorArgs(),
        robot=SimpleNamespace(),
        runtime=motion_runtime,
    )
    controller = SimpleNamespace(
        pick_planning=PickPlanningPort(
            scene_port=scene_port,
            collision_scene_manager=manager,
            update_pose_cost_metric=lambda _value: None,
            build_commands=lambda **kwargs: kwargs,
            arm_base_transform=lambda: "base",
            frame_debug=lambda: {},
            capture_reference=lambda _name: None,
            retarget_commands=lambda _name, commands: commands,
            replan_after_safety=lambda _name, _command, _commands: True,
            execution_ee_pose=lambda: ("position", "orientation"),
            phase_complete=lambda _command: True,
            robot_file="panda_right.yml",
            batch_capability=True,
            plan_pose_batch=motion_runtime.plan_pose_batch,
            plan_pose_result=motion_runtime.plan_pose,
        )
    )
    planning = controller.pick_planning

    planning.prepare_world("target")
    batch_request = planning.plan_pose_batch(
        [[0.1, 0.2, 0.3]], [[1.0, 0.0, 0.0, 0.0]]
    )
    single_request = planning.plan_pose_result(
        [0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]
    )

    assert isinstance(batch_request, BatchPosePlanRequest)
    assert isinstance(single_request, PosePlanRequest)
    assert batch_request.world_revision == single_request.world_revision == 17
    assert batch_request.metadata["world_revision"] == single_request.metadata["world_revision"] == 17
    assert batch_request.phase_id == "pick_pregrasp_batch"
    assert single_request.phase_id == "pick_pregrasp"
    assert batch_request.active_target == single_request.active_target == "target"


def test_template_composes_pick_placement_and_skill_runtime_ports():
    """Template composition keeps all typed operation ports together."""

    path = (
        ROOT
        / "workflows"
        / "simbox"
        / "core"
        / "controllers"
        / "curobo"
        / "controller.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    init = next(
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    init_source = ast.unparse(init)

    assert "self.pick_planning = compose_pick_planning_port(" in init_source
    assert "self.placement_planning = compose_placement_planning_port(" in init_source
    assert "self.skill_runtime = compose_skill_runtime_port(" in init_source
    assert "self.runtime.plan_pose_batch" in init_source
    assert "self._planning_queries.plan_pose_result" in init_source
    assert "self._planning_queries.plan_pose_from_path" in init_source
    assert "self._planning_queries.measure_cartesian_path" in init_source

    # The façade must not regain the removed whole-controller planner aliases.
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name in {"plan_pose", "plan_pose_batch"}
        for node in controller.body
    )
