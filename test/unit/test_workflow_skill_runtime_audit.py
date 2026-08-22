"""Static contracts for workflow safety's narrow runtime boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

import numpy as np

from core.controllers.curobo.components import MutableExecutionState
from core.controllers.curobo.skill_runtime import (
    SkillRuntimePort,
    compose_skill_runtime_port,
)
from core.execution.execution_supervisor import ExecutionSupervisor
from core.execution.safety_monitor import SafetyMonitor
from core.planning.motion_command import MotionPhase, MotionPhaseCommand


def test_workflow_safety_does_not_reach_through_controller_facade():
    source = (ROOT / "workflows/simbox_dual_workflow.py").read_text(encoding="utf-8")
    forbidden = (
        "controller.forward_kinematic",
        "controller.get_ee_pose",
        "controller.get_armbase_pose",
        "controller.arm_indices",
        "controller._last_commanded_arm_position",
        "controller._active_phase_command",
        "controller.complete_terminal_place_on_contact",
        "controller.hold_action",
    )
    assert all(token not in source for token in forbidden)
    assert "runtime = skill.skill_runtime" in source
    assert "skill.placement_planning.complete_terminal_place_on_contact" in source


def test_execution_supervisor_has_no_legacy_hold_fallback():
    source = (
        ROOT / "workflows/simbox/core/execution/execution_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "hold_action" not in source
    assert "runtime.hold(reason)" in source
    assert "runtime.clear_plan_and_hold()" in source
    assert "runtime.execute(command)" in source


def test_supervisor_accepts_narrow_runtime_fake():
    calls = []
    state = MutableExecutionState()
    status = SimpleNamespace(
        phase="transit_pregrasp", plan_id="plan", replan_allowed=True
    )
    runtime = SkillRuntimePort(
        robot=SimpleNamespace(),
        runtime=None,
        execution_state=state,
        arm_spec=None,
        arm_indices=[0],
        gripper_indices=[1],
        name="robot",
        arm_name="left",
        ee_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        arm_base_pose=lambda: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        compute_fk=lambda joints: (joints, np.array([1.0, 0.0, 0.0, 0.0])),
        execution_status=lambda _command=None: status,
        execute=lambda _command: calls.append("execute") or "motion",
        hold=lambda: calls.append("hold") or "hold",
        clear_plan_and_hold=lambda: calls.append("clear"),
    )
    command = MotionPhaseCommand(phase=MotionPhase.TRANSIT_PREGRASP)
    supervisor = ExecutionSupervisor(SafetyMonitor())

    assert supervisor.forward_or_hold(runtime, command) == "motion"
    assert calls == ["execute"]


def test_composed_runtime_port_calls_each_explicit_owner_once():
    """Exercise every composed callback instead of checking attributes only."""

    events = []

    class _ExecutionOwner:
        def ee_pose(self):
            events.append("ee_pose")
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def arm_base_pose(self):
            events.append("arm_base_pose")
            return np.ones(3), np.array([1.0, 0.0, 0.0, 0.0])

        def compute_fk(self, joints, *, joint_names=None):
            events.append(("compute_fk", tuple(joint_names or ())))
            return np.asarray(joints), np.array([1.0, 0.0, 0.0, 0.0])

        def execution_status(self, command=None):
            events.append(("execution_status", command))
            return {"complete": True}

        def command_status(self, command=None):
            events.append(("command_status", command))
            return "completed"

        def execute(self, command):
            events.append(("execute", command))
            return "executed"

        def hold(self):
            events.append("hold")
            return "held"

        def clear_plan_and_hold(self):
            events.append("clear_plan_and_hold")
            return "cleared"

    class _PlannerRuntime:
        scene_revision = 7

        def plan_pose(self, position, orientation, **kwargs):
            events.append(("plan_pose", position, orientation, kwargs))
            return "pose-plan"

        def plan_cspace(self, goal, **kwargs):
            events.append(("plan_cspace", goal, kwargs))
            return "cspace-plan"

    class _SetupOwner:
        def refresh(self):
            events.append("refresh")

        def __init__(self):
            self.scope = None

        def push(self, scope):
            previous = self.scope
            self.scope = scope
            events.append(("push_timing_scope", scope))
            return previous

        def restore(self, previous):
            self.scope = previous
            events.append(("restore_timing_scope", previous))

        def clear(self, scope=None):
            if scope is None or self.scope is scope:
                self.scope = None
            events.append(("clear_timing_scope", scope))

    class _SceneManager:
        def assert_attached_owner(self, entity, robot, arm):
            events.append(("assert_attached_owner", entity, robot, arm))
            return "attached"

        def get_source_support_entity(self, entity):
            events.append(("get_source_support_entity", entity))
            return "support"

        def get_attached_entity(self, robot, arm):
            events.append(("get_attached_entity", robot, arm))
            return "object"

        def has_native_obstacle(self, scene_port, path):
            events.append(("has_native_obstacle", scene_port, path))
            return path == "/World/obstacle"

        def get_attached_object_slip(self, entity):
            events.append(("attached_object_slip", entity))
            return {"slip": 0.0}

    execution = _ExecutionOwner()
    planner_runtime = _PlannerRuntime()
    setup = _SetupOwner()
    manager = _SceneManager()
    state = MutableExecutionState()
    robot = SimpleNamespace()
    scene_port = SimpleNamespace(name="robot", lr_name="right")
    port = compose_skill_runtime_port(
        robot=robot,
        runtime=planner_runtime,
        execution_state=state,
        arm_spec=SimpleNamespace(name="arm"),
        arm_indices=[0, 1],
        gripper_indices=[2],
        raw_joint_names=["joint0", "joint1"],
        control_joint_names=["joint0", "joint1"],
        robot_file="right_robot.yml",
        robot_config={"family": "fake"},
        robot_base_path="/World/robot/base",
        robot_ee_path="/World/robot/ee",
        reference_prim_path="/World/robot/base",
        name="robot",
        arm_name="right",
        batch_capability=True,
        interpolation_dt=0.01,
        ee_pose=execution.ee_pose,
        arm_base_pose=execution.arm_base_pose,
        compute_fk=execution.compute_fk,
        initial_ee_pose=execution.ee_pose,
        execution_status=execution.execution_status,
        command_status=execution.command_status,
        execute=execution.execute,
        hold=execution.hold,
        clear_plan_and_hold=execution.clear_plan_and_hold,
        push_timing_scope=setup.push,
        restore_timing_scope=setup.restore,
        clear_timing_scope=setup.clear,
        collision_scene_manager=manager,
        scene_port=scene_port,
        refresh_reference_world=setup.refresh,
    )

    # Read every immutable/state property exposed to Skills.
    assert port.robot is robot
    assert port.runtime is planner_runtime
    assert port.execution_state is state
    assert port.arm_spec.name == "arm"
    assert port.arm_indices.tolist() == [0, 1]
    assert port.gripper_indices.tolist() == [2]
    assert port.raw_joint_names == ("joint0", "joint1")
    assert port.control_joint_names == ("joint0", "joint1")
    assert port.robot_file == "right_robot.yml"
    assert port.robot_config["family"] == "fake"
    assert port.robot_cfg["family"] == "fake"
    assert port.robot_base_path == "/World/robot/base"
    assert port.robot_ee_path == "/World/robot/ee"
    assert port.reference_prim_path == "/World/robot/base"
    assert port.name == "robot"
    assert port.arm_name == port.lr_name == "right"
    assert port.batch_capability is True
    assert port.interpolation_dt == 0.01
    assert port.num_last_cmd == 0
    assert port.num_plan_failed == 0
    assert port.step_idx == 0
    assert port.active_phase_command is None
    assert port.last_commanded_arm_position is None
    assert port.phase_base_pose() is None

    state.last_commanded_arm_position = [1.0, 2.0]
    state.phase_base_position = [3.0, 4.0, 5.0]
    state.phase_base_orientation = [1.0, 0.0, 0.0, 0.0]
    assert port.last_commanded_arm_position.tolist() == [1.0, 2.0]
    assert port.phase_base_pose()[0].tolist() == [3.0, 4.0, 5.0]

    assert port.record_plan_failure() == 1
    port.num_plan_failed = 3
    assert port.num_plan_failed == 3
    port.reset_plan_failures()
    assert port.num_plan_failed == 0

    # Call every runtime/execution callback.
    assert port.ee_pose()[0].shape == (3,)
    assert port.arm_base_pose()[0].shape == (3,)
    assert port.compute_fk([0.0, 1.0], joint_names=["joint0", "joint1"])[0].shape == (2,)
    assert port.initial_ee_pose()[0].shape == (3,)
    assert port.plan_pose("position", "orientation", context="test") == "pose-plan"
    assert port.plan_cspace("goal", context="test") == "cspace-plan"
    assert port.execution_status("command") == {"complete": True}
    assert port.command_status("command") == "completed"
    assert port.execute("command") == "executed"
    assert port.hold("test") == "held"
    assert port.clear_plan_and_hold() == "cleared"

    previous = port.push_timing_scope("scope")
    assert previous is None
    port.restore_timing_scope(previous)
    port.clear_timing_scope("scope")

    # Call every scene ownership callback.
    assert port.assert_attached_owner("object") == "attached"
    assert port.get_source_support_entity("object") == "support"
    assert port.get_attached_entity() == "object"
    assert port.has_native_obstacle("/World/obstacle") is True
    assert port.attached_object_slip("object") == {"slip": 0.0}

    assert "refresh" in events
    assert any(event[0] == "push_timing_scope" for event in events if isinstance(event, tuple))


def test_skill_runtime_timing_callbacks_use_template_owner_not_setup_copy():
    template_source = (ROOT / "workflows/simbox/core/controllers/curobo/controller.py").read_text(
        encoding="utf-8"
    )
    setup_source = (ROOT / "workflows/simbox/core/controllers/curobo/scene_setup.py").read_text(
        encoding="utf-8"
    )
    assert "push_timing_scope=self.push_timing_scope" in template_source
    assert "restore_timing_scope=self.restore_timing_scope" in template_source
    assert "clear_timing_scope=self.clear_timing_scope" in template_source
    assert "_setup.push_timing_scope" not in template_source
    assert "_setup.restore_timing_scope" not in template_source
    assert "_setup.clear_timing_scope" not in template_source
    assert "def push_timing_scope" not in setup_source
    assert "def restore_timing_scope" not in setup_source
    assert "def clear_timing_scope" not in setup_source
