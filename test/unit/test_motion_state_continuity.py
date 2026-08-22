"""Tests for preserving measured motion state across manipulation replans."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.planning.motion_command import MotionPhase, MotionPhaseCommand  # noqa: E402


@pytest.fixture(scope="module")
def _simulation_app():
    """Bootstrap Isaac extensions for the real phase-geometry assertion."""

    pytest.importorskip("isaacsim")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    yield app
    app.close()


_CONTROLLERS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
)
_STATE_PLANNING_PATH = _CONTROLLERS_ROOT / "controller_state_planning.py"
_PHASES_PATH = _CONTROLLERS_ROOT / "controller_phases.py"


def _load_derivative_helper():
    tree = ast.parse(_STATE_PLANNING_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControllerStatePlanning"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_joint_state_derivatives"
    )
    namespace = {"np": np}
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _STATE_PLANNING_PATH, "exec"), namespace)
    return namespace["_joint_state_derivatives"]


def test_joint_state_derivatives_preserve_measured_values():
    helper = _load_derivative_helper()
    state = SimpleNamespace(
        positions=np.zeros(3),
        velocities=np.array([0.1, -0.2, 0.3]),
        accelerations=np.array([0.4, 0.5, -0.6]),
        jerks=np.array([-0.7, 0.8, 0.9]),
    )

    velocity, acceleration, jerk = helper(state)

    np.testing.assert_allclose(velocity, state.velocities)
    np.testing.assert_allclose(acceleration, state.accelerations)
    np.testing.assert_allclose(jerk, state.jerks)


def test_joint_state_derivatives_fall_back_for_missing_or_invalid_fields():
    helper = _load_derivative_helper()
    state = SimpleNamespace(
        positions=np.zeros(3),
        velocities=np.array([0.1, np.nan, 0.3]),
    )

    velocity, acceleration, jerk = helper(state)

    np.testing.assert_array_equal(velocity, np.zeros(3))
    np.testing.assert_array_equal(acceleration, np.zeros(3))
    np.testing.assert_array_equal(jerk, np.zeros(3))


def _load_retarget_helper():
    tree = ast.parse(_PHASES_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ControllerPhases"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "retarget_pick_phase_commands"
    )

    def tf_matrix_from_pose(translation, orientation):
        x, y, z, w = np.asarray(orientation, dtype=float)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        matrix[:3, 3] = np.asarray(translation, dtype=float)
        return matrix

    def pose_from_tf_matrix(matrix):
        return np.asarray(matrix[:3, 3]), np.array([0.0, 0.0, 0.0, 1.0])

    namespace = {
        "np": np,
        "MotionPhase": MotionPhase,
        "MotionPhaseCommand": MotionPhaseCommand,
        "tf_matrix_from_pose": tf_matrix_from_pose,
        "pose_from_tf_matrix": pose_from_tf_matrix,
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _PHASES_PATH, "exec"), namespace)
    return namespace["retarget_pick_phase_commands"]


def test_retarget_translation_uses_object_center_delta_not_rotation_lever_arm():
    helper = _load_retarget_helper()
    old_translation = np.array([-0.176, 0.114, 0.784])
    current_translation = old_translation + np.array([0.007, -0.003, 0.005])
    angle = np.deg2rad(9.0)
    old_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    current_orientation = np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

    controller = SimpleNamespace(
        _pick_plan_references={
            "target": {
                "object_pose": (old_translation.copy(), old_orientation.copy()),
                "world_armbase_tf": np.eye(4),
            }
        },
        _get_pick_object_world_pose=lambda _name: (
            current_translation.copy(),
            current_orientation.copy(),
        ),
        get_pick_armbase_transform=lambda: np.eye(4),
    )
    command = MotionPhaseCommand(
        MotionPhase.TRANSIT_PREGRASP,
        target_position=np.array([0.2, 0.1, 0.3]),
        target_orientation=old_orientation.copy(),
        active_object="target",
        params={
            "preplanned_joint_path": object(),
            "path_length_ratio": 1.2,
            "path_max_deviation_m": 0.01,
        },
    )

    translation_delta, rotation_delta = helper(controller, "target", [command])

    np.testing.assert_allclose(translation_delta, np.linalg.norm(current_translation - old_translation))
    assert abs(translation_delta - 0.00911043358) < 1e-8
    assert abs(rotation_delta - 9.0) < 1e-6
    assert isinstance(command, MotionPhaseCommand)
    assert not np.allclose(command.target_position, [0.2, 0.1, 0.3])
    assert "preplanned_joint_path" not in command.params
    assert "path_length_ratio" not in command.params
    assert "path_max_deviation_m" not in command.params


def test_real_controller_phases_retarget_non_identity_object_pose(_simulation_app):
    """Exercise the composed phase component's real geometry imports."""

    del _simulation_app
    from core.controllers.controller_component import (  # noqa: PLC0415
        MutableExecutionState,
        PhasesPort,
    )
    from core.controllers.controller_phases import ControllerPhases
    from core.controllers.pick_planning import PickPlanningPort
    from core.controllers.skill_runtime import SkillRuntimePort
    from core.planning.collision_scene_manager import PlannerScenePort

    old_translation = np.array([1.7, -0.8, 0.6])
    old_orientation = np.array([0.0, 0.0, 0.0, 1.0])
    current_translation = old_translation + np.array([0.007, -0.003, 0.005])
    angle = np.deg2rad(9.0)
    current_orientation = np.array(
        [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
    )
    execution_state = MutableExecutionState()
    controller = ControllerPhases(
        PhasesPort(
            {
                "execution_state": execution_state,
                "_pick_plan_references": {
                    "target": {
                        "object_pose": (old_translation.copy(), old_orientation.copy()),
                        "world_armbase_tf": np.eye(4),
                    }
                },
            }
        )
    )
    controller._get_pick_object_world_pose = lambda _name: (
        current_translation.copy(),
        current_orientation.copy(),
    )
    controller.get_pick_armbase_transform = lambda: np.eye(4)
    command = MotionPhaseCommand(
        MotionPhase.TRANSIT_PREGRASP,
        target_position=np.array([0.2, 0.1, 0.3]),
        target_orientation=old_orientation.copy(),
        active_object="target",
        params={
            "preplanned_joint_path": object(),
            "path_length_ratio": 1.2,
            "path_max_deviation_m": 0.01,
        },
    )

    runtime = SimpleNamespace(
        scene_revision=0,
        robot_port=SimpleNamespace(interpolation_dt=0.01),
    )
    scene_port = PlannerScenePort(
        name="robot",
        lr_name="right",
        reference_prim_path="/World/robot/base",
        robot_ee_path="/World/robot/ee",
        tensor_args=SimpleNamespace(),
        robot=SimpleNamespace(),
        runtime=runtime,
    )
    planning = PickPlanningPort(
        scene_port=scene_port,
        collision_scene_manager=SimpleNamespace(),
        update_pose_cost_metric=lambda _value: None,
        build_commands=lambda **kwargs: kwargs,
        arm_base_transform=controller.get_pick_armbase_transform,
        frame_debug=lambda: {},
        capture_reference=lambda _name: None,
        retarget_commands=controller.retarget_pick_phase_commands,
        replan_after_safety=lambda _name, _command, _commands: True,
        execution_ee_pose=lambda: (np.zeros(3), old_orientation.copy()),
        phase_complete=lambda _command: True,
    )
    skill_runtime = SkillRuntimePort(
        robot=scene_port.robot,
        runtime=runtime,
        execution_state=execution_state,
        arm_spec=SimpleNamespace(name="right"),
        arm_indices=[0],
        gripper_indices=[1],
        ee_pose=lambda: (np.zeros(3), old_orientation.copy()),
        arm_base_pose=lambda: (np.zeros(3), old_orientation.copy()),
        compute_fk=lambda joints: (np.asarray(joints), old_orientation.copy()),
        initial_ee_pose=lambda: (np.zeros(3), old_orientation.copy()),
    )

    assert planning.scene_port is scene_port
    assert skill_runtime.execution_state is execution_state
    translation_delta, rotation_delta = planning.retarget_commands(
        "target", [command]
    )

    np.testing.assert_allclose(
        translation_delta, np.linalg.norm(current_translation - old_translation)
    )
    assert rotation_delta == pytest.approx(9.0, abs=1e-6)
    assert not np.allclose(command.target_position, [0.2, 0.1, 0.3])
    assert "preplanned_joint_path" not in command.params
    assert "path_length_ratio" not in command.params
    assert "path_max_deviation_m" not in command.params
