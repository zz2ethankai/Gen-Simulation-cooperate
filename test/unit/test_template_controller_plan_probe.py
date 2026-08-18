"""Regression tests for side-effect-free controller planning probes."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pytest
import torch


_CONTROLLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "simbox"
    / "core"
    / "controllers"
    / "template_controller.py"
)


def _load_plan_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_forward_from_joint_positions"
    )
    namespace = {
        "np": np,
        "Optional": Optional,
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["test_forward_from_joint_positions"]


class _NamedJointState:
    def __init__(self, position, velocity=None, acceleration=None, jerk=None, joint_names=None):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        self.joint_names = joint_names

    def unsqueeze(self, dimension):
        return _NamedJointState(
            self.position.unsqueeze(dimension),
            velocity=None if self.velocity is None else self.velocity.unsqueeze(dimension),
            acceleration=None if self.acceleration is None else self.acceleration.unsqueeze(dimension),
            jerk=None if self.jerk is None else self.jerk.unsqueeze(dimension),
            joint_names=list(self.joint_names) if self.joint_names is not None else None,
        )

    @classmethod
    def from_position(cls, position, joint_names):
        return cls(position, joint_names=list(joint_names))

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]

        def _reorder(value):
            return None if value is None else value[..., indices]

        return _NamedJointState(
            _reorder(self.position),
            velocity=_reorder(self.velocity),
            acceleration=_reorder(self.acceleration),
            jerk=_reorder(self.jerk),
            joint_names=list(joint_names),
        )

    def __getitem__(self, index):
        return _NamedJointState(
            self.position[index],
            velocity=None if self.velocity is None else self.velocity[index],
            acceleration=None if self.acceleration is None else self.acceleration[index],
            jerk=None if self.jerk is None else self.jerk[index],
            joint_names=list(self.joint_names),
        )


class _PathPose:
    def __init__(self, position=None, quaternion=None, batch=None, batch_size=None):
        self.position = position
        self.quaternion = quaternion
        self.batch = batch if batch is not None else batch_size


def _load_forward_from_path_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "test_single_forward_from_path")
    helper_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "_plan_pose_from_state")
    namespace = {
        "np": np,
        "torch": torch,
        "JointState": _NamedJointState,
        "Pose": _PathPose,
        "Optional": Optional,
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[helper_node, method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["test_single_forward_from_path"], namespace["_plan_pose_from_state"]


def _load_batch_forward_from_paths_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "test_batch_forward_from_paths")
    helper_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "_plan_batch_from_state")
    namespace = {
        "torch": torch,
        "JointState": _NamedJointState,
        "Pose": _PathPose,
        "Optional": Optional,
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[helper_node, method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["test_batch_forward_from_paths"], namespace["_plan_batch_from_state"]


def _load_measure_cartesian_path_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(
        node
        for node in controller_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure_cartesian_path"
    )
    method_module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["measure_cartesian_path"]


def _load_attach_objects_probe():
    tree = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    controller_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TemplateController"
    )
    method_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "attach_objects")
    helper_node = next(node for node in controller_node.body if isinstance(node, ast.FunctionDef) and node.name == "_attach_native_planner")

    class _SphereFitType:
        VOXEL = "voxel"

    namespace = {
        "JointState": _NamedJointState,
        "List": List,
        "SphereFitType": _SphereFitType,
        "LOGGER": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    method_module = ast.fix_missing_locations(ast.Module(body=[helper_node, method_node], type_ignores=[]))
    exec(compile(method_module, _CONTROLLER_PATH, "exec"), namespace)
    return namespace["attach_objects"], namespace["_attach_native_planner"]


class _Plan:
    def __init__(self, endpoint):
        if isinstance(endpoint, _Plan):
            self.position = endpoint.position.clone()
            self.joint_names = list(endpoint.joint_names)
        else:
            self.position = torch.as_tensor(endpoint, dtype=torch.float32).reshape(1, -1)
            self.joint_names = ["arm_0", "arm_1"]

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]
        path = _Plan(self)
        path.position = self.position[..., indices]
        path.joint_names = list(joint_names)
        return path

    def __getitem__(self, index):
        return _NamedJointState(self.position[index], joint_names=list(self.joint_names))


class _Result:
    def __init__(self, success, endpoint):
        self.success = torch.tensor([[success]])
        self.path = _Plan(endpoint)


class _Controller:
    test_forward_from_joint_positions = _load_plan_probe()

    def __init__(self, success=True):
        self.arm_indices = np.array([1, 3])
        self.raw_js_names = ["arm_0", "arm_1"]
        self.use_batch = False
        self.num_plan_failed = 4
        self.cmd_plan = "runtime-plan"
        self.robot = SimpleNamespace(
            dof_names=["base", "arm_0", "gripper", "arm_1"],
            get_joints_state=lambda: SimpleNamespace(
                positions=np.array([9.0, 0.1, 0.02, 0.2]),
                velocities=np.zeros(4),
            ),
        )
        self.success = success
        self.planning_start = None

    def plan(self, ee_trans, ee_ori, sim_js, js_names):
        self.planning_start = np.asarray(sim_js.positions).copy()
        return _Result(self.success, [0.7, 0.8])

    @staticmethod
    def _result_path(result):
        return result.path

    @staticmethod
    def _result_success(result):
        return bool(result.success.reshape(-1)[0].item())

    @staticmethod
    def _command_path(path):
        return path.reorder(["arm_0", "arm_1"])


def test_plan_probe_uses_requested_start_and_preserves_runtime_state():
    controller = _Controller(success=True)

    success, endpoint, _ = controller.test_forward_from_joint_positions(
        np.ones(3),
        np.ones(4),
        start_arm_positions=np.array([0.4, 0.5]),
    )

    assert success is True
    np.testing.assert_allclose(controller.planning_start, [9.0, 0.4, 0.02, 0.5])
    np.testing.assert_allclose(endpoint, [0.7, 0.8])
    assert controller.num_plan_failed == 4
    assert controller.cmd_plan == "runtime-plan"


def test_failed_plan_probe_returns_no_endpoint_or_runtime_failure_side_effect():
    controller = _Controller(success=False)

    success, endpoint, _ = controller.test_forward_from_joint_positions(np.ones(3), np.ones(4))

    assert success is False
    assert endpoint is None
    assert controller.num_plan_failed == 4
    assert controller.cmd_plan == "runtime-plan"


def test_plan_probe_uses_single_native_planner_when_batch_candidates_are_enabled():
    controller = _Controller(success=True)
    controller.use_batch = True

    success, endpoint, _ = controller.test_forward_from_joint_positions(np.ones(3), np.ones(4))

    assert success is True
    np.testing.assert_allclose(endpoint, [0.7, 0.8])


def test_terminal_probe_propagates_named_full_trajectory_endpoint():
    class _NativePlanner:
        def __init__(self):
            self.start_state = None

        def plan_pose(self, _goal, start_state, **_kwargs):
            self.start_state = start_state
            return "result"

    class _Controller:
        test_single_forward_from_path, _plan_pose_from_state = _load_forward_from_path_probe()

        def __init__(self):
            self.cmd_js_names = [f"active_{index}" for index in range(7)]
            self.planner = _NativePlanner()
            self.tensor_args = SimpleNamespace(to_device=lambda value: value)
            self._max_plan_attempts = 4
            self._single_graph_attempt = 1
            self.logged = None

        @staticmethod
        def _planner_state(state):
            return state

        @staticmethod
        def _goal_tool_pose(*_args, **_kwargs):
            return "goal"

        @staticmethod
        def _run_timed_curobo_call(_operation, call):
            return call()

        def _log_plan_result(self, *args, **kwargs):
            self.logged = (args, kwargs)

    full_names = [
        *(f"panda_joint{index}" for index in range(1, 8)),
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    path = _NamedJointState(
        torch.arange(9.0).reshape(1, 9),
        joint_names=full_names,
    )
    controller = _Controller()

    assert controller.test_single_forward_from_path(
        np.zeros(3), np.ones(4), path
    ) == "result"
    assert controller.planner.start_state.joint_names == full_names
    assert controller.planner.start_state.position.shape == (1, 9)
    assert torch.equal(
        controller.planner.start_state.position[0], torch.arange(9.0)
    )


def test_batch_terminal_probe_preserves_and_reorders_named_full_endpoints():
    class _NativeBatchPlanner:
        def __init__(self):
            self.start_state = None
            self.batch_size = 20

        def plan_pose(self, _goal, start_state, **_kwargs):
            self.start_state = start_state
            return "batch-result"

    class _Controller:
        test_batch_forward_from_paths, _plan_batch_from_state = _load_batch_forward_from_paths_probe()

        def __init__(self):
            self.planner_names = names
            self.batch_planner = _NativeBatchPlanner()
            self.tensor_args = SimpleNamespace(to_device=lambda value: value)
            self._batch_max_attempts = 4
            self._batch_graph_attempt = 3

        def _planner_joint_names(self):
            return list(self.planner_names)

        def _planner_state(self, state):
            return state.reorder(self.planner_names)

        @staticmethod
        def _goal_tool_pose(*_args, **_kwargs):
            return "goal"

        @staticmethod
        def _run_timed_curobo_call(_operation, call):
            return call()

        def _log_plan_result(self, *args, **kwargs):
            del args, kwargs

    names = [
        *(f"panda_joint{index}" for index in range(1, 8)),
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    reversed_names = list(reversed(names))
    first = _NamedJointState(torch.arange(9.0).reshape(1, 9), joint_names=names)
    second_values = torch.arange(9.0, 18.0).reshape(1, 9)
    second = _NamedJointState(second_values, joint_names=reversed_names)
    controller = _Controller()

    assert controller.test_batch_forward_from_paths(
        np.zeros((2, 3)), np.ones((2, 4)), [first, second]
    ) == "batch-result"
    state = controller.batch_planner.start_state
    assert state.joint_names == names
    assert state.position.shape == (2, 9)
    assert torch.equal(state.position[0], torch.arange(9.0))
    assert torch.equal(state.position[1], torch.arange(9.0, 18.0).flip(0))


def test_batch_terminal_probe_rejects_unnamed_or_mismatched_endpoints():
    class _NativeBatchPlanner:
        batch_size = 20

        def plan_pose(self, *_args, **_kwargs):
            raise AssertionError("invalid endpoint must fail before planning")

    class _Controller:
        test_batch_forward_from_paths, _plan_batch_from_state = _load_batch_forward_from_paths_probe()

        def __init__(self):
            self.planner_names = ["a", "b"]
            self.batch_planner = _NativeBatchPlanner()
            self.tensor_args = SimpleNamespace(to_device=lambda value: value)
            self._batch_max_attempts = 4
            self._batch_graph_attempt = 3

        def _planner_joint_names(self):
            return list(self.planner_names)

        def _planner_state(self, state):
            return state.reorder(self.planner_names)

        @staticmethod
        def _goal_tool_pose(*_args, **_kwargs):
            return "goal"

        @staticmethod
        def _run_timed_curobo_call(_operation, call):
            return call()

    controller = _Controller()
    unnamed = _NamedJointState(torch.zeros(1, 2), joint_names=None)
    with pytest.raises(ValueError, match="explicit joint_names"):
        controller.test_batch_forward_from_paths(
            np.zeros((1, 3)), np.ones((1, 4)), [unnamed]
        )

    named = _NamedJointState(torch.zeros(1, 2), joint_names=["a", "b"])
    wrong = _NamedJointState(torch.zeros(1, 3), joint_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="same named joint contract"):
        controller.test_batch_forward_from_paths(
            np.zeros((2, 3)), np.ones((2, 4)), [named, wrong]
        )


def test_cartesian_measurement_reduces_full_path_by_explicit_active_names():
    class _Controller:
        measure_cartesian_path = _load_measure_cartesian_path_probe()

        def __init__(self):
            self.raw_js_names = ["a", "b"]
            self.fk_inputs = []

        def forward_kinematic(self, joints):
            self.fk_inputs.append(np.asarray(joints).copy())
            return np.array([joints[0], joints[1], 0.0], dtype=float), np.array(
                [1.0, 0.0, 0.0, 0.0]
            )

    path = _NamedJointState(
        torch.tensor([[0.0, 0.0, 0.9], [1.0, 0.2, 0.8]]),
        joint_names=["a", "b", "locked_finger"],
    )
    controller = _Controller()

    ratio, deviation = controller.measure_cartesian_path(
        path, np.zeros(3), np.array([1.0, 0.2, 0.0])
    )

    assert ratio >= 1.0
    assert deviation >= 0.0
    assert len(controller.fk_inputs) == 2
    assert all(input_state.shape == (2,) for input_state in controller.fk_inputs)
    np.testing.assert_allclose(controller.fk_inputs[1], [1.0, 0.2])


def test_attach_objects_passes_native_active_arm_state_to_attachment_manager():
    class _NativeAttachmentManager:
        def __init__(self):
            self.received = None

        def attach(self, joint_state, meshes, **kwargs):
            self.received = (joint_state, meshes, kwargs)

    class _Controller:
        attach_objects, _attach_native_planner = _load_attach_objects_probe()

        def __init__(self):
            self.robot = SimpleNamespace(
                dof_names=[
                    "mobilebase0_joint_mobile_forward",
                    "mobilebase0_joint_mobile_side",
                    "mobilebase0_joint_mobile_yaw",
                    "robot0_joint1",
                    "robot0_joint2",
                    "robot0_joint3",
                    "robot0_joint4",
                    "robot0_joint5",
                    "robot0_joint6",
                    "robot0_joint7",
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ],
                get_joints_state=lambda: SimpleNamespace(
                    positions=torch.arange(12.0), velocities=torch.zeros(12)
                ),
            )
            self.tensor_args = SimpleNamespace(to_device=lambda value: value)
            self.raw_js_names = [f"panda_joint{index}" for index in range(1, 8)]
            self.cmd_js_names = [f"robot0_joint{index}" for index in range(1, 8)]
            self.planner = SimpleNamespace(
                joint_names=list(self.raw_js_names),
                attachment_manager=_NativeAttachmentManager(),
            )
            self.batch_planner = SimpleNamespace(
                joint_names=list(self.raw_js_names),
                attachment_manager=_NativeAttachmentManager(),
                kinematics=SimpleNamespace(
                    config=SimpleNamespace(
                        kinematics_config=SimpleNamespace(
                            get_number_of_spheres=lambda _link_name: 8
                        )
                    )
                ),
            )
            self.world_cfg = SimpleNamespace(get_obstacle=lambda _name: object())
            self.name = "panda_omron"
            self.lr_name = "left"
            self._native_batch_attached_obstacle_names = []

        def _arm_joint_state(self, _sim_js):
            return _NamedJointState(
                torch.arange(3.0, 10.0),
                velocity=torch.zeros(7),
                acceleration=torch.zeros(7),
                jerk=torch.zeros(7),
                joint_names=list(self.raw_js_names),
            )

        def _planner_joint_names(self):
            return list(self.planner.joint_names)

        def _native_planners(self):
            return (self.planner, self.batch_planner)

        @staticmethod
        def _native_attachment_geometry(_paths):
            return [object()], "native-offset"

        @staticmethod
        def _attached_sphere_count(_link_name, _object_count, *, planner=None):
            return 4

    controller = _Controller()
    assert controller.attach_objects(["/World/apple"])
    received_state = controller.planner.attachment_manager.received[0]
    assert received_state.joint_names == controller.raw_js_names
    assert received_state.position.shape == (7,)
    assert torch.equal(received_state.position, torch.arange(3.0, 10.0))
    assert controller._native_batch_attached_obstacle_names == []
