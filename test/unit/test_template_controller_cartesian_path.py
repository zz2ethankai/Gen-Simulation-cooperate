"""Focused host tests for batched Cartesian FK validation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.controllers.controller_execution import ControllerExecution  # noqa: E402
from core.controllers.controller_component import ComponentPort  # noqa: E402
from core.controllers.controller_planning_queries import (  # noqa: E402
    ControllerPlanningQueries,
)
from core.planning.domain_types import JointTrajectory  # noqa: E402
import core.controllers.controller_execution as execution_module  # noqa: E402


class _NamedPath:
    def __init__(self, position, joint_names):
        self.position = position
        self.joint_names = list(joint_names)

    def reorder(self, joint_names):
        indices = [self.joint_names.index(name) for name in joint_names]
        return _NamedPath(self.position[..., indices], joint_names)


def test_measure_cartesian_path_uses_one_batched_fk_after_name_reorder():
    class _Controller:
        def __init__(self):
            self.raw_js_names = ["a", "b"]
            self.batch_inputs = []

        def _forward_kinematic_batch(self, joint_positions, **kwargs):
            del kwargs
            self.batch_inputs.append(joint_positions)
            points = np.asarray(joint_positions.detach().cpu())
            return np.concatenate([points, np.zeros((len(points), 1))], axis=1)

    path = _NamedPath(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
        ["b", "a", "locked_finger"],
    )
    controller = _Controller()
    queries = ControllerPlanningQueries(
        ComponentPort(
            {
                "raw_js_names": controller.raw_js_names,
                "tensor_args": SimpleNamespace(
                    to_device=lambda value: value
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, dtype=torch.float32)
                ),
                "_forward_kinematic_batch": controller._forward_kinematic_batch,
            }
        )
    )

    ratio, deviation = queries.measure_cartesian_path(
        path, np.zeros(3), np.array([0.5, 0.5, 0.0])
    )

    assert ratio == pytest.approx(1.0)
    assert deviation == pytest.approx(0.0)
    assert len(controller.batch_inputs) == 1
    assert tuple(controller.batch_inputs[0].shape) == (2, 2)
    torch.testing.assert_close(
        controller.batch_inputs[0], torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    )


def test_continuous_place_validation_uses_one_batched_fk():
    class _Controller:
        name = "stub"
        lr_name = "left"
        ds_ratio = 2

        def __init__(self):
            self.batch_inputs = []

        @staticmethod
        def get_ee_pose():
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def _forward_kinematic_batch(self, joint_positions, **kwargs):
            del kwargs
            self.batch_inputs.append(joint_positions)
            return np.asarray(joint_positions.detach().cpu())[:, :3]

    command = SimpleNamespace(
        target_position=np.array([0.0, 0.0, -0.08]),
        params={
            "max_cartesian_step_m": 0.01,
            "max_path_length_ratio": 1.5,
            "max_path_deviation_m": 0.01,
        },
    )
    plan = JointTrajectory(
        positions=torch.tensor(
            [[0.0, 0.0, -0.002 * index] for index in range(41)]
        ),
        joint_names=("joint_0", "joint_1", "joint_2"),
    )
    controller = _Controller()
    execution = ControllerExecution(
        ComponentPort(
            {
                "name": controller.name,
                "lr_name": controller.lr_name,
                "ds_ratio": controller.ds_ratio,
                "get_ee_pose": controller.get_ee_pose,
                "tensor_args": SimpleNamespace(
                    to_device=lambda value: value
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, dtype=torch.float32)
                ),
                "_forward_kinematic_batch": controller._forward_kinematic_batch,
            }
        )
    )
    validate = execution._validate_continuous_place_plan

    assert validate(command, plan) is True
    assert len(controller.batch_inputs) == 1
    assert tuple(controller.batch_inputs[0].shape) == (41, 3)


def test_real_controller_execution_armbase_pose_reads_nontrivial_usd_transform():
    """The execution component uses Isaac's public transform helpers."""

    pytest.importorskip("isaacsim.core.utils.transformations")
    import omni.usd
    from pxr import Gf, UsdGeom

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.Xform.Define(stage, "/World/Task")
    base = UsdGeom.Xform.Define(stage, "/World/Task/ArmBase")
    base.AddTranslateOp().Set(Gf.Vec3d(0.3, -0.4, 0.5))
    base.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 30.0))

    # Use the same narrow ComponentPort that production wiring supplies.  The
    # execution component receives frame paths, never the whole controller or
    # task façade.
    execution = ControllerExecution(
        ComponentPort(
            {
                "robot_base_path": "/World/Task/ArmBase",
                "robot_ee_path": "/World/Task/ArmBase/EE",
                "task_root_prim_path": "/World/Task",
            }
        )
    )

    translation, orientation = execution.get_armbase_pose()

    np.testing.assert_allclose(translation, [0.3, -0.4, 0.5], atol=1e-6)
    expected_orientation = np.array(
        [np.cos(np.deg2rad(15.0)), 0.0, 0.0, np.sin(np.deg2rad(15.0))]
    )
    np.testing.assert_allclose(orientation, expected_orientation, atol=1e-5)


def test_batch_fk_helper_builds_named_state_and_transfers_once(monkeypatch):
    class _JointState:
        received = None

        @classmethod
        def from_position(cls, position, joint_names):
            cls.received = (position, list(joint_names))
            return SimpleNamespace(position=position, joint_names=list(joint_names))

    class _ToolPoses:
        @staticmethod
        def get_link_pose(_link_name):
            return SimpleNamespace(
                position=torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=torch.float32,
                )
            )

    class _Kinematics:
        tool_frames = ["tool"]

        def __init__(self):
            self.calls = []

        def compute_kinematics(self, state):
            self.calls.append(state)
            return SimpleNamespace(tool_poses=_ToolPoses())

    controller = SimpleNamespace(
        tensor_args=SimpleNamespace(
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
        ),
        kin_model=_Kinematics(),
        _planner_joint_names=lambda: ["joint_a", "joint_b"],
    )
    monkeypatch.setattr(execution_module, "JointState", _JointState)
    helper = ControllerExecution(
        ComponentPort(
            {
                "tensor_args": controller.tensor_args,
                "kin_model": controller.kin_model,
                "_planner_joint_names": controller._planner_joint_names,
            }
        )
    )._forward_kinematic_batch

    result = helper(np.zeros((2, 2), dtype=np.float64))

    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert len(controller.kin_model.calls) == 1
    assert _JointState.received[1] == ["joint_a", "joint_b"]
    assert tuple(_JointState.received[0].shape) == (2, 2)


def test_batch_fk_helper_reorders_source_names_before_native_fk(monkeypatch):
    class _JointState:
        received = None

        @classmethod
        def from_position(cls, position, joint_names):
            cls.received = (position, list(joint_names))
            return SimpleNamespace(position=position, joint_names=list(joint_names))

    class _ToolPoses:
        @staticmethod
        def get_link_pose(_link_name):
            return SimpleNamespace(position=torch.zeros((2, 3)))

    class _Kinematics:
        tool_frames = ["tool"]

        @staticmethod
        def compute_kinematics(_state):
            return SimpleNamespace(tool_poses=_ToolPoses())

    controller = SimpleNamespace(
        tensor_args=SimpleNamespace(
            to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)
        ),
        kin_model=_Kinematics(),
        _planner_joint_names=lambda: ["joint_a", "joint_b"],
    )
    monkeypatch.setattr(execution_module, "JointState", _JointState)
    helper = ControllerExecution(
        ComponentPort(
            {
                "tensor_args": controller.tensor_args,
                "kin_model": controller.kin_model,
                "_planner_joint_names": controller._planner_joint_names,
            }
        )
    )._forward_kinematic_batch

    helper(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        joint_names=["joint_b", "joint_a"],
    )

    torch.testing.assert_close(
        _JointState.received[0], torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    )
    assert _JointState.received[1] == ["joint_a", "joint_b"]
