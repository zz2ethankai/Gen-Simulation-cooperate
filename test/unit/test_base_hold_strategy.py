"""Unit/fake coverage for the explicit mobile-base hold boundary."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.runtime import ArmSpec, BaseHoldStrategy, RobotRuntime  # noqa: E402


class _FakeBaseHoldPort:
    def __init__(self):
        self.dof_names = (
            "robot0_joint1", "robot0_joint2", "panda_finger_joint1",
            "base_x", "base_y", "base_yaw",
        )
        self.positions = np.asarray([0.1, -0.2, 0.04, 1.25, -0.5, 0.15], dtype=float)
        self.kps = np.asarray([10.0] * len(self.dof_names), dtype=float)
        self.kds = np.asarray([2.0] * len(self.dof_names), dtype=float)
        self.max_efforts = np.asarray([30.0] * len(self.dof_names), dtype=float)
        self.position_targets = []
        self.velocity_targets = []
        self.drive_writes = []

    def resolve_joint_indices(self, joint_names):
        return [self.dof_names.index(name) for name in joint_names]

    def read_joint_positions(self, indices):
        return self.positions[np.asarray(indices, dtype=np.int64)]

    def get_drive_state(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return self.kps[indices], self.kds[indices], self.max_efforts[indices]

    def set_drive_state(self, indices, kps, kds, max_efforts):
        indices = np.asarray(indices, dtype=np.int64)
        self.kps[indices] = np.asarray(kps, dtype=float)
        self.kds[indices] = np.asarray(kds, dtype=float)
        self.max_efforts[indices] = np.asarray(max_efforts, dtype=float)
        self.drive_writes.append(
            (indices.copy(), self.kps[indices].copy(), self.kds[indices].copy(), self.max_efforts[indices].copy())
        )

    def set_position_targets(self, indices, positions):
        self.position_targets.append(
            (np.asarray(indices, dtype=np.int64).copy(), np.asarray(positions, dtype=float).copy())
        )

    def set_velocity_targets(self, indices, velocities):
        self.velocity_targets.append(
            (np.asarray(indices, dtype=np.int64).copy(), np.asarray(velocities, dtype=float).copy())
        )


def _strategy(port=None):
    port = port or _FakeBaseHoldPort()
    return port, BaseHoldStrategy(
        {
            "enabled": True,
            "joint_names": ["base_x", "base_y", "base_yaw"],
            "stiffness": 100.0,
            "damping": 20.0,
            "max_effort": 50.0,
        },
        port=port,
    )


def test_hold_captures_and_reapplies_position_and_velocity_targets():
    port, strategy = _strategy()

    assert strategy.enable() is True
    assert strategy.active is True
    assert strategy.indices == (3, 4, 5)
    assert strategy.target_positions == (1.25, -0.5, 0.15)
    np.testing.assert_allclose(port.kps[[3, 4, 5]], [100.0, 100.0, 100.0])
    np.testing.assert_allclose(port.position_targets[-1][1], [1.25, -0.5, 0.15])
    np.testing.assert_allclose(port.velocity_targets[-1][1], [0.0, 0.0, 0.0])

    port.positions[[3, 4, 5]] += [0.2, -0.1, 0.03]
    assert strategy.reapply() is True
    np.testing.assert_allclose(port.position_targets[-1][1], [1.25, -0.5, 0.15])
    np.testing.assert_allclose(port.velocity_targets[-1][1], [0.0, 0.0, 0.0])


def test_suspend_restores_navigation_drive_and_resume_recaptures_pose():
    port, strategy = _strategy()
    strategy.enable()
    assert strategy.suspend() is True
    assert strategy.active is False
    np.testing.assert_allclose(port.kps[[3, 4, 5]], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(port.kds[[3, 4, 5]], [2.0, 2.0, 2.0])

    port.positions[[3, 4, 5]] = [2.0, 3.0, -0.4]
    assert strategy.resume() is True
    assert strategy.active is True
    assert strategy.target_positions == (2.0, 3.0, -0.4)
    np.testing.assert_allclose(port.position_targets[-1][1], [2.0, 3.0, -0.4])


def test_arm_contract_stays_disjoint_from_base_hold_indices():
    port, strategy = _strategy()
    strategy.enable()
    runtime = RobotRuntime(
        port.dof_names,
        {
            "left": ArmSpec(
                "left",
                ["robot0_joint1", "robot0_joint2"],
                gripper_names=["panda_finger_joint1"],
            )
        },
    )
    arm_action_indices = set(runtime.indices("left", include_gripper=True))
    assert arm_action_indices.isdisjoint(strategy.indices)


def test_disabled_hold_is_a_noop_and_invalid_enabled_config_is_rejected():
    port = _FakeBaseHoldPort()
    disabled = BaseHoldStrategy({}, port=port)
    assert disabled.enable() is False
    assert port.drive_writes == []
    with pytest.raises(ValueError, match="requires joint_names"):
        BaseHoldStrategy({"enabled": True}, port=port)
