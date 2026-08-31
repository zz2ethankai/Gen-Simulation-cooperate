from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from core.utils.transformation_utils import get_fk_solution, pose_to_6d

from .lmdb_logger import LmdbLogger


class _LoggingBaseBridge(Protocol):
    """Narrow local-base logging contract used by :func:`log_dual_obs`."""

    def get_logging_action_snapshot(self) -> Mapping[str, Any]:
        """Return the most recent body-twist command snapshot."""


RuntimePorts = Mapping[str, Mapping[str, Any]]


def _robot_has_keys(robot_infos, *keys):
    return all(key in robot_infos for key in keys)


def _gripper_openness(runtime_ports: RuntimePorts, robot_name: str, lr_name: str = "left") -> float:
    """Read gripper state through the explicit runtime port only.

    A missing arm is a valid configuration for single-arm/passive robots, and
    is represented as the neutral open value.  Active arms must expose the
    typed execution state; no controller
    façade or private state is inspected here.
    """

    robot_ports = runtime_ports.get(robot_name, {})
    runtime = robot_ports.get(lr_name)
    if runtime is None:
        return 1.0
    status = runtime.execution.execution_status()
    return 1.0 if float(status.gripper_state) > 0.0 else 0.0


def _master_key(action_name: str, field: str) -> str:
    prefix = f"{action_name}_" if action_name else ""
    return f"master_actions.{prefix}{field}"


# pylint: disable=line-too-long,unused-argument
def log_dual_obs(
    logger: LmdbLogger,
    obs,
    action_dict,
    runtime_ports: RuntimePorts,
    base_bridges: Mapping[str, _LoggingBaseBridge] | None = None,
    step_idx: int = 0,
):
    """Record one observation using explicit runtime/state inputs.

    ``runtime_ports`` is a robot-to-arm mapping of typed runtimes
    instances.  The logger intentionally receives neither the controller
    façade nor a private gripper field; command actions remain in
    ``action_dict`` and measured robot state remains in ``obs``.
    """

    base_bridges = base_bridges or {}

    # Add robots' proprio
    for robot_name, robot_infos in obs["robots"].items():
        for key in robot_infos.keys():
            logger.add_proprio_data(robot_name, key, robot_infos[key])

        # Add objects' data (if exists)
        if "objects" in obs:
            for object_name in obs["objects"].keys():
                for attr_name, attr_value in obs["objects"][object_name].items():
                    logger.add_object_data(robot_name, f"{object_name}/{attr_name}", attr_value)

        base_bridge = base_bridges.get(robot_name)
        if base_bridge is not None:
            base_action = base_bridge.get_logging_action_snapshot()
            logger.add_action_data(robot_name, "base_actions.vx_body", base_action["vx_body"])
            logger.add_action_data(robot_name, "base_actions.vy_body", base_action["vy_body"])
            logger.add_action_data(robot_name, "base_actions.wz_body", base_action["wz_body"])
            logger.add_action_data(robot_name, "base_actions.requested_steering", base_action["requested_steering"])
            logger.add_action_data(
                robot_name,
                "base_actions.requested_wheel_velocities",
                base_action["requested_wheel_velocities"],
            )
            logger.add_action_data(
                robot_name,
                "base_actions.applied_wheel_velocities",
                base_action.get("applied_wheel_velocities", base_action["requested_wheel_velocities"]),
            )

        # Canonical robot profiles describe observation/action bindings via a
        # data adapter. Keep this profile-driven path on the typed runtime
        # boundary so logging never needs a controller private field.
        data_adapter = logger.robot_data_adapters.get(robot_name)
        if data_adapter is not None:
            raw_actions = {
                action["lr_name"]: action["arm_action"]
                for action in (action_dict.get(robot_name) or {}).get("raw_action", [])
                if action.get("lr_name") in runtime_ports.get(robot_name, {})
            }
            for arm_id, arm_adapter in data_adapter["arms"].items():
                action_name = str(arm_adapter.get("action_name", ""))
                joint_position = raw_actions.get(
                    arm_id, robot_infos[arm_adapter["joint_position_key"]]
                )
                logger.add_action_data(
                    robot_name,
                    _master_key(action_name, "joint.position"),
                    joint_position,
                )
                logger.add_action_data(
                    robot_name,
                    _master_key(action_name, "gripper.position"),
                    robot_infos[arm_adapter["gripper_position_key"]],
                )
                logger.add_action_data(
                    robot_name,
                    _master_key(action_name, "gripper.openness"),
                    _gripper_openness(runtime_ports, robot_name, arm_id),
                )
                gripper_pose_key = arm_adapter.get("gripper_pose_key")
                if gripper_pose_key:
                    logger.add_action_data(
                        robot_name,
                        _master_key(action_name, "gripper.pose"),
                        robot_infos[gripper_pose_key],
                    )
            continue

        # Add robots' action data (very very important)
        if _robot_has_keys(
            robot_infos,
            "states.left_joint.position",
            "states.right_joint.position",
            "states.left_gripper.position",
            "states.right_gripper.position",
        ):
            left_joint_position = obs["robots"][robot_name]["states.left_joint.position"]
            right_joint_position = obs["robots"][robot_name]["states.right_joint.position"]
            left_gripper_position = obs["robots"][robot_name]["states.left_gripper.position"]
            right_gripper_position = obs["robots"][robot_name]["states.right_gripper.position"]
            left_gripper_openness = _gripper_openness(runtime_ports, robot_name, "left")
            right_gripper_openness = _gripper_openness(runtime_ports, robot_name, "right")

            # Use raw action to udpate if one arm is not static
            robot_action = action_dict.get(robot_name, None)
            if robot_action is not None:
                raw_action = robot_action["raw_action"]
                for action in raw_action:
                    lr_name = action["lr_name"]
                    if lr_name == "left":
                        arm_action = action["arm_action"]
                        left_joint_position = arm_action
                    elif lr_name == "right":
                        arm_action = action["arm_action"]
                        right_joint_position = arm_action
                    else:
                        pass

            logger.add_action_data(robot_name, "master_actions.left_joint.position", left_joint_position)
            logger.add_action_data(robot_name, "master_actions.right_joint.position", right_joint_position)
            logger.add_action_data(robot_name, "master_actions.left_gripper.position", left_gripper_position)
            logger.add_action_data(robot_name, "master_actions.right_gripper.position", right_gripper_position)
            logger.add_action_data(robot_name, "master_actions.left_gripper.openness", left_gripper_openness)
            logger.add_action_data(robot_name, "master_actions.right_gripper.openness", right_gripper_openness)
        elif _robot_has_keys(
            robot_infos,
            "states.joint.position",
            "states.gripper.position",
            "states.gripper.pose",
        ):
            joint_position = obs["robots"][robot_name]["states.joint.position"]
            gripper_pose = obs["robots"][robot_name]["states.gripper.pose"]
            gripper_openness = _gripper_openness(runtime_ports, robot_name, "left")
            gripper_position = obs["robots"][robot_name]["states.gripper.position"]

            # Use raw action to udpate if one arm is not static
            robot_action = action_dict.get(robot_name, None)
            if robot_action is not None:
                raw_action = robot_action["raw_action"]
                for action in raw_action:
                    lr_name = action["lr_name"]
                    if lr_name == "left":
                        arm_action = action["arm_action"]
                        joint_position = arm_action
                        gripper_pose = pose_to_6d(get_fk_solution(joint_position[:7]))
                    else:
                        pass

            logger.add_action_data(robot_name, "master_actions.joint.position", joint_position)
            logger.add_action_data(robot_name, "master_actions.gripper.position", gripper_position)
            logger.add_action_data(robot_name, "master_actions.gripper.openness", gripper_openness)
            logger.add_action_data(robot_name, "master_actions.gripper.pose", gripper_pose)
        else:
            raise NotImplementedError

    # Count time steps
    logger.count_timestep()
