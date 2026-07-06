from core.utils.transformation_utils import get_fk_solution, pose_to_6d

from .lmdb_logger import LmdbLogger


def _robot_has_keys(robot_infos, *keys):
    return all(key in robot_infos for key in keys)


def _get_controller(controllers, robot_name, lr_name="left"):
    robot_controllers = controllers.get(robot_name, {})
    controller = robot_controllers.get(lr_name)
    if controller is not None:
        return controller
    if robot_controllers:
        return next(iter(robot_controllers.values()))
    return None


def _gripper_openness(controllers, robot_name, lr_name="left"):
    controller = _get_controller(controllers, robot_name, lr_name)
    if controller is None:
        return 1.0
    return 1.0 if getattr(controller, "_gripper_state", 1.0) > 0.0 else 0.0


# pylint: disable=line-too-long,unused-argument
def log_dual_obs(logger: LmdbLogger, obs, action_dict, controllers, base_bridges=None, step_idx=0):
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
        if base_bridge is not None and hasattr(base_bridge, "get_logging_action_snapshot"):
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
            left_gripper_openness = _gripper_openness(controllers, robot_name, "left")
            right_gripper_openness = _gripper_openness(controllers, robot_name, "right")

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
            gripper_openness = _gripper_openness(controllers, robot_name, "left")
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
