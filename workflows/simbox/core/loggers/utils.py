from .lmdb_logger import LmdbLogger


def _master_key(action_name: str, field: str) -> str:
    prefix = f"{action_name}_" if action_name else ""
    return f"master_actions.{prefix}{field}"


def log_dual_obs(logger: LmdbLogger, obs, action_dict, controllers, step_idx=0):
    del step_idx
    for robot_name, robot_info in obs["robots"].items():
        for key, value in robot_info.items():
            logger.add_proprio_data(robot_name, key, value)

        for object_name, object_info in obs.get("objects", {}).items():
            for attribute_name, value in object_info.items():
                logger.add_object_data(
                    robot_name, f"{object_name}/{attribute_name}", value
                )

        try:
            data_adapter = logger.robot_data_adapters[robot_name]
        except KeyError as exc:
            raise ValueError(
                f"robot {robot_name!r} has no canonical data_adapter"
            ) from exc
        raw_actions = {
            action["lr_name"]: action["arm_action"]
            for action in (action_dict.get(robot_name) or {}).get("raw_action", [])
            if action.get("lr_name") in controllers[robot_name]
        }
        for arm_id, arm_adapter in data_adapter["arms"].items():
            controller = controllers[robot_name][arm_id]
            action_name = str(arm_adapter.get("action_name", ""))
            joint_position = raw_actions.get(
                arm_id, robot_info[arm_adapter["joint_position_key"]]
            )
            gripper_position = robot_info[arm_adapter["gripper_position_key"]]
            gripper_openness = 1.0 if controller._gripper_state > 0.0 else 0.0

            logger.add_action_data(
                robot_name,
                _master_key(action_name, "joint.position"),
                joint_position,
            )
            logger.add_action_data(
                robot_name,
                _master_key(action_name, "gripper.position"),
                gripper_position,
            )
            logger.add_action_data(
                robot_name,
                _master_key(action_name, "gripper.openness"),
                gripper_openness,
            )
            gripper_pose_key = arm_adapter.get("gripper_pose_key")
            if gripper_pose_key:
                logger.add_action_data(
                    robot_name,
                    _master_key(action_name, "gripper.pose"),
                    robot_info[gripper_pose_key],
                )

    logger.count_timestep()
