from __future__ import annotations

from typing import Any


class SimBoxEvalEnv:
    """Online evaluation wrapper for a single SimBox task.

    This adapter is intentionally thin: it owns reset/observe/step/metrics and
    leaves policy-specific action mapping in config.
    """

    def __init__(
        self,
        task_config: str,
        workflow_type: str,
        task_index: int,
        max_steps: int,
        simulator: dict[str, Any],
        env_args: dict[str, Any],
    ):
        self.task_config = task_config
        self.workflow_type = workflow_type
        self.task_index = task_index
        self.max_steps = max_steps
        self.simulator = simulator
        self.env_args = env_args
        self.simulation_app = None
        self.world = None
        self.workflow = None
        self.step_count = 0
        self.initial_obs: dict[str, Any] | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        print("[eval] simbox.ensure_workflow start", flush=True)
        self._ensure_workflow(seed)
        print("[eval] simbox.init_task start", flush=True)
        self.workflow.init_task(self.task_index, self.env_args.get("need_preload", False))
        if self.env_args.get("randomize", True):
            print("[eval] simbox.randomization start", flush=True)
            self.workflow.randomization()
            print("[eval] simbox.randomization done", flush=True)
        self.step_count = 0
        print("[eval] simbox.observe initial start", flush=True)
        self.initial_obs = self.observe()
        print("[eval] simbox.observe initial done", flush=True)
        return self.initial_obs

    def observe(self) -> dict[str, Any]:
        task = self.workflow.task
        obs = task.get_observations()
        return {
            "raw": obs,
            "prompt": getattr(task, "language_instruction", ""),
            "detailed_prompt": getattr(task, "detailed_language_instruction", ""),
            "step": self.step_count,
        }

    def step(self, action: Any) -> dict[str, Any]:
        print("[eval] simbox.apply_action start", flush=True)
        action_dict = self._to_simbox_action(action)
        self.workflow.task.apply_action(action_dict)
        self.workflow.world.step(render=bool(self.env_args.get("render", False)))
        self.step_count += 1
        return self.observe()

    def is_done(self) -> bool:
        return self.step_count >= self.max_steps or bool(self._predicate_metrics().get("success", False))

    def get_metrics(self) -> dict[str, Any]:
        predicate = self._predicate_metrics()
        success = bool(predicate.get("success", False))
        return {
            "success": success,
            "failure_reason": None if success else predicate.get("failure_reason", "predicate_not_satisfied"),
            "steps": self.step_count,
            "predicate": predicate,
        }

    def close(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None

    def _ensure_workflow(self, seed: int) -> None:
        if self.workflow is not None:
            return

        from fractions import Fraction

        from isaacsim import SimulationApp
        from nimbus.utils.flags import set_random_seed
        from nimbus.utils.utils import init_env

        init_env()
        set_random_seed(seed)

        app_config = {
            "headless": self.simulator.get("headless", True),
            "anti_aliasing": self.simulator.get("anti_aliasing", 0),
            "renderer": self.simulator.get("renderer", "RayTracedLighting"),
        }
        for key in ("multi_gpu", "active_gpu", "physics_gpu", "max_gpu_count"):
            if key in self.simulator:
                app_config[key] = self.simulator[key]

        self.simulation_app = SimulationApp(app_config)

        from omni.isaac.core import World
        from workflows import import_extensions
        from workflows.base import create_workflow

        physics_dt = self.simulator.get("physics_dt", "1/30")
        rendering_dt = self.simulator.get("rendering_dt", "1/30")
        physics_dt = float(Fraction(physics_dt)) if isinstance(physics_dt, str) else float(physics_dt)
        rendering_dt = float(Fraction(rendering_dt)) if isinstance(rendering_dt, str) else float(rendering_dt)

        self.world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=self.simulator.get("stage_units_in_meters", 1.0),
        )
        import_extensions(self.workflow_type)
        self.workflow = create_workflow(
            self.workflow_type,
            self.world,
            self.task_config,
            scene_info=self.env_args.get("scene_info", "dining_room_scene_info"),
            random_seed=seed,
        )

    def _to_simbox_action(self, action: Any) -> dict[str, dict[str, Any]]:
        if isinstance(action, dict) and "action_dict" in action:
            return action["action_dict"]

        action_cfg = dict(self.env_args.get("action", {}))
        action_mode = action_cfg.get("mode", "joint_positions")
        if action_mode == "franka_eef_delta":
            return self._franka_eef_delta_to_action_dict(action, action_cfg)

        robot_name = action_cfg.get("robot_name")
        joint_indices = action_cfg.get("joint_indices")
        if not robot_name or joint_indices is None:
            raise ValueError("SimBox array actions require env_args.action.robot_name and joint_indices.")

        import numpy as np

        joint_positions = np.asarray(_extract_action_array(action), dtype=float)
        return {
            robot_name: {
                "joint_positions": joint_positions,
                "joint_indices": np.asarray(joint_indices, dtype=int),
            }
        }

    def _franka_eef_delta_to_action_dict(self, action: Any, action_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        robot_name = action_cfg.get("robot_name", "franka")
        arm_name = action_cfg.get("arm", "left")
        controller = self.workflow.controllers[robot_name][arm_name]

        action_array = np.asarray(_extract_action_array(action), dtype=float).reshape(-1)
        if action_array.size < 7:
            raise ValueError(f"franka_eef_delta action expects at least 7 values, got {action_array.size}.")

        delta_pos = action_array[:3]
        delta_rot = action_array[3:6]
        gripper_close = float(action_array[6])

        max_position_delta = action_cfg.get("max_position_delta")
        if max_position_delta is not None:
            delta_pos = np.clip(delta_pos, -float(max_position_delta), float(max_position_delta))
        max_rotation_delta = action_cfg.get("max_rotation_delta")
        if max_rotation_delta is not None:
            delta_rot = np.clip(delta_rot, -float(max_rotation_delta), float(max_rotation_delta))

        current_pos, current_quat = controller.get_ee_pose()
        current_rot = R.from_quat(current_quat, scalar_first=True)

        position_frame = action_cfg.get("position_frame", "base")
        if position_frame == "eef":
            delta_pos = current_rot.apply(delta_pos)
        elif position_frame != "base":
            raise ValueError(f"Unsupported franka_eef_delta position_frame: {position_frame}")
        target_pos = np.asarray(current_pos, dtype=float) + delta_pos

        delta_rotation = R.from_euler(
            "xyz",
            delta_rot,
            degrees=bool(action_cfg.get("rotation_degrees", False)),
        )
        rotation_frame = action_cfg.get("rotation_frame", "base")
        if rotation_frame == "eef":
            target_rot = current_rot * delta_rotation
        elif rotation_frame == "base":
            target_rot = delta_rotation * current_rot
        else:
            raise ValueError(f"Unsupported franka_eef_delta rotation_frame: {rotation_frame}")
        target_quat = target_rot.as_quat(scalar_first=True)

        gripper_threshold = float(action_cfg.get("gripper_close_threshold", 0.5))
        gripper_fn = "close_gripper" if gripper_close >= gripper_threshold else "open_gripper"
        controller_action = controller.forward(
            (target_pos, target_quat, gripper_fn, {}),
            eps=float(action_cfg.get("controller_eps", 1e-4)),
        )
        controller_action = _select_controller_plan_action(
            controller,
            controller_action,
            action_cfg.get("controller_plan_step"),
        )
        self._maybe_log_franka_eef_action(action_cfg, action_array, controller_action)
        return {
            robot_name: {
                "joint_positions": controller_action["joint_positions"],
                "joint_indices": controller_action["joint_indices"],
                "raw_action": [controller_action],
            }
        }

    def _maybe_log_franka_eef_action(
        self,
        action_cfg: dict[str, Any],
        action_array: Any,
        controller_action: dict[str, Any],
    ) -> None:
        log_every = int(action_cfg.get("debug_action_every", 0) or 0)
        if log_every <= 0 or self.step_count % log_every != 0:
            return

        import numpy as np

        arm_action = np.asarray(controller_action.get("arm_action", []), dtype=float)
        gripper_action = np.asarray(controller_action.get("gripper_action", []), dtype=float)
        print(
            "[eval] franka_eef_delta "
            f"step={self.step_count} "
            f"delta={np.array2string(np.asarray(action_array[:7], dtype=float), precision=4, suppress_small=True)} "
            f"arm_target={np.array2string(arm_action, precision=4, suppress_small=True)} "
            f"gripper_target={np.array2string(gripper_action, precision=4, suppress_small=True)}",
            flush=True,
        )

    def _predicate_metrics(self) -> dict[str, Any]:
        predicate = dict(self.env_args.get("success_predicate", {"type": "none"}))
        if predicate.get("type") == "none":
            return {"success": False, "failure_reason": "no_success_predicate_configured"}
        if predicate.get("type") != "object_lifted":
            raise ValueError(f"Unsupported success predicate: {predicate.get('type')}")

        object_name = predicate["object"]
        min_delta_z = float(predicate.get("min_delta_z", 0.02))
        initial = self.initial_obs["raw"]["objects"][object_name]["translation"][2]
        current = self.observe()["raw"]["objects"][object_name]["translation"][2]
        delta = float(current - initial)
        return {
            "success": delta >= min_delta_z,
            "failure_reason": None if delta >= min_delta_z else "object_not_lifted",
            "object": object_name,
            "delta_z": delta,
            "min_delta_z": min_delta_z,
        }


def _extract_action_array(action: Any) -> Any:
    if isinstance(action, dict):
        return action.get("actions", action.get("action"))
    return action


def _select_controller_plan_action(controller: Any, controller_action: dict[str, Any], plan_step: Any) -> dict[str, Any]:
    if plan_step in (None, False, "default"):
        return controller_action

    cmd_plan = getattr(controller, "cmd_plan", None)
    if not cmd_plan:
        return controller_action

    plan_len = len(cmd_plan)
    if plan_len <= 0:
        return controller_action

    if plan_step == "next":
        index = min(int(getattr(controller, "cmd_idx", 0)), plan_len - 1)
    elif plan_step == "goal":
        index = plan_len - 1
    else:
        index = int(plan_step)
        if index < 0:
            index = plan_len + index
        index = max(0, min(index, plan_len - 1))

    cmd_state = cmd_plan[index]
    arm_action = cmd_state.position.cpu().numpy()
    gripper_action = controller_action.get("gripper_action")
    if gripper_action is None:
        return controller_action

    import numpy as np

    selected = dict(controller_action)
    selected["arm_action"] = arm_action
    selected["joint_positions"] = np.concatenate([arm_action, gripper_action])

    controller.cmd_idx = index + 1
    if controller.cmd_idx >= plan_len:
        controller.cmd_idx = 0
        controller.cmd_plan = None
    return selected
