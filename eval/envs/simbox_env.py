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
        self.step_count = 0
        print("[eval] simbox.observe initial start", flush=True)
        self.initial_obs = self.observe()
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
        robot_name = action_cfg.get("robot_name")
        joint_indices = action_cfg.get("joint_indices")
        if not robot_name or joint_indices is None:
            raise ValueError("SimBox array actions require env_args.action.robot_name and joint_indices.")

        import numpy as np

        if isinstance(action, dict):
            action = action.get("actions", action.get("action"))
        joint_positions = np.asarray(action, dtype=float)
        return {
            robot_name: {
                "joint_positions": joint_positions,
                "joint_indices": np.asarray(joint_indices, dtype=int),
            }
        }

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
