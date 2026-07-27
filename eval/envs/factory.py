from __future__ import annotations

from eval.envs.mock_env import MockEvalEnv
from eval.envs.simbox_env import SimBoxEvalEnv
from eval.specs import TaskSpec


def create_env(spec: TaskSpec):
    if spec.env_type == "mock":
        return MockEvalEnv(max_steps=spec.max_steps, target=float(spec.env_args.get("target", 1.0)))
    if spec.env_type == "simbox":
        if not spec.task_config:
            raise ValueError("SimBox eval requires task.task_config.")
        return SimBoxEvalEnv(
            task_config=spec.task_config,
            workflow_type=spec.workflow_type,
            task_index=spec.task_index,
            max_steps=spec.max_steps,
            simulator=spec.simulator,
            env_args=spec.env_args,
        )
    raise ValueError(f"Unsupported env_type: {spec.env_type}")
