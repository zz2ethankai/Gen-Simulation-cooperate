from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_type: str
    max_steps: int
    max_episode_seconds: float | None = None
    task_config: str | None = None
    workflow_type: str = "SimBoxDualWorkFlow"
    task_index: int = 0
    simulator: dict[str, Any] = field(default_factory=dict)
    env_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    policy_type: str
    action_dim: int | None = None
    endpoint: str | None = None
    host: str | None = None
    port: int | None = None
    open_loop_horizon: int = 1
    policy_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSpec:
    name: str
    task: TaskSpec
    policy: PolicySpec
    seeds: list[int]
    output_dir: Path
    run_args: dict[str, Any] = field(default_factory=dict)


def eval_spec_from_dict(data: dict[str, Any]) -> EvalSpec:
    task_data = dict(data["task"])
    policy_data = dict(data["policy"])
    output_dir = Path(data.get("output_dir", "outputs/eval"))
    max_episode_seconds = task_data.get("max_episode_seconds")

    task = TaskSpec(
        name=task_data["name"],
        env_type=task_data.get("env_type", "simbox"),
        max_steps=_resolve_max_steps(task_data),
        max_episode_seconds=float(max_episode_seconds) if max_episode_seconds is not None else None,
        task_config=task_data.get("task_config"),
        workflow_type=task_data.get("workflow_type", "SimBoxDualWorkFlow"),
        task_index=int(task_data.get("task_index", 0)),
        simulator=dict(task_data.get("simulator", {})),
        env_args=dict(task_data.get("env_args", {})),
    )

    policy = PolicySpec(
        name=policy_data["name"],
        policy_type=policy_data.get("policy_type", "zero_action"),
        action_dim=policy_data.get("action_dim"),
        endpoint=policy_data.get("endpoint"),
        host=policy_data.get("host"),
        port=policy_data.get("port"),
        open_loop_horizon=int(policy_data.get("open_loop_horizon", 1)),
        policy_args=dict(policy_data.get("policy_args", {})),
    )

    return EvalSpec(
        name=data["name"],
        task=task,
        policy=policy,
        seeds=[int(seed) for seed in data.get("seeds", [0])],
        output_dir=output_dir,
        run_args=dict(data.get("run_args", {})),
    )


def _resolve_max_steps(task_data: dict[str, Any]) -> int:
    if "max_episode_seconds" not in task_data:
        return int(task_data.get("max_steps", 500))

    simulator = dict(task_data.get("simulator", {}))
    rendering_dt = _parse_seconds(simulator.get("rendering_dt", "1/30"))
    return max(1, math.ceil(float(task_data["max_episode_seconds"]) / rendering_dt))


def _parse_seconds(value: Any) -> float:
    if isinstance(value, str):
        return float(Fraction(value))
    return float(value)
