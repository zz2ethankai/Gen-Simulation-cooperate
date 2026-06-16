from __future__ import annotations

from typing import Any


class MockEvalEnv:
    """Small deterministic env used to test the eval loop outside Isaac Sim."""

    def __init__(self, max_steps: int, target: float = 1.0):
        self.max_steps = max_steps
        self.target = target
        self.value = 0.0
        self.step_count = 0

    def reset(self, seed: int) -> dict[str, Any]:
        self.value = 0.0
        self.step_count = 0
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {
            "observation/state": [self.value],
            "prompt": "Move the scalar state to the target.",
            "step": self.step_count,
        }

    def step(self, action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            action = action.get("actions", action.get("action", [0.0]))
        delta = float(action[0]) if isinstance(action, (list, tuple)) else float(action)
        self.value += delta
        self.step_count += 1
        return self.observe()

    def is_done(self) -> bool:
        return self.value >= self.target or self.step_count >= self.max_steps

    def get_metrics(self) -> dict[str, Any]:
        success = self.value >= self.target
        return {
            "success": success,
            "failure_reason": None if success else "target_not_reached",
            "steps": self.step_count,
            "final_value": self.value,
            "target": self.target,
        }

    def close(self) -> None:
        return None

