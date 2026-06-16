from __future__ import annotations

from typing import Any


class ZeroActionPolicy:
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def reset(self, task_meta: dict[str, Any]) -> None:
        return None

    def infer(self, observation: dict[str, Any]) -> dict[str, list[float]]:
        return {"actions": [0.0] * self.action_dim}

    def close(self) -> None:
        return None


class ConstantActionPolicy:
    def __init__(self, action: list[float]):
        self.action = action

    def reset(self, task_meta: dict[str, Any]) -> None:
        return None

    def infer(self, observation: dict[str, Any]) -> dict[str, list[float]]:
        return {"actions": list(self.action)}

    def close(self) -> None:
        return None

