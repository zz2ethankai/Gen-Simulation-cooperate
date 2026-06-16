from __future__ import annotations

from typing import Any, Protocol


class PolicyClient(Protocol):
    def reset(self, task_meta: dict[str, Any]) -> None:
        ...

    def infer(self, observation: dict[str, Any]) -> Any:
        ...

    def close(self) -> None:
        ...

