from __future__ import annotations

from typing import Any, Protocol


class EvalEnv(Protocol):
    def reset(self, seed: int) -> dict[str, Any]:
        ...

    def observe(self) -> dict[str, Any]:
        ...

    def step(self, action: Any) -> dict[str, Any]:
        ...

    def is_done(self) -> bool:
        ...

    def get_metrics(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
