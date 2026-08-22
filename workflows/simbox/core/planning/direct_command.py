"""Small simulator-independent helpers for direct Skill commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def dummy_forward_params(command: Any) -> dict[str, Any] | None:
    """Return parameters for a legacy ``dummy_forward`` command.

    The direct command deliberately remains a small compatibility boundary:
    ``(ee_position, ee_orientation, "dummy_forward", params)``.  It is
    execution-only and does not describe a Physics-schema planning phase.
    Returning ``None`` for all other values lets typed commands continue to
    use the normal ``MotionPhaseCommand`` path.
    """

    if not isinstance(command, (tuple, list)) or len(command) < 4:
        return None
    if command[2] != "dummy_forward":
        return None
    params = command[3]
    if not isinstance(params, Mapping):
        raise TypeError("dummy_forward command parameters must be a mapping")
    return dict(params)


__all__ = ["dummy_forward_params"]
