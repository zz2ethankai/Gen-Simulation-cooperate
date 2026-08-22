"""Controller registration and arm-level configuration contracts.

The old controller subclasses each duplicated the same runtime joint-index
bookkeeping.  ``ArmSpec`` is deliberately a small, immutable description of
the arm-specific bits; the TemplateController owns all planning and
execution behaviour.  Keeping this module free of Isaac/CuRobo imports also
lets config/registry tests inspect the contract without starting a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from core.planning.domain_types import CommandStatus


CONTROLLER_DICT = {}


@dataclass(frozen=True)
class ArmSpec:
    """Static arm configuration consumed by :class:`TemplateController`.

    ``planner_joints`` are the names used by the CuRobo kinematics model;
    ``control_joints`` are the articulation names in the loaded USD.  The
    left/right maps contain one entry for every supported controller arm and
    are selected from the robot-file name by the template.
    """

    planner_joints: tuple[str, ...] | Mapping[str, tuple[str, ...]]
    control_joints: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    default_ignore_substring: tuple[str, ...] = ()
    gripper_clip_max: float = 0.04
    gripper_scale: float = 1.0
    gripper_invert: bool = False
    gripper_home: tuple[float, ...] = (1.0,)
    collision_cache: Mapping[str, int] | None = None
    grasp_approach_axis: int | None = None
    sort_path_weights: tuple[float, ...] | None = None
    supported_arms: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        planner_values = (
            tuple(self.planner_joints.values())
            if isinstance(self.planner_joints, Mapping)
            else (self.planner_joints,)
        )
        if not planner_values or any(not names for names in planner_values):
            raise ValueError("ArmSpec.planner_joints must not be empty")
        if any(
            any(not isinstance(name, str) or not name for name in names)
            for names in planner_values
        ):
            raise ValueError("ArmSpec planner joint names must be non-empty strings")
        if self.gripper_clip_max <= 0:
            raise ValueError("ArmSpec.gripper_clip_max must be positive")
        if self.gripper_scale <= 0:
            raise ValueError("ArmSpec.gripper_scale must be positive")

    def control_joint_names(self, arm: str) -> tuple[str, ...]:
        """Return articulation names for ``arm`` with a clear config error."""

        try:
            names = tuple(self.control_joints[arm])
        except KeyError as exc:
            raise ValueError(f"arm {arm!r} is not configured in ArmSpec") from exc
        planner_count = len(self.planner_joint_names(arm))
        if len(names) != planner_count:
            raise ValueError(
                f"ArmSpec planner/control joint count mismatch for {arm}: "
                f"planner={planner_count}, control={len(names)}"
            )
        return names

    def planner_joint_names(self, arm: str) -> tuple[str, ...]:
        """Return planner names for ``arm`` (dual-arm models may differ)."""

        if isinstance(self.planner_joints, Mapping):
            try:
                return tuple(self.planner_joints[arm])
            except KeyError as exc:
                raise ValueError(f"arm {arm!r} is not configured in ArmSpec") from exc
        return tuple(self.planner_joints)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ArmSpec":
        """Build a spec from plain config data used by a few downstream tools."""

        data = dict(values)
        planner_raw = data.pop("planner_joints", ())
        if isinstance(planner_raw, Mapping):
            planner = {
                str(arm): tuple(str(item) for item in names)
                for arm, names in planner_raw.items()
            }
        else:
            planner = tuple(str(item) for item in planner_raw)
        control = {
            str(arm): tuple(str(item) for item in names)
            for arm, names in dict(data.pop("control_joints", {})).items()
        }
        return cls(planner_joints=planner, control_joints=control, **data)

def register_controller(target_class):
    # key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    key = target_class.__name__
    assert key.endswith("Controller")
    key = key.removesuffix("Controller")
    # assert key not in CONTROLLER_DICT
    CONTROLLER_DICT[key] = target_class
    return target_class
