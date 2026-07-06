"""Mobile base platform abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy


class MobileBasePlatform(ABC):
    """Base class for chassis-specific mobile base semantics."""

    profile_name = ""
    aliases: tuple[str, ...] = ()

    @classmethod
    def matches(cls, profile_name: str) -> bool:
        normalized = str(profile_name).strip().lower()
        return normalized == cls.profile_name or normalized in cls.aliases

    def normalize_base_cfg(self, base_cfg: dict) -> dict:
        normalized = deepcopy(base_cfg)
        normalized["ackermann_split_steering"] = self._require_bool(normalized, "ackermann_split_steering")
        normalized["ackermann_split_wheel_speeds"] = self._require_bool(normalized, "ackermann_split_wheel_speeds")
        return normalized

    @staticmethod
    def _require_bool(mapping: dict, key: str) -> bool:
        if key not in mapping:
            raise KeyError(f"Missing required bool config: {key}")
        value = mapping[key]
        if not isinstance(value, bool):
            raise TypeError(f"Config must be a bool: {key}")
        return value

    @abstractmethod
    def default_nav2_footprint_points(self, base_cfg: dict) -> list[list[float]]:
        """Return Nav2 footprint points for this chassis."""

    @abstractmethod
    def default_nav2_inflation_radius_m(self, base_cfg: dict) -> float:
        """Return Nav2 inflation radius for this chassis."""

    @abstractmethod
    def default_nav2_minimum_turning_radius_m(self, base_cfg: dict) -> float:
        """Return Nav2 minimum turning radius for this chassis."""

    @abstractmethod
    def max_steer_angle_ackermann(self, base_cfg: dict) -> float:
        """Return the chassis Ackermann steering cap."""

    @abstractmethod
    def nav2_controller_hard_limits(self, base_cfg: dict) -> dict:
        """Return the required controller hard limits shared across Nav2 control layers."""

    @abstractmethod
    def build_bridge(self, robot, *, node_name: str):
        """Construct the Isaac-side ROS bridge for this chassis."""
