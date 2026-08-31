"""Unitree G1 learned-locomotion platform profile."""

from __future__ import annotations

from copy import deepcopy

from .base_platform import MobileBasePlatform


class UnitreeG1Platform(MobileBasePlatform):
    """Expose G1 footprint and body-command limits to local navigation."""

    profile_name = "unitree_g1_decoupled_wbc"
    aliases = ("unitree_g1", "g1_decoupled_wbc")

    def normalize_base_cfg(self, base_cfg: dict) -> dict:
        normalized = deepcopy(base_cfg)
        platform_cfg = self._require_mapping(normalized, "platform")
        platform_cfg["profile"] = self.profile_name
        self.default_navigation_footprint_points(normalized)
        self.default_navigation_inflation_radius_m(normalized)
        self.default_navigation_minimum_turning_radius_m(normalized)
        self.max_steer_angle_ackermann(normalized)
        self.navigation_controller_hard_limits(normalized)
        return normalized

    @staticmethod
    def _require_mapping(mapping: dict, key: str) -> dict:
        value = mapping.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"Missing required mapping config: {key}")
        return value

    def _navigation_cfg(self, base_cfg: dict) -> dict:
        platform_cfg = self._require_mapping(base_cfg, "platform")
        value = platform_cfg.get("local_navigation")
        if not isinstance(value, dict):
            raise KeyError("Missing required mapping config: platform.local_navigation")
        return value

    @staticmethod
    def _require_float(mapping: dict, key: str) -> float:
        if key not in mapping:
            raise KeyError(f"Missing required numeric config: {key}")
        return float(mapping[key])

    def default_navigation_footprint_points(self, base_cfg: dict) -> list[list[float]]:
        navigation_cfg = self._navigation_cfg(base_cfg)
        points = navigation_cfg.get("footprint_points")
        if not isinstance(points, list) or len(points) < 3:
            raise KeyError(
                "Missing required list config: platform.local_navigation.footprint_points"
            )
        return [[float(x), float(y)] for x, y in points]

    def default_navigation_inflation_radius_m(self, base_cfg: dict) -> float:
        return self._require_float(
            self._navigation_cfg(base_cfg),
            "inflation_radius_m",
        )

    def default_navigation_minimum_turning_radius_m(self, base_cfg: dict) -> float:
        return self._require_float(
            self._navigation_cfg(base_cfg),
            "minimum_turning_radius_m",
        )

    def max_steer_angle_ackermann(self, base_cfg: dict) -> float:
        return self._require_float(
            self._navigation_cfg(base_cfg),
            "max_steer_angle_ackermann",
        )

    def navigation_controller_hard_limits(self, base_cfg: dict) -> dict:
        limits = self._navigation_cfg(base_cfg).get("controller_hard_limits")
        if not isinstance(limits, dict):
            raise KeyError(
                "Missing required mapping config: "
                "platform.local_navigation.controller_hard_limits"
            )
        result = {}
        for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
            values = limits.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise KeyError(
                    "Missing required 3-element list config: "
                    f"platform.local_navigation.controller_hard_limits.{key}"
                )
            result[key] = [float(value) for value in values]
        return result
