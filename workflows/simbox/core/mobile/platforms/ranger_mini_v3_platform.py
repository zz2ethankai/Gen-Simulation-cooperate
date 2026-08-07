"""Ranger Mini V3 4WIS platform profile."""

from __future__ import annotations

from .base_platform import MobileBasePlatform


class RangerMiniV3Platform(MobileBasePlatform):
    profile_name = "ranger_mini_v3"
    aliases = ("ranger_mini_v3_4wis", "ranger_mini", "split_aloha_base")

    def normalize_base_cfg(self, base_cfg: dict) -> dict:
        normalized = super().normalize_base_cfg(base_cfg)
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

    def _platform_navigation_cfg(self, base_cfg: dict) -> dict:
        platform_cfg = self._require_mapping(base_cfg, "platform")
        navigation_cfg = platform_cfg.get("local_navigation")
        if not isinstance(navigation_cfg, dict):
            raise KeyError("Missing required mapping config: platform.local_navigation")
        return navigation_cfg

    @staticmethod
    def _require_float(mapping: dict, key: str, *, path: str) -> float:
        if key not in mapping:
            raise KeyError(f"Missing required numeric config: {path}")
        return float(mapping[key])

    def default_navigation_footprint_points(self, base_cfg: dict) -> list[list[float]]:
        navigation_cfg = self._platform_navigation_cfg(base_cfg)
        configured = navigation_cfg.get("footprint_points")
        if not isinstance(configured, list):
            raise KeyError("Missing required list config: platform.local_navigation.footprint_points")
        return [[float(x), float(y)] for x, y in configured]

    def default_navigation_inflation_radius_m(self, base_cfg: dict) -> float:
        navigation_cfg = self._platform_navigation_cfg(base_cfg)
        return self._require_float(navigation_cfg, "inflation_radius_m", path="platform.local_navigation.inflation_radius_m")

    def default_navigation_minimum_turning_radius_m(self, base_cfg: dict) -> float:
        navigation_cfg = self._platform_navigation_cfg(base_cfg)
        return self._require_float(
            navigation_cfg,
            "minimum_turning_radius_m",
            path="platform.local_navigation.minimum_turning_radius_m",
        )

    def max_steer_angle_ackermann(self, base_cfg: dict) -> float:
        navigation_cfg = self._platform_navigation_cfg(base_cfg)
        return self._require_float(
            navigation_cfg,
            "max_steer_angle_ackermann",
            path="platform.local_navigation.max_steer_angle_ackermann",
        )

    def navigation_controller_hard_limits(self, base_cfg: dict) -> dict:
        navigation_cfg = self._platform_navigation_cfg(base_cfg)
        limits_cfg = navigation_cfg.get("controller_hard_limits")
        if not isinstance(limits_cfg, dict):
            raise KeyError("Missing required mapping config: platform.local_navigation.controller_hard_limits")

        result = {}
        for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
            values = limits_cfg.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise KeyError(f"Missing required 3-element list config: platform.local_navigation.controller_hard_limits.{key}")
            result[key] = [float(value) for value in values]
        return result
