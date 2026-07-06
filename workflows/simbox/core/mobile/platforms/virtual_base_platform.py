"""Virtual holonomic mobile-base platform profile."""

from __future__ import annotations

from .base_platform import MobileBasePlatform


class VirtualBasePlatform(MobileBasePlatform):
    """Platform profile for X/Y/yaw virtual-base articulation joints."""

    profile_name = "virtual_base"
    aliases = ("omni_virtual_base", "panda_omron_virtual", "panda_omron_virtual_base")

    def normalize_base_cfg(self, base_cfg: dict) -> dict:
        normalized = super().normalize_base_cfg(base_cfg)
        platform_cfg = self._require_mapping(normalized, "platform")
        platform_cfg["profile"] = self.profile_name
        self.default_nav2_footprint_points(normalized)
        self.default_nav2_inflation_radius_m(normalized)
        self.default_nav2_minimum_turning_radius_m(normalized)
        self.max_steer_angle_ackermann(normalized)
        self.nav2_controller_hard_limits(normalized)
        return normalized

    @staticmethod
    def _require_mapping(mapping: dict, key: str) -> dict:
        value = mapping.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"Missing required mapping config: {key}")
        return value

    def _platform_nav2_cfg(self, base_cfg: dict) -> dict:
        platform_cfg = self._require_mapping(base_cfg, "platform")
        nav2_cfg = platform_cfg.get("nav2")
        if not isinstance(nav2_cfg, dict):
            raise KeyError("Missing required mapping config: platform.nav2")
        return nav2_cfg

    @staticmethod
    def _require_float(mapping: dict, key: str, *, path: str) -> float:
        if key not in mapping:
            raise KeyError(f"Missing required numeric config: {path}")
        return float(mapping[key])

    def default_nav2_footprint_points(self, base_cfg: dict) -> list[list[float]]:
        nav2_cfg = self._platform_nav2_cfg(base_cfg)
        configured = nav2_cfg.get("footprint_points")
        if not isinstance(configured, list):
            raise KeyError("Missing required list config: platform.nav2.footprint_points")
        return [[float(x), float(y)] for x, y in configured]

    def default_nav2_inflation_radius_m(self, base_cfg: dict) -> float:
        nav2_cfg = self._platform_nav2_cfg(base_cfg)
        return self._require_float(nav2_cfg, "inflation_radius_m", path="platform.nav2.inflation_radius_m")

    def default_nav2_minimum_turning_radius_m(self, base_cfg: dict) -> float:
        nav2_cfg = self._platform_nav2_cfg(base_cfg)
        return self._require_float(
            nav2_cfg,
            "minimum_turning_radius_m",
            path="platform.nav2.minimum_turning_radius_m",
        )

    def max_steer_angle_ackermann(self, base_cfg: dict) -> float:
        nav2_cfg = self._platform_nav2_cfg(base_cfg)
        return self._require_float(
            nav2_cfg,
            "max_steer_angle_ackermann",
            path="platform.nav2.max_steer_angle_ackermann",
        )

    def nav2_controller_hard_limits(self, base_cfg: dict) -> dict:
        nav2_cfg = self._platform_nav2_cfg(base_cfg)
        limits_cfg = nav2_cfg.get("controller_hard_limits")
        if not isinstance(limits_cfg, dict):
            raise KeyError("Missing required mapping config: platform.nav2.controller_hard_limits")

        result = {}
        for key in ("max_velocity", "min_velocity", "max_accel", "max_decel"):
            values = limits_cfg.get(key)
            if not isinstance(values, list) or len(values) != 3:
                raise KeyError(f"Missing required 3-element list config: platform.nav2.controller_hard_limits.{key}")
            result[key] = [float(value) for value in values]
        return result

    def build_bridge(self, robot, *, node_name: str):
        from workflows.simbox.core.mobile.bridge.virtual_base_bridge import VirtualBaseBridge

        return VirtualBaseBridge(robot, node_name=node_name)
