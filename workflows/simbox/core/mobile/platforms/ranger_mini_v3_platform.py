"""Ranger Mini V3 4WIS platform profile."""

from __future__ import annotations

from .base_platform import MobileBasePlatform


class RangerMiniV3Platform(MobileBasePlatform):
    profile_name = "ranger_mini_v3"
    aliases = ("ranger_mini_v3_4wis", "ranger_mini", "split_aloha_base")

    DEFAULT_NAV2_CFG = {
        "footprint_points": [
            [0.36, 0.24],
            [0.32, 0.29],
            [-0.32, 0.29],
            [-0.36, 0.24],
            [-0.36, -0.24],
            [-0.32, -0.29],
            [0.32, -0.29],
            [0.36, -0.24],
        ],
        "inflation_radius_m": 0.34,
        "minimum_turning_radius_m": 0.47644,
        "max_steer_angle_ackermann": 0.6981,
        "controller_hard_limits": {
            "max_velocity": [0.35, 0.25, 0.60],
            "min_velocity": [-0.35, -0.25, -0.60],
            "max_accel": [0.35, 0.35, 0.70],
            "max_decel": [-0.35, -0.35, -0.70],
        },
    }

    def normalize_base_cfg(self, base_cfg: dict) -> dict:
        normalized = super().normalize_base_cfg(base_cfg)
        platform_cfg = normalized.setdefault("platform", {})
        platform_cfg["profile"] = self.profile_name
        ros_cfg = self._require_mapping(normalized, "ros")
        ranger_model = str(ros_cfg.get("ranger_model", self.profile_name)).strip().lower()
        if ranger_model and ranger_model != "ranger_mini_v3":
            raise ValueError(f"Unsupported ros.ranger_model for RangerMiniV3Platform: {ranger_model or '<missing>'}")
        ros_cfg["ranger_model"] = self.profile_name
        ros_cfg.pop("command_topic", None)
        ros_cfg.pop("motion_mode_topic", None)
        ros_cfg.pop("command_type", None)
        ros_cfg.pop("internal_cmdvel_controller_enabled", None)

        nav2_cfg = platform_cfg.setdefault("nav2", {})
        for key, value in self.DEFAULT_NAV2_CFG.items():
            nav2_cfg.setdefault(key, value if not isinstance(value, dict) else dict(value))
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
        from workflows.simbox.core.mobile.bridge.ranger_mini_v3_bridge import RangerMiniV3Bridge

        return RangerMiniV3Bridge(robot, node_name=node_name)
