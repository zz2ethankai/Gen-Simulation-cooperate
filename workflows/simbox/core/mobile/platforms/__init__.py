"""Platform registry for mobile base chassis profiles."""

from __future__ import annotations

from .base_platform import MobileBasePlatform
from .ranger_mini_v3_platform import RangerMiniV3Platform


def _profile_name_from_base_cfg(base_cfg: dict) -> str:
    platform_cfg = dict(base_cfg.get("platform", {}))
    value = str(platform_cfg.get("profile") or "").strip()
    if value:
        return value.lower()
    ros_cfg = dict(base_cfg.get("ros", {}))
    ranger_model = str(ros_cfg.get("ranger_model") or "").strip().lower()
    if ranger_model == "ranger_mini_v3":
        return ranger_model
    raise KeyError("Missing required mobile base config field: platform.profile")


def get_mobile_base_platform(base_cfg: dict) -> MobileBasePlatform:
    profile_name = _profile_name_from_base_cfg(base_cfg)
    if RangerMiniV3Platform.matches(profile_name):
        return RangerMiniV3Platform()
    raise KeyError(f"Unsupported mobile base platform profile: {profile_name}")


def build_mobile_base_bridge(robot, *, node_name: str):
    base_interface = robot.get_base_interface()
    base_cfg = base_interface.get("base_cfg", {}) if isinstance(base_interface, dict) else {}
    platform = get_mobile_base_platform(base_cfg)
    return platform.build_bridge(robot, node_name=node_name)


__all__ = [
    "MobileBasePlatform",
    "RangerMiniV3Platform",
    "build_mobile_base_bridge",
    "get_mobile_base_platform",
]
