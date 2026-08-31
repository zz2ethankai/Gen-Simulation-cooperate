"""Platform registry for mobile base chassis profiles."""

from __future__ import annotations

from .base_platform import MobileBasePlatform
from .differential_drive_platform import DifferentialDrivePlatform
from .ranger_mini_v3_platform import RangerMiniV3Platform
from .unitree_g1_platform import UnitreeG1Platform
from .virtual_base_platform import VirtualBasePlatform


def _profile_name_from_base_cfg(base_cfg: dict) -> str:
    platform_cfg = dict(base_cfg.get("platform", {}))
    value = str(platform_cfg.get("profile") or "").strip()
    if value:
        return value.lower()
    raise KeyError("Missing required mobile base config field: platform.profile")


def get_mobile_base_platform(base_cfg: dict) -> MobileBasePlatform:
    profile_name = _profile_name_from_base_cfg(base_cfg)
    if DifferentialDrivePlatform.matches(profile_name):
        return DifferentialDrivePlatform()
    if RangerMiniV3Platform.matches(profile_name):
        return RangerMiniV3Platform()
    if UnitreeG1Platform.matches(profile_name):
        return UnitreeG1Platform()
    if VirtualBasePlatform.matches(profile_name):
        return VirtualBasePlatform()
    raise KeyError(f"Unsupported mobile base platform profile: {profile_name}")


__all__ = [
    "DifferentialDrivePlatform",
    "MobileBasePlatform",
    "RangerMiniV3Platform",
    "UnitreeG1Platform",
    "VirtualBasePlatform",
    "get_mobile_base_platform",
]
