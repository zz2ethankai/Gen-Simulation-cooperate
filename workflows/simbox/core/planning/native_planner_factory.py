"""CuRobo-native planner construction isolated behind PlannerRuntime factories."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.utils.constants import CUROBO_BATCH_SIZE
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from core.planning.native_bridge import RobotCfg

from .domain_types import PlannerKind, PlannerRuntimeProfile


def resolve_native_robot_config(robot_file: str) -> dict[str, Any]:
    """Resolve the simulator robot YAML into CuRobo's native schema."""

    from curobo.config_io import load_yaml

    robot_path = Path(robot_file).expanduser().resolve()
    raw = load_yaml(str(robot_path))
    if not isinstance(raw, dict):
        raise TypeError(f"robot config must be a mapping, got {type(raw)!r}")
    raw = deepcopy(raw.get("robot_cfg", raw))
    kinematics = deepcopy(raw.get("kinematics", raw))
    if not isinstance(kinematics, dict):
        raise TypeError("robot_cfg.kinematics must be a mapping")
    config_dir = robot_path.parent
    asset_root = config_dir.parents[1] / "assets" if len(config_dir.parents) > 1 else config_dir
    ee_link = kinematics.get("ee_link")
    if "tool_frames" not in kinematics and ee_link:
        kinematics["tool_frames"] = [ee_link]
    for key in (
        "use_usd_kinematics", "isaac_usd_path", "usd_path", "usd_robot_root",
        "usd_flip_joints", "usd_flip_joint_limits", "ee_link",
    ):
        kinematics.pop(key, None)

    def resolve_path(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        candidate = asset_root / path
        if candidate.exists():
            return str(candidate)
        candidate = config_dir / path
        return str(candidate if candidate.exists() else asset_root / path)

    if "urdf_path" in kinematics:
        kinematics["urdf_path"] = resolve_path(kinematics["urdf_path"])
    if "asset_root_path" in kinematics:
        path = Path(kinematics["asset_root_path"])
        if not path.is_absolute():
            kinematics["asset_root_path"] = str(asset_root / path)
    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, str):
        sphere_path = Path(spheres)
        if not sphere_path.is_absolute():
            local = config_dir / sphere_path
            sphere_path = local if local.exists() else asset_root / sphere_path
        if sphere_path.exists():
            sphere_data = load_yaml(str(sphere_path)) or {}
            kinematics["collision_spheres"] = sphere_data.get("collision_spheres", sphere_data)
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict) and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    return {"kinematics": kinematics}


@dataclass(frozen=True)
class PlannerBuildConfig:
    """Explicit native-planner inputs owned by :class:`PlannerRuntime`."""

    robot_file: str
    tensor_args: Any
    collision_activation_distance: float


class NativePlannerFactory:
    """Build the native planner for one controller."""

    def __init__(
        self,
        build_config: PlannerBuildConfig,
        *,
        robot_config: Callable[[str], dict[str, Any]],
        pose_criteria: Callable[[Any, Any], Any],
        collision_cache: dict[str, int],
        interpolation_dt: float,
    ) -> None:
        self.build_config = build_config
        self.robot_config = robot_config
        self.pose_criteria = pose_criteria
        self.collision_cache = dict(collision_cache)
        self.interpolation_dt = float(interpolation_dt)

    def _robot_cfg(self) -> RobotCfg:
        return RobotCfg.create(
            self.robot_config(self.build_config.robot_file),
            device_cfg=self.build_config.tensor_args,
            num_envs=1,
        )

    def _planner_cfg(
        self,
        robot_cfg: RobotCfg,
        *,
        batch_size: int,
        trajopt_seeds: int,
    ):
        config = MotionPlannerCfg.create(
            robot=robot_cfg,
            device_cfg=self.build_config.tensor_args,
            collision_cache=self.collision_cache,
            max_goalset=1,
            max_batch_size=batch_size,
            num_ik_seeds=6,
            num_trajopt_seeds=trajopt_seeds,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=self.build_config.collision_activation_distance,
            self_collision_check=True,
            use_cuda_graph=True,
        )
        config.trajopt_solver_config.interpolation_dt = self.interpolation_dt
        return config

    def build_single(self, profile: PlannerRuntimeProfile | None = None, kind: PlannerKind | None = None):
        del profile, kind
        planner = MotionPlanner(
            self._planner_cfg(
                self._robot_cfg(),
                batch_size=1,
                trajopt_seeds=6,
            )
        )
        self.pose_criteria(planner, None)
        # PlannerRuntime injects the current SceneCfg before warming the
        # planner.  Warmup captures IK/TrajOpt/graph state, so doing it here
        # would compile against an empty collision world and the subsequent
        # scene update would leave those captures stale.
        return planner

    def build_batch(self, profile: PlannerRuntimeProfile | None = None, kind: PlannerKind | None = None):
        del profile, kind
        planner = BatchMotionPlanner(
            self._planner_cfg(
                self._robot_cfg(),
                batch_size=CUROBO_BATCH_SIZE,
                trajopt_seeds=6,
            )
        )
        self.pose_criteria(planner, None)
        return planner

__all__ = ["NativePlannerFactory", "PlannerBuildConfig"]
