"""CuRobo-native planner construction isolated behind PlannerRuntime factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.utils.constants import CUROBO_BATCH_SIZE
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from core.planning.native_bridge import RobotCfg

from .domain_types import PlannerKind, PlannerRuntimeProfile


@dataclass(frozen=True)
class PlannerBuildConfig:
    """Explicit native-planner inputs owned by :class:`PlannerRuntime`."""

    robot_file: str
    tensor_args: Any
    collision_activation_distance: float


class NativePlannerFactory:
    """Build the single and lazy batch native planners for one controller."""

    def __init__(
        self,
        build_config: PlannerBuildConfig,
        *,
        robot_config: Callable[[str], dict[str, Any]],
        pose_criteria: Callable[[Any, Any], Any],
        collision_cache: dict[str, int],
        graph_enabled: bool,
        warmup_iterations: int,
        interpolation_dt: float,
    ) -> None:
        self.build_config = build_config
        self.robot_config = robot_config
        self.pose_criteria = pose_criteria
        self.collision_cache = dict(collision_cache)
        self.graph_enabled = bool(graph_enabled)
        self.warmup_iterations = int(warmup_iterations)
        self.interpolation_dt = float(interpolation_dt)

    def _robot_cfg(self) -> RobotCfg:
        return RobotCfg.create(
            self.robot_config(self.build_config.robot_file),
            device_cfg=self.build_config.tensor_args,
            num_envs=1,
        )

    def _planner_cfg(self, robot_cfg: RobotCfg, *, batch_size: int, trajopt_seeds: int):
        config = MotionPlannerCfg.create(
            robot=robot_cfg,
            device_cfg=self.build_config.tensor_args,
            collision_cache=self.collision_cache,
            max_goalset=1,
            max_batch_size=batch_size,
            num_ik_seeds=20,
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
        planner = MotionPlanner(self._planner_cfg(self._robot_cfg(), batch_size=1, trajopt_seeds=12))
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
                # Candidate batches still need the same local-optimization
                # budget as the historical CuRobo path.  One seed can reach
                # a pose while remaining infeasible because of the attached
                # object; keep the batch parallelism, but do not trade away
                # the alternate collision-free solutions.
                trajopt_seeds=12,
            )
        )
        self.pose_criteria(planner, None)
        # Keep the batch planner lazy; PlannerRuntime performs world
        # materialization and then the configured warmup on first use.
        return planner


__all__ = ["NativePlannerFactory", "PlannerBuildConfig"]
