"""Controller-owned runtime composition.

The controller façade owns no CuRobo planner.  ``PlannerRuntime`` owns the
native single/batch instances and this class supplies the simulator-specific
factories, state conversion and attachment/scene adapters around it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from core.planning.attachment_runtime import AttachmentRuntime
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.controllers.curobo.phase_execution import ExecutionStatus
from core.planning.domain_types import (
    AttachmentSpec,
    BatchPosePlanRequest,
    CollisionOptions,
    CollisionPolicy,
    CommandStatus,
    CspacePlanRequest,
    PlanResult,
    PosePlanRequest,
    PlanningProfile,
    PlannerRuntimeProfile,
)
from core.planning.native_scene_adapter import NativeSceneAdapter
from core.planning.planner_runtime import PlannerRuntime
from core.planning.scene_runtime import SceneRuntime
from core.utils.constants import CUROBO_BATCH_SIZE
from core.controllers.curobo.trajectory import normalize_named_trajectory

if TYPE_CHECKING:
    from core.planning.native_planner_factory import PlannerBuildConfig
else:
    # Keep annotations resolvable without importing CuRobo/Warp at module load.
    PlannerBuildConfig = Any


def resolve_native_robot_config(robot_file: str) -> dict[str, Any]:
    """Normalize a SimBox robot YAML for native planner construction."""

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
    if isinstance(cspace, dict):
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace.pop("retract_config")
        else:
            cspace.pop("retract_config", None)
    return {"kinematics": kinematics}


class _NativeAttachmentAdapter:
    """Adapt one PlannerRuntime-owned native planner to AttachmentRuntime."""

    def __init__(self, runtime: "MotionPlannerRuntime", *, batch: bool = False) -> None:
        self.runtime = runtime
        self.batch = bool(batch)

    def _planner(self):
        if self.batch:
            return self.runtime.ensure_batch_planner()
        return self.runtime.native_planner

    def attach(
        self,
        state,
        meshes,
        *,
        link_name="attached_object",
        world_objects_pose_offset=None,
        disable_obstacle_names=(),
        **_kwargs,
    ):
        return self.runtime.attach_native(
            list(disable_obstacle_names),
            state,
            meshes,
            link_name=link_name,
            world_objects_pose_offset=world_objects_pose_offset,
            planner=self._planner(),
        )

    def detach(self):
        return self._planner().attachment_manager.detach()


class PlanningSceneRuntime(SceneRuntime):
    """Controller-facing scene state with no simulator ownership."""


@dataclass
class RobotPort:
    """Narrow robot/state inputs consumed by :class:`MotionPlannerRuntime`."""

    name: str
    lr_name: str
    task_cfg: Mapping[str, Any]
    robot: Any
    tensor_args: Any
    arm_spec: Any
    arm_indices: Any
    raw_js_names: list[str]
    batch_capability: bool
    constrain_grasp_approach: bool
    collision_scene_manager: Any = None
    collision_world_mode: str = "physics_schema"
    obstacle_pose: Callable[[str], Any] | None = None
    interpolation_dt: float = 0.01
    ik_solver: Any = None
    kin_model: Any = None


@dataclass(frozen=True)
class ExecutionPort:
    """Execution callbacks kept separate from planner/runtime state."""

    forward_phase_command: Callable[[Any], Any]
    dummy_forward: Callable[..., Any]
    execution_status: Callable[[Any], ExecutionStatus]
    hold_action: Callable[[], Any]


class MotionPlannerRuntime:
    """Own planner construction and simulator-independent native operations."""

    batch_attachment_runtime: AttachmentRuntime | None = None

    def __init__(
        self,
        planner_build_config: PlannerBuildConfig,
        robot_port: RobotPort,
        execution_port: ExecutionPort,
        *,
        world: Any = None,
        phase_executor: PhaseExecutor | None = None,
    ) -> None:
        self.planner_build_config = planner_build_config
        self.robot_port = robot_port
        self.execution_port = execution_port
        self.scene_runtime = PlanningSceneRuntime(world)
        self.phase_executor = phase_executor or PhaseExecutor()
        self._pending_pose_criteria = None
        self.world_update_signature = None
        self._configure_native_runtime()
        self.attachment_runtime = AttachmentRuntime(
            manager=_NativeAttachmentAdapter(self), strict=False
        )
        self.batch_attachment_runtime: AttachmentRuntime | None = None
        # Keep the controller-local geometry adapter on the same revision
        # fanout as every collision-scene adapter.  It is used by attachment
        # fitting and diagnostics, so leaving it outside PlannerRuntime would
        # leave stale world provenance after a scene update (and after lazy
        # batch materialization).
        self.native_scene_adapter = NativeSceneAdapter(
            self.native_planner,
            strict=True,
            world=self.scene_runtime.world,
            world_revision=self.scene_revision,
        )
        self.planner_runtime.register_scene_adapter(self.native_scene_adapter)

    @property
    def scene_revision(self) -> int:
        """Revision shared by the controller scene and native planners."""

        return int(self.planner_runtime.scene_revision)

    @property
    def world_revision(self) -> int:
        """Alias used by scene ports and collision audits."""

        return self.scene_revision

    @property
    def world(self):
        return self.scene_runtime.world

    def adopt_scene_revision(self, revision: int) -> int:
        return self.planner_runtime.adopt_scene_revision(revision)

    @property
    def native_planner(self):
        return self.planner_runtime.ensure_planner()

    @property
    def batch_planner(self):
        return self.planner_runtime.batch_planner

    @property
    def planner_names(self) -> list[str]:
        return list(self.native_planner.joint_names)

    def _configure_native_runtime(self) -> None:
        # CuRobo's native factory imports Warp.  Keep that simulator-only
        # dependency behind runtime construction so the typed request façade
        # remains importable for host-side contract tests.
        from core.planning.native_planner_factory import NativePlannerFactory

        port = self.robot_port
        pick_cfg = dict(port.task_cfg).get("planning", {}).get("pick_place", {})
        graph_enabled = bool(pick_cfg.get("enable_graph", False))
        # Candidate batches need one viable path, not a proof that every
        # sampled pose is solvable.  Keep the batch graph seed available by
        # default; the old controller used graph-assisted batch planning even
        # when the single-query path did not.
        batch_graph_enabled = bool(pick_cfg.get("batch_enable_graph", True))
        try:
            max_attempts = max(1, int(pick_cfg.get("max_plan_attempts", 4)))
        except (TypeError, ValueError):
            max_attempts = 4
        try:
            batch_attempts = max(1, int(pick_cfg.get("batch_max_plan_attempts", min(max_attempts, 4))))
        except (TypeError, ValueError):
            batch_attempts = min(max_attempts, 4)
        try:
            interpolation_dt = float(pick_cfg.get("interpolation_dt", 0.01))
            if not np.isfinite(interpolation_dt) or interpolation_dt <= 0:
                raise ValueError
        except (TypeError, ValueError):
            interpolation_dt = 0.01
        try:
            warmup_iterations = max(1, int(pick_cfg.get("warmup_iterations", 1)))
        except (TypeError, ValueError):
            warmup_iterations = 1
        self.max_plan_attempts = max_attempts
        self.batch_max_attempts = batch_attempts
        self.graph_enabled = graph_enabled
        self.single_graph_attempt = max(0, min(1, max_attempts - 1)) if graph_enabled else max_attempts
        self.batch_graph_enabled = batch_graph_enabled
        self.batch_graph_attempt = max(0, min(3, batch_attempts - 1)) if batch_graph_enabled else batch_attempts

        factory = NativePlannerFactory(
            self.planner_build_config,
            robot_config=resolve_native_robot_config,
            pose_criteria=self._set_pose_criteria,
            collision_cache=self._collision_cache(),
            graph_enabled=graph_enabled,
            warmup_iterations=warmup_iterations,
            interpolation_dt=interpolation_dt,
        )

        self.planner_runtime = PlannerRuntime(
            name=f"{port.name}.{port.lr_name}",
            profile=PlannerRuntimeProfile(
                name=f"{port.name}.{port.lr_name}",
                batch_enabled=bool(port.batch_capability),
                max_batch_size=CUROBO_BATCH_SIZE,
                lazy_batch=True,
                warmup_config={
                    "enable_graph": graph_enabled or batch_graph_enabled,
                    "num_warmup_iterations": warmup_iterations,
                },
            ),
            planner_factory=factory.build_single,
            batch_planner_factory=factory.build_batch,
            scene_revision=self.scene_runtime.revision,
            # Bind before the first native planner is materialized.  The
            # replay installs the initial world, then PlannerRuntime's
            # factory/update/warmup transaction builds the solver against it.
            scene=self.scene_runtime,
        )
        if self.scene_runtime.world is None:
            raise RuntimeError("MotionPlannerRuntime requires an explicit planning world")
        # SceneRuntime owns revision increments.  Binding here makes its
        # replay/update fanout the one path into PlannerRuntime, including the
        # revision on every world update and on lazy batch materialization.
        self.planner_runtime.bind_scene(self.scene_runtime)
        port.ik_solver = self.native_planner.ik_solver
        port.kin_model = self.native_planner.kinematics
        port.interpolation_dt = float(self.native_planner.trajopt_solver.config.interpolation_dt)
        self.world_update_signature = self._world_signature(self.scene_runtime.world)

    @staticmethod
    def _world_signature(world):
        """Return a stable, native-independent signature for a scene config."""

        def value(item):
            if item is None or isinstance(item, (str, bool, int, float)):
                return item
            if isinstance(item, np.ndarray):
                return tuple(np.asarray(item).reshape(-1).tolist())
            if isinstance(item, (list, tuple)):
                return tuple(value(entry) for entry in item)
            return str(item)

        objects = getattr(world, "objects", None) or []
        return tuple(
            (
                type(obj).__name__,
                getattr(obj, "name", None),
                value(getattr(obj, "pose", None)),
                value(getattr(obj, "dims", None)),
                value(getattr(obj, "scale", None)),
                getattr(obj, "file_path", None),
                value(getattr(obj, "vertices", None)),
                value(getattr(obj, "faces", None)),
            )
            for obj in objects
        )

    def _collision_cache(self):
        spec = getattr(self.robot_port, "arm_spec", None)
        cache = getattr(spec, "collision_cache", None)
        return dict(cache) if cache is not None else {"cuboid": 700, "mesh": 700}

    def _approach_axis(self) -> int:
        spec = getattr(self.robot_port, "arm_spec", None)
        axis = getattr(spec, "grasp_approach_axis", None)
        if axis is not None:
            return int(axis)
        return {"x": 0, "y": 1, "z": 2}[self.robot_port.robot.cfg["ee_axis"]]

    def _goal_tool_pose(self, ee_translation, ee_orientation, batch_size=1):
        from curobo.types import GoalToolPose

        position = self.robot_port.tensor_args.to_device(ee_translation)
        quaternion = self.robot_port.tensor_args.to_device(ee_orientation)
        return GoalToolPose(
            tool_frames=[self.native_planner.tool_frames[0]],
            position=position.reshape(batch_size, 1, 1, 1, 3),
            quaternion=quaternion.reshape(batch_size, 1, 1, 1, 4),
        )

    def _set_pose_criteria(self, planner, criteria=None):
        from core.planning.native_bridge import ToolPoseCriteria

        port = self.robot_port
        if criteria is None:
            non_terminal = [0.0] * 6
            if port.constrain_grasp_approach:
                non_terminal = [1.0] * 6
                non_terminal[self._approach_axis()] = 0.0
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=non_terminal,
                device_cfg=port.tensor_args,
            )
        planner.update_tool_pose_criteria({frame: criteria.clone() for frame in planner.tool_frames})
        self._pending_pose_criteria = criteria

    def ensure_batch_planner(self):
        batch = self.planner_runtime.ensure_batch_planner()
        if self.batch_attachment_runtime is None:
            self.batch_attachment_runtime = AttachmentRuntime(
                manager=_NativeAttachmentAdapter(self, batch=True), strict=False
            )
        return batch

    def update_world(self, world):
        update = self.scene_runtime.update_world(world)
        if update.changed:
            self.world_update_signature = self._world_signature(self.scene_runtime.world)
        return update

    def update_obstacle_poses(self, poses, *, force: bool = False):
        """Publish dynamic collider poses with the shared scene revision."""

        update = self.scene_runtime.update_poses(poses, force=force)
        return update

    def _request_common_kwargs(
        self,
        request_metadata: Mapping[str, Any] | None,
        *,
        default_profile: PlanningProfile,
        attachment_runtime: AttachmentRuntime | None = None,
    ) -> dict[str, Any]:
        """Normalize command metadata before constructing a typed request.

        MotionPhaseCommand is the execution boundary, while the public planner
        request types are the native boundary.  Keep the conversion explicit
        here so phase identity, collision policy, target entities and
        completion/replan declarations cannot disappear between those layers.
        """

        supplied = dict(request_metadata or {})
        metadata = dict(supplied.get("metadata", {}))
        metadata.setdefault("world_revision", self.scene_revision)
        profile = supplied.get("profile", default_profile)
        if profile is None:
            profile = default_profile
        collision_policy = supplied.get(
            "collision_policy", CollisionPolicy.WORLD_TRANSIT
        )
        if collision_policy is None:
            collision_policy = CollisionPolicy.WORLD_TRANSIT
        if not isinstance(collision_policy, CollisionPolicy):
            collision_policy = CollisionPolicy(str(collision_policy).lower())

        # Native CuRobo only addresses exact world collider paths.  Phase
        # metadata carries logical entity names, so resolve those names once
        # at this simulator-owned request builder and keep the native adapter
        # free of scene-manager lookups or substring/path heuristics.
        collision_options = CollisionOptions.from_mapping(
            supplied.get("collision_options"),
            default_policy=collision_policy,
        )
        collision_scene = getattr(self.robot_port, "collision_scene_manager", None)
        target_entity = supplied.get("active_target", supplied.get("active_object"))
        support_entity = supplied.get("support", supplied.get("support_object"))
        target_paths = self._collision_entity_paths(collision_scene, target_entity)
        support_paths = self._collision_entity_paths(collision_scene, support_entity)
        attachment_runtime = attachment_runtime or getattr(self, "attachment_runtime", None)
        attached_paths = tuple(
            str(path)
            for path in (getattr(attachment_runtime, "attached_obstacle_names", ()) or ())
        )
        option_mapping = collision_options.to_dict()
        if not option_mapping["target_obstacles"] and target_paths:
            option_mapping["target_obstacles"] = list(target_paths)
        if not option_mapping["support_obstacles"] and support_paths:
            option_mapping["support_obstacles"] = list(support_paths)
        if not option_mapping["attached_obstacles"] and attached_paths:
            option_mapping["attached_obstacles"] = list(attached_paths)
        collision_options = CollisionOptions.from_mapping(
            option_mapping,
            default_policy=collision_policy,
        )
        return {
            "phase_id": str(supplied.get("phase_id", "phase")),
            "completion_policy": supplied.get("completion_policy", "default"),
            "replan_policy": supplied.get("replan_policy", "allowed"),
            "collision_policy": collision_policy,
            "collision_options": collision_options,
            "active_target": target_entity,
            "support": support_entity,
            "profile": profile,
            "preplanned_joint_path": supplied.get(
                "preplanned_joint_path", supplied.get("preplanned_trajectory")
            ),
            "metadata": metadata,
            "world_revision": self.scene_revision,
        }

    @staticmethod
    def _collision_entity_paths(scene_manager: Any, entity_name: Any) -> tuple[str, ...]:
        """Resolve one logical scene entity to its exact native collider paths."""

        if scene_manager is None or entity_name is None:
            return ()
        records = getattr(scene_manager, "records", None)
        if not isinstance(records, Mapping):
            return ()
        record = records.get(str(entity_name))
        if record is None:
            return ()
        paths = (
            record.get("collision_prim_paths", ())
            if isinstance(record, Mapping)
            else getattr(record, "collision_prim_paths", ())
        )
        if paths is None or isinstance(paths, (str, bytes)):
            paths = (paths,) if paths else ()
        return tuple(dict.fromkeys(str(path) for path in paths if str(path)))

    def _single_pose_native_kwargs(self) -> dict[str, Any]:
        """Map the typed single-pose profile to CuRobo V2 parameters."""

        return {
            # Keep the native-v2 pose query equivalent to the pre-migration
            # controller call.  CuRobo's implicit-goal mode is required for
            # the GoalToolPose contract used by Pick terminal queries; relying
            # on a backend default makes typed single fallbacks diverge from
            # the old ``plan_pose(..., use_implicit_goal=True)`` call.
            "use_implicit_goal": True,
            "max_attempts": self.max_plan_attempts,
            "enable_graph_attempt": self.single_graph_attempt,
        }

    def _batch_pose_native_kwargs(self) -> dict[str, Any]:
        """Map the typed batch-pose profile to CuRobo V2 parameters."""

        return {
            "use_implicit_goal": True,
            "max_attempts": self.batch_max_attempts,
            # Pick/Place batches are candidate searches.  Requiring all
            # sampled candidates to converge makes one bad sample invalidate
            # otherwise usable paths and needlessly burns every retry.
            "success_ratio": 1.0 / float(CUROBO_BATCH_SIZE),
            "enable_graph_attempt": self.batch_graph_attempt,
        }

    def _single_cspace_native_kwargs(self) -> dict[str, Any]:
        """Map the typed single-cspace profile to CuRobo V2 parameters."""

        return {
            "max_attempts": self.max_plan_attempts,
            "enable_graph_attempt": self.single_graph_attempt,
        }

    def plan_pose(
        self,
        position,
        orientation,
        *,
        start_state=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        del context
        if start_state is None:
            start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        goal = self._goal_tool_pose(position, orientation)
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.TRANSIT,
            attachment_runtime=getattr(self, "attachment_runtime", None),
        )
        return self.planner_runtime.plan_pose(
            PosePlanRequest(
                goal=goal,
                start_state=start_state,
                kwargs=self._single_pose_native_kwargs(),
                **common,
            )
        )

    def plan_pose_batch(
        self,
        positions,
        orientations,
        *,
        start_state=None,
        batch_size=None,
        start_paths=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        del context
        batch = self.ensure_batch_planner()
        if start_paths is not None:
            start_state = self.batch_start_state_from_paths(start_paths)
        elif start_state is None:
            start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state(), repeat=len(positions))
        goals = self._goal_tool_pose(positions, orientations, batch_size=len(positions))
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.TRANSIT,
            attachment_runtime=getattr(self, "batch_attachment_runtime", None),
        )
        return self.planner_runtime.plan_pose_batch(
            BatchPosePlanRequest(
                goals=goals,
                start_state=start_state,
                batch_size=batch_size or len(positions),
                kwargs=self._batch_pose_native_kwargs(),
                **common,
            )
        )

    def plan_cspace(
        self,
        goal_positions,
        *,
        start_state=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        from curobo.types import JointState

        del context
        goal = JointState.from_position(
            self.robot_port.tensor_args.to_device(goal_positions),
            joint_names=self.planner_names,
        ).unsqueeze(0)
        if start_state is None:
            start_state = self.arm_joint_state(
                self.robot_port.robot.get_joints_state()
            ).unsqueeze(0)
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.CSPACE,
        )
        return self.planner_runtime.plan_cspace(
            CspacePlanRequest(
                goal_positions=goal,
                start_state=start_state,
                kwargs=self._single_cspace_native_kwargs(),
                **common,
            )
        )

    def compute_fk(self, joint_positions, *, joint_names=None):
        from curobo.types import JointState

        names = list(joint_names or self.planner_names)
        state = JointState.from_position(self.robot_port.tensor_args.to_device(joint_positions), joint_names=names)
        state = state.reorder(self.planner_names)
        out = self.native_planner.compute_kinematics(state)
        pose = out.tool_poses.get_link_pose(self.native_planner.tool_frames[0])
        return pose.position.detach().cpu().numpy(), pose.quaternion.detach().cpu().numpy()

    def arm_joint_state(self, sim_state, *, repeat=1):
        from curobo.types import JointState

        positions = np.asarray(sim_state.positions, dtype=float)
        names = list(self.robot_port.robot.dof_names)
        values = positions[self.robot_port.arm_indices]
        state = JointState.from_position(
            self.robot_port.tensor_args.to_device(values), joint_names=self.robot_port.raw_js_names
        )
        state = state.reorder(self.planner_names)
        if repeat > 1:
            state = state.unsqueeze(0).repeat((repeat, 1))
        return state

    def joint_state_from_trajectory(self, path):
        """Convert one normalized trajectory to a planner JointState.

        ``PlannerRuntime`` deliberately publishes native-independent
        ``JointTrajectory`` values.  The conversion back to CuRobo belongs at
        this runtime boundary: native list, NumPy, and tensor positions all
        pass through the controller's ``DeviceCfg`` before the public
        ``JointState.from_position`` constructor is used.
        """

        from curobo.types import JointState

        positions = getattr(path, "positions", None)
        if positions is None:
            positions = getattr(path, "position", None)
        position, names = normalize_named_trajectory(
            positions,
            getattr(path, "joint_names", None),
            self.robot_port.tensor_args,
        )
        state = JointState.from_position(position, joint_names=names)
        return state.reorder(self.planner_names)

    def joint_state_from_path_endpoint(self, path):
        """Convert one normalized trajectory endpoint to a planner JointState."""

        from curobo.types import JointState

        trajectory = self.joint_state_from_trajectory(path)
        endpoint = trajectory.position[-1]
        return JointState.from_position(
            endpoint,
            joint_names=list(trajectory.joint_names),
        )

    def batch_start_state_from_paths(self, paths):
        from curobo.types import JointState

        states = []
        for path in paths:
            if path is None:
                states.append(self.arm_joint_state(self.robot_port.robot.get_joints_state()))
                continue
            states.append(self.joint_state_from_path_endpoint(path))
        positions = self.robot_port.tensor_args.to_device(np.stack([state.position.detach().cpu().numpy() for state in states]))
        return JointState.from_position(positions, joint_names=self.planner_names)

    def attach_native(
        self,
        paths,
        joint_state,
        meshes,
        *,
        link_name="attached_object",
        world_objects_pose_offset=None,
        planner=None,
    ):
        from curobo.sphere_fit import SphereFitType

        planner = planner or self.native_planner
        if not paths:
            raise ValueError("native attachment requires non-empty obstacle paths")
        if joint_state is None:
            joint_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        if meshes is None:
            meshes, world_objects_pose_offset = self._attachment_geometry(paths)
        planner.attachment_manager.attach(
            joint_state,
            meshes,
            link_name=link_name,
            num_spheres=max(1, self._attached_sphere_count(link_name, 1, planner=planner)),
            surface_radius=0.001,
            sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=world_objects_pose_offset,
            disable_obstacle_names=paths,
        )
        return paths

    def _attached_sphere_count(self, link_name, object_count, *, planner=None):
        planner = planner or self.native_planner
        total = planner.kinematics.config.kinematics_config.get_number_of_spheres(
            link_name
        )
        return max(1, int(total) // max(1, int(object_count)))

    def _attachment_geometry(self, object_names):
        """Build local meshes and the current pose for native attachment."""

        from curobo.types import Pose
        from core.planning.native_bridge import Mesh

        obstacles = []
        current_poses = []
        for object_name in object_names:
            try:
                obstacle = self.native_scene_adapter.get_obstacle_geometry(object_name)
            except Exception as exc:
                raise ValueError(
                    f"attach collision prim is not in native scene model: {object_name}"
                ) from exc
            obstacles.append(obstacle)
            if (
                self.robot_port.obstacle_pose is not None
                and self.robot_port.collision_world_mode == "physics_schema"
            ):
                current_poses.append(self.robot_port.obstacle_pose(object_name))
            else:
                pose = getattr(obstacle, "pose", None)
                if pose is None:
                    raise ValueError(f"native obstacle has no pose: {object_name}")
                current_poses.append(
                    Pose.from_list(list(pose), device_cfg=self.robot_port.tensor_args)
                )

        anchor_pose = current_poses[0]
        anchor_inverse = np.linalg.inv(anchor_pose.get_numpy_matrix()[0])
        vertices = []
        faces = []
        vertex_offset = 0
        for obstacle, current_pose in zip(obstacles, current_poses):
            mesh = obstacle.get_trimesh_mesh(transform_with_pose=False)
            local_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                raise ValueError(
                    f"native attachment mesh has invalid vertices: {obstacle.name}"
                )
            object_matrix = current_pose.get_numpy_matrix()[0]
            world_vertices = (
                object_matrix[:3, :3] @ local_vertices.T
            ).T + object_matrix[:3, 3]
            anchor_vertices = (
                anchor_inverse[:3, :3] @ world_vertices.T
            ).T + anchor_inverse[:3, 3]
            obstacle_faces = np.asarray(mesh.faces, dtype=np.int64)
            if obstacle_faces.ndim != 2 or obstacle_faces.shape[1] != 3:
                raise ValueError(
                    f"native attachment mesh has invalid faces: {obstacle.name}"
                )
            vertices.append(anchor_vertices)
            faces.append(obstacle_faces + vertex_offset)
            vertex_offset += anchor_vertices.shape[0]

        if not vertices:
            raise ValueError("native attachment requires at least one mesh")
        return [
            Mesh(
                name="__native_attached_object__",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=np.concatenate(vertices, axis=0),
                faces=np.concatenate(faces, axis=0),
            )
        ], anchor_pose

    def execute(self, command):
        return self.execution_port.forward_phase_command(command)

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        """Forward one legacy direct joint action to the articulation owner."""

        return self.execution_port.dummy_forward(
            arm_action, gripper_state, *args, **kwargs
        )

    def execution_status(self, command=None) -> ExecutionStatus:
        """Expose detailed execution state without leaking planner storage."""

        return self.execution_port.execution_status(command)

    def command_status(self, command=None) -> CommandStatus:
        """Expose only the finite public command-status enum."""

        return self.execution_status(command).status

    def hold(self):
        return self.execution_port.hold_action()

    def attach_object(self, spec: AttachmentSpec):
        return self.attachment_runtime.attach(spec)

    def detach_object(self):
        return self.attachment_runtime.detach()

    def reset_attachments(self) -> None:
        """Detach execution and candidate attachments through their owners."""

        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        self.attachment_runtime.detach()

    def destroy(self) -> None:
        """Release planner resources and detach all scene subscriptions."""

        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        self.attachment_runtime.detach()
        self.planner_runtime.unregister_scene_adapter(self.native_scene_adapter)
        # PlannerRuntime owns the native single/batch instances and its scene
        # subscription.  Clear the lightweight scene registry as well so a
        # runtime that is retained by a test/worker cannot fan out updates
        # after native teardown.
        try:
            self.planner_runtime.destroy()
        finally:
            self.scene_runtime.clear()

    close = destroy

    def __enter__(self) -> "MotionPlannerRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


__all__ = [
    "ExecutionPort",
    "MotionPlannerRuntime",
    "PlanningSceneRuntime",
    "RobotPort",
]
