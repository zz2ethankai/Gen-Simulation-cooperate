"""Small controller façade composed from explicitly wired operation ports."""

from __future__ import annotations
import logging
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from curobo.types import DeviceCfg
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask

from core.controllers.controller_registry import ArmSpec
from core.controllers.curobo.components import (
    ComponentPort,
    MutableExecutionState,
    PlanningConfig,
    wire_controller_components,
)
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.controllers.curobo.runtime import MotionPlannerRuntime, RobotPort
from core.controllers.curobo.skill_runtime import SkillRuntimePort
from core.planning.motion_command import MotionPhaseCommand
from core.planning.native_planner_factory import PlannerBuildConfig
from core.planning.collision_scene_manager import PlannerScenePort
LOGGER = logging.getLogger("de_logger")


class _TypedIsaacController(BaseController):
    """Bridge Isaac's lifecycle hook to the typed SimBox command contract.

    Isaac's ``BaseController`` requires ``forward`` for its articulation
    controller interface.  The structured operation is ``execute`` with a
    :class:`MotionPhaseCommand`; the separate public ``dummy_forward`` method
    is retained for Skills that own direct joint interpolation.  Keeping this
    adapter private lets robot subclasses remain configuration-only while
    still producing concrete classes for Isaac's ABC.
    """

    @abstractmethod
    def execute(self, command: MotionPhaseCommand):
        """Execute one typed motion command."""

    def forward(self, command: MotionPhaseCommand):
        """Adapt Isaac's hook for structured commands."""

        return self.execute(command)


class TemplateController(_TypedIsaacController):
    """Public typed planning façade; native planners live in ``PlannerRuntime``."""
    arm_spec: ArmSpec | None = None

    def __init__(
        self,
        name: str,
        robot_file: str,
        task: BaseTask,
        world: World,
        constrain_grasp_approach: bool = False,
        collision_activation_distance: float = 0.03,
        batch_capability: bool = False,
        trajectory_visualizer=None,
        skill_target_visualizer=None,
        collision_scene_manager=None,
        timing_recorder=None,
        **kwargs: Any,
    ) -> None:
        del kwargs, skill_target_visualizer
        super().__init__(name=name)
        self.name = name
        self.robot_file = robot_file
        self.task = task
        self.world = world
        self.robot = task.robots[name]
        self.execution_state = MutableExecutionState()
        self.planning_config = PlanningConfig()
        self.phase_executor = PhaseExecutor()
        self.trajectory_visualizer = trajectory_visualizer
        self.collision_scene_manager = collision_scene_manager
        self.timing_recorder = timing_recorder
        self._timing_scope = None
        self.batch_capability = bool(batch_capability)
        self.batch_enabled = self.batch_capability
        self.constrain_grasp_approach = bool(constrain_grasp_approach)
        self.collision_activation_distance = float(collision_activation_distance)
        self.tensor_args = DeviceCfg(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )
        self.raw_js_names: list[str] = []
        self.cmd_js_names: list[str] = []
        self.arm_indices = np.array([], dtype=np.int64)
        self.gripper_indices = np.array([], dtype=np.int64)
        self.reference_prim_path = None
        self.robot_base_path = None
        self.robot_ee_path = None
        self.lr_name = None
        self._gripper_state = 1.0
        self._gripper_joint_position = np.array([1.0])
        self._pick_mobile_base_prim_path = getattr(self.robot, "mobile_base_prim_path", None)
        self._pick_cached_mobile_to_armbase_tf = None
        mount_prefix = "fr" if "right" in robot_file else "fl"
        self._pick_configured_mobile_to_armbase_translation = np.asarray(
            self.robot.cfg.get(f"{mount_prefix}_base_mount_translation", []), dtype=np.float32
        )
        self._pick_configured_mobile_to_armbase_orientation = np.asarray(
            self.robot.cfg.get(
                f"{mount_prefix}_base_mount_orientation", [1.0, 0.0, 0.0, 0.0]
            ),
            dtype=np.float32,
        )
        self.runtime = None
        self.skill_runtime = None
        self.interpolation_dt = 0.01
        self._curobo_plan_debug_counter = 0
        self._curobo_plan_debug_dir = os.environ.get(
            "SIMBOX_CUROBO_PLAN_DEBUG_DIR",
            os.path.join("output", "local_navigation", "skills", "curobo_plan_debug"),
        )
        components = wire_controller_components(
            self._port,
            prepare_setup=self._prepare_setup,
        )
        self._setup = components.setup
        self._execution = components.execution
        for name in (
            "raw_js_names", "cmd_js_names", "arm_indices", "gripper_indices",
            "reference_prim_path", "robot_base_path", "robot_ee_path", "lr_name",
            "_gripper_state", "_gripper_joint_position",
        ):
            if hasattr(self._setup, name):
                setattr(self, name, getattr(self._setup, name))
        planning_world = self._setup._load_world()
        self.runtime = self._build_runtime(planning_world)
        # A zero-candidate batch is a controller failure, not a skill-level
        # planning concern.  Let the runtime reuse the native single-planner
        # collision audit on that failure without exposing native objects to
        # Pick/Place.
        for component in (self._setup, self._execution):
            component.runtime = self.runtime
            component.phase_executor = self.phase_executor
        self.runtime.setup = self._setup
        self.runtime.execution_state = self.execution_state
        self._execution.setup = self._setup
        self.interpolation_dt = self.runtime.robot_port.interpolation_dt
        self._setup.interpolation_dt = self.interpolation_dt
        self._setup._configure_execution_stride()
        if self.collision_scene_manager is not None:
            require_batch_scene_adapter = getattr(
                self.collision_scene_manager,
                "require_batch_scene_adapter",
                None,
            )
            self._scene_port = PlannerScenePort(
                name=self.name, lr_name=self.lr_name, reference_prim_path=self.reference_prim_path,
                robot_ee_path=self.robot_ee_path, tensor_args=self.tensor_args, robot=self.robot,
                runtime=self.runtime, check_current_start_state=self.runtime.check_current_start_state,
                native_scene_adapter=self.runtime.native_scene_adapter,
                adopt_scene_revision=self.runtime.adopt_scene_revision,
                attach_collision_object=self.runtime.attach_collision_object,
                detach_attachment=self.runtime.detach_attachment,
                has_attached_collision_spheres=self.runtime.has_attached_collision_spheres,
                require_batch_scene_adapter=require_batch_scene_adapter)
            if callable(require_batch_scene_adapter):
                self._require_batch_scene_adapter = (
                    lambda: require_batch_scene_adapter(self._scene_port)
                )
                # The runtime receives the strict target-batch façade
                # explicitly after the scene port exists.
                self.runtime._require_batch_scene_adapter = self._require_batch_scene_adapter
            self.runtime.robot_port.obstacle_pose = lambda path: self.collision_scene_manager._port_obstacle_pose(self._scene_port, path)
            self._setup.scene_port = self._scene_port
            self.collision_scene_manager.bind_scene_port(self._scene_port)
        # Skills receive one narrow, immutable runtime view after scene binding.
        # The port binds the two concrete components directly; it does not
        # retain a callback-built controller façade.
        self.skill_runtime = SkillRuntimePort(
            robot=self.robot,
            runtime=self.runtime,
            execution=self._execution,
            execution_state=self.execution_state,
            arm_indices=self.arm_indices,
            gripper_indices=self.gripper_indices,
            robot_file=self.robot_file,
            robot_config=getattr(self.robot, "cfg", {}),
            robot_base_path=self.robot_base_path,
            robot_ee_path=self.robot_ee_path,
            reference_prim_path=self.reference_prim_path,
            name=self.name,
            arm_name=self.lr_name,
            batch_capability=self.batch_capability,
            interpolation_dt=self.interpolation_dt,
            collision_scene_manager=self.collision_scene_manager,
            timing_owner=self,
        )

    def bind_timing_scope(self, scope):
        """Bind the workflow-owned scope used by this controller runtime."""

        self._timing_scope = scope
        return scope

    def push_timing_scope(self, scope):
        """Temporarily bind one Skill timing scope to the runtime callbacks."""

        previous = self._timing_scope
        self._timing_scope = scope
        return previous

    def restore_timing_scope(self, previous):
        """Restore the scope active before a Skill planner call."""

        self._timing_scope = previous

    def clear_timing_scope(self, scope=None):
        """Clear only the scope owned by the completed Skill invocation."""

        if scope is None or self._timing_scope is scope:
            self._timing_scope = None

    def _prepare_setup(self, setup) -> None:
        """Resolve robot joints and frame paths before sibling port wiring."""

        setup._configure_joint_indices(self.robot_file)
        setup._resolve_runtime_control_indices()

    def _port(self, names: tuple[str, ...], *, port_type=ComponentPort, **overrides: Any) -> ComponentPort:
        # Component wiring may provide a value from an already-created owner
        # (notably ``ControllerSetup.task_root_prim_path``) before the façade
        # has a same-named attribute.  Prefer those explicit port values and
        # only read the façade for construction-time inputs that have no
        # component owner yet.  Do not restore aliases on TemplateController.
        values = {
            name: getattr(self, name)
            for name in names
            if name not in overrides
        }
        values.update(overrides)
        return port_type(values)
    def _build_runtime(self, planning_world: Any) -> MotionPlannerRuntime:
        return MotionPlannerRuntime(
            PlannerBuildConfig(
                robot_file=self.robot_file,
                tensor_args=self.tensor_args,
                collision_activation_distance=self.collision_activation_distance,
            ),
            RobotPort(
                name=self.name,
                lr_name=self.lr_name,
                task_cfg=self.task.cfg,
                robot=self.robot,
                tensor_args=self.tensor_args,
                arm_spec=self.arm_spec,
                arm_indices=self.arm_indices,
                raw_js_names=self.raw_js_names,
                batch_capability=self.batch_capability,
                constrain_grasp_approach=self.constrain_grasp_approach,
                collision_scene_manager=self.collision_scene_manager,
            ),
            world=planning_world,
            phase_executor=self.phase_executor,
            execution_state=self.execution_state,
            setup=self._setup,
        )
    def reset(self):
        return self._setup.reset()
    def execute(self, command: MotionPhaseCommand):
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("TemplateController.execute accepts MotionPhaseCommand only")
        return self._execution.forward_phase_command(command)
    def dummy_forward(self, arm_action, gripper_state):
        """Public direct-joint interface for Skills that own interpolation."""

        return self._execution.dummy_forward(arm_action, gripper_state)
    def execution_status(self, command: Any = None):
        return self._execution.execution_status(command)
    def hold(self):
        return self._execution.hold_action()
