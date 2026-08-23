"""Isaac controller with a legacy lane and a Pick/Place Physics-schema lane."""

from __future__ import annotations
import logging
import os
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from curobo.types import DeviceCfg
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask

from core.controllers.controller_registry import ArmSpec
from core.controllers.curobo.components import MutableExecutionState, PlanningConfig
from core.controllers.curobo.execution import ControllerExecution
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.controllers.curobo.runtime import MotionPlannerRuntime, RobotPort
from core.controllers.curobo.scene_setup import ControllerSetup
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
    """Own lifecycle assembly and the single public legacy dispatcher."""
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
        self._timing_scope = None
        self.batch_capability = bool(batch_capability)
        self.constrain_grasp_approach = bool(constrain_grasp_approach)
        self.collision_activation_distance = float(collision_activation_distance)
        self.tensor_args = DeviceCfg(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )
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
        debug_cfg = self.task.cfg.get("planning", {}).get("debug", {})
        if not isinstance(debug_cfg, Mapping):
            debug_cfg = {}
        debug_plan_json = bool(
            debug_cfg.get("curobo_plan_json", False)
            or os.environ.get("SIMBOX_CUROBO_PLAN_DEBUG", "").lower()
            in {"1", "true", "yes", "on"}
        )
        debug_plan_dir = os.environ.get(
            "SIMBOX_CUROBO_PLAN_DEBUG_DIR",
            os.path.join("output", "local_navigation", "skills", "curobo_plan_debug"),
        )
        self._setup = ControllerSetup(
            name=self.name,
            world=self.world,
            task=self.task,
            robot=self.robot,
            arm_spec=self.arm_spec,
            robot_file=self.robot_file,
            tensor_args=self.tensor_args,
            phase_executor=self.phase_executor,
            execution_state=self.execution_state,
            planning_config=self.planning_config,
            trajectory_visualizer=self.trajectory_visualizer,
            collision_scene_manager=self.collision_scene_manager,
            batch_capability=self.batch_capability,
            debug_plan_json=debug_plan_json,
            debug_plan_dir=debug_plan_dir,
        )
        self._prepare_setup(self._setup)
        self._execution = ControllerExecution(
            name=self._setup.name,
            lr_name=self._setup.lr_name,
            robot=self.robot,
            arm_spec=self.arm_spec,
            tensor_args=self.tensor_args,
            raw_js_names=self._setup.raw_js_names,
            arm_indices=self._setup.arm_indices,
            gripper_indices=self._setup.gripper_indices,
            phase_executor=self.phase_executor,
            setup=self._setup,
            robot_base_path=self._setup.robot_base_path,
            robot_ee_path=self._setup.robot_ee_path,
            task_root_prim_path=self._setup.task_root_prim_path,
            reference_prim_path=self._setup.reference_prim_path,
            pick_mobile_base_prim_path=self._pick_mobile_base_prim_path,
            pick_cached_mobile_to_armbase_tf=self._pick_cached_mobile_to_armbase_tf,
            pick_configured_mobile_to_armbase_translation=(
                self._pick_configured_mobile_to_armbase_translation
            ),
            pick_configured_mobile_to_armbase_orientation=(
                self._pick_configured_mobile_to_armbase_orientation
            ),
            execution_state=self.execution_state,
            planning_config=self.planning_config,
        )
        self._setup._get_ee_pose = self._execution.get_ee_pose
        planning_world = self._setup._load_world()
        self.runtime = self._build_runtime(planning_world)
        # A zero-candidate batch is a controller failure, not a skill-level
        # planning concern.  Let the runtime reuse the native single-planner
        # collision audit on that failure without exposing native objects to
        # Pick/Place.
        self._setup.runtime = self.runtime
        self._execution.runtime = self.runtime
        self.runtime.setup = self._setup
        self.runtime.execution_state = self.execution_state
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
                name=self.name,
                lr_name=self._setup.lr_name,
                reference_prim_path=self._setup.reference_prim_path,
                robot_ee_path=self._setup.robot_ee_path,
                tensor_args=self.tensor_args,
                robot=self.robot,
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
                # The runtime receives the strict target-batch scene hook
                # explicitly after the scene port exists.
                self.runtime._require_batch_scene_adapter = self._require_batch_scene_adapter
            self.runtime.robot_port.obstacle_pose = lambda path: self.collision_scene_manager._port_obstacle_pose(self._scene_port, path)
            self._setup.scene_port = self._scene_port
            self.collision_scene_manager.bind_scene_port(self._scene_port)
        # Skills receive one narrow, immutable runtime view after scene binding.
        # The port binds the concrete runtime and execution owners directly;
        # it does not retain a callback-built controller façade.
        self.skill_runtime = SkillRuntimePort(
            robot=self.robot,
            runtime=self.runtime,
            execution=self._execution,
            execution_state=self.execution_state,
            arm_indices=self._setup.arm_indices,
            gripper_indices=self._setup.gripper_indices,
            robot_file=self.robot_file,
            robot_config=getattr(self.robot, "cfg", {}),
            robot_base_path=self._setup.robot_base_path,
            robot_ee_path=self._setup.robot_ee_path,
            reference_prim_path=self._setup.reference_prim_path,
            name=self.name,
            arm_name=self._setup.lr_name,
            batch_capability=self.batch_capability,
            interpolation_dt=self.interpolation_dt,
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
        """Resolve robot joints and frame paths before runtime construction."""

        setup._configure_joint_indices(self.robot_file)
        setup._resolve_runtime_control_indices()

    def _build_runtime(self, planning_world: Any) -> MotionPlannerRuntime:
        return MotionPlannerRuntime(
            PlannerBuildConfig(
                robot_file=self.robot_file,
                tensor_args=self.tensor_args,
                collision_activation_distance=self.collision_activation_distance,
            ),
            RobotPort(
                name=self.name,
                lr_name=self._setup.lr_name,
                task_cfg=self.task.cfg,
                robot=self.robot,
                tensor_args=self.tensor_args,
                arm_spec=self.arm_spec,
                arm_indices=self._setup.arm_indices,
                raw_js_names=self._setup.raw_js_names,
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

    def forward(self, command, eps=5e-3):
        """Dispatch typed Pick/Place commands or the legacy tuple lane.

        ``MotionPhaseCommand`` is reserved for the Physics-schema path.  A
        tuple is accepted only for the historical non-Pick/Place controller
        contract and is decoded here, at the single public legacy boundary.
        No tuple parsing is performed by the Pick/Place runtime port.
        """

        if isinstance(command, MotionPhaseCommand):
            return self.execute(command)
        if not isinstance(command, (tuple, list)) or len(command) < 4:
            raise TypeError(
                "TemplateController.forward accepts MotionPhaseCommand or "
                "a legacy (ee_position, ee_orientation, method, params) tuple"
            )

        ee_trans, ee_ori, method_name, raw_params = command[:4]
        if not isinstance(method_name, str):
            raise TypeError("legacy controller command method must be a string")
        if raw_params is None:
            params = {}
        elif isinstance(raw_params, dict):
            params = dict(raw_params)
        else:
            try:
                params = dict(raw_params)
            except (TypeError, ValueError) as exc:
                raise TypeError("legacy controller command params must be a mapping") from exc

        self._execution._last_command_name = method_name
        skip_plan = bool(params.pop("skip_plan", False))
        gripper_action = params.pop("gripper_action", None)
        if isinstance(gripper_action, str) and gripper_action in {
            "open_gripper",
            "close_gripper",
        }:
            self._execution._apply_gripper_action(gripper_action)
            gripper_action = None
        forward_eps = float(params.pop("eps", eps))
        params.pop("t_eps", None)
        params.pop("o_eps", None)
        method = getattr(self, method_name, None)
        legacy_direct_methods = {
            "in_plane_rotation",
            "mobile_move",
            "dummy_forward",
            "observe_hold",
        }
        if method_name in legacy_direct_methods:
            if not callable(method):
                raise AttributeError(f"unknown legacy controller operation: {method_name}")
            return method(**params)

        if method_name == "pre_forward":
            return self._execution.pre_forward(ee_trans, ee_ori, **params)

        if method_name == "ee_forward":
            return self._execution.legacy_ee_forward(
                ee_trans,
                ee_ori,
                command_name=method_name,
                eps=forward_eps,
                skip_plan=skip_plan,
                gripper_action=gripper_action,
            )

        if method_name == "update_specific":
            if not callable(method):
                raise AttributeError(f"unknown legacy controller operation: {method_name}")
            method(**params)
            return self._execution.legacy_ee_forward(
                ee_trans,
                ee_ori,
                command_name=method_name,
                eps=forward_eps,
                skip_plan=True,
                gripper_action=gripper_action,
            )

        if method_name not in {"open_gripper", "close_gripper"} or not callable(method):
            raise AttributeError(f"unknown legacy controller operation: {method_name}")
        method(**params)
        return self._execution.legacy_ee_forward(
            ee_trans,
            ee_ori,
            command_name=method_name,
            eps=forward_eps,
            skip_plan=skip_plan,
            gripper_action=gripper_action,
        )

    # The following methods are deliberately thin legacy-lane delegates.  No
    # Pick/Place code receives the controller object, and these do not expose
    # native planners or scene-manager internals to SkillRuntimePort.
    def ee_forward(self, *args, **kwargs):
        return self._execution.legacy_ee_forward(*args, **kwargs)

    def pre_forward(self, *args, **kwargs):
        return self._execution.pre_forward(*args, **kwargs)

    def in_plane_rotation(self, *args, **kwargs):
        return self._execution.in_plane_rotation(*args, **kwargs)

    def mobile_move(self, *args, **kwargs):
        return self._execution.mobile_move(*args, **kwargs)

    def observe_hold(self):
        return self._execution.observe_hold()

    def update_specific(self, *args, **kwargs):
        return self._setup.update_specific(*args, **kwargs)

    def open_gripper(self):
        return self._execution.open_gripper()

    def close_gripper(self):
        return self._execution.close_gripper()

    def get_ee_pose(self):
        return self._execution.get_ee_pose()

    def get_armbase_pose(self):
        return self._execution.get_armbase_pose()

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
