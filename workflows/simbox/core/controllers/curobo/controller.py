from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
import torch
from curobo.types import DeviceCfg
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask
from core.controllers.controller_registry import ArmSpec
from core.controllers.curobo.components import MutableExecutionState
from core.execution.curobo_execution import ControllerExecution
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.controllers.curobo.runtime import MotionPlannerRuntime, RobotPort
from core.controllers.curobo.scene_setup import ControllerSetup
from core.planning.motion_command import MotionPhaseCommand
from core.planning.native_planner_factory import PlannerBuildConfig
from core.planning.collision_scene_manager import PlannerScenePort
class _TypedIsaacController(BaseController):
    @abstractmethod
    def execute(self, command: MotionPhaseCommand):
        raise NotImplementedError
    def forward(self, command: MotionPhaseCommand):
        return self.execute(command)
class TemplateController(_TypedIsaacController):
    arm_spec: ArmSpec | None = None
    def __init__(
        self,
        name: str,
        robot_file: str,
        task: BaseTask,
        world: World,
        constrain_grasp_approach: bool = False,
        collision_activation_distance: float = 0.03,
        collision_scene_manager=None,
        ignore_substring=None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(name=name)
        self.name = name
        self.robot_file = robot_file
        self.task = task
        self.world = world
        self.robot = task.robots[name]
        self.execution_state = MutableExecutionState()
        self.phase_executor = PhaseExecutor()
        self.collision_scene_manager = collision_scene_manager
        self.constrain_grasp_approach = bool(constrain_grasp_approach)
        self.collision_activation_distance = float(collision_activation_distance)
        configured_ignore = getattr(self.arm_spec, "default_ignore_substring", ())
        if ignore_substring is not None:
            if isinstance(ignore_substring, (str, bytes)) or not isinstance(
                ignore_substring, Sequence
            ):
                raise TypeError("ignore_substring must be a YAML list of strings")
            configured_ignore = ignore_substring
        if any(not isinstance(value, str) or not value for value in configured_ignore):
            raise TypeError("ignore_substring must contain non-empty strings")
        self.ignore_substring = tuple(dict.fromkeys((*configured_ignore, self.name)))
        self.tensor_args = DeviceCfg(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )
        self.runtime = None
        self.skill_runtime = None
        self.interpolation_dt = 0.01
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
            collision_scene_manager=self.collision_scene_manager,
            ignore_substring=self.ignore_substring,
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
            execution_state=self.execution_state,
        )
        planning_world = self._setup._load_world()
        self.runtime = self._build_runtime(planning_world)
        self.interpolation_dt = self.runtime.robot_port.interpolation_dt
        self._setup.interpolation_dt = self.interpolation_dt
        self._setup._configure_execution_stride()
        if self.collision_scene_manager is not None:
            self._scene_port = PlannerScenePort(
                name=self.name,
                lr_name=self._setup.lr_name,
                reference_prim_path=self._setup.reference_prim_path,
                robot_ee_path=self._setup.robot_ee_path,
                tensor_args=self.tensor_args,
                robot=self.robot,
                runtime=self.runtime,
                check_current_start_state=self.runtime.check_current_start_state,
                attach_collision_object=self.runtime.attach_collision_object,
                detach_attachment=self.runtime.detach_attachment,
                has_attached_collision_spheres=self.runtime.has_attached_collision_spheres,
                ignore_substring=self.ignore_substring,
            )
            self.runtime.robot_port.obstacle_pose = lambda path: self.collision_scene_manager._port_obstacle_pose(self._scene_port, path)
            self._setup.scene_port = self._scene_port
            self.runtime.scene_port = self._scene_port
            self.collision_scene_manager.bind_scene_port(self._scene_port)
        self.skill_runtime = self.runtime
    def _prepare_setup(self, setup) -> None:
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
                constrain_grasp_approach=self.constrain_grasp_approach,
                collision_scene_manager=self.collision_scene_manager,
            ),
            world=planning_world,
            phase_executor=self.phase_executor,
            execution_state=self.execution_state,
            setup=self._setup,
            execution=self._execution,
        )
    def reset(self):
        return self._setup.reset(self.runtime, self._execution.get_ee_pose)
    def execute(self, command: MotionPhaseCommand):
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("TemplateController.execute accepts MotionPhaseCommand only")
        return self._execution.forward_phase_command(command)
