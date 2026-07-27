import glob
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

import numpy as np
import yaml
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.physx import acquire_physx_interface
from tqdm import tqdm
from yaml import Loader

from deps.world_toolkit.world_recorder import WorldRecorder
from workflows.simbox.utils.task_config_parser import TaskConfigParser

from .base import NimbusWorkFlow
from core.controllers import get_controller_cls
from core.execution.safety_monitor import (
    SafetyDecision,
    SafetyMeasurements,
    SafetyMonitor,
    quaternion_angle,
)
from core.execution.execution_supervisor import ExecutionSupervisor
from core.planning.config_contract import validate_planning_contract
from .simbox.core.utils.camera_utils import capture_topdown_screenshot
from core.loggers.lmdb_logger import LmdbLogger
from core.planning.collision_scene_manager import (
    CollisionObjectState,
    CollisionSceneError,
    CollisionSceneManager,
)
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.loggers.utils import log_dual_obs
from core.skills import get_skill_cls
from core.tasks import get_task_cls
from core.utils.collision_utils import filter_collisions
from core.utils.episode_event_writer import emit_episode_saved
from core.utils.utils import set_random_seed
from core.visualization.curobo_trajectory import create_curobo_trajectory_visualizer
from core.visualization.skill_targets import create_skill_target_visualizer


LOGGER = logging.getLogger("de_logger")


# pylint: disable=unused-argument
@NimbusWorkFlow.register("SimBoxDualWorkFlow")
class SimBoxDualWorkFlow(NimbusWorkFlow):
    def __init__(
        self,
        world,
        task_cfg_path: str,
        scene_info: str = "dining_room_scene_info",
        random_seed: int = None,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        super().__init__(world, task_cfg_path)

    def parse_task_cfgs(self, task_cfg_path: str) -> list:
        task_cfgs = TaskConfigParser(task_cfg_path).parse_tasks()
        # Merge robot configs for each task
        for task_cfg in task_cfgs:
            self._merge_robot_configs(task_cfg)
        return task_cfgs

    def _merge_robot_configs(self, task_cfg: dict):
        """Merge robot configs from robot_config_file into task_cfg['robots']."""
        robots = task_cfg.get("robots", [])

        for robot in robots:
            robot_config_file = robot.get("robot_config_file")
            if robot_config_file:
                with open(robot_config_file, "r", encoding="utf-8") as f:
                    robot_base_cfg = yaml.load(f, Loader=Loader)

                # Merge: robot_base_cfg as base, task_cfg['robots'][i] overrides
                merged_cfg = deepcopy(robot_base_cfg)
                merged_cfg.update(robot)
                robot.clear()
                robot.update(merged_cfg)

    def reset(self, need_preload: bool = True):
        self.close()
        # source code noted this as debug, so it could be removed later
        from omni.isaac.core.utils.viewports import set_camera_view

        set_camera_view(eye=[1.3, 0.7, 2.7], target=[0.0, 0, 1.5], camera_prim_path="/OmniverseKit_Persp")
        # Modify config
        arena_file_path = self.task_cfg.get("arena_file", None)
        if arena_file_path is None:
            arena_file_path = getattr(self, "_saved_arena_file", None)
        else:
            self._saved_arena_file = arena_file_path
        if arena_file_path is None:
            raise FileNotFoundError(
                f"arena_file not found in task_cfg. Keys: {list(self.task_cfg.keys())}"
            )
        with open(arena_file_path, "r", encoding="utf-8") as arena_file:
            arena = yaml.load(arena_file, Loader=Loader)

        # if "involved_scenes" in arena:
        #     arena["involved_scenes"] = self.scene_info

        self.task_cfg["arena"] = arena

        for obj_cfg in self.task_cfg["objects"]:
            if obj_cfg["target_class"] == "ArticulatedObject":
                if obj_cfg.get("apply_randomization", False):
                    asset_root = self.task_cfg["asset_root"]
                    art_paths = glob.glob(os.path.join(asset_root, obj_cfg["art_cat"], "*"))
                    art_paths.sort()
                    path = random.choice(art_paths)
                    info_name = obj_cfg["info_name"]
                    info_path = f"{path}/Kps/{info_name}/info.json"
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    scale = info["object_scale"][:3]

                    obj_cfg["path"] = path.replace(f"{asset_root}/", "", 1) + "/instance.usd"
                    obj_cfg["category"] = path.split("/")[-2]
                    obj_cfg["obj_info_path"] = info_path.replace(f"{asset_root}/", "", 1)
                    obj_cfg["scale"] = scale
                    self.task_cfg["data"]["collect_info"] = obj_cfg["category"]

        self.task_cfg.pop("arena_file", None)
        self.task_cfg.pop("camera_file", None)
        self.task_cfg.pop("logger_file", None)
        # Modify config done
        if self.task_cfg.get("fluid", None):
            # for fluid manipulation, only gpu mode is supportive
            physx_interface = acquire_physx_interface()
            physx_interface.overwrite_gpu_setting(1)

        self.task = get_task_cls(self.task_cfg["task"])(self.task_cfg)
        self.stage = self.world.stage
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))

        task_name = self.task.name
        if hasattr(self.world, '_current_tasks') and task_name in self.world._current_tasks:
            self.world._current_tasks.pop(task_name)

        root_prim = self.stage.GetPrimAtPath(self.task.root_prim_path)
        if root_prim.IsValid():
            self.stage.RemovePrim(self.task.root_prim_path)
        collision_prim = self.stage.GetPrimAtPath("/World/collisions")
        if collision_prim.IsValid():
            self.stage.RemovePrim("/World/collisions")

        self.world.add_task(self.task)

        # # Add hidden ground plane for physics simulation
        # from omni.isaac.core.objects import GroundPlane
        # plane = GroundPlane(
        #     prim_path="/World/GroundPlane",
        #     z_position=0.0,
        #     visible=False,
        # )

        prim_paths = []  # do not collide with each other
        global_collision_paths = []  # collide with everything

        self.robots_prim_paths = []
        for robot in self.task_cfg["robots"]:
            robot_prim_path = self.task.root_prim_path + "/" + robot["name"]
            prim_paths.append(robot_prim_path)
            self.robots_prim_paths.append(robot_prim_path)
        neglect_collision_names = self.task_cfg.get("neglect_collision_names", [])
        candidates = self.task_cfg["objects"] + self.task_cfg["arena"]["fixtures"]
        for candidate in candidates:
            candidate_prim_path = self.task.root_prim_path + "/" + candidate["name"]
            global_collision_paths.append(candidate_prim_path)
            for neglect_collision_name in neglect_collision_names:
                if neglect_collision_name in candidate["name"]:
                    prim_paths.append(candidate_prim_path)
                    global_collision_paths.remove(candidate_prim_path)

        collision_root_path = "/World/collisions"
        filter_collisions(
            self.stage,
            self.world.get_physics_context().prim_path,
            collision_root_path,
            prim_paths,
            global_collision_paths,
        )
        planning_cfg = self.task_cfg.get("planning", {})
        collision_cfg = planning_cfg.get("collision_world", {})
        safety_cfg = planning_cfg.get("execution_safety", {})
        self.collision_world_mode = str(collision_cfg.get("mode", "physics_schema"))
        self._validate_planning_contract(self.task_cfg, self.collision_world_mode)
        self.world.reset()
        # Base hold must be active before the very first Physics step. The
        # delivered mobile-base USD otherwise has zero-gain planar joints and
        # can move while CuRobo/controller initialization is still running.
        self._enable_manipulation_base_holds()
        self.world.step(render=True)
        if self.collision_world_mode == "physics_schema":
            self.collision_scene_manager = CollisionSceneManager(
                self.stage, self.task, collision_cfg, safety_cfg
            )
        elif self.collision_world_mode == "legacy_stage_scan":
            self.collision_scene_manager = None
        else:
            raise ValueError(
                f"unsupported planning.collision_world.mode: {self.collision_world_mode!r}"
            )
        self.execution_safety_enabled = bool(
            safety_cfg.get("enabled", self.collision_world_mode == "physics_schema")
        )
        LOGGER.warning(
            "[ExecutionSafety] initialized enabled=%s collision_world_mode=%s manager=%s",
            self.execution_safety_enabled,
            self.collision_world_mode,
            self.collision_scene_manager is not None,
        )
        self.safety_monitor = SafetyMonitor(safety_cfg)
        self.execution_supervisor = ExecutionSupervisor(self.safety_monitor, safety_cfg)
        self._safety_failure_reason = ""
        self._safety_abort_requested = False
        self._active_execution_step_id = -1
        self.trajectory_visualizer = create_curobo_trajectory_visualizer(
            self.stage,
            self.task.root_prim_path,
            self.task_cfg,
        )
        self.skill_target_visualizer = create_skill_target_visualizer(
            self.stage,
            self.task.root_prim_path,
            self.task_cfg,
        )
        self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.initialize_contact_views()
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        for _ in range(50):
            self._init_static_objects(self.task)
            self.world.step(render=False)
        if self.task_cfg.get("debug_topdown_check") or os.environ.get("INTERNDATA_DEBUG_TOPDOWN") == "1":
            capture_topdown_screenshot(
                self.task_cfg["data"]["task_dir"],
                self.world,
                task_cameras=getattr(self.task, "cameras", None),
            )
        self.logger = LmdbLogger(
            task_dir=self.task_cfg["data"]["task_dir"],
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
            collect_info=self.task_cfg["data"]["collect_info"],
            version=self.task_cfg["data"].get("version", "v1.0"),
        )
        # Motion vectors are large dense tensors; keep LMDB logging opt-in.
        self.log_motion_vectors = bool(self.task_cfg["data"].get("log_motion_vectors", False))

        if self.random_seed is not None:
            seed = self.random_seed
        else:
            seed = time.time_ns() % (2**32)
        set_random_seed(seed)

        # while True:
        #     self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

    @staticmethod
    def _validate_planning_contract(task_cfg, collision_world_mode):
        """Reject silent fallback of unmigrated Skills in the Physics world."""
        validate_planning_contract(task_cfg, collision_world_mode)

    def _initialize_skills(self, task, task_cfg, controllers, world):
        draw_points = False
        if draw_points:
            from omni.isaac.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        else:
            draw = None

        # Initialize skills for each robot and bind optional target diagnostics.
        skills = []
        skill_index = 0
        for cfg_skill_dict in task_cfg["skills"]:
            skill_dict = defaultdict(list)
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                robot = task.robots[robot_name]
                controller = controllers[robot_name]

                for lr_skill_dict in robot_skill_list:
                    skill_sequence = []
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        arm_skills = []
                        for skill_cfg in lr_skill_list:
                            skill = get_skill_cls(skill_cfg["name"])(
                                robot,
                                controller[lr_name],
                                task,
                                skill_cfg,
                                world=world,
                                draw=draw,
                            )
                            skill.bind_target_visualizer(
                                self.skill_target_visualizer,
                                robot=robot_name,
                                arm=lr_name,
                                skill=skill_cfg["name"],
                                skill_index=skill_index,
                            )
                            arm_skills.append(skill)
                            skill_index += 1
                        skill_sequence.append(arm_skills)
                    skill_dict[robot_name].append(skill_sequence)
            skills.append(skill_dict)
        return skills

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            controllers[robot["name"]] = {}
            for robot_file in robot["robot_file"]:
                controller_name = "left" if "left" in robot_file else "right"
                controllers[robot["name"]][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot["name"],
                    robot_file=robot_file,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    task=task,
                    world=world,
                    ignore_substring=robot.get("ignore_substring", ["material", "Plane", "conveyor", "scene", "table"]),
                    use_batch=robot.get("use_batch", False),
                    trajectory_visualizer=self.trajectory_visualizer,
                    skill_target_visualizer=self.skill_target_visualizer,
                    collision_scene_manager=self.collision_scene_manager,
                    collision_world_mode=self.collision_world_mode,
                )
                controllers[robot["name"]][controller_name].reset()
        return controllers

    def _initialize_world_recorder(self):
        """
        Initialize WorldRecorder with appropriate mode based on configuration.

        Supports two modes:
        - step_replay=False: Records prim poses for fast geometric replay (compatible with old workflow)
        - step_replay=True: Uses preprocessed joint position data for physics-accurate replay (new default)
        """
        self.world_recorder = WorldRecorder(
            self.world,
            self.task.robots,
            self.task.objects | self.task.distractors | self.task.visuals,
            step_replay=self.step_replay,
        )
        self.world_recorder.reset()

    def _reset_controllers(self, controllers):
        """Reset all controllers."""
        for _, controller in controllers.items():
            for _, ctrl in controller.items():
                ctrl.reset()

    def _enable_manipulation_base_holds(self):
        """Freeze configured mobile-base DOFs for Physics-schema Pick/Place."""

        if self.collision_world_mode != "physics_schema":
            return
        for robot in self.task.robots.values():
            enable = getattr(robot, "enable_manipulation_base_hold", None)
            if enable is not None:
                enable()

    def _init_static_objects(self, task):
        for _, obj in task.objects.items():
            try:
                init_translation = obj.init_translation
                init_orientation = obj.init_orientation
                init_parent = obj.init_parent
                if init_translation and init_orientation and init_parent:
                    parent_world_pose = get_relative_transform(
                        get_prim_at_path(task.root_prim_path + "/" + init_parent), get_prim_at_path(task.root_prim_path)
                    )
                    parent_translation, _ = pose_from_tf_matrix(parent_world_pose)
                    obj.set_local_pose(
                        translation=(parent_translation + init_translation), orientation=init_orientation
                    )
                    obj.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
                    obj.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            except Exception:
                pass

    def _randomization_layout_mem(self):
        # Reset world
        self.world.reset()
        if self.trajectory_visualizer is not None:
            self.trajectory_visualizer.clear()
        if self.skill_target_visualizer is not None:
            self.skill_target_visualizer.clear()

        # Individual initialize
        self.task.individual_randomize_from_mem()
        self.task.post_reset()

        self._enable_manipulation_base_holds()
        self.world.step(render=False)

        # Reset controllers
        self._reset_controllers(self.controllers)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.reset_episode()
            self.collision_scene_manager.initialize_contact_views()
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._safety_abort_requested = False

        # Reset skills
        del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def _randomization_layout(self):
        # Reset world
        self.world.reset()
        if self.trajectory_visualizer is not None:
            self.trajectory_visualizer.clear()
        if self.skill_target_visualizer is not None:
            self.skill_target_visualizer.clear()

        # Individual initialize
        self.task.individual_randomize()
        self.task.post_reset()

        self._enable_manipulation_base_holds()
        self.world.step(render=False)

        # Reset controllers
        if self.task_cfg.get("fluid", None):
            # Fluid, Bug, Why !!!!!!
            # For fluid manipulation, only delete controllers and reinitialize controllers can plan successfully
            if hasattr(self, "controllers"):
                del self.controllers
            self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)

        # del self.controllers
        # self.controllers = self._initialize_controllers(self.task, self.task_cfg, self.world)
        self._reset_controllers(self.controllers)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.reset_episode()
            self.collision_scene_manager.initialize_contact_views()
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._safety_abort_requested = False

        # Reset skills
        if hasattr(self, "skills"):
            del self.skills

        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)

        # Warmup
        for _ in range(20):
            self.world.get_observations()
            self._init_static_objects(self.task)
            self.world.step(render=False)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self.world.step(render=False)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def randomization(self, layout_path=None) -> bool:
        try:
            if layout_path is None:
                # Individual Reset
                self.task.individual_reset()
                self._randomization_layout()
            else:
                with open(layout_path, "rb") as f:
                    data = pickle.load(f)
                self.data = data
                self.randomization_from_mem(data)
            return True
        except Exception as e:
            raise e

    def update_skill_states(self, skills, episode_success, should_continue):
        """Update and manage skill states."""
        current_skills = skills[0]

        # Check if any skills remain
        if not any(current_skills.values()):
            skills.pop(0)
            if skills:
                should_continue = self.plan_first_skill(skills, should_continue)
            return episode_success, should_continue

        # Update each robot's skills
        for _, skill_sequences in current_skills.items():
            if not skill_sequences:
                continue

            # Update all skills first
            for lr_skill_list in skill_sequences[0]:
                if lr_skill_list:
                    start_lr_skill = lr_skill_list[0]
                    start_lr_skill.update()  # Must update regardless of completion
                    if start_lr_skill.is_done():
                        skill_success = bool(start_lr_skill.is_success())
                        start_lr_skill.complete_target_intent(skill_success)
                        if not skill_success:
                            episode_success = False
                            should_continue = False
                        lr_skill_list.remove(start_lr_skill)

                        # Do not plan or publish the next target after a failed
                        # prerequisite Skill.  The episode is already stopping,
                        # and showing a Place intent in that state would imply a
                        # target that will never be executed.
                        if skill_success and lr_skill_list:
                            next_skill = lr_skill_list[0]
                            next_skill.simple_generate_manip_cmds()
                            if hasattr(next_skill, "visualize_target"):
                                next_skill.visualize_target(self.world)
                            if len(next_skill.manip_list) == 0:
                                self._safety_failure_reason = getattr(
                                    next_skill,
                                    "failure_reason",
                                    "NO_COLLISION_FREE_PLAN",
                                ) or "NO_COLLISION_FREE_PLAN"
                                should_continue = not next_skill.is_ready()
                    if hasattr(start_lr_skill, "visualize_target"):
                        start_lr_skill.visualize_target(self.world)

            # Remove empty skill sequences
            completed_skills = []
            for lr_skill_list in skill_sequences[0]:
                if not lr_skill_list:
                    completed_skills.append(lr_skill_list)
            for completed_skill in completed_skills:
                skill_sequences[0].remove(completed_skill)

            # Move to next sequence if current is empty
            if not skill_sequences[0]:
                skill_sequences.pop(0)
                if skill_sequences:
                    for skill in skill_sequences[0]:
                        if not skill:
                            continue
                        skill[0].simple_generate_manip_cmds()
                        if len(skill[0].manip_list) == 0:
                            self._safety_failure_reason = getattr(
                                skill[0],
                                "failure_reason",
                                "NO_COLLISION_FREE_PLAN",
                            ) or "NO_COLLISION_FREE_PLAN"
                            should_continue = not skill[0].is_ready()
        return episode_success, should_continue

    def plan_first_skill(self, skills, should_continue):
        for _, robot_skill_list in skills[0].items():
            for lr_skill_list in robot_skill_list[0]:
                # Dual-arm configs legitimately use an empty list for the idle
                # arm.  Do not index it as if it contained a first Skill.
                if not lr_skill_list:
                    continue
                lr_skill_list[0].simple_generate_manip_cmds()
                if hasattr(lr_skill_list[0], "visualize_target"):
                    lr_skill_list[0].visualize_target(self.world)
                if len(lr_skill_list[0].manip_list) == 0:
                    plan_result = getattr(
                        getattr(lr_skill_list[0], "plan_evaluation", None), "result", None
                    )
                    self._safety_failure_reason = getattr(
                        lr_skill_list[0],
                        "failure_reason",
                        getattr(plan_result, "failure_code", "") or "NO_COLLISION_FREE_PLAN",
                    )
                    should_continue = not lr_skill_list[0].is_ready()
        return should_continue

    def _iter_active_skills(self):
        if not self.skills:
            return
        for robot_name, skill_sequences in self.skills[0].items():
            if not skill_sequences or not skill_sequences[0]:
                continue
            for arm_skill_list in skill_sequences[0]:
                if arm_skill_list:
                    yield robot_name, arm_skill_list[0]

    def _safety_measurements(self, skill, dynamic_changed: bool) -> SafetyMeasurements:
        controller = skill.controller
        joint_state = controller.robot.get_joints_state()
        actual_arm = np.asarray(joint_state.positions[controller.arm_indices], dtype=float)
        commanded_arm = controller._last_commanded_arm_position
        joint_error = (
            float(np.max(np.abs(actual_arm - commanded_arm)))
            if commanded_arm is not None and len(commanded_arm) == len(actual_arm)
            else 0.0
        )
        ee_position_error = 0.0
        ee_orientation_error = 0.0
        if commanded_arm is not None and len(commanded_arm) == len(actual_arm):
            expected_position, expected_orientation = controller.forward_kinematic(commanded_arm)
            actual_position, actual_orientation = controller.get_ee_pose()
            ee_position_error = float(np.linalg.norm(expected_position - actual_position))
            ee_orientation_error = quaternion_angle(expected_orientation, actual_orientation)

        base_position, base_orientation = controller.get_armbase_pose()
        if hasattr(controller, "_phase_base_position"):
            initial_position = controller._phase_base_position
            initial_orientation = controller._phase_base_orientation
        else:
            initial_position, initial_orientation = pose_from_tf_matrix(controller.T_world_base_init)
        base_translation = float(np.linalg.norm(np.asarray(base_position) - np.asarray(initial_position)))
        base_rotation = float(np.degrees(quaternion_angle(base_orientation, initial_orientation)))
        velocity = np.asarray(joint_state.velocities, dtype=float)
        arm_velocity = velocity[controller.arm_indices]
        joint_limit_violation = False
        try:
            limits = np.asarray(controller.robot.get_dof_limits(), dtype=float)
            arm_limits = limits[controller.arm_indices]
            joint_limit_violation = bool(
                np.any(actual_arm < arm_limits[:, 0] - 1e-4)
                or np.any(actual_arm > arm_limits[:, 1] + 1e-4)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            # Some legacy robot wrappers do not expose limits.  Physics-schema
            # SplitAloha does; strict world audit remains the primary guard.
            joint_limit_violation = False
        arrays = (actual_arm, velocity, np.asarray(base_position), np.asarray(base_orientation))
        nan_detected = any(not np.all(np.isfinite(value)) for value in arrays)

        illegal_state = False
        try:
            self.collision_scene_manager.assert_invariants()
        except CollisionSceneError:
            illegal_state = True
        dropped = False
        command = skill.manip_list[0]
        if command.active_object:
            record = self.collision_scene_manager.records.get(command.active_object)
            if record is not None and record.state == CollisionObjectState.ATTACHED and hasattr(skill, "get_contact"):
                _, indices = skill.get_contact()
                dropped = len(indices) == 0
        unexpected_contact = self.collision_scene_manager.get_unexpected_robot_contact_force(
            controller.name, controller.lr_name
        )
        _, unexpected_finger_contact = (
            self.collision_scene_manager.get_finger_environment_contact_forces(
                controller.name,
                controller.lr_name,
                command.active_object if command.allow_target_finger_contact else None,
            )
        )
        unexpected_contact = max(unexpected_contact, unexpected_finger_contact)
        allowed_support_contact = 0.0
        unexpected_object_contact = 0.0
        attached_slip_translation = 0.0
        attached_slip_rotation = 0.0
        if command.active_object and record is not None and record.state in {
            CollisionObjectState.ATTACHED,
            CollisionObjectState.PLACEMENT_CONTACT,
        } and command.phase != MotionPhase.DETACH_AND_SETTLE:
            attached_slip_translation, attached_slip_rotation = (
                self.collision_scene_manager.get_attached_object_slip(command.active_object)
            )
            allowed_support_contact, unexpected_object_contact = (
                self.collision_scene_manager.get_object_environment_contact_forces(
                    command.active_object,
                    command.support_object
                    if command.allow_object_support_contact
                    else None,
                )
            )
            if (
                command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
                and command.allow_object_support_contact
                and allowed_support_contact > 0.0
            ):
                command.params["contact_complete"] = True
                skill.manip_list[:] = [command] + [
                    later
                    for later in skill.manip_list[1:]
                    if not (
                        isinstance(later, MotionPhaseCommand)
                        and later.phase == MotionPhase.TERMINAL_PLACE_DESCENT
                    )
                ]
        return SafetyMeasurements(
            joint_error_rad=joint_error,
            ee_position_error_m=ee_position_error,
            ee_orientation_error_rad=ee_orientation_error,
            base_translation_m=base_translation,
            base_rotation_deg=base_rotation,
            unexpected_contact_n=unexpected_contact,
            unexpected_object_contact_n=unexpected_object_contact,
            allowed_object_support_contact_n=allowed_support_contact,
            attached_slip_translation_m=attached_slip_translation,
            attached_slip_rotation_deg=attached_slip_rotation,
            dynamic_obstacle_changed=dynamic_changed,
            nan_detected=nan_detected,
            joint_limit_violation=joint_limit_violation,
            # Base pose drift and attached-object/gripper behavior have their
            # own monitors.  CuRobo commands this arm's six joints, so a fast
            # mimic gripper or passive/base DOF must not masquerade as an arm
            # trajectory velocity failure.
            arm_velocity_rad_s=float(
                np.max(np.abs(arm_velocity)) if arm_velocity.size else 0.0
            ),
            # Reserved for explicit simulator validity/fault-injection errors.
            # Ordinary finite velocity is evaluated numerically below using
            # soft/hard thresholds and the standard consecutive-step debounce.
            abnormal_velocity=False,
            illegal_object_state=illegal_state,
            attached_object_dropped=dropped,
            plan_failed=bool(controller._phase_plan_failed),
            tracking_completion_failed=bool(controller._phase_tracking_failed),
        )

    def _execution_safety_precheck(self, step_id: int, action_dict=None) -> bool:
        """Evaluate the previous step before producing the next command.

        On a hard abort the articulation must receive a measured hold target
        in this same simulation step. Merely clearing ``cmd_plan`` leaves the
        previous PhysX drive target active and can move the robot once more
        after the safety decision.
        """

        if step_id % 100 == 0:
            LOGGER.warning(
                "[ExecutionHeartbeat] precheck-entry step=%d enabled=%s manager=%s skills=%s",
                step_id,
                self.execution_safety_enabled,
                self.collision_scene_manager is not None,
                bool(self.skills),
            )
        if not self.execution_safety_enabled or self.collision_scene_manager is None:
            return True
        safety_cfg = self.task_cfg.get("planning", {}).get("execution_safety", {})
        interval = int(safety_cfg.get("dynamic_sync_interval_steps", 5))
        changed = set(
            self.collision_scene_manager.sync_dynamic_poses(step_id, interval_steps=interval)
        )
        for robot_name, skill in self._iter_active_skills() or []:
            if not skill.manip_list or not isinstance(skill.manip_list[0], MotionPhaseCommand):
                continue
            command = skill.manip_list[0]
            controller = skill.controller
            # This precheck evaluates the result of the *previous* physics
            # step.  A newly selected command has not run yet, so its phase
            # baseline and first commanded joint target do not exist.  Using
            # the controller's initialization pose here caused step-0 and
            # phase-transition false base-drift aborts.
            if command is not controller._active_phase_command:
                continue
            if self.execution_supervisor.is_holding(controller):
                continue
            if (
                step_id % 100 == 0
                or controller._phase_plan_failed
                or controller._phase_tracking_failed
            ):
                LOGGER.warning(
                    "[ExecutionHeartbeat] step=%d robot=%s arm=%s phase=%s plan_active=%s plan_index=%d phase_finished=%s plan_failed=%s tracking_failed=%s",
                    step_id,
                    robot_name,
                    controller.lr_name,
                    command.phase.value,
                    controller.cmd_plan is not None,
                    int(controller.cmd_idx),
                    bool(controller._phase_plan_finished),
                    bool(controller._phase_plan_failed),
                    bool(controller._phase_tracking_failed),
                )
            try:
                measurements = self._safety_measurements(
                    skill, bool(changed - {command.active_object})
                )
            except CollisionSceneError as exc:
                # Contact-view and invariant failures are themselves hard
                # safety failures. Convert them to an auditable event instead
                # of escaping to the outer workflow exception handler, which
                # would otherwise retry without first applying a hold target.
                LOGGER.exception(
                    "[ExecutionSafety] collision/contact audit failed for %s/%s: %s",
                    robot_name,
                    controller.lr_name,
                    exc,
                )
                measurements = SafetyMeasurements(illegal_object_state=True)
            decision = self.execution_supervisor.evaluate(
                measurements,
                step_id=step_id,
                robot=robot_name,
                skill=skill,
                command=command,
                world_revision=self.collision_scene_manager.world_revision,
            )
            if decision == SafetyDecision.ABORT:
                self._safety_failure_reason = self.execution_supervisor.failure_reason
                if action_dict is not None:
                    hold = controller.hold_action()
                    action_dict[robot_name] = {
                        "joint_positions": np.asarray(hold["joint_positions"]),
                        "joint_indices": np.asarray(hold["joint_indices"]),
                        "raw_action": [hold],
                    }
                return False
        return True

    def _forward_or_hold(self, skill):
        controller = skill.controller
        command = skill.manip_list[0]
        try:
            return self.execution_supervisor.forward_or_hold(
                controller,
                lambda: controller.forward(command),
            )
        except CollisionSceneError as exc:
            LOGGER.exception(
                "[ExecutionSafety] collision-state operation failed for %s/%s: %s",
                controller.name,
                controller.lr_name,
                exc,
            )
            self.execution_supervisor.evaluate(
                SafetyMeasurements(illegal_object_state=True),
                step_id=self._active_execution_step_id,
                robot=controller.name,
                skill=skill,
                command=command,
                world_revision=self.collision_scene_manager.world_revision,
            )
            self._safety_failure_reason = (
                f"COLLISION_SCENE_ERROR:{exc}"
            )
            self._safety_abort_requested = True
            return controller.hold_action()

    def generate_seq(self) -> list:
        end = False

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        step_id = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=False)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            self._active_execution_step_id = step_id
            record_flag = True
            if self.skills and should_continue and not self._execution_safety_precheck(
                step_id, action_dict
            ):
                episode_success = False
                should_continue = False
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            self._forward_or_hold(skill[0])
                            for skill in skill_sequences[0]
                            if skill and skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [
                            skill[0].is_feasible() for skill in skill_sequences[0] if skill and skill[0]
                        ]
                        record_labels = [
                            skill[0].is_record() for skill in skill_sequences[0] if skill and skill[0]
                        ]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
                        if self._safety_abort_requested:
                            episode_success = False
                            should_continue = False
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self.world.step(render=False)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self.world_recorder.record()

                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self.world_recorder.record()
            self.task.apply_action(action_dict)
            self.world.step(render=False)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        if end:
            if self.step_replay:
                return [None] * step_id
            else:
                # Prim poses mode: return recorded poses for compatibility
                return self.world_recorder.prim_poses
        else:
            return []

    def recover_seq(self, seq_path):
        data = self.data
        return self.recover_seq_from_mem(data)

    def _record_rgb_depth(self, step_idx: int):
        for key, value in self.task.cameras.items():
            camera_info = self.task.cameras_info.get(key, {})
            record_to = camera_info.get("record_to")
            if record_to is None:
                robot_names = [robot_name for robot_name in self.task.robots if robot_name in key]
            elif isinstance(record_to, str):
                robot_names = [record_to]
            else:
                robot_names = list(record_to)

            unknown_robots = [name for name in robot_names if name not in self.task.robots]
            if unknown_robots:
                raise ValueError(f"camera {key} records to unknown robots: {unknown_robots}")
            if not robot_names:
                continue

            record_mode = camera_info.get("record_mode", "lmdb_and_video")
            if record_mode == "video_only":
                rgb_img = value.get_rgb_observation()
                for robot_name in robot_names:
                    save_camera_name = camera_info.get("save_name") or key.replace(f"{robot_name}_", "")
                    self.logger.add_video_frame(
                        robot_name,
                        "images.rgb." + save_camera_name,
                        rgb_img,
                        step_idx=step_idx,
                    )
                continue
            if record_mode != "lmdb_and_video":
                raise ValueError(f"unsupported camera record_mode {record_mode!r}: {key}")

            camera_obs = value.get_observations()
            for robot_name in robot_names:
                rgb_img = camera_obs["color_image"]
                camera2env_pose = camera_obs["camera2env_pose"]
                save_camera_name = camera_info.get("save_name") or key.replace(f"{robot_name}_", "")
                self.logger.add_color_image(
                    robot_name, "images.rgb." + save_camera_name, rgb_img, step_idx=step_idx
                )
                if "depth_image" in camera_obs:
                    depth_img = np.nan_to_num(
                        np.asarray(camera_obs["depth_image"]), nan=0.0, posinf=0.0, neginf=0.0
                    )
                    self.logger.add_depth_image(
                        robot_name, "images.depth." + save_camera_name, depth_img, step_idx=step_idx
                    )
                if "semantic_mask" in camera_obs:
                    self.logger.add_seg_image(
                        robot_name,
                        "images.seg." + save_camera_name,
                        camera_obs["semantic_mask"],
                        step_idx=step_idx,
                    )
                    if "semantic_mask_id2labels" in camera_obs:
                        self.logger.add_scalar_data(
                            robot_name,
                            "labels.seg." + save_camera_name,
                            camera_obs["semantic_mask_id2labels"],
                        )
                if "bbox2d_tight" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name, "labels.bbox2d_tight." + save_camera_name, camera_obs["bbox2d_tight"]
                    )
                if "bbox2d_tight_id2labels" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name,
                        "labels.bbox2d_tight_id2labels." + save_camera_name,
                        camera_obs["bbox2d_tight_id2labels"],
                    )
                if "bbox2d_loose" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name, "labels.bbox2d_loose." + save_camera_name, camera_obs["bbox2d_loose"]
                    )
                if "bbox2d_loose_id2labels" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name,
                        "labels.bbox2d_loose_id2labels." + save_camera_name,
                        camera_obs["bbox2d_loose_id2labels"],
                    )
                if "bbox3d" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name, "labels.bbox3d." + save_camera_name, camera_obs["bbox3d"]
                    )
                if "bbox3d_id2labels" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name, "labels.bbox3d_id2labels." + save_camera_name, camera_obs["bbox3d_id2labels"]
                    )
                if self.log_motion_vectors and "motion_vectors" in camera_obs:
                    self.logger.add_scalar_data(
                        robot_name, "labels.motion_vectors." + save_camera_name, camera_obs["motion_vectors"]
                    )
                self.logger.add_scalar_data(robot_name, "camera2env_pose." + save_camera_name, camera2env_pose)
                if step_idx == 0:
                    self.logger.add_json_data(
                        robot_name, f"{save_camera_name}_camera_params", camera_obs["camera_params"]
                    )

    def seq_replay(self, sequence: list) -> int:
        """
        Replay recorded sequence with mode-specific data preparation.

        Returns:
            int: Number of steps replayed
        """
        if not self.step_replay:
            self.world_recorder.prim_poses = sequence

        # warmup before replay formally
        self.world_recorder.warmup()

        # Get total steps from WorldRecorder
        total_steps = self.world_recorder.get_total_steps()
        step_idx = 0

        # Unified replay loop - WorldRecorder handles rendering internally
        with tqdm(total=total_steps, desc="Replay Progress") as pbar:
            while not self.world_recorder.replay():
                # Record RGB/depth at current step
                self._record_rgb_depth(step_idx)
                step_idx += 1
                pbar.update(1)

        self.length = total_steps
        print("Replay finished.")
        return total_steps

    def get_task_name(self):
        return self.task_cfg["task"]

    def save_seq(self, save_path: str) -> int:
        ser_bytes = self.dump_plan_info()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        save_path = os.path.join(save_path, "plan")
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, f"{timestamp}.pkl")
        with open(path, "wb") as f:
            f.write(ser_bytes)
        return self.world_recorder.get_total_steps()

    def save(self, save_path: str) -> int:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        episode_failed = getattr(self, "_episode_failed", False)
        if episode_failed:
            timestamp = f"fail_{timestamp}"
        saved_dirs = self.logger.save(save_path, timestamp, save_img=True)
        if self.trajectory_visualizer is not None:
            for episode_dir in saved_dirs or []:
                self.trajectory_visualizer.export(episode_dir)
        if self.skill_target_visualizer is not None:
            for episode_dir in saved_dirs or []:
                self.skill_target_visualizer.export(episode_dir)
        if self.collision_scene_manager is not None:
            for episode_dir in saved_dirs or []:
                self.collision_scene_manager.export(episode_dir)
        if self.safety_monitor is not None:
            for episode_dir in saved_dirs or []:
                self.safety_monitor.export(episode_dir)
        emit_episode_saved(
            status="failed" if episode_failed else "success",
            episode_dirs=saved_dirs or [],
            num_steps=self.length,
            failure_reason=getattr(self, "_episode_failure_reason", "") if episode_failed else "",
            task_name=self.task_cfg.get("task"),
            task_dir=self.task_cfg.get("data", {}).get("task_dir"),
            collect_info=self.task_cfg.get("data", {}).get("collect_info"),
        )

        return self.length

    def close(self):
        """Release episode-local streaming writers and the anonymous debug layer."""
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.close()
        visualizer = getattr(self, "trajectory_visualizer", None)
        if visualizer is not None:
            visualizer.close()
        self.trajectory_visualizer = None
        target_visualizer = getattr(self, "skill_target_visualizer", None)
        if target_visualizer is not None:
            target_visualizer.close()
        self.skill_target_visualizer = None

    def plan_with_render(self):
        end = False

        step_id = 0
        length = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        should_continue = self.plan_first_skill(self.skills, should_continue)

        # Warmup
        for _ in range(10):
            obs = self.world.get_observations()
            # self._init_static_objects(self.task)
            self.world.step(render=True)

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        while not (step_id >= max_episode_length or (not self.skills and not episode_success) or (not should_continue)):
            obs = self.world.get_observations()
            action_dict = {}
            self._active_execution_step_id = step_id
            record_flag = True
            if self.skills and should_continue and not self._execution_safety_precheck(
                step_id, action_dict
            ):
                episode_success = False
                should_continue = False
            if self.skills and should_continue:
                # Process current skills
                current_skills = self.skills[0]
                for robot_name, skill_sequences in current_skills.items():
                    if skill_sequences and skill_sequences[0]:
                        action = [
                            self._forward_or_hold(skill[0])
                            for skill in skill_sequences[0]
                            if skill and skill[0] and skill[0].is_ready()
                        ]

                        feasible_labels = [
                            skill[0].is_feasible() for skill in skill_sequences[0] if skill and skill[0]
                        ]
                        record_labels = [
                            skill[0].is_record() for skill in skill_sequences[0] if skill and skill[0]
                        ]

                        if False in feasible_labels:
                            should_continue = False
                        if False in record_labels:
                            record_flag = False

                        if action:
                            action_dict[robot_name] = {
                                "joint_positions": np.concatenate([a["joint_positions"] for a in action]),
                                "joint_indices": np.concatenate([a["joint_indices"] for a in action]),
                                "raw_action": action,
                            }
                        if self._safety_abort_requested:
                            episode_success = False
                            should_continue = False
            elif not self.skills and episode_success:
                print("Task is successful")
                end = True
                for j_idx in range(1, 7):
                    self.world.step(render=True)
                    obs = self.world.get_observations()
                    log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id + j_idx)
                    self._record_rgb_depth(step_id + j_idx)
                    self.world_recorder.record()
                length = step_id + 6
                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(self.logger, obs, action_dict, self.controllers, step_idx=step_id)
                self._record_rgb_depth(step_id)
            self.task.apply_action(action_dict)
            self.world.step(render=True)

            step_id += 1
            if self.skills:
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        if end:
            self._episode_failed = False
            self._episode_failure_reason = ""
            self.length = length
            return length
        elif getattr(self, "save_failed", False):
            if step_id == 0:
                # LMDB converts state/action labels by shifting one sample.
                # Persist two measured hold frames so a planning-time safe
                # failure remains a valid, inspectable episode after that
                # shift instead of producing an empty gripper/action list.
                action_dict = {}
                for failure_step in range(2):
                    obs = self.world.get_observations()
                    log_dual_obs(
                        self.logger,
                        obs,
                        action_dict,
                        self.controllers,
                        step_idx=failure_step,
                    )
                    self._record_rgb_depth(failure_step)
                    self.world_recorder.record()
                    self.world.step(render=True)
                step_id = 2
            self._episode_failed = True
            if step_id >= max_episode_length:
                self._episode_failure_reason = "max_episode_length_reached"
            elif not episode_success:
                self._episode_failure_reason = (
                    self._safety_failure_reason or "episode_success_false"
                )
            elif not should_continue:
                self._episode_failure_reason = self._safety_failure_reason or "skill_or_record_stopped"
            else:
                self._episode_failure_reason = "failed_episode_saved"
            if self.skill_target_visualizer is not None:
                self.skill_target_visualizer.abort_active(self._episode_failure_reason)
            self.length = step_id
            return step_id
        else:
            self._episode_failed = False
            self._episode_failure_reason = ""
            self.length = 0
            return 0

    def _dump_task_cfg(self, task_cfg):
        task_cfg_copy = deepcopy(task_cfg)
        return pickle.dumps(task_cfg_copy)

    def dump_plan_info(self) -> bytes:
        logger_ser = self.logger.dump()
        cfg_ser = self._dump_task_cfg(self.task_cfg)
        ser = pickle.dumps((cfg_ser, self.world_recorder.dumps(), logger_ser))
        return ser

    def dedump_plan_info(self, ser_obj: bytes) -> object:
        res = pickle.loads(ser_obj)
        return res

    def randomization_from_mem(self, data) -> bool:
        try:
            cfg_ser, _, _ = data
            task_cfg = pickle.loads(cfg_ser)
            self.task_cfg = task_cfg
            self.task.cfg = task_cfg

            # Individual Reset
            self.task.individual_reset_from_mem()
            self._randomization_layout_mem()
            return True
        except Exception as e:
            raise e

    def recover_seq_from_mem(self, data) -> list:
        """
        Recover sequence from memory based on WorldRecorder mode.

        Returns:
            - step_replay=False: Returns prim_poses list
            - step_replay=True: Returns placeholder list (replay data is in WorldRecorder)
        """
        try:
            _, wr_ser, logger_ser = data
            self.logger.dedump(logger_ser)

            if wr_ser:
                self.world_recorder.loads(wr_ser)

            if self.step_replay:
                return [None] * self.world_recorder.num_steps
            else:
                return self.world_recorder.prim_poses

        except Exception as e:
            raise e
