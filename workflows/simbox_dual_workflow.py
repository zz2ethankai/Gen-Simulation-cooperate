import glob
import json
import logging
import os
import pickle
import random
import time
from collections import defaultdict, deque
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
from core.planning.config_contract import (
    PASSTHROUGH_MODE,
    resolve_collision_world_mode,
    resolve_runtime_skill_collision_world_mode,
    resolve_skill_collision_world_mode,
    task_uses_physics_schema,
    validate_planning_contract,
)
from core.utils.camera_utils import capture_topdown_screenshot
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


class _PassiveSkillController:
    """Placeholder controller for skills that do not emit manipulator actions."""

    def __init__(self, *, robot_name: str, controller_name: str):
        self.name = robot_name
        self.robot_file = f"{controller_name}_passive_skill_controller"
        self._gripper_state = 1.0

    def reset(self):
        return None

    def forward(self, _command):
        return {
            "joint_positions": np.array([], dtype=np.float32),
            "joint_indices": np.array([], dtype=np.int64),
        }


# pylint: disable=unused-argument
@NimbusWorkFlow.register("SimBoxDualWorkFlow")
class SimBoxDualWorkFlow(NimbusWorkFlow):
    def __init__(
        self,
        world,
        task_cfg_path: str,
        scene_info: str = "dining_room_scene_info",
        random_seed: int = None,
        planning_step_render: bool = False,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        self.planning_step_render = bool(planning_step_render)
        self._local_base_drivers = {}
        super().__init__(world, task_cfg_path)

    @staticmethod
    def _skill_requires_controller(skill_cfg: dict) -> bool:
        if not isinstance(skill_cfg, dict):
            return True
        return str(skill_cfg.get("name", "")).strip() != "navigate"

    def _skill_controller_names(self, task_cfg: dict, robot_name: str) -> set[str]:
        controller_names = set()
        skills = task_cfg.get("skills", [])
        if not isinstance(skills, list):
            return controller_names
        for cfg_skill_dict in skills:
            if not isinstance(cfg_skill_dict, dict):
                continue
            robot_skill_list = cfg_skill_dict.get(robot_name, [])
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, dict):
                    continue
                for lr_name in lr_skill_dict.keys():
                    controller_names.add(str(lr_name))
        return controller_names

    def _required_controller_names(self, task_cfg: dict, robot_name: str) -> set[str]:
        required = set()
        skills = task_cfg.get("skills", [])
        if not isinstance(skills, list):
            return required
        for cfg_skill_dict in skills:
            if not isinstance(cfg_skill_dict, dict):
                continue
            robot_skill_list = cfg_skill_dict.get(robot_name, [])
            if not isinstance(robot_skill_list, list):
                continue
            for lr_skill_dict in robot_skill_list:
                if not isinstance(lr_skill_dict, dict):
                    continue
                for lr_name, lr_skill_list in lr_skill_dict.items():
                    if not isinstance(lr_skill_list, list):
                        continue
                    if any(self._skill_requires_controller(skill_cfg) for skill_cfg in lr_skill_list):
                        required.add(str(lr_name))
        return required

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
                base_cfg = merged_cfg.get("base")
                if isinstance(base_cfg, dict):
                    self._merge_base_configs(base_cfg)
                robot.clear()
                robot.update(merged_cfg)

    def _merge_base_configs(self, base_cfg: dict):
        """Merge chassis and ROS-free local-navigation configs in-place."""
        override_cfg = deepcopy(base_cfg)
        merged_base_cfg = {}
        base_config_file = override_cfg.get("base_config_file")
        local_navigation_config_file = override_cfg.get("local_navigation_config_file")

        if base_config_file:
            with open(base_config_file, "r", encoding="utf-8") as f:
                loaded_base_cfg = yaml.load(f, Loader=Loader)
            if isinstance(loaded_base_cfg, dict):
                merged_base_cfg = deepcopy(loaded_base_cfg)

        if local_navigation_config_file:
            config_files = (
                local_navigation_config_file
                if isinstance(local_navigation_config_file, list)
                else [local_navigation_config_file]
            )
            for config_path in config_files:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded_navigation_cfg = yaml.load(f, Loader=Loader)
                if isinstance(loaded_navigation_cfg, dict):
                    self._deep_update_dict(merged_base_cfg, loaded_navigation_cfg)

        self._deep_update_dict(merged_base_cfg, override_cfg)
        base_cfg.clear()
        base_cfg.update(merged_base_cfg)

    def _deep_update_dict(self, base: dict, override: dict):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update_dict(base[key], value)
            else:
                base[key] = value

    def _resolve_arena_file_path(self, arena_file_path: str | None) -> str | None:
        if arena_file_path and os.path.exists(arena_file_path):
            return arena_file_path

        if arena_file_path:
            asset_root = self.task_cfg.get("asset_root")
            if asset_root:
                arena_from_asset_root = os.path.join(asset_root, arena_file_path)
                if os.path.exists(arena_from_asset_root):
                    return arena_from_asset_root
            raise FileNotFoundError(f"arena_file does not exist: {arena_file_path}")
        raise ValueError("task config must define arena_file")

    def reset(self, need_preload: bool = True):
        self.close()
        self._destroy_local_base_drivers()

        # A previous task can remain registered if scene setup fails during world.reset().
        # Clear the world before constructing the next task so retries do not trip the
        # duplicate-name guard in omni.isaac.core.world.World.add_task().
        if self.world.get_current_tasks() or self.world.is_tasks_scene_built():
            self.world.clear()
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
        arena_file_path = self._resolve_arena_file_path(arena_file_path)
        self._saved_arena_file = arena_file_path
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
        if hasattr(self.world, "_current_tasks") and task_name in self.world._current_tasks:
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
        self.requested_collision_world_mode = str(collision_cfg.get("mode", "auto"))
        self.collision_world_mode, collision_mode_reason = resolve_collision_world_mode(
            self.task_cfg, self.requested_collision_world_mode
        )
        LOGGER.warning(
            "[CollisionWorld] requested_mode=%s resolved_mode=%s reason=%s",
            self.requested_collision_world_mode,
            self.collision_world_mode,
            collision_mode_reason,
        )
        self._validate_planning_contract(self.task_cfg, self.collision_world_mode)
        self.world.reset()
        # Hold configured virtual-base DOFs before the first Physics step, then
        # keep the local world-step and fixed-start-pose sequence authoritative.
        self._enable_manipulation_base_holds()
        self._step_world(render=True)
        self._set_fixed_robot_start_poses_after_reset()
        if task_uses_physics_schema(self.collision_world_mode):
            # The task-level mode may be ``auto``/``hybrid``, but this manager
            # owns only the exact Physics-schema subset of the collision world.
            collision_manager_cfg = dict(collision_cfg)
            collision_manager_cfg["mode"] = "physics_schema"
            self.collision_scene_manager = CollisionSceneManager(
                self.stage, self.task, collision_manager_cfg, safety_cfg
            )
        elif self.collision_world_mode == "legacy_stage_scan":
            self.collision_scene_manager = None
        else:
            raise ValueError(
                f"unsupported planning.collision_world.mode: {self.collision_world_mode!r}"
            )
        self.execution_safety_enabled = bool(
            safety_cfg.get(
                "enabled", task_uses_physics_schema(self.collision_world_mode)
            )
        )
        LOGGER.info(
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
        self._initialize_local_base_drivers()

        self._run_reset_warmup(50)

        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        if callable(get_physics_dt):
            physics_dt = float(get_physics_dt())
        else:
            physics_dt = float(getattr(self.world, "physics_dt", 1.0 / 30.0))
        video_fps = int(round(1.0 / physics_dt)) if physics_dt > 0 else 30

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
            video_fps=video_fps,
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

        self._skills_use_dag = self._task_uses_skill_dag(task_cfg)
        if self._skills_use_dag:
            return self._initialize_skill_dag(task, task_cfg, controllers, world, draw)
        return self._initialize_legacy_skills(task, task_cfg, controllers, world, draw)

    def _bind_skill_collision_world_mode(self, skill, skill_cfg):
        mode = resolve_skill_collision_world_mode(
            skill_cfg.get("name", ""), self.requested_collision_world_mode
        )
        setattr(skill, "collision_world_mode", mode)
        return mode

    @staticmethod
    def _task_uses_skill_dag(task_cfg):
        for cfg_skill_dict in task_cfg.get("skills", []):
            if not isinstance(cfg_skill_dict, dict):
                continue
            for robot_skill_list in cfg_skill_dict.values():
                if not isinstance(robot_skill_list, list):
                    continue
                for lr_skill_dict in robot_skill_list:
                    if not isinstance(lr_skill_dict, dict):
                        continue
                    for lr_skill_list in lr_skill_dict.values():
                        if not isinstance(lr_skill_list, list):
                            continue
                        for skill_cfg in lr_skill_list:
                            if isinstance(skill_cfg, dict) and ("id" in skill_cfg or "depends_on" in skill_cfg):
                                return True
        return False

    def _initialize_legacy_skills(self, task, task_cfg, controllers, world, draw):
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
                                workflow=self,
                                draw=draw,
                            )
                            self._bind_skill_collision_world_mode(skill, skill_cfg)
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

    def _initialize_skill_dag(self, task, task_cfg, controllers, world, draw):
        nodes_by_id = {}
        nodes = []

        for phase_idx, cfg_skill_dict in enumerate(task_cfg["skills"]):
            for robot_name, robot_skill_list in cfg_skill_dict.items():
                robot = task.robots[robot_name]
                controller = controllers[robot_name]

                for sequence_idx, lr_skill_dict in enumerate(robot_skill_list):
                    for lr_name, lr_skill_list in lr_skill_dict.items():
                        for skill_idx, skill_cfg in enumerate(lr_skill_list):
                            if "id" not in skill_cfg:
                                raise ValueError(
                                    f"DAG skill config requires 'id': robot={robot_name}, "
                                    f"controller={lr_name}, phase={phase_idx}, sequence={sequence_idx}, skill={skill_idx}"
                                )
                            skill_id = str(skill_cfg["id"])
                            if skill_id in nodes_by_id:
                                raise ValueError(f"Duplicate skill id in DAG config: {skill_id}")

                            depends_on = skill_cfg.get("depends_on", [])
                            if depends_on is None:
                                depends_on = []
                            if not isinstance(depends_on, list):
                                raise TypeError(f"Skill '{skill_id}' depends_on must be a list")
                            depends_on = [str(dep_id) for dep_id in depends_on]

                            skill = get_skill_cls(skill_cfg["name"])(
                                robot,
                                controller[lr_name],
                                task,
                                skill_cfg,
                                world=world,
                                workflow=self,
                                draw=draw,
                            )
                            skill_collision_mode = self._bind_skill_collision_world_mode(
                                skill, skill_cfg
                            )
                            skill.bind_target_visualizer(
                                self.skill_target_visualizer,
                                robot=robot_name,
                                arm=lr_name,
                                skill=skill_cfg["name"],
                                skill_index=len(nodes),
                            )
                            setattr(skill, "skill_id", skill_id)
                            node = {
                                "id": skill_id,
                                "depends_on": depends_on,
                                "robot_name": robot_name,
                                "controller_name": str(lr_name),
                                "collision_world_mode": skill_collision_mode,
                                "skill": skill,
                                "state": "pending",
                            }
                            nodes_by_id[skill_id] = node
                            nodes.append(node)

        ordered_nodes = self._toposort_skill_nodes(nodes, nodes_by_id)
        return {"nodes": ordered_nodes, "nodes_by_id": nodes_by_id}

    @staticmethod
    def _toposort_skill_nodes(nodes, nodes_by_id):
        indegree = {node["id"]: 0 for node in nodes}
        children = defaultdict(list)

        for node in nodes:
            for dep_id in node["depends_on"]:
                if dep_id not in nodes_by_id:
                    raise ValueError(f"Skill '{node['id']}' depends on unknown skill '{dep_id}'")
                children[dep_id].append(node["id"])
                indegree[node["id"]] += 1

        ready = deque([node["id"] for node in nodes if indegree[node["id"]] == 0])
        ordered = []
        while ready:
            skill_id = ready.popleft()
            ordered.append(nodes_by_id[skill_id])
            for child_id in children[skill_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)

        if len(ordered) != len(nodes):
            raise ValueError("Cycle detected in skill DAG config")
        return ordered

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            robot_name = robot["name"]
            controllers[robot_name] = {}
            required_controller_names = self._required_controller_names(task_cfg, robot_name)
            declared_controller_names = self._skill_controller_names(task_cfg, robot_name)

            robot_files = robot.get("robot_file", [])
            if isinstance(robot_files, str):
                robot_files = [robot_files]
            robot_files_by_name = {}
            for robot_file in robot_files:
                controller_name = "left" if "left" in robot_file else "right"
                robot_files_by_name[controller_name] = robot_file

            for controller_name in required_controller_names:
                robot_file = robot_files_by_name.get(controller_name)
                if robot_file is None:
                    raise KeyError(
                        f"Robot '{robot_name}' is missing robot_file for controller '{controller_name}'"
                    )
                controllers[robot_name][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot_name,
                    robot_file=robot_file,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    task=task,
                    world=world,
                    # Let each controller provide its own collision-filter defaults.
                    # In particular, fluid-capable controllers must ignore the
                    # runtime particle isosurface during legacy Stage scans.
                    ignore_substring=robot.get("ignore_substring"),
                    use_batch=robot.get("use_batch", False),
                    trajectory_visualizer=self.trajectory_visualizer,
                    skill_target_visualizer=self.skill_target_visualizer,
                    collision_scene_manager=self.collision_scene_manager,
                    collision_world_mode=(
                        "physics_schema"
                        if task_uses_physics_schema(self.collision_world_mode)
                        else "legacy_stage_scan"
                    ),
                )
                controllers[robot_name][controller_name].reset()

            passive_controller_names = (
                declared_controller_names | set(robot_files_by_name.keys())
            ) - required_controller_names
            for controller_name in passive_controller_names:
                controllers[robot_name][controller_name] = _PassiveSkillController(
                    robot_name=robot_name,
                    controller_name=controller_name,
                )
        return controllers

    def _initialize_local_base_drivers(self):
        self._destroy_local_base_drivers()
        try:
            from core.mobile import build_local_base_driver
        except Exception as exc:
            raise RuntimeError("Local base driver module import failed") from exc
        for robot_name, robot in self.task.robots.items():
            if not hasattr(robot, "get_base_interface") or not hasattr(robot, "apply_base_command"):
                continue
            try:
                driver = build_local_base_driver(robot, world=self.world)
            except KeyError:
                raise RuntimeError(
                    f"Unsupported local base profile for mobile robot '{robot_name}'"
                ) from None
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize local base driver for '{robot_name}'") from exc
            self._local_base_drivers[robot_name] = driver
            setattr(robot, "_simbox_local_base_driver", driver)
        if self._local_base_drivers:
            print(f"[local-base-driver] Initialized {sorted(self._local_base_drivers.keys())}")

    def _ensure_local_base_driver_bindings(self):
        """Restore robot-to-driver bindings before reading observations.

        Isaac task resets can recreate or reinitialize robot wrappers while the
        workflow-level driver registry remains alive.  Observation logging must
        see the same driver object that the workflow steps, so repair the
        binding at this lifecycle boundary instead of weakening robot logging.
        """
        drivers = getattr(self, "_local_base_drivers", {})
        needs_reinit = False
        for robot_name, robot in getattr(self.task, "robots", {}).items():
            if not hasattr(robot, "get_base_interface") or not hasattr(robot, "apply_base_command"):
                continue
            driver = drivers.get(robot_name)
            if driver is None or not hasattr(driver, "get_logging_state_snapshot"):
                needs_reinit = True
                break
            # Isaac can recreate the robot wrapper during a reset while the
            # workflow-owned driver remains valid. Rebind that driver instead
            # of destroying it and invalidating active Navigate skills.
            if getattr(robot, "_simbox_local_base_driver", None) is not driver:
                setattr(robot, "_simbox_local_base_driver", driver)
        if needs_reinit:
            self._initialize_local_base_drivers()

        for robot_name, robot in getattr(self.task, "robots", {}).items():
            if not hasattr(robot, "get_base_interface") or not hasattr(robot, "apply_base_command"):
                continue
            driver = getattr(robot, "_simbox_local_base_driver", None)
            if driver is None or not hasattr(driver, "get_logging_state_snapshot"):
                raise RuntimeError(
                    f"Local base driver binding missing for mobile robot '{robot_name}'"
                )

    def _get_observations(self):
        """Read observations only after repairing mobile-base driver bindings."""

        self._ensure_local_base_driver_bindings()
        return self.world.get_observations()

    def _step_local_base_drivers(self):
        if not self._local_base_drivers:
            return
        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        step_dt = float(get_physics_dt()) if callable(get_physics_dt) else float(getattr(self.world, "physics_dt", 1.0 / 60.0))
        for robot_name, driver in list(self._local_base_drivers.items()):
            try:
                driver.step(step_dt=step_dt)
            except Exception as exc:
                raise RuntimeError(f"Local base driver step failed for '{robot_name}'") from exc

    def get_local_base_driver(self, robot_name: str):
        return self._local_base_drivers.get(robot_name)

    def _destroy_local_base_drivers(self):
        for robot_name, driver in list(getattr(self, "_local_base_drivers", {}).items()):
            try:
                driver.finalize_after_navigation()
            except Exception:
                pass
            robot = self.task.robots.get(robot_name) if hasattr(self, "task") else None
            if robot is not None:
                setattr(robot, "_simbox_local_base_driver", None)
        self._local_base_drivers = {}

    def _reset_local_base_drivers(self, *, clear_debug_history: bool):
        for robot_name, driver in list(self._local_base_drivers.items()):
            try:
                driver.reset(clear_debug_history=clear_debug_history)
            except Exception as exc:
                raise RuntimeError(f"Failed to reset local base driver for '{robot_name}'") from exc

    def _step_world(self, render: bool = False):
        # Apply the skill's body twist immediately before the physics step.
        self._step_local_base_drivers()
        self.world.step(render=render)

    def __del__(self):
        try:
            self._destroy_local_base_drivers()
        except Exception:
            pass

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
        # Randomized retry resets can replace an object's USD subtree and its
        # exact attach collider path.  Rebuild the Physics-schema collision
        # records/world before TemplateController.reset() audits them.
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.refresh_after_task_reset()
        for _, controller in controllers.items():
            for _, ctrl in controller.items():
                ctrl.reset()

    def _enable_manipulation_base_holds(self):
        """Freeze configured mobile-base DOFs for Physics-schema Pick/Place."""

        if not task_uses_physics_schema(self.collision_world_mode):
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

    def _set_fixed_robot_start_poses_after_reset(self):
        self.task.set_fixed_robot_start_poses()
        self._step_world(render=False)

    def _reset_fixed_robot_start_states_after_physics(self, *, clear_debug_history: bool):
        self.task.set_fixed_robot_start_poses()
        self._reset_local_base_drivers(clear_debug_history=clear_debug_history)

    def _run_reset_warmup(self, step_count: int):
        for _ in range(int(step_count)):
            self._init_static_objects(self.task)
            self._step_world(render=False)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

    def _randomization_layout_mem(self):
        self._destroy_local_base_drivers()

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
        self._step_world(render=False)
        self._set_fixed_robot_start_poses_after_reset()

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
        self._initialize_local_base_drivers()

        # Warmup
        for _ in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def _randomization_layout(self):
        self._destroy_local_base_drivers()

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
        self._step_world(render=False)
        self._set_fixed_robot_start_poses_after_reset()

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
        self._initialize_local_base_drivers()

        # Warmup
        for _ in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self._step_world(render=False)
            self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def reset_after_failed_generation(self):
        self._destroy_local_base_drivers()

        self.task.individual_reset()
        self.world.reset()
        if hasattr(self.task, "reset_fixed_rigid_objects"):
            self.task.reset_fixed_rigid_objects()
        self.task.post_reset()
        self._enable_manipulation_base_holds()
        self._step_world(render=False)
        self._set_fixed_robot_start_poses_after_reset()

        self._reset_controllers(self.controllers)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.reset_episode()
            self.collision_scene_manager.initialize_contact_views()
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._safety_abort_requested = False
        if hasattr(self, "skills"):
            del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_local_base_drivers()

        for _ in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        self._initialize_world_recorder()

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
        if self._skills_use_dag:
            return self.update_dag_skill_states(skills, episode_success, should_continue)

        current_skills = skills[0]

        # Check if any skills remain
        if not any(current_skills.values()):
            skills.pop(0)
            if skills:
                should_continue = self.plan_first_skill(skills, should_continue)
            return episode_success, should_continue

        # Update each robot's skills
        for robot_name, skill_sequences in current_skills.items():
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
                            self._record_skill_failure(
                                robot_name,
                                start_lr_skill,
                                fallback_reason="skill_reported_unsuccessful",
                                fallback_message="Skill completed but reported unsuccessful status.",
                            )
                            episode_success = False
                            should_continue = False
                        lr_skill_list.remove(start_lr_skill)

                        # Do not plan or publish the next target after a failed
                        # prerequisite Skill.  The episode is already stopping,
                        # and showing a Place intent in that state would imply a
                        # target that will never be executed.
                        if skill_success and lr_skill_list:
                            next_skill = lr_skill_list[0]
                            self._activate_skill_collision_world(next_skill)
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
                        self._activate_skill_collision_world(skill[0])
                        skill[0].simple_generate_manip_cmds()
                        if len(skill[0].manip_list) == 0:
                            self._safety_failure_reason = getattr(
                                skill[0],
                                "failure_reason",
                                "NO_COLLISION_FREE_PLAN",
                            ) or "NO_COLLISION_FREE_PLAN"
                            should_continue = not skill[0].is_ready()
        return episode_success, should_continue

    @staticmethod
    def _dag_skill_done(skills):
        return all(node["state"] == "succeeded" for node in skills["nodes"])

    def _skills_complete(self):
        if self._skills_use_dag:
            return self._dag_skill_done(self.skills)
        return not self.skills

    def _dag_ready_to_start(self, node, nodes_by_id):
        return node["state"] == "pending" and all(
            nodes_by_id[dep_id]["state"] == "succeeded" for dep_id in node["depends_on"]
        )

    def _start_dag_ready_skills(self, skills, should_continue):
        nodes_by_id = skills["nodes_by_id"]
        arm_manipulation_running = any(
            node["state"] == "running"
            and node["controller_name"] in {"left", "right"}
            and node.get("collision_world_mode") != PASSTHROUGH_MODE
            for node in skills["nodes"]
        )
        for node in skills["nodes"]:
            if not self._dag_ready_to_start(node, nodes_by_id):
                continue

            is_arm_manipulation = (
                node["controller_name"] in {"left", "right"}
                and node.get("collision_world_mode") != PASSTHROUGH_MODE
            )
            if arm_manipulation_running and is_arm_manipulation:
                continue

            skill = node["skill"]
            self._activate_skill_collision_world(skill)
            skill.simple_generate_manip_cmds()
            if hasattr(skill, "visualize_target"):
                skill.visualize_target(self.world)
            if len(skill.manip_list) == 0 and skill.is_ready():
                self._record_skill_failure(
                    node["robot_name"],
                    skill,
                    fallback_reason="empty_manip_list",
                    fallback_message="Skill planning produced no manipulation commands.",
                )
                node["state"] = "failed"
                should_continue = False
                continue
            node["state"] = "running"
            arm_manipulation_running = arm_manipulation_running or is_arm_manipulation
            if len(skill.manip_list) == 0 and not skill.is_ready():
                should_continue = True
        return should_continue

    def _collect_dag_skill_actions(self, skills):
        actions_by_robot = defaultdict(list)
        running_nodes = [node for node in skills["nodes"] if node["state"] == "running"]
        record_flag = True

        for node in running_nodes:
            skill = node["skill"]
            if not skill.is_feasible():
                self._record_skill_failure(
                    node["robot_name"],
                    skill,
                    fallback_reason="skill_not_feasible",
                    fallback_message="Skill feasibility check failed before completion.",
                )
                node["state"] = "failed"
                return {}, False, True
            if not skill.is_record():
                record_flag = False

            if skill.is_ready():
                if not skill.manip_list:
                    self._record_skill_failure(
                        node["robot_name"],
                        skill,
                        fallback_reason="empty_manip_list",
                        fallback_message="Running skill has no manipulation commands to execute.",
                    )
                    node["state"] = "failed"
                    return {}, False, True
                actions_by_robot[node["robot_name"]].append(self._forward_or_hold(skill))

        action_dict = {}
        for robot_name, actions in actions_by_robot.items():
            if actions:
                action_dict[robot_name] = {
                    "joint_positions": np.concatenate([a["joint_positions"] for a in actions]),
                    "joint_indices": np.concatenate([a["joint_indices"] for a in actions]),
                    "raw_action": actions,
                }

        return action_dict, record_flag, False

    def _validate_robot_action_indices(self, action_dict: dict):
        for robot_name, robot_action in (action_dict or {}).items():
            robot = self.task.robots.get(robot_name)
            if robot is None:
                continue
            base_indices = set(getattr(robot, "base_steering_joint_indices", []) or []) | set(
                getattr(robot, "base_wheel_joint_indices", []) or []
            )
            if not base_indices:
                continue
            action_indices = np.asarray(robot_action.get("joint_indices", []), dtype=np.int64).reshape(-1)
            overlap = sorted(base_indices & {int(value) for value in action_indices.tolist()})
            if not overlap:
                continue
            dof_names = list(getattr(getattr(robot, "_articulation_view", None), "dof_names", []) or [])
            overlap_names = [
                dof_names[index] if 0 <= int(index) < len(dof_names) else str(index)
                for index in overlap
            ]
            raise RuntimeError(
                "Manipulator action joint mapping overlaps mobile base joints: "
                f"robot={robot_name}, indices={overlap}, names={overlap_names}"
            )

    def update_dag_skill_states(self, skills, episode_success, should_continue):
        for node in skills["nodes"]:
            if node["state"] != "running":
                continue

            skill = node["skill"]
            skill.update()
            if skill.is_done():
                skill_success = bool(skill.is_success())
                skill.complete_target_intent(skill_success)
                if not skill_success:
                    self._record_skill_failure(
                        node["robot_name"],
                        skill,
                        fallback_reason="skill_reported_unsuccessful",
                        fallback_message="Skill completed but reported unsuccessful status.",
                    )
                    node["state"] = "failed"
                    episode_success = False
                    should_continue = False
                else:
                    node["state"] = "succeeded"

            if hasattr(skill, "visualize_target"):
                skill.visualize_target(self.world)

        if should_continue:
            should_continue = self._start_dag_ready_skills(skills, should_continue)

        if any(node["state"] == "failed" for node in skills["nodes"]):
            return False, False
        return episode_success, should_continue

    @staticmethod
    def _skill_display_name(skill) -> str:
        skill_cfg = getattr(skill, "skill_cfg", None)
        if skill_cfg is not None:
            cfg_name = skill_cfg.get("name", None)
            if cfg_name:
                return str(cfg_name)
        return str(skill.__class__.__name__).lower()

    def _activate_skill_collision_world(self, skill) -> str:
        mode = getattr(skill, "collision_world_mode", PASSTHROUGH_MODE)
        if mode == PASSTHROUGH_MODE:
            return mode
        controller = getattr(skill, "controller", None)
        activate = getattr(controller, "activate_collision_world_mode", None)
        if not callable(activate):
            if mode == "physics_schema":
                raise RuntimeError(
                    f"Physics-schema Skill {self._skill_display_name(skill)!r} "
                    "requires an active manipulator controller"
                )
            return mode
        attached_entity = None
        if self.collision_scene_manager is not None:
            attached_entity = self.collision_scene_manager.get_attached_entity(
                controller.name, controller.lr_name
            )
        runtime_mode = resolve_runtime_skill_collision_world_mode(
            getattr(skill, "skill_cfg", {}),
            self.requested_collision_world_mode,
            attached_object=attached_entity is not None,
        )
        setattr(skill, "_physics_schema_active_object", attached_entity)
        setattr(skill, "effective_collision_world_mode", runtime_mode)
        adapter_state = (mode, runtime_mode, attached_entity)
        if runtime_mode != mode and getattr(
            skill, "_last_collision_world_adapter_log_state", None
        ) != adapter_state:
            LOGGER.info(
                "[CollisionWorld] runtime-adapter skill=%s configured_mode=%s "
                "effective_mode=%s attached_object=%s controller=%s/%s",
                self._skill_display_name(skill),
                mode,
                runtime_mode,
                attached_entity,
                controller.name,
                controller.lr_name,
            )
        setattr(skill, "_last_collision_world_adapter_log_state", adapter_state)
        mode = runtime_mode
        activate(mode)
        return mode

    def get_failure_context(self) -> dict:
        """Return the recorded failure fields for the current episode."""
        json_data_logger = getattr(getattr(self, "logger", None), "json_data_logger", {})
        for robot_name, metadata in json_data_logger.items():
            if not isinstance(metadata, dict):
                continue
            context = {
                "robot": robot_name,
                "failed_skill": metadata.get("failed_skill"),
                "failed_skill_id": metadata.get("failed_skill_id"),
                "failure_reason": metadata.get("failure_reason"),
                "failure_message": metadata.get("failure_message"),
            }
            if any(value not in (None, "") for key, value in context.items() if key != "robot"):
                return context
        return {}

    def _record_skill_failure(self, robot_name: str, skill, fallback_reason: str, fallback_message: str):
        failure_reason = str(getattr(skill, "failure_reason", "") or fallback_reason)
        error_message = str(getattr(skill, "error_message", "") or fallback_message)
        failed_skill = self._skill_display_name(skill)
        skill_id = getattr(skill, "skill_id", None)
        self.logger.add_json_data(robot_name, "episode_success", False)
        self.logger.add_json_data(robot_name, "failed_skill", failed_skill)
        if skill_id is not None:
            self.logger.add_json_data(robot_name, "failed_skill_id", str(skill_id))
        self.logger.add_json_data(robot_name, "failure_reason", failure_reason)
        self.logger.add_json_data(robot_name, "failure_message", error_message)
        self.logger.info(
            "Episode failed: robot=%s skill=%s skill_id=%s reason=%s message=%s",
            robot_name,
            failed_skill,
            str(skill_id) if skill_id is not None else "",
            failure_reason,
            error_message,
        )

    def plan_first_skill(self, skills, should_continue):
        if self._skills_use_dag:
            return self._start_dag_ready_skills(skills, should_continue)

        for _, robot_skill_list in skills[0].items():
            for lr_skill_list in robot_skill_list[0]:
                # Dual-arm configs legitimately use an empty list for the idle
                # arm.  Do not index it as if it contained a first Skill.
                if not lr_skill_list:
                    continue
                self._activate_skill_collision_world(lr_skill_list[0])
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

    def _dump_navigation_debug_snapshots(self, tag: str):
        for robot_name, driver in getattr(self, "_local_base_drivers", {}).items():
            try:
                output_dir = os.path.join("output", "local_navigation", str(getattr(self.task, "name", "scene")))
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, f"workflow_{tag}_{robot_name}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({"robot_name": robot_name, "tag": tag, "base_action": driver.get_logging_action_snapshot(), "base_state": driver.get_logging_state_snapshot()}, handle, indent=2)
            except Exception as exc:
                print(f"[local-navigation] Failed to dump debug snapshot for '{robot_name}': {exc}")

    def _iter_active_skills(self):
        if not self.skills:
            return
        if self._skills_use_dag:
            for node in self.skills["nodes"]:
                if node["state"] == "running":
                    yield node["robot_name"], node["skill"]
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
                controller.complete_terminal_place_on_contact(command)
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
            LOGGER.info(
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
                LOGGER.info(
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
        self._activate_skill_collision_world(skill)
        command = skill.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            return controller.forward(command)
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
        step_render = bool(getattr(self, "planning_step_render", False))

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        step_id = 0
        episode_success = True
        should_continue = True
        max_episode_length = self.task_cfg["data"]["max_episode_length"]
        episode_stats = {"succeed_times": 0, "current_times": 0}

        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        for _ in range(10):
            obs = self._get_observations()
            # self._init_static_objects(self.task)
            self._step_world(render=step_render)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        should_continue = self.plan_first_skill(self.skills, should_continue)

        while not (
            step_id >= max_episode_length
            or (self._skills_complete() and not episode_success)
            or (not should_continue)
        ):
            obs = self._get_observations()
            action_dict = {}
            self._active_execution_step_id = step_id
            record_flag = True
            if self.skills and should_continue and not self._execution_safety_precheck(
                step_id, action_dict
            ):
                episode_success = False
                should_continue = False
            if self._skills_use_dag and not self._skills_complete() and should_continue:
                action_dict, record_flag, skill_failed = self._collect_dag_skill_actions(self.skills)
                if skill_failed:
                    episode_success = False
                    should_continue = False
            elif not self._skills_use_dag and self.skills and should_continue:
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
                            failed_skill = next(
                                (skill[0] for skill in skill_sequences[0] if skill[0] and not skill[0].is_feasible()),
                                None,
                            )
                            if failed_skill is not None:
                                self._record_skill_failure(
                                    robot_name,
                                    failed_skill,
                                    fallback_reason="skill_not_feasible",
                                    fallback_message="Skill feasibility check failed before completion.",
                                )
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
            elif self._skills_complete() and episode_success:
                end = True
                for j_idx in range(1, 7):
                    self._step_world(render=step_render)
                    obs = self._get_observations()
                    log_dual_obs(
                        self.logger,
                        obs,
                        action_dict,
                        self.controllers,
                        self._local_base_drivers,
                        step_idx=step_id + j_idx,
                    )
                    self.world_recorder.record()

                self.logger.info(
                    "Task is successful, mode=generate_seq, final_step=%d, recorded_tail_frames=%d",
                    step_id,
                    6,
                )
                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(
                    self.logger,
                    obs,
                    action_dict,
                    self.controllers,
                    self._local_base_drivers,
                    step_idx=step_id,
                )
                self.world_recorder.record()
            self._validate_robot_action_indices(action_dict)
            self.task.apply_action(action_dict)
            self._step_world(render=step_render)

            step_id += 1
            if not self._skills_complete():
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
            if step_id >= max_episode_length:
                self._dump_navigation_debug_snapshots("step_limit")
            elif not should_continue:
                self._dump_navigation_debug_snapshots("skill_stop")
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
        self.logger.save(save_path, timestamp, save_img=True)
        saved_dirs = [
            os.path.join(
                save_path,
                str(robot_name),
                str(self.logger.task_dir),
                str(self.logger.collect_info),
                timestamp,
            )
            for robot_name in self.logger.proprio_data_logger
        ]
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
        self._destroy_local_base_drivers()
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

        self._ensure_local_base_driver_bindings()
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        for _ in range(10):
            obs = self._get_observations()
            # self._init_static_objects(self.task)
            self._step_world(render=True)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        should_continue = self.plan_first_skill(self.skills, should_continue)

        # while True:
        #     obs = self.world.get_observations()
        #     # self._init_static_objects(self.task)
        #     self.world.step(render=True)

        while not (
            step_id >= max_episode_length
            or (self._skills_complete() and not episode_success)
            or (not should_continue)
        ):
            obs = self._get_observations()
            action_dict = {}
            self._active_execution_step_id = step_id
            record_flag = True
            if self.skills and should_continue and not self._execution_safety_precheck(
                step_id, action_dict
            ):
                episode_success = False
                should_continue = False
            if self._skills_use_dag and not self._skills_complete() and should_continue:
                action_dict, record_flag, skill_failed = self._collect_dag_skill_actions(self.skills)
                if skill_failed:
                    episode_success = False
                    should_continue = False
            elif not self._skills_use_dag and self.skills and should_continue:
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
                            failed_skill = next(
                                (skill[0] for skill in skill_sequences[0] if skill[0] and not skill[0].is_feasible()),
                                None,
                            )
                            if failed_skill is not None:
                                self._record_skill_failure(
                                    robot_name,
                                    failed_skill,
                                    fallback_reason="skill_not_feasible",
                                    fallback_message="Skill feasibility check failed before completion.",
                                )
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
            elif self._skills_complete() and episode_success:
                end = True
                for j_idx in range(1, 7):
                    self._step_world(render=True)
                    obs = self._get_observations()
                    log_dual_obs(
                        self.logger,
                        obs,
                        action_dict,
                        self.controllers,
                        self._local_base_drivers,
                        step_idx=step_id + j_idx,
                    )
                    self._record_rgb_depth(step_id + j_idx)
                    self.world_recorder.record()
                length = step_id + 6
                self.logger.info(
                    "Task is successful, mode=plan_with_render, final_step=%d, recorded_length=%d",
                    step_id,
                    length,
                )
                episode_stats["succeed_times"] += 1
                should_continue = False

            if record_flag:
                log_dual_obs(
                    self.logger,
                    obs,
                    action_dict,
                    self.controllers,
                    self._local_base_drivers,
                    step_idx=step_id,
                )
                self._record_rgb_depth(step_id)
            self._validate_robot_action_indices(action_dict)
            self.task.apply_action(action_dict)
            self._step_world(render=True)

            step_id += 1
            if not self._skills_complete():
                episode_success, should_continue = self.update_skill_states(
                    self.skills, episode_success, should_continue
                )

        self.length = length
        if end:
            self._episode_failed = False
            self._episode_failure_reason = ""
            return length
        else:
            self._dump_navigation_debug_snapshots("incomplete")
            self.length = step_id
        if getattr(self, "save_failed", False):
            if step_id == 0:
                # LMDB converts state/action labels by shifting one sample.
                # Persist two measured hold frames so a planning-time safe
                # failure remains a valid, inspectable episode after that
                # shift instead of producing an empty gripper/action list.
                action_dict = {}
                for failure_step in range(2):
                    obs = self._get_observations()
                    log_dual_obs(
                        self.logger,
                        obs,
                        action_dict,
                        self.controllers,
                        step_idx=failure_step,
                    )
                    self._record_rgb_depth(failure_step)
                    self.world_recorder.record()
                    self._step_world(render=True)
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
