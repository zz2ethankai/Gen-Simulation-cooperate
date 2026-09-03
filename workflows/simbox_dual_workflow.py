import glob
import json
import logging
import math
import os
import pickle
import random
import time
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
try:
    from omni.physx import acquire_physx_interface
except ImportError:
    # Isaac Sim 6 renamed the public PhysX accessor.
    from omni.physx import get_physx_interface as acquire_physx_interface
from tqdm import tqdm
from yaml import Loader
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

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
    DIRECT_EXECUTION_MODE,
    PHYSICS_SCHEMA_MODE,
    PASSTHROUGH_MODE,
    canonicalize_planning_config,
    is_passthrough_skill,
    resolve_collision_world_mode,
    resolve_skill_collision_world_mode,
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
from core.planning.skill_dag_compiler import compile_skill_dag_configs
from core.loggers.utils import log_dual_obs
from core.robots.profile import load_robot_profile, project_runtime_config
from core.skills import get_skill_cls
from core.tasks import get_task_cls
from core.telemetry import SkillTimingRecorder
from core.utils.collision_utils import filter_collisions
from core.utils.episode_event_writer import emit_episode_saved
from core.utils.relation_predicates import evaluate_compiled_place_relations
from core.utils.task_data import normalize_runtime_data_config
from core.utils.utils import set_random_seed
from core.visualization.curobo_trajectory import create_curobo_trajectory_visualizer
from core.visualization.skill_targets import create_skill_target_visualizer


LOGGER = logging.getLogger("de_logger")
VELOCITY_TRACE_LOGGER = logging.getLogger("de_velocity_trace")


class _PassiveSkillController:
    """Placeholder controller for skills that do not emit manipulator actions."""

    def __init__(self, *, robot_name: str, controller_name: str):
        self.name = robot_name
        self.robot_file = f"{controller_name}_passive_skill_controller"
        self._gripper_state = 1.0
        # Passthrough Skills are compiled without a manipulator controller.
        # Keep the port slots explicit so the compiler never has to reflect
        # through a controller façade while constructing them.
        self.skill_runtime = None

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
        timing_recorder=None,
    ):
        self.scene_info = scene_info
        self.step_replay = False
        self.random_seed = random_seed
        self.planning_step_render = bool(planning_step_render)
        self.timing_recorder = (
            timing_recorder
            if timing_recorder is not None
            else SkillTimingRecorder(retain_records=True, best_effort=True)
        )
        self._timing_record_simulation_steps = False
        self._local_base_drivers = {}
        self._static_map_cache = {}
        self._static_map_debug_by_robot = {}
        self._static_map_layout_epoch = 0
        super().__init__(world, task_cfg_path)

    @staticmethod
    def _skill_requires_controller(skill_cfg: dict) -> bool:
        return not is_passthrough_skill(skill_cfg)

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
        task_cfg_dir = os.path.dirname(os.path.abspath(task_cfg_path))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Merge robot configs for each task
        for task_cfg in task_cfgs:
            normalize_runtime_data_config(task_cfg, task_cfg_path)
            asset_root = task_cfg.get("asset_root")
            if asset_root and not os.path.isabs(str(asset_root)):
                # Both downloaded task formats exist in the workspace:
                # older files store asset_root relative to the task YAML,
                # while Scene-8 files use a repository-relative path such as
                # ``InternDataAssets/assets/custom/scene_8/01_kitchen``.
                # Resolve an existing repository-relative candidate first,
                # then retain compatibility with task-relative files.
                raw_asset_root = str(asset_root)
                repo_asset_root = os.path.abspath(os.path.join(repo_root, raw_asset_root))
                task_asset_root = os.path.abspath(os.path.join(task_cfg_dir, raw_asset_root))
                if os.path.isdir(repo_asset_root):
                    task_cfg["asset_root"] = repo_asset_root
                else:
                    task_cfg["asset_root"] = task_asset_root

            env_map = task_cfg.get("env_map")
            envmap_lib = env_map.get("envmap_lib") if isinstance(env_map, dict) else None
            if envmap_lib and not os.path.isabs(str(envmap_lib)):
                envmap_candidates = [
                    os.path.join(str(task_cfg.get("asset_root", "")), str(envmap_lib)),
                    os.path.join(task_cfg_dir, str(envmap_lib)),
                ]
                repo_relative = str(envmap_lib)
                while repo_relative.startswith("../"):
                    repo_relative = repo_relative[3:]
                envmap_candidates.append(os.path.join(repo_root, repo_relative))
                for candidate in envmap_candidates:
                    if os.path.isdir(candidate):
                        env_map["envmap_lib"] = os.path.abspath(candidate)
                        break
            self._merge_robot_configs(task_cfg, Path(task_cfg_path))
        return task_cfgs

    def _merge_robot_configs(self, task_cfg: dict, task_cfg_path: Path):
        """Resolve each task robot from its canonical profile."""
        for robot in task_cfg.get("robots", []):
            config_path = robot.get("robot_config_file")
            if not config_path:
                raise ValueError(
                    f"robot instance {robot.get('name')!r} must declare robot_config_file"
                )
            profile = load_robot_profile(config_path)
            runtime_config = project_runtime_config(
                profile,
                overrides=robot,
                task_path=task_cfg_path,
                asset_root=task_cfg.get("asset_root"),
            )
            robot.clear()
            robot.update(runtime_config)
            # Merge the referenced mobile-base config file (wheel/chassis and
            # local-navigation parameters) into the projected base section.
            base_cfg = robot.get("base")
            if isinstance(base_cfg, dict) and base_cfg.get("base_config_file"):
                self._merge_base_configs(base_cfg)

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

    @staticmethod
    def _create_native_support_collision_proxy(stage, collision_root_path: str, source_prim_path: str, index: int):
        """Create a static sibling collider for a neglected referenced fixture.

        Isaac Sim 6 can compose a referenced child after collision collections
        have been expanded.  A collection target below that reference is then
        present in USD but absent from the PhysX group.  Authoring this small
        native collider in the non-referenced collision scope makes the group
        membership deterministic while preserving the table's world-space
        support footprint.
        """

        source_prim = stage.GetPrimAtPath(source_prim_path)
        if not source_prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        try:
            bbox = bbox_cache.ComputeWorldBound(source_prim).ComputeAlignedBox()
            min_point = bbox.GetMin()
            max_point = bbox.GetMax()
            min_values = [float(value) for value in min_point]
            max_values = [float(value) for value in max_point]
        except Exception:
            LOGGER.warning(
                "[CollisionGroups] cannot compute support bbox for %s",
                source_prim_path,
                exc_info=True,
            )
            return None

        size = [max_value - min_value for min_value, max_value in zip(min_values, max_values)]
        if any(value <= 1.0e-6 for value in size):
            LOGGER.warning(
                "[CollisionGroups] empty support bbox for %s: min=%s max=%s",
                source_prim_path,
                min_values,
                max_values,
            )
            return None

        UsdGeom.Scope.Define(stage, collision_root_path)
        proxy_path = f"{collision_root_path}/support_proxy_{index}"
        proxy_geom = UsdGeom.Cube.Define(stage, proxy_path)
        proxy_geom.CreateSizeAttr().Set(1.0)
        proxy_xform = UsdGeom.Xformable(proxy_geom.GetPrim())
        proxy_xform.AddTranslateOp().Set(
            tuple((min_value + max_value) * 0.5 for min_value, max_value in zip(min_values, max_values))
        )
        proxy_xform.AddScaleOp().Set(tuple(size))
        UsdPhysics.CollisionAPI.Apply(proxy_geom.GetPrim())
        UsdGeom.Imageable(proxy_geom.GetPrim()).MakeInvisible()
        LOGGER.info(
            "[CollisionGroups] created native support proxy=%s source=%s min=%s max=%s",
            proxy_path,
            source_prim_path,
            min_values,
            max_values,
        )
        return proxy_path

    def _configure_collision_groups(self):
        """Create collision groups after Isaac 6 has built the task scene.

        ``World.add_task`` only registers the task.  Isaac Sim 6 creates the
        task objects during ``World.reset``; configuring collections before
        that lifecycle point leaves referenced fixture children unresolved.
        """

        prim_paths = []  # do not collide with each other
        global_collision_paths = []  # collide with everything
        collision_root_path = "/World/collisions"

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
            collision_filter_names = list(neglect_collision_names)
            collision_approximation = str(candidate.get("collision_approximation", ""))
            is_explicit_support_body = (
                candidate.get("target_class") == "GeometryObject"
                and collision_approximation.replace("_", "").lower() == "supportbodybbox"
            )
            if is_explicit_support_body and not any(
                name in candidate["name"] for name in collision_filter_names
            ):
                # Referenced support meshes can remain visible in USD while
                # their descendant shape is absent from the first PhysX
                # collection expansion.  The explicit supportBodyBBox
                # contract already provides a bounded native proxy; include
                # it in the global group for dynamic-object contact too.
                collision_filter_names.append(candidate["name"])
            for neglect_collision_name in collision_filter_names:
                if neglect_collision_name not in candidate["name"]:
                    continue
                # Preserve the original task contract: a neglected fixture is
                # in the robot's filtered group, while dynamic objects stay in
                # global_group and can collide with the fixture's authored or
                # task-local proxy descendants.  Keeping the fixture in
                # global_group disables object/support contact under Isaac 6's
                # inverted collision-group filtering and only becomes visible
                # after a failed-generation reset.
                if candidate_prim_path not in prim_paths:
                    prim_paths.append(candidate_prim_path)
                global_collision_paths.remove(candidate_prim_path)

        filter_collisions(
            self.stage,
            self.world.get_physics_context().prim_path,
            collision_root_path,
            prim_paths,
            global_collision_paths,
        )
        if os.environ.get("INTERNDATA_DEBUG_RESET_LIFECYCLE") == "1":
            collision_group_debug = {}
            collision_root = self.stage.GetPrimAtPath(collision_root_path)
            for group_prim in collision_root.GetChildren() if collision_root.IsValid() else []:
                if str(group_prim.GetTypeName()) != "PhysicsCollisionGroup":
                    continue
                includes = group_prim.GetRelationship("collection:colliders:includes")
                filtered_groups = group_prim.GetRelationship("physics:filteredGroups")
                collision_group_debug[str(group_prim.GetPath())] = {
                    "includes": [str(path) for path in includes.GetTargets()] if includes else [],
                    "filtered_groups": [
                        str(path) for path in filtered_groups.GetTargets()
                    ]
                    if filtered_groups
                    else [],
                }
            collision_table_debug = {}
            try:
                collision_table = UsdPhysics.CollisionGroup.ComputeCollisionGroupTable(self.stage)
                group_paths = sorted(collision_group_debug)
                for index, group_a in enumerate(group_paths):
                    for group_b in group_paths[index:]:
                        collision_table_debug[f"{group_a}|{group_b}"] = bool(
                            collision_table.IsCollisionEnabled(Sdf.Path(group_a), Sdf.Path(group_b))
                        )
            except Exception as exc:
                collision_table_debug = {"error": repr(exc)}
            LOGGER.warning(
                "[CollisionGroups] prim_paths=%s global_paths=%s groups=%s table=%s",
                prim_paths,
                global_collision_paths,
                collision_group_debug,
                collision_table_debug,
            )

    def _resolve_arena_file_path(self, arena_file_path: str | None) -> str | None:
        if arena_file_path and os.path.exists(arena_file_path):
            return arena_file_path

        if arena_file_path:
            task_cfg_dir = os.path.dirname(os.path.abspath(self.task_cfg_path))
            arena_from_task_cfg = os.path.join(task_cfg_dir, arena_file_path)
            if os.path.exists(arena_from_task_cfg):
                return arena_from_task_cfg

            asset_root = self.task_cfg.get("asset_root")
            if asset_root:
                arena_from_asset_root = os.path.join(asset_root, arena_file_path)
                if os.path.exists(arena_from_asset_root):
                    return arena_from_asset_root
            raise FileNotFoundError(f"arena_file does not exist: {arena_file_path}")
        raise ValueError("task config must define arena_file")

    def reset(self, need_preload: bool = True):
        self.invalidate_static_map_cache("reset")
        self._finish_active_skill_timings("episode_reset")
        self._timing_record_simulation_steps = False
        try:
            self.timing_recorder.reset()
        except Exception:
            LOGGER.debug("Failed to reset skill timing recorder", exc_info=True)
        self.close()
        self._destroy_local_base_drivers()

        if SimulationManager.get_physics_sim_view() is not None:
            self.world.stop()
        if SimulationManager.get_physics_sim_view() is not None:
            SimulationManager.invalidate_physics()
        if SimulationManager.get_physics_sim_view() is not None:
            raise RuntimeError("Physics simulation view remains after stop/invalidate")

        # A previous task can remain registered if scene setup fails during world.reset().
        # Clear the world before constructing the next task so retries do not trip the
        # duplicate-name guard in omni.isaac.core.world.World.add_task().
        if self.world.get_current_tasks() or self.world.is_tasks_scene_built():
            self.world.clear()
        # source code noted this as debug, so it could be removed later
        from isaacsim.core.utils.viewports import set_camera_view

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

        # Arena generators express fixture textures relative to the arena
        # YAML, whereas object asset paths are resolved from ``asset_root``.
        # Resolve both an explicit texture_file and a texture_lib here so a
        # scene remains portable when its runs directory is copied elsewhere.
        arena_dir = os.path.dirname(arena_file_path)
        for fixture_cfg in arena.get("fixtures", []):
            texture_cfg = fixture_cfg.get("texture")
            if not isinstance(texture_cfg, dict):
                continue

            texture_file = texture_cfg.get("texture_file")
            if texture_file and not os.path.isabs(str(texture_file)):
                arena_texture_file = os.path.abspath(
                    os.path.join(arena_dir, str(texture_file))
                )
                if os.path.isfile(arena_texture_file):
                    texture_cfg["texture_file"] = arena_texture_file

            texture_lib = texture_cfg.get("texture_lib")
            if not texture_lib or os.path.isabs(str(texture_lib)):
                continue
            arena_texture_path = os.path.abspath(
                os.path.join(arena_dir, str(texture_lib))
            )
            if os.path.isdir(arena_texture_path):
                texture_cfg["texture_lib"] = arena_texture_path

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

        # Normalize before constructing the task so every downstream owner
        # (collision groups, task.cfg, controllers, and Skills) observes the
        # same single-world contract.
        self.task_cfg = canonicalize_planning_config(
            self.task_cfg,
            config_path=self.task_cfg.get("metadata", {}).get("source_yaml")
            if isinstance(self.task_cfg.get("metadata", {}), dict)
            else self.task_cfg_path,
        )
        planning_cfg = self.task_cfg.get("planning", {})
        collision_cfg = planning_cfg.get("collision_world", {})
        safety_cfg = planning_cfg.get("execution_safety", {})
        self.requested_collision_world_mode = PHYSICS_SCHEMA_MODE
        self.collision_world_mode, collision_mode_reason = resolve_collision_world_mode(
            self.task_cfg, PHYSICS_SCHEMA_MODE
        )
        LOGGER.warning(
            "[CollisionWorld] requested_mode=%s resolved_mode=%s reason=%s",
            self.requested_collision_world_mode,
            self.collision_world_mode,
            collision_mode_reason,
        )
        self._validate_planning_contract(self.task_cfg, self.collision_world_mode)

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
        # World.reset() invokes BananaBaseTask.set_up_scene() before it
        # finalizes the PhysX scene.  Register the collision setup there so
        # native support proxies and collision-group relationships are part of
        # the first physics build, rather than being authored into an already
        # initialized simulation view.
        self.task._before_physics_scene_finalize = self._configure_collision_groups
        self.world.add_task(self.task)

        self.world.reset()
        # Hold configured virtual-base DOFs before the first Physics step, then
        # keep the local world-step and fixed-start-pose sequence authoritative.
        self._enable_manipulation_base_holds()
        self._step_world(render=True)
        self._initialize_task_physics_views()
        self._set_fixed_robot_start_poses_after_reset()
        # Pass the canonical task-entity names to the collision compiler.  It
        # resolves each name to its unique collider; no Prim-path or reason
        # mapping is performed by the workflow.
        collision_manager_cfg = dict(collision_cfg)
        collision_manager_cfg["mode"] = PHYSICS_SCHEMA_MODE
        planning_exclusions = list(planning_cfg.get("planning_exclusions", []))
        collision_manager_cfg["planning_exclusions"] = planning_exclusions
        self.collision_scene_manager = CollisionSceneManager(
            self.stage, self.task, collision_manager_cfg, safety_cfg
        )
        self.execution_safety_enabled = bool(
            safety_cfg.get(
                "enabled", True
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
        self._failure_subtask_ids = set()
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
            self.collision_scene_manager.initialize_contact_views(self.world.physics_sim_view)
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_local_base_drivers()

        self._completed_relation_skills = []

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
            )
        self.logger = LmdbLogger(
            task_dir=self.task_cfg["data"]["task_dir"],
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
            collect_info=self.task_cfg["data"]["collect_info"],
            version=self.task_cfg["data"].get("version", "v1.0"),
            video_fps=video_fps,
            robot_data_adapters={
                robot["name"]: robot["data_adapter"]
                for robot in self.task_cfg["robots"]
                if robot.get("data_adapter")
            },
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
            from isaacsim.util.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        else:
            draw = None

        return self._initialize_skill_dag(task, task_cfg, controllers, world, draw)

    def _start_skill_timing(self, robot_name, skill):
        """Bind one recorder scope to one skill/controller pair."""

        scope = getattr(skill, "_timing_scope", None)
        if scope is not None and not bool(getattr(scope, "_finished", False)):
            return scope
        recorder = getattr(self, "timing_recorder", None)
        if recorder is None:
            return None
        runtime = skill.skill_runtime
        metadata = {
            "robot": str(robot_name),
            "controller": str(runtime.arm_name if runtime is not None else ""),
            "skill_id": str(getattr(skill, "skill_id", "")),
        }
        try:
            scope = recorder.start_skill(self._skill_display_name(skill), metadata=metadata)
        except Exception:
            LOGGER.debug("Failed to start skill timing", exc_info=True)
            return None
        setattr(skill, "_timing_scope", scope)
        setattr(skill, "_timing_recorder", recorder)
        return scope

    def _start_skill_execution_phase(self, skill, name=None):
        phase = getattr(skill, "_timing_execution_phase", None)
        if phase is not None and not bool(getattr(phase, "_finished", False)):
            return phase
        scope = getattr(skill, "_timing_scope", None)
        if scope is None:
            return None
        phase_name = name or f"{self._skill_display_name(skill)}.execution"
        try:
            phase = scope.execution(
                phase_name,
                metadata={"skill": self._skill_display_name(skill)},
            ).start()
        except Exception:
            LOGGER.debug("Failed to start skill execution timing", exc_info=True)
            return None
        setattr(skill, "_timing_execution_phase", phase)
        setattr(skill, "_timing_motion_phase", None)
        setattr(skill, "_timing_motion_phase_key", None)
        return phase

    def _start_skill_motion_phase(self, skill, command):
        """Track one MotionPhaseCommand inside the Skill execution span."""

        execution_phase = getattr(skill, "_timing_execution_phase", None)
        scope = getattr(skill, "_timing_scope", None)
        if execution_phase is None or scope is None:
            return None
        key = (id(command), str(getattr(getattr(command, "phase", None), "value", "")))
        current = getattr(skill, "_timing_motion_phase", None)
        if current is not None and getattr(skill, "_timing_motion_phase_key", None) == key:
            return current
        if current is not None and not bool(getattr(current, "_finished", False)):
            try:
                current.finish(success=True)
            except Exception:
                LOGGER.debug("Failed to close previous motion timing phase", exc_info=True)
        try:
            phase_name = f"motion.{getattr(getattr(command, 'phase', None), 'value', 'unknown')}"
            phase = scope.phase(
                phase_name,
                category="execution",
                metadata={
                    "phase": getattr(getattr(command, "phase", None), "value", "unknown"),
                    "plan_id": getattr(command, "plan_id", None),
                },
            ).start()
        except Exception:
            LOGGER.debug("Failed to start motion timing phase", exc_info=True)
            return None
        setattr(skill, "_timing_motion_phase", phase)
        setattr(skill, "_timing_motion_phase_key", key)
        return phase

    def _run_skill_planner(self, robot_name, skill):
        """Call simple_generate under an explicit planner phase."""

        scope = self._start_skill_timing(robot_name, skill)
        phase = None
        if scope is not None:
            try:
                phase = scope.phase(
                    f"{self._skill_display_name(skill)}.plan",
                    category="planner",
                    metadata={"skill": self._skill_display_name(skill)},
                ).start()
            except Exception:
                LOGGER.debug("Failed to start skill planner timing", exc_info=True)
        runtime = skill.skill_runtime
        previous_scope = runtime.push_timing_scope(scope) if runtime is not None else None
        try:
            planner = getattr(skill, "generate_manip_cmds", None)
            if not callable(planner):
                # Other legacy skills still expose the generic compatibility
                # hook.  Pick and Place use the explicit current planner
                # entrypoint above.
                planner = skill.simple_generate_manip_cmds
            planner()
        except Exception as exc:
            if phase is not None:
                try:
                    phase.finish(success=False, reason=str(exc), error=exc)
                except Exception:
                    pass
            self._finish_skill_timing(skill, False, reason=str(exc), error=exc)
            self._log_planner_timing(
                robot_name, skill, scope, phase, success=False, reason=str(exc)
            )
            raise
        finally:
            if runtime is not None:
                runtime.restore_timing_scope(previous_scope)
        if phase is not None:
            try:
                phase.finish(success=True)
            except Exception:
                pass
        if getattr(skill, "manip_list", None):
            self._log_planner_timing(robot_name, skill, scope, phase, success=True)
            self._start_skill_execution_phase(skill)
        elif bool(getattr(skill, "is_ready", lambda: True)()):
            # A manipulator Skill with no commands is a planning failure in
            # both schedulers.  Close its timing scope now so the failure is
            # retained even when the loop exits before the next update tick.
            reason = str(getattr(skill, "failure_reason", "") or "empty_manip_list")
            self._finish_skill_timing(skill, False, reason=reason)
            self._log_planner_timing(
                robot_name, skill, scope, phase, success=False, reason=reason
            )

    def _log_planner_timing(
        self, robot_name, skill, scope, phase, *, success: bool, reason: str | None = None
    ):
        """Log one planner invocation and all of its nested timing phases."""

        if scope is None:
            return
        try:
            payload = scope.to_dict()
            if phase is not None and phase.record is not None:
                duration = float(phase.record.duration_sec)
            else:
                duration = float(payload.get("duration_sec", 0.0))
            segments = {
                str(item.get("name", "unknown")): round(
                    float(item.get("duration_sec", 0.0)), 3
                )
                for item in payload.get("phases", [])
            }
            LOGGER.info(
                "[PlannerTiming] robot=%s skill=%s status=%s duration=%.3fs "
                "segments=%s reason=%s",
                robot_name,
                self._skill_display_name(skill),
                "success" if success else "failed",
                duration,
                segments,
                reason or "",
            )
        except Exception:
            # Diagnostic timing must never change planning or episode behavior.
            LOGGER.debug("Failed to log planner timing", exc_info=True)

    def _finish_skill_timing(self, skill, success, reason=None, error=None):
        scope = getattr(skill, "_timing_scope", None)
        if scope is None:
            return
        motion_phase = getattr(skill, "_timing_motion_phase", None)
        if motion_phase is not None and not bool(getattr(motion_phase, "_finished", False)):
            try:
                motion_phase.finish(success=bool(success), reason=reason, error=error)
            except Exception:
                LOGGER.debug("Failed to finish motion timing phase", exc_info=True)
        phase = getattr(skill, "_timing_execution_phase", None)
        if phase is not None and not bool(getattr(phase, "_finished", False)):
            try:
                phase.finish(success=bool(success), reason=reason, error=error)
            except Exception:
                pass
        try:
            scope.finish(success=bool(success), reason=reason, error=error)
        except Exception:
            LOGGER.debug("Failed to finish skill timing", exc_info=True)
        runtime = skill.skill_runtime
        if runtime is not None:
            try:
                runtime.clear_timing_scope(scope)
            except Exception:
                pass
        setattr(skill, "_timing_execution_phase", None)
        setattr(skill, "_timing_motion_phase", None)
        setattr(skill, "_timing_motion_phase_key", None)

    def _finish_active_skill_timings(self, reason="episode_failed"):
        seen = set()
        try:
            active_skills = self._iter_active_skills()
            for _, skill in active_skills or ():
                if id(skill) in seen:
                    continue
                seen.add(id(skill))
                self._finish_skill_timing(skill, False, reason=reason)
        except Exception:
            LOGGER.debug("Failed to finish active skill timings", exc_info=True)

    def _record_skill_simulation_steps(self):
        """Feed physics dt to the optional recorder extension, if available."""

        get_dt = getattr(self.world, "get_physics_dt", None)
        try:
            dt = float(get_dt()) if callable(get_dt) else float(getattr(self.world, "physics_dt", 0.0))
        except Exception:
            dt = 0.0
        try:
            active_skills = list(self._iter_active_skills() or ())
            if not active_skills:
                return
            record_episode_step = getattr(
                getattr(self, "timing_recorder", None),
                "record_episode_simulation_step",
                None,
            )
            if callable(record_episode_step):
                try:
                    record_episode_step(dt)
                except Exception:
                    LOGGER.debug("Failed to record episode simulation step", exc_info=True)
            for _, skill in active_skills:
                phase = getattr(skill, "_timing_motion_phase", None)
                if phase is None or bool(getattr(phase, "_finished", False)):
                    phase = getattr(skill, "_timing_execution_phase", None)
                record_step = getattr(phase, "record_simulation_step", None)
                if callable(record_step):
                    try:
                        record_step(dt)
                    except Exception:
                        LOGGER.debug("Failed to record skill simulation step", exc_info=True)
        except Exception:
            LOGGER.debug("Failed to enumerate active skill timing phases", exc_info=True)

    def _bind_skill_collision_world_mode(self, skill, skill_cfg):
        mode = resolve_skill_collision_world_mode(
            skill_cfg.get("name", ""), PHYSICS_SCHEMA_MODE
        )
        setattr(skill, "collision_world_mode", mode)
        if getattr(skill, "execution_mode", None) == DIRECT_EXECUTION_MODE:
            # Direct Skills do not have a planner world, even though the DAG
            # keeps the canonical operation metadata for scheduling.
            setattr(skill, "effective_collision_world_mode", DIRECT_EXECUTION_MODE)
        return mode

    def _initialize_skill_dag(self, task, task_cfg, controllers, world, draw):
        nodes_by_id = {}
        nodes = []
        # Compile all IDs and dependency metadata before constructing a Skill.
        # Legacy YAMLs often omit ``id``; the compiler supplies a stable
        # source-location ID and reconstructs phase/sequence barriers without
        # mutating the source task or reintroducing the old controller API.
        compiled_skills = compile_skill_dag_configs(task_cfg)

        for compiled in compiled_skills:
            robot_name = compiled.robot_name
            lr_name = compiled.controller_name
            skill_cfg = compiled.skill_cfg
            robot = task.robots[robot_name]
            arm_controllers = controllers[robot_name]
            controller_ports = arm_controllers[lr_name]
            skill_id = compiled.skill_id
            depends_on = list(compiled.depends_on)

            skill_cls = get_skill_cls(skill_cfg["name"])
            skill = skill_cls(
                robot,
                controller_ports.skill_runtime,
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
                "controller_name": lr_name,
                "collision_world_mode": skill_collision_mode,
                "execution_mode": getattr(skill, "execution_mode", None),
                "skill": skill,
                "state": "pending",
            }
            nodes_by_id[skill_id] = node
            nodes.append(node)

        self._append_return_to_initial_nodes(
            task, task_cfg, controllers, nodes, nodes_by_id
        )
        ordered_nodes = self._toposort_skill_nodes(nodes, nodes_by_id)
        return {"nodes": ordered_nodes, "nodes_by_id": nodes_by_id}

    def _append_return_to_initial_nodes(
        self, task, task_cfg, controllers, nodes, nodes_by_id
    ):
        """Gate every downstream node on a typed post-Place arm reset."""

        pick_place_cfg = task_cfg.get("planning", {}).get("pick_place", {})
        if os.environ.get("SIMBOX_RETURN_TO_EPISODE_INITIAL", "1") == "0":
            return
        if not bool(pick_place_cfg.get("return_to_episode_initial", True)):
            return
        original_nodes = list(nodes)
        for place_node in original_nodes:
            place_skill = place_node["skill"]
            if type(place_skill).__name__.lower() not in {"place", "dexplace"}:
                continue
            reset_id = f"{place_node['id']}:return_to_episode_initial"
            if reset_id in nodes_by_id:
                raise ValueError(f"duplicate synthetic Skill id: {reset_id}")
            for node in original_nodes:
                if node is place_node:
                    continue
                node["depends_on"] = [
                    reset_id if dep_id == place_node["id"] else dep_id
                    for dep_id in node["depends_on"]
                ]
            robot_name = place_node["robot_name"]
            controller_name = place_node["controller_name"]
            runtime = controllers[robot_name][controller_name].skill_runtime
            reset_cfg = {
                "name": "return_to_episode_initial",
                "agent_subtask_id": getattr(place_skill, "skill_cfg", {}).get(
                    "agent_subtask_id"
                ),
                "joint_tolerance_rad": float(
                    pick_place_cfg.get("return_joint_tolerance_rad", 0.03)
                ),
            }
            reset_skill = get_skill_cls("return_to_episode_initial")(
                task.robots[robot_name], runtime, task, reset_cfg
            )
            reset_mode = self._bind_skill_collision_world_mode(
                reset_skill, reset_cfg
            )
            reset_skill.bind_target_visualizer(
                self.skill_target_visualizer,
                robot=robot_name,
                arm=controller_name,
                skill=reset_cfg["name"],
                skill_index=len(nodes),
            )
            setattr(reset_skill, "skill_id", reset_id)
            reset_node = {
                "id": reset_id,
                "depends_on": [place_node["id"]],
                "robot_name": robot_name,
                "controller_name": controller_name,
                "collision_world_mode": reset_mode,
                "execution_mode": getattr(reset_skill, "execution_mode", None),
                "skill": reset_skill,
                "state": "pending",
            }
            nodes_by_id[reset_id] = reset_node
            nodes.append(reset_node)

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
    def _record_failure_subtask(self, skill):
        subtask_id = str(
            getattr(skill, "skill_cfg", {}).get("agent_subtask_id") or ""
        ).strip()
        if subtask_id:
            self._failure_subtask_ids.add(subtask_id)

    def _episode_failing_subtask_id(self, predicate_results=None):
        failing_subtasks = set(self._failure_subtask_ids)
        for result in predicate_results or []:
            subtask_id = str(result.get("subtask_id") or "").strip()
            if result.get("success") is False and subtask_id:
                failing_subtasks.add(subtask_id)
        if len(failing_subtasks) != 1:
            return None
        return next(iter(failing_subtasks))

    def _initialize_controllers(self, task, task_cfg, world):
        """Initialize controllers for each robot."""
        controllers = {}
        for robot in task_cfg["robots"]:
            robot_name = robot["name"]
            controllers[robot_name] = {}
            required_controller_names = self._required_controller_names(task_cfg, robot_name)
            declared_controller_names = self._skill_controller_names(task_cfg, robot_name)

            arm_profiles = robot.get("arms", {})
            if not isinstance(arm_profiles, dict):
                raise TypeError(
                    f"Robot '{robot_name}' canonical arms profile must be a mapping"
                )

            for controller_name in required_controller_names:
                arm_profile = arm_profiles.get(controller_name)
                robot_file = (
                    arm_profile.get("curobo_file")
                    if isinstance(arm_profile, dict)
                    else None
                )
                if not robot_file:
                    raise KeyError(
                        f"Robot '{robot_name}' is missing arms.{controller_name}.curobo_file"
                    )
                controllers[robot_name][controller_name] = get_controller_cls(robot["target_class"])(
                    name=robot_name,
                    robot_file=robot_file,
                    arm_name=controller_name,
                    constrain_grasp_approach=robot.get("constrain_grasp_approach", False),
                    collision_activation_distance=robot.get("collision_activation_distance", 0.03),
                    ignore_substring=robot.get("ignore_substring"),
                    task=task,
                    world=world,
                    trajectory_visualizer=self.trajectory_visualizer,
                    skill_target_visualizer=self.skill_target_visualizer,
                    collision_scene_manager=self.collision_scene_manager,
                    collision_world_mode=PHYSICS_SCHEMA_MODE,
                    timing_recorder=self.timing_recorder,
                )
                controllers[robot_name][controller_name].reset()

            passive_controller_names = (
                declared_controller_names | set(arm_profiles)
            ) - required_controller_names
            for controller_name in passive_controller_names:
                controllers[robot_name][controller_name] = _PassiveSkillController(
                    robot_name=robot_name,
                    controller_name=controller_name,
                )
        return controllers
    def _settle_scene_before_planning(self) -> None:
        """Synchronize randomized robot poses and mounted cameras before planning."""

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

    def _skill_runtime_ports_for_logging(self):
        """Return the narrow runtime mapping consumed by observation logging."""

        return {
            robot_name: {
                arm_name: arm_controller.skill_runtime
                for arm_name, arm_controller in arm_controllers.items()
                if arm_controller.skill_runtime is not None
            }
            for robot_name, arm_controllers in self.controllers.items()
        }

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

    def _reapply_manipulation_base_holds(self):
        """Refresh active base position/velocity holds at each physics boundary."""

        for robot_name, robot in getattr(self.task, "robots", {}).items():
            reapply = getattr(robot, "reapply_manipulation_base_hold", None)
            if not callable(reapply):
                continue
            try:
                reapply()
            except Exception as exc:
                raise RuntimeError(
                    f"Manipulation base hold reapply failed for '{robot_name}'"
                ) from exc

    def _recapture_manipulation_base_holds(self):
        """Refresh active hold targets after a fixed robot pose/reset write."""

        for robot_name, robot in getattr(self.task, "robots", {}).items():
            recapture = getattr(robot, "recapture_manipulation_base_hold", None)
            if not callable(recapture):
                continue
            try:
                recapture()
            except Exception as exc:
                raise RuntimeError(
                    f"Manipulation base hold recapture failed for '{robot_name}'"
                ) from exc

    def get_local_base_driver(self, robot_name: str):
        return self._local_base_drivers.get(robot_name)

    @staticmethod
    def _freeze_static_map_config(value):
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), SimBoxDualWorkFlow._freeze_static_map_config(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(SimBoxDualWorkFlow._freeze_static_map_config(item) for item in value)
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    def _static_map_revision(self):
        collision_manager = getattr(self, "collision_scene_manager", None)
        revision = getattr(collision_manager, "world_revision", None)
        if isinstance(revision, (int, np.integer)) and not isinstance(revision, bool) and revision >= 0:
            return ("collision_world", int(revision))
        return ("layout_epoch", int(self._static_map_layout_epoch))

    def invalidate_static_map_cache(self, reason: str = ""):
        """Drop canonical maps after a layout, reset, or movable-object change."""
        self._static_map_cache.clear()
        self._static_map_debug_by_robot.clear()
        self._static_map_layout_epoch += 1
        LOGGER.debug(
            "[local-navigation] invalidated static-map cache reason=%s layout_epoch=%d",
            reason,
            self._static_map_layout_epoch,
        )

    def get_or_export_static_map(self, *, robot, map_cfg: dict):
        """Return a per-Navigate read-only copy of the canonical runtime map."""
        from core.skills.navigation_geometry import StaticMap
        from core.skills.static_map_exporter import IsaacStaticMapExporter

        robot_name = str(getattr(robot, "name", "") or "")
        robot_path = str(getattr(robot, "robot_prim_path", "") or "")
        key = (
            robot_name,
            robot_path,
            self._freeze_static_map_config(map_cfg),
            self._static_map_revision(),
        )
        cached = self._static_map_cache.get(key)
        if cached is None:
            base_interface = robot.get_base_interface()
            exporter = IsaacStaticMapExporter(
                self,
                robot,
                base_interface["base_cfg"],
                map_cfg=map_cfg,
            )
            canonical = exporter.export_map()
            if not isinstance(canonical, StaticMap):
                raise TypeError("IsaacStaticMapExporter.export_map() must return StaticMap")
            debug = dict(exporter.last_export_debug)
            self._static_map_cache[key] = (canonical, debug)
            LOGGER.debug(
                "[local-navigation] generated static map robot=%s revision=%s",
                robot_name,
                key[-1],
            )
        else:
            canonical, debug = cached

        self._static_map_debug_by_robot[robot_name] = dict(debug)
        return StaticMap(
            occupancy=np.array(canonical.occupancy, dtype=np.uint8, copy=True),
            resolution=canonical.resolution,
            origin=canonical.origin,
        )

    def get_static_map_debug(self, robot):
        return dict(self._static_map_debug_by_robot.get(str(getattr(robot, "name", "") or ""), {}))

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
        # Navigation suspends the hold; manipulation resumes it and this
        # boundary reapplies both position and zero-velocity targets.
        self._reapply_manipulation_base_holds()
        self.world.step(render=render)
        if self._timing_record_simulation_steps:
            self._record_skill_simulation_steps()

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

    def _initialize_task_physics_views(self):
        """Initialize task views, then restore dynamic prim state.

        Isaac Sim 6 resets the active physics backend before the task's
        region placement is applied.  Restoring a dynamic object's pose
        before its current tensor/contact views exist can be overwritten by
        the first physics tick.  Every robot/task reset path therefore uses
        this single ordering: physics view -> contact view -> object state.
        """

        physics_sim_view = self.world.physics_sim_view
        if hasattr(self.task, "initialize_rigid_objects"):
            self.task.initialize_rigid_objects(physics_sim_view)
        if hasattr(self.task, "initialize_contact_views"):
            self.task.initialize_contact_views(physics_sim_view)
        restore_states = getattr(self.task, "restore_rigid_object_states", None)
        if callable(restore_states):
            restore_states()
        debug_reset_dynamics = getattr(self.task, "debug_reset_dynamics", None)
        if callable(debug_reset_dynamics):
            debug_reset_dynamics("after_initialize_and_restore")

    def _restore_fixed_rigid_objects_after_warmup(self, label):
        """Restore fixed rigid state before exposing the reset to planning.

        Fixed assets are still dynamic bodies; this boundary only restores the
        captured pose/scale/visibility and clears residual velocity after the
        reset warmup.  The collision manager then republishes the exact USD
        poses through the typed PlannerRuntime scene owner with a fresh
        monotonic world revision.
        """

        task_type = f"{type(self.task).__module__}.{type(self.task).__qualname__}"
        audit = getattr(self.task, "audit_fixed_rigid_object_reset", None)
        if callable(audit):
            audit(label=f"{label}:dispatch")
        else:
            LOGGER.warning(
                "[ResetLifecycle] fixed rigid audit unavailable label=%s task_type=%s",
                label,
                task_type,
            )

        restore = getattr(self.task, "restore_fixed_rigid_object_states", None)
        if not callable(restore):
            LOGGER.warning(
                "[ResetLifecycle] fixed rigid restore unavailable label=%s task_type=%s",
                label,
                task_type,
            )
            return []
        restored = restore(label=label)
        if callable(audit):
            audit(label=f"{label}:after")
        if self.collision_scene_manager is not None:
            sync = getattr(self.collision_scene_manager, "sync_after_task_state_restore", None)
            if callable(sync):
                sync(label=label)
            else:
                LOGGER.warning(
                    "[ResetLifecycle] fixed rigid scene sync unavailable label=%s task_type=%s",
                    label,
                    task_type,
                )
        else:
            LOGGER.warning(
                "[ResetLifecycle] fixed rigid scene manager unavailable label=%s task_type=%s",
                label,
                task_type,
            )
        LOGGER.warning(
            "[ResetLifecycle] fixed rigid restore dispatched label=%s task_type=%s restored=%d",
            label,
            task_type,
            len(restored),
        )
        return restored

    def _refresh_task_rigid_views_after_world_reset(self):
        """Rebind task rigid wrappers before any post-reset pose writes.

        ``World.reset()`` replaces the active PhysX tensor view while keeping
        the Python ``SingleRigidPrim`` wrappers alive.  Isaac Sim 6 therefore
        leaves those wrappers pointing at the previous view until
        ``initialize`` is called again.  Region sampling and fixed-object
        restoration both write through the wrapper, so they must run only
        after this refresh.  Contact views intentionally remain deferred
        until the first post-reset simulation step.
        """

        physics_sim_view = self.world.physics_sim_view
        initialize_rigid_objects = getattr(self.task, "initialize_rigid_objects", None)
        if callable(initialize_rigid_objects):
            initialize_rigid_objects(physics_sim_view)

    def _refresh_task_robot_views_after_world_reset(self):
        """Rebind task robot articulation views after ``World.reset()``.

        Isaac Sim keeps the Python ``Robot`` wrappers alive across a hard
        world reset, but their internal ``Articulation`` view can still hold
        the old PhysX tensor handle.  ``Articulation.initialize`` is a no-op
        while that handle is non-null, so the next observation may read an
        invalid transform buffer (including an all-zero quaternion).  Clear
        only the cached handle, then let the normal robot initialize path
        rebuild it; the USD robot prim and its configuration are preserved.
        """

        physics_sim_view = self.world.physics_sim_view
        robots = getattr(self.task, "robots", {})
        for robot in robots.values():
            articulation_view = getattr(robot, "_articulation_view", None)
            if articulation_view is None:
                continue
            is_handle_valid = getattr(articulation_view, "is_physics_handle_valid", None)
            if callable(is_handle_valid):
                try:
                    if is_handle_valid():
                        continue
                except Exception:
                    pass
            if hasattr(articulation_view, "_physics_view"):
                articulation_view._physics_view = None
            robot.initialize(physics_sim_view=physics_sim_view)

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
        """Freeze profile-declared base and lift DOFs for manipulation."""

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
        self._recapture_manipulation_base_holds()
        self._step_world(render=False)

    def _reset_fixed_robot_start_states_after_physics(self, *, clear_debug_history: bool):
        self.task.set_fixed_robot_start_poses()
        self._reset_local_base_drivers(clear_debug_history=clear_debug_history)
        self._recapture_manipulation_base_holds()

    def _run_reset_warmup(self, step_count: int):
        debug_reset_dynamics = getattr(self.task, "debug_reset_dynamics", None)
        for step_index in range(int(step_count)):
            self._init_static_objects(self.task)
            self._step_world(render=False)
            if callable(debug_reset_dynamics) and (
                step_index in {0, 1, 2, 4, 9, 19, 49}
                or step_index == int(step_count) - 1
            ):
                debug_reset_dynamics(f"initial_warmup_{step_index + 1}")
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

    def _debug_reset_warmup_step(self, debug_reset_dynamics, label, step_index, step_count):
        if callable(debug_reset_dynamics) and (
            step_index in {0, 1, 2, 4, 9, 19, 49}
            or step_index == int(step_count) - 1
        ):
            debug_reset_dynamics(f"{label}_{step_index + 1}")

    def _randomization_layout_mem(self):
        self._destroy_local_base_drivers()

        # Reset world
        self.world.reset()
        self._refresh_task_rigid_views_after_world_reset()
        self._refresh_task_robot_views_after_world_reset()
        if self.trajectory_visualizer is not None:
            self.trajectory_visualizer.clear()
        if self.skill_target_visualizer is not None:
            self.skill_target_visualizer.clear()

        # Individual initialize
        self.task.individual_randomize_from_mem()
        self.task.post_reset()

        self._enable_manipulation_base_holds()
        self._step_world(render=False)
        self._initialize_task_physics_views()
        self._set_fixed_robot_start_poses_after_reset()

        # Reset controllers
        self._reset_controllers(self.controllers)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.reset_episode()
            self.collision_scene_manager.initialize_contact_views(self.world.physics_sim_view)
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._failure_subtask_ids.clear()
        self._safety_abort_requested = False

        # Reset skills
        del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_local_base_drivers()
        self._completed_relation_skills = []

        # Warmup
        debug_reset_dynamics = getattr(self.task, "debug_reset_dynamics", None)
        for step_index in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
            self._debug_reset_warmup_step(debug_reset_dynamics, "layout_mem_warmup", step_index, 20)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        self._restore_fixed_rigid_objects_after_warmup("layout_mem_warmup_complete")

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
        self._refresh_task_rigid_views_after_world_reset()
        self._refresh_task_robot_views_after_world_reset()
        if self.trajectory_visualizer is not None:
            self.trajectory_visualizer.clear()
        if self.skill_target_visualizer is not None:
            self.skill_target_visualizer.clear()

        # Individual initialize
        self.task.individual_randomize()
        self.task.post_reset()

        self._enable_manipulation_base_holds()
        self._step_world(render=False)
        self._initialize_task_physics_views()
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
            self.collision_scene_manager.initialize_contact_views(self.world.physics_sim_view)
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._failure_subtask_ids.clear()
        self._safety_abort_requested = False

        # Reset skills
        if hasattr(self, "skills"):
            del self.skills

        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_local_base_drivers()
        self._completed_relation_skills = []

        # Warmup
        debug_reset_dynamics = getattr(self.task, "debug_reset_dynamics", None)
        for step_index in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
            self._debug_reset_warmup_step(debug_reset_dynamics, "layout_warmup", step_index, 20)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        if self.task_cfg.get("fluid", None):
            self.task._set_fluid()
            # Fluid need additional warmup
            for _ in range(150):
                self._step_world(render=False)
            self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)

        self._restore_fixed_rigid_objects_after_warmup("layout_warmup_complete")

        self._initialize_world_recorder()

        self.logger.clear(
            language_instruction=self.task.language_instruction,
            detailed_language_instruction=self.task.detailed_language_instruction,
        )

        # episode_stats["current_times"] += 1

    def reset_after_failed_generation(self):
        self.invalidate_static_map_cache("reset_after_failed_generation")
        self._finish_active_skill_timings("reset_after_failed_generation")
        self._timing_record_simulation_steps = False
        try:
            self.timing_recorder.reset()
        except Exception:
            LOGGER.debug("Failed to reset skill timing recorder", exc_info=True)
        self._destroy_local_base_drivers()

        self.task.individual_reset()
        # A failed planning attempt does not require rebuilding the PhysX
        # scene.  A hard World.reset() tears down the articulation tensor
        # view and then runs BananaBaseTask.post_reset() while the split
        # Aloha virtual-base joints are being reattached; on Isaac Sim 6 that
        # path can publish NaN link transforms and leave observations with a
        # zero quaternion.  Keep the live physics view for this retry and
        # restore the already-loaded rigid bodies in place.
        self.world.reset(soft=True)
        self._refresh_task_rigid_views_after_world_reset()
        if hasattr(self.task, "reset_fixed_rigid_objects"):
            self.task.reset_fixed_rigid_objects()
        self.task.post_reset()
        self._enable_manipulation_base_holds()
        self._step_world(render=False)
        self._initialize_task_physics_views()
        self._set_fixed_robot_start_poses_after_reset()

        self._reset_controllers(self.controllers)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.reset_episode()
            self.collision_scene_manager.initialize_contact_views(self.world.physics_sim_view)
        self.safety_monitor.reset()
        self.execution_supervisor.reset()
        self._safety_failure_reason = ""
        self._safety_abort_requested = False
        if hasattr(self, "skills"):
            del self.skills
        self.skills = self._initialize_skills(self.task, self.task_cfg, self.controllers, self.world)
        self._initialize_local_base_drivers()

        debug_reset_dynamics = getattr(self.task, "debug_reset_dynamics", None)
        for step_index in range(20):
            self._get_observations()
            self._init_static_objects(self.task)
            self._step_world(render=False)
            self._debug_reset_warmup_step(debug_reset_dynamics, "failed_generation_warmup", step_index, 20)
        self._reset_fixed_robot_start_states_after_physics(clear_debug_history=True)
        self._restore_fixed_rigid_objects_after_warmup("failed_generation_warmup_complete")

        self._initialize_world_recorder()

    def randomization(self, layout_path=None) -> bool:
        try:
            self.invalidate_static_map_cache("randomization")
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
        """Update the compiled Skill DAG and start newly-unblocked nodes."""

        return self.update_dag_skill_states(skills, episode_success, should_continue)

    @staticmethod
    def _dag_skill_done(skills):
        return all(node["state"] == "succeeded" for node in skills["nodes"])

    def _skills_complete(self):
        return self._dag_skill_done(self.skills)

    @staticmethod
    def _skill_changes_navigation_collision(skill) -> bool:
        skill_name = type(skill).__name__.lower()
        return ("pick" in skill_name and "probe" not in skill_name) or "place" in skill_name

    def _dag_ready_to_start(self, node, nodes_by_id):
        return node["state"] == "pending" and all(
            nodes_by_id[dep_id]["state"] == "succeeded" for dep_id in node["depends_on"]
        )

    def _start_dag_ready_skills(self, skills, should_continue):
        nodes_by_id = skills["nodes_by_id"]
        operation_running = any(
            node["state"] == "running"
            and node.get("collision_world_mode") != PASSTHROUGH_MODE
            for node in skills["nodes"]
        )
        for node in skills["nodes"]:
            if not self._dag_ready_to_start(node, nodes_by_id):
                continue

            is_operation = node.get("collision_world_mode") != PASSTHROUGH_MODE
            if operation_running and is_operation:
                continue

            skill = node["skill"]
            if node.get("execution_mode") != DIRECT_EXECUTION_MODE:
                self._activate_skill_collision_world(skill)
            self._run_skill_planner(node["robot_name"], skill)
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
            operation_running = operation_running or is_operation
            if len(skill.manip_list) == 0 and not skill.is_ready():
                should_continue = True
        return should_continue

    def _collect_dag_skill_actions(self, skills):
        actions_by_robot = defaultdict(list)
        running_nodes = [node for node in skills["nodes"] if node["state"] == "running"]
        record_flag = True
        operation_node_id = None

        for node in running_nodes:
            skill = node["skill"]
            if node.get("collision_world_mode") == PASSTHROUGH_MODE:
                # Navigation/observation Skills update their own state and do
                # not emit an arm action into the typed MotionPlanner path.
                continue
            if operation_node_id is not None:
                raise RuntimeError(
                    "DAG scheduler invariant violated: multiple operation "
                    f"commands are running ({operation_node_id!r}, {node['id']!r})"
                )
            operation_node_id = node["id"]
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
                command = skill.manip_list[0]
                if not isinstance(command, MotionPhaseCommand):
                    raise TypeError(
                        f"operation Skill {self._skill_display_name(skill)!r} must emit "
                        "MotionPhaseCommand values"
                    )
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
                skill_name = type(skill).__name__.lower()
                if skill_name in {"pick", "place"}:
                    self._completed_relation_skills.append(
                        {
                            "skill": skill,
                            "skill_name": skill_name,
                            "objects": list(
                                getattr(skill, "skill_cfg", {}).get("objects", [])
                            ),
                            "terminal_success": skill_success,
                        }
                    )
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
                    self._finish_skill_timing(skill, True)
                    node["state"] = "succeeded"
                    if self._skill_changes_navigation_collision(skill):
                        self.invalidate_static_map_cache(f"completed:{node['id']}")

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
        """Bind execution metadata without switching planner worlds.

        The workflow owns one Physics-schema world for the episode.  A
        passthrough Skill (for example local navigation or observe-hold) does
        not need a planner activation; an operation Skill simply reuses the
        controller's already initialized runtime.
        """

        mode = getattr(skill, "collision_world_mode", PASSTHROUGH_MODE)
        if getattr(skill, "execution_mode", None) == DIRECT_EXECUTION_MODE:
            setattr(skill, "effective_collision_world_mode", DIRECT_EXECUTION_MODE)
            return DIRECT_EXECUTION_MODE
        if mode == PASSTHROUGH_MODE:
            setattr(skill, "effective_collision_world_mode", PASSTHROUGH_MODE)
            return mode
        runtime = skill.skill_runtime
        if runtime is None:
            raise RuntimeError(
                f"Physics-schema Skill {self._skill_display_name(skill)!r} "
                "requires an active manipulator controller"
            )
        if mode != PHYSICS_SCHEMA_MODE:
            raise RuntimeError(
                "operation Skills must use the canonical Physics-schema planner"
            )
        attached_entity = None
        if self.collision_scene_manager is not None:
            attached_entity = self.collision_scene_manager.get_attached_entity(
                runtime.name, runtime.arm_name
            )
        setattr(skill, "_physics_schema_active_object", attached_entity)
        setattr(skill, "effective_collision_world_mode", PHYSICS_SCHEMA_MODE)
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
        self._record_failure_subtask(skill)
        self._finish_skill_timing(skill, False, reason=failure_reason)

    def plan_first_skill(self, skills, should_continue):
        return self._start_dag_ready_skills(skills, should_continue)

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
        for node in self.skills["nodes"]:
            if node["state"] == "running":
                yield node["robot_name"], node["skill"]

    @staticmethod
    def _execution_status(runtime, command=None):
        """Read the detailed execution snapshot from the typed runtime port."""

        return runtime.execution_status(command)

    @staticmethod
    def _status_value(status, name, default=None):
        if isinstance(status, dict):
            return status.get(name, default)
        return getattr(status, name, default)

    @classmethod
    def _status_tracking_failed(cls, status) -> bool:
        return bool(
            cls._status_value(status, "tracking_failed", False)
            or cls._status_value(status, "tracking_completion_failed", False)
            or str(cls._status_value(status, "reason", "")).lower()
            in {"tracking_failed", "tracking_completion_failed"}
        )

    def _debug_finger_contacts(self, runtime):
        """Return one-shot finger contact details for a failing terminal pose."""

        manager = self.collision_scene_manager
        filters = list(getattr(manager, "collision_prim_paths", ()))
        views = getattr(manager, "_finger_environment_contact_views", {}).get(
            (str(runtime.name), str(runtime.arm_name)), ()
        )
        if not filters or not views:
            return []
        contacts = []
        for view_index, view in enumerate(views):
            try:
                values = np.asarray(view.get_contact_force_matrix(), dtype=float)
                if not values.size:
                    continue
                magnitudes = np.linalg.norm(values, axis=-1).reshape(-1, len(filters))
                maxima = np.max(magnitudes, axis=0)
                contacts.extend(
                    {
                        "view": int(view_index),
                        "filter": filters[index],
                        "force": float(force),
                    }
                    for index, force in enumerate(maxima)
                    if float(force) > 0.0
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                contacts.append({"view": int(view_index), "error": repr(exc)})
        return sorted(contacts, key=lambda item: float(item.get("force", 0.0)), reverse=True)

    def _safety_measurements(self, skill, dynamic_changed: bool) -> SafetyMeasurements:
        runtime = skill.skill_runtime
        if runtime is None:
            raise RuntimeError("execution safety requires a bound typed runtime")
        robot = runtime.robot
        arm_indices = runtime.arm_indices
        joint_state = robot.get_joints_state()
        actual_arm = np.asarray(joint_state.positions[arm_indices], dtype=float)
        command = skill.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError(
                f"operation Skill {self._skill_display_name(skill)!r} must emit "
                "MotionPhaseCommand values"
            )
        execution_status = runtime.execution_status(command)
        commanded_arm = self._status_value(
            execution_status, "last_commanded_arm_position", None
        )
        joint_error = (
            float(np.max(np.abs(actual_arm - commanded_arm)))
            if commanded_arm is not None and len(commanded_arm) == len(actual_arm)
            else 0.0
        )
        ee_position_error = 0.0
        ee_orientation_error = 0.0
        if commanded_arm is not None and len(commanded_arm) == len(actual_arm):
            expected_position, expected_orientation = runtime.compute_fk(commanded_arm)
            actual_position, actual_orientation = runtime.ee_pose()
            ee_position_error = float(np.linalg.norm(expected_position - actual_position))
            ee_orientation_error = quaternion_angle(expected_orientation, actual_orientation)

        base_position, base_orientation = runtime.arm_base_pose()
        phase_base_position = self._status_value(
            execution_status, "phase_base_position", None
        )
        phase_base_orientation = self._status_value(
            execution_status, "phase_base_orientation", None
        )
        if phase_base_position is None or phase_base_orientation is None:
            initial_position, initial_orientation = base_position, base_orientation
        else:
            initial_position, initial_orientation = (
                phase_base_position,
                phase_base_orientation,
            )
        base_translation = float(np.linalg.norm(np.asarray(base_position) - np.asarray(initial_position)))
        base_rotation = float(np.degrees(quaternion_angle(base_orientation, initial_orientation)))
        velocity = np.asarray(joint_state.velocities, dtype=float)
        arm_velocity = velocity[arm_indices]
        VELOCITY_TRACE_LOGGER.info(
            "[VelocityTrace] step=%d phase=%s plan_active=%s steps_remaining=%s "
            "actual=%s commanded=%s velocity=%s",
            self._active_execution_step_id,
            self._status_value(execution_status, "phase", command.phase.value),
            bool(self._status_value(execution_status, "plan_active", False)),
            int(self._status_value(execution_status, "plan_steps_remaining", 0)),
            np.array2string(actual_arm, precision=5),
            np.array2string(np.asarray(commanded_arm, dtype=float), precision=5)
            if commanded_arm is not None
            else None,
            np.array2string(arm_velocity, precision=5),
        )
        joint_limit_violation = False
        limits = None
        arm_limits = None
        limit_error = None
        try:
            limit_owner = robot
            get_dof_limits = getattr(robot, "get_dof_limits", None)
            if not callable(get_dof_limits):
                limit_owner = getattr(robot, "_articulation_view", None)
                get_dof_limits = getattr(limit_owner, "get_dof_limits", None)
            if not callable(get_dof_limits):
                raise AttributeError("robot and articulation view expose no get_dof_limits")
            limits = np.asarray(get_dof_limits(), dtype=float)
            arm_limits = limits[arm_indices]
            joint_limit_violation = bool(
                np.any(actual_arm < arm_limits[:, 0] - 1e-4)
                or np.any(actual_arm > arm_limits[:, 1] + 1e-4)
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            # Some robot wrappers do not expose limits.  Physics-schema
            # SplitAloha does; strict world audit remains the primary guard.
            joint_limit_violation = False
            limit_error = repr(exc)
        if (
            os.environ.get("INTERNDATA_DEBUG_JOINT_LIMITS") == "1"
            and command.phase == MotionPhase.TERMINAL_GRASP_APPROACH
            and not bool(self._status_value(execution_status, "plan_active", False))
            and not self._status_value(execution_status, "complete", False)
        ):
            try:
                controller = robot.get_articulation_controller()
                try:
                    gains = controller.get_gains()
                except Exception as exc:
                    gains = f"unavailable:{exc!r}"
                try:
                    max_efforts = controller.get_max_efforts()
                except Exception as exc:
                    max_efforts = f"unavailable:{exc!r}"
                LOGGER.warning(
                    "[DofLimitDebug] robot=%s arm=%s dof_names=%s arm_indices=%s "
                    "limits=%s arm_limits=%s limit_error=%s actual=%s commanded=%s "
                    "gains=%s max_efforts=%s finger_contacts=%s",
                    robot.name,
                    runtime.arm_name,
                    repr(getattr(robot, "dof_names", None)),
                    list(arm_indices),
                    repr(limits),
                    repr(arm_limits),
                    limit_error,
                    np.array2string(actual_arm, precision=7),
                    np.array2string(np.asarray(commanded_arm, dtype=float), precision=7)
                    if commanded_arm is not None
                    else None,
                    gains,
                    max_efforts,
                    self._debug_finger_contacts(runtime),
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                LOGGER.warning("[DofLimitDebug] unavailable error=%r", exc)
        arrays = (actual_arm, velocity, np.asarray(base_position), np.asarray(base_orientation))
        nan_detected = any(not np.all(np.isfinite(value)) for value in arrays)

        illegal_state = False
        try:
            self.collision_scene_manager.assert_invariants()
        except CollisionSceneError:
            illegal_state = True
        dropped = False
        if command.phase in {
            MotionPhase.SYNC_WORLD,
            MotionPhase.GRIPPER_CLOSE,
            MotionPhase.ATTACH,
            MotionPhase.GRIPPER_OPEN,
            MotionPhase.DETACH_AND_SETTLE,
            MotionPhase.RESTORE_WORLD,
        }:
            # These phases do not command a moving arm trajectory.  A
            # passive rigid object settling elsewhere in the scene cannot
            # invalidate the already-reached gripper/contact state here.
            dynamic_changed = False
        record = None
        if command.active_object:
            record = self.collision_scene_manager.records.get(command.active_object)
            if record is not None and record.state == CollisionObjectState.ATTACHED:
                try:
                    get_contact = skill.get_contact
                except AttributeError:
                    get_contact = None
                if get_contact is not None:
                    _, contact_indices = get_contact()
                    dropped = len(contact_indices) == 0
        target_owner_matches = bool(
            record is not None
            and (str(record.owner_robot), str(record.owner_arm))
            == (str(runtime.name), str(runtime.arm_name))
        )
        pending_detach = bool(
            command.phase == MotionPhase.DETACH_AND_SETTLE
            and self.collision_scene_manager.is_pending_detach(command.active_object)
        )
        allow_target_robot_contact = bool(
            command.allow_target_robot_contact
            and command.active_object
            and record is not None
            and target_owner_matches
            and (
                record.state
                in {
                    CollisionObjectState.ATTACHED,
                    CollisionObjectState.PLACEMENT_CONTACT,
                }
                or pending_detach
            )
        )
        unexpected_contact = self.collision_scene_manager.get_unexpected_robot_contact_force(
            runtime.name,
            runtime.arm_name,
            command.active_object if allow_target_robot_contact else None,
        )
        allowed_finger_contact, unexpected_finger_contact = (
            self.collision_scene_manager.get_finger_environment_contact_forces(
                runtime.name,
                runtime.arm_name,
                command.active_object if command.allow_target_finger_contact else None,
            )
        )
        unexpected_contact = max(unexpected_contact, unexpected_finger_contact)
        if (
            os.environ.get("SIMBOX_DEBUG_PICK") == "1"
            and command.phase == MotionPhase.TERMINAL_GRASP_APPROACH
            and command.active_object
            and (dynamic_changed or allowed_finger_contact > 0.0)
        ):
            LOGGER.warning(
                "[PickDebug] terminal contact object=%s dynamic_changed=%s "
                "allowed_finger_contact_n=%s unexpected_finger_contact_n=%s details=%s",
                command.active_object,
                bool(dynamic_changed),
                float(allowed_finger_contact),
                float(unexpected_finger_contact),
                self._debug_finger_contacts(runtime),
            )
        if (
            dynamic_changed
            and command.phase == MotionPhase.TERMINAL_GRASP_APPROACH
            and command.allow_target_finger_contact
            and allowed_finger_contact > 0.0
        ):
            # Target motion caused by the explicitly allowed finger contact is
            # part of the grasp approach contract.  Keep the safety check for
            # unrelated robot/world contact and for target motion without any
            # measured finger contact.
            dynamic_changed = False
        allowed_support_contact = 0.0
        unexpected_object_contact = 0.0
        attached_slip_translation = 0.0
        attached_slip_rotation = 0.0
        if command.active_object and record is not None and record.state in {
            CollisionObjectState.ATTACHED,
            CollisionObjectState.PLACEMENT_CONTACT,
        }:
            if command.phase not in {
                MotionPhase.GRIPPER_OPEN,
                MotionPhase.DETACH_AND_SETTLE,
            }:
                # Opening the gripper is the release boundary.  The target
                # can rotate or translate as finger contact is removed; that
                # expected motion must not be evaluated as attached-carry
                # slip before DETACH_AND_SETTLE transfers ownership to the
                # support/world state.
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
                runtime.complete_contact_phase(command)
                skill.manip_list[:] = [command] + [
                    later
                    for later in skill.manip_list[1:]
                    if later.phase != MotionPhase.TERMINAL_PLACE_DESCENT
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
            plan_failed=bool(self._status_value(execution_status, "plan_failed", False)),
            tracking_completion_failed=self._status_tracking_failed(execution_status),
        )

    @staticmethod
    def _dynamic_changes_relevant_to_command(
        changed: set[str], command: MotionPhaseCommand, safety_cfg
    ) -> set[str]:
        """Select dynamic changes that can invalidate this phase's contract.

        Physics-schema synchronization still updates every exact collider in
        CuRobo.  Execution replan signals are narrower: an unrelated rigid
        task object settling on its own support must not invalidate an arm
        path that is carrying a different object.  Contact, attachment and
        tracking checks remain active for that path, and tasks can explicitly
        add dynamic entities when they are genuine moving obstacles.
        """

        changed = {str(name) for name in changed}
        configured = safety_cfg.get("dynamic_replan_entities", ())
        if isinstance(configured, str):
            configured = (configured,)
        try:
            configured_entities = {
                str(name).strip() for name in configured if str(name).strip()
            }
        except TypeError:
            configured_entities = set()

        owned_entities = {
            str(name).strip()
            for name in (command.active_object, command.support_object)
            if name
        }
        relevant_entities = owned_entities | configured_entities
        # A structured motion command without an object owner has no semantic
        # way to narrow its world dependency, so retain the conservative
        # all-entity behavior for that diagnostic case.
        if not relevant_entities:
            return changed
        # During terminal grasp approach the active target is intentionally
        # allowed to move under finger contact.  Its pose is still synced to
        # the exact native collider above, but that expected motion must not
        # invalidate the approach path as if an unrelated obstacle moved.
        if command.phase is MotionPhase.TERMINAL_GRASP_APPROACH:
            changed = changed - {
                str(command.active_object)
            } if command.active_object else changed
        return changed & relevant_entities

    def _execution_safety_precheck(self, step_id: int, action_dict=None) -> bool:
        """Evaluate the previous step before producing the next command.

        On a hard abort the articulation must receive a measured hold target
        in this same simulation step. The runtime status is evaluated before
        another command is consumed, so a failed plan cannot advance its
        trajectory consumer after the safety decision.
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
        if changed:
            LOGGER.info(
                "[ExecutionSafety] dynamic obstacle pose synced step=%d entities=%s",
                step_id,
                sorted(changed),
        )
        for robot_name, skill in self._iter_active_skills() or []:
            if (
                not skill.manip_list
                or getattr(skill, "collision_world_mode", PHYSICS_SCHEMA_MODE)
                == PASSTHROUGH_MODE
            ):
                continue
            command = skill.manip_list[0]
            if not isinstance(command, MotionPhaseCommand):
                raise TypeError(
                    f"operation Skill {self._skill_display_name(skill)!r} must emit "
                    "MotionPhaseCommand values"
                )
            if command.is_direct:
                # Direct interpolation is intentionally outside the Physics
                # schema safety/replan loop. The typed command owns the joint
                # path and the controller only forwards it.
                continue
            runtime = skill.skill_runtime
            if runtime is None:
                raise RuntimeError("execution safety requires a bound typed runtime")
            # This precheck evaluates the result of the *previous* physics
            # step.  A newly selected command has not run yet, so its phase
            # baseline and first commanded joint target do not exist.  Using
            # the controller's initialization pose here caused step-0 and
            # phase-transition false base-drift aborts.
            execution_status = self._execution_status(runtime, command)
            if not bool(self._status_value(execution_status, "active", False)):
                continue
            if self.execution_supervisor.is_holding(runtime):
                continue
            if (
                step_id % 100 == 0
                or bool(self._status_value(
                    execution_status, "plan_failed", False
                ))
                or self._status_tracking_failed(execution_status)
            ):
                LOGGER.info(
                    "[ExecutionHeartbeat] step=%d robot=%s arm=%s phase=%s plan_active=%s steps_remaining=%d complete=%s plan_failed=%s tracking_failed=%s",
                    step_id,
                    robot_name,
                    runtime.arm_name,
                    self._status_value(execution_status, "phase", command.phase.value),
                    bool(self._status_value(execution_status, "plan_active", False)),
                    int(self._status_value(execution_status, "plan_steps_remaining", 0)),
                    bool(self._status_value(execution_status, "complete", False)),
                    bool(self._status_value(execution_status, "plan_failed", False)),
                    self._status_tracking_failed(execution_status),
                )
            relevant_changed = self._dynamic_changes_relevant_to_command(
                changed, command, safety_cfg
            )
            if relevant_changed:
                LOGGER.warning(
                    "[ExecutionSafety] relevant dynamic obstacle changed step=%d "
                    "phase=%s entities=%s",
                    step_id,
                    command.phase.value,
                    sorted(relevant_changed),
                )
            try:
                measurements = self._safety_measurements(
                    skill,
                    bool(relevant_changed),
                )
            except CollisionSceneError as exc:
                # Contact-view and invariant failures are themselves hard
                # safety failures. Convert them to an auditable event instead
                # of escaping to the outer workflow exception handler, which
                # would otherwise retry without first applying a hold target.
                LOGGER.exception(
                    "[ExecutionSafety] collision/contact audit failed for %s/%s: %s",
                    robot_name,
                    runtime.arm_name,
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
                self._record_failure_subtask(skill)
                self._safety_failure_reason = self.execution_supervisor.failure_reason
                failure_reason = str(self.execution_supervisor.failure_reason or "").lower()
                if action_dict is not None:
                    hold = runtime.hold("execution_safety_abort")
                    action_dict[robot_name] = {
                        "joint_positions": np.asarray(hold["joint_positions"]),
                        "joint_indices": np.asarray(hold["joint_indices"]),
                        "raw_action": [hold],
                    }
                return False
        return True

    def _forward_or_hold(self, skill):
        runtime = skill.skill_runtime
        if runtime is None:
            raise RuntimeError("operation Skill requires a bound typed runtime")
        command = skill.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError(
                f"operation Skill {self._skill_display_name(skill)!r} must emit "
                "MotionPhaseCommand values"
            )
        if not command.is_direct:
            self._activate_skill_collision_world(skill)
        self._start_skill_motion_phase(skill, command)
        scope = getattr(skill, "_timing_scope", None)
        previous_scope = runtime.push_timing_scope(scope)
        try:
            try:
                return self.execution_supervisor.forward_or_hold(
                    runtime,
                    command,
                )
            except CollisionSceneError as exc:
                LOGGER.exception(
                    "[ExecutionSafety] collision-state operation failed for %s/%s: %s",
                    runtime.name,
                    runtime.arm_name,
                    exc,
                )
                self.execution_supervisor.evaluate(
                    SafetyMeasurements(illegal_object_state=True),
                    step_id=self._active_execution_step_id,
                    robot=runtime.name,
                    skill=skill,
                    command=command,
                    world_revision=self.collision_scene_manager.world_revision,
                )
                self._safety_failure_reason = f"COLLISION_SCENE_ERROR:{exc}"
                self._safety_abort_requested = True
                return runtime.hold("collision_scene_error")
        finally:
            runtime.restore_timing_scope(previous_scope)

    def generate_seq(self) -> list:
        end = False
        step_render = bool(getattr(self, "planning_step_render", False))
        self._timing_record_simulation_steps = False

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
        self._timing_record_simulation_steps = True
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
            if not self._skills_complete() and should_continue:
                action_dict, record_flag, skill_failed = self._collect_dag_skill_actions(self.skills)
                if skill_failed:
                    episode_success = False
                    should_continue = False
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
                        self._skill_runtime_ports_for_logging(),
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
                    self._skill_runtime_ports_for_logging(),
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

        if not end:
            self._finish_active_skill_timings("episode_incomplete")
        self._timing_record_simulation_steps = False
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
        self._finish_active_skill_timings("episode_saved_before_completion")
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
        # Timing is an optional diagnostic.  It is deliberately written only
        # after the logger has created the episode directory, and any timing
        # failure must not turn a saved episode into a failed save.
        try:
            timing_summary = self.timing_recorder.to_dict()
            for episode_dir in saved_dirs or []:
                try:
                    with open(
                        os.path.join(episode_dir, "skill_timing.json"),
                        "w",
                        encoding="utf-8",
                    ) as timing_file:
                        json.dump(timing_summary, timing_file, indent=2, allow_nan=False)
                except Exception:
                    LOGGER.warning(
                        "Failed to write skill_timing.json for %s",
                        episode_dir,
                        exc_info=True,
                    )
        except Exception:
            LOGGER.warning("Failed to serialize skill timing summary", exc_info=True)
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
        predicate_results = self._final_predicate_results()
        emit_episode_saved(
            status="failed" if episode_failed else "success",
            episode_dirs=saved_dirs or [],
            num_steps=self.length,
            failure_reason=getattr(self, "_episode_failure_reason", "") if episode_failed else "",
            failing_subtask_id=self._episode_failing_subtask_id(predicate_results),
            task_name=self.task_cfg.get("task"),
            task_dir=self.task_cfg.get("data", {}).get("task_dir"),
            collect_info=self.task_cfg.get("data", {}).get("collect_info"),
            predicate_results=predicate_results,
            task_predicate_success=(
                bool(predicate_results)
                and all(bool(item["success"]) for item in predicate_results)
            ),
            world_revision=(
                self.collision_scene_manager.world_revision
                if self.collision_scene_manager is not None
                else None
            ),
        )

        return self.length

    def _final_predicate_results(self) -> list[dict]:
        """Re-evaluate final object relations after all settling and return motion."""

        return evaluate_compiled_place_relations(
            list(getattr(self, "_completed_relation_skills", []))
        )

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
        self._timing_record_simulation_steps = False

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
        self._timing_record_simulation_steps = True
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
            if not self._skills_complete() and should_continue:
                action_dict, record_flag, skill_failed = self._collect_dag_skill_actions(self.skills)
                if skill_failed:
                    episode_success = False
                    should_continue = False
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
                        self._skill_runtime_ports_for_logging(),
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
                    self._skill_runtime_ports_for_logging(),
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

        if not end:
            self._finish_active_skill_timings("episode_incomplete")
        self._timing_record_simulation_steps = False
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
                        self._skill_runtime_ports_for_logging(),
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

        else:
            self._episode_failed = False
            self._episode_failure_reason = ""
            self.length = 0

        if (
            self.task_cfg.get("debug_topdown_check")
            or os.environ.get("INTERNDATA_DEBUG_TOPDOWN") == "1"
        ):
            screenshot_dir = os.environ.get("INTERNDATA_SCREENSHOT_DIR")
            if not screenshot_dir:
                runtime_output_dir = os.environ.get("OUTPUT_DIR") or os.environ.get(
                    "SEQ_OUTPUT_DIR"
                )
                if not runtime_output_dir:
                    raise RuntimeError(
                        "topdown capture requires INTERNDATA_SCREENSHOT_DIR or a runtime output directory"
                    )
                screenshot_dir = str(
                    Path(runtime_output_dir).expanduser().resolve().parent
                    / "screenshots"
                )
            camera = self.task_cfg.get("debug_topdown_camera") or {}
            if camera.get("template") and camera.get("eye") is None:
                from core.utils.camera_template import resolve_camera_template_pose

                robot_name = str(self.task_cfg["robots"][0]["name"])
                target_name = str(camera["target_object"])
                robot_position, robot_orientation = self.task.robots[
                    robot_name
                ].get_world_pose()
                target_position, _ = self.task.objects[target_name].get_world_pose()
                w, x, y, z = [float(value) for value in robot_orientation]
                robot_yaw_deg = math.degrees(
                    math.atan2(
                        2.0 * (w * z + x * y),
                        1.0 - 2.0 * (y * y + z * z),
                    )
                )
                camera = {
                    **camera,
                    **resolve_camera_template_pose(
                        str(camera["template"]),
                        robot_position,
                        robot_yaw_deg,
                        target_position,
                        camera.get("template_params"),
                        camera.get("room_bounds_xy"),
                    ),
                }
            capture_topdown_screenshot(
                screenshot_dir,
                eye=camera.get("eye"),
                target=camera.get("target"),
                width=int((camera.get("resolution") or [640, 480])[0]),
                height=int((camera.get("resolution") or [640, 480])[1]),
                focal_length_mm=float(camera.get("focal_length_mm", 16.0)),
            )
        return self.length

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
            self.invalidate_static_map_cache("randomization_from_mem")
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
