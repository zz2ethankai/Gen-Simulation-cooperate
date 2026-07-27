"""
Template Controller base class for robot motion planning.

Common functionality extracted from FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha.
Subclasses implement _get_default_ignore_substring() and _configure_joint_indices().
"""

import json
import numbers
import os
import random
import time
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    sort_by_difference_js,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.isaac.core.utils.types import ArticulationAction


# pylint: disable=line-too-long,unused-argument
class TemplateController(BaseController):
    """Base controller for CuRobo-based motion planning. Supports single and batch planning."""

    def __init__(
        self,
        name: str,
        robot_file: str,
        task: BaseTask,
        world: World,
        constrain_grasp_approach: bool = False,
        collision_activation_distance: float = 0.03,
        ignore_substring: Optional[List[str]] = None,
        use_batch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.name = name
        self.world = world
        self.task = task
        self.robot = self.task.robots[name]
        self.ignore_substring = self._get_default_ignore_substring()
        if ignore_substring is not None:
            self.ignore_substring = ignore_substring
        self.ignore_substring.append(name)
        self.use_batch = use_batch
        self.constrain_grasp_approach = constrain_grasp_approach
        self.collision_activation_distance = collision_activation_distance
        self.usd_help = UsdHelper()
        self.tensor_args = TensorDeviceType()
        self.init_curobo = False
        self.robot_file = robot_file
        self.num_plan_failed = 0
        self.raw_js_names = []
        self.cmd_js_names = []
        self.arm_indices = np.array([])
        self.gripper_indices = np.array([])
        self.reference_prim_path = None
        self.lr_name = None
        self._ee_trans = 0.0
        self._ee_ori = 0.0
        self._gripper_state = 1.0
        self._gripper_joint_position = np.array([1.0])
        self.idx_list = None

        self._configure_joint_indices(robot_file)
        self._load_robot(robot_file)
        self._load_kin_model()
        self._load_world()
        self._init_motion_gen()

        self.usd_help.load_stage(self.world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.num_last_cmd = 0
        self.ds_ratio = 1
        self._last_arm_action = None
        self._curobo_plan_debug_counter = 0
        self._curobo_plan_debug_dir = os.environ.get(
            "SIMBOX_CUROBO_PLAN_DEBUG_DIR",
            os.path.join("output", "ros_bridge", "skills", "curobo_plan_debug"),
        )

    def _get_default_ignore_substring(self) -> List[str]:
        return ["material", "Plane", "conveyor", "scene", "table"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        raise NotImplementedError

    def _load_robot(self, robot_file: str) -> None:
        self.robot_cfg = load_yaml(robot_file)["robot_cfg"]

    def _load_kin_model(self) -> None:
        urdf_file = self.robot_cfg["kinematics"]["urdf_path"]
        base_link = self.robot_cfg["kinematics"]["base_link"]
        ee_link = self.robot_cfg["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

    def _load_world(self, use_default: bool = True) -> None:
        if use_default:
            self.world_cfg = WorldConfig()
        else:
            world_cfg_table = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            )
            self._world_cfg_table = world_cfg_table
            self._world_cfg_table.cuboid[0].pose[2] -= 10.5
            world_cfg1 = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            ).get_mesh_world()
            world_cfg1.mesh[0].name += "_mesh"
            world_cfg1.mesh[0].pose[2] = -10.5
            self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    def _get_motion_gen_collision_cache(self):
        """Override in subclasses to use different cache size (e.g. FR3 uses 1000)."""
        return {"obb": 700, "mesh": 700}

    def _get_grasp_approach_linear_axis(self) -> int:
        """Axis for grasp approach constraint (0=x, 1=y, 2=z). Override in subclasses (e.g. Lift2 uses 0)."""
        if self.robot.cfg["ee_axis"] == "x":
            return 0
        elif self.robot.cfg["ee_axis"] == "y":
            return 1
        elif self.robot.cfg["ee_axis"] == "z":
            return 2
        else:
            raise NotImplementedError

    def _get_sort_path_weights(self) -> Optional[List[float]]:
        """Optional per-joint weights for sort_by_difference_js.

        Used when selecting among batch paths. None means equal weights.
        Override in subclasses (e.g. Genie1).
        """
        return None

    def _init_motion_gen(self) -> None:
        pose_metric = None
        if self.constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1,
                linear_axis=self._get_grasp_approach_linear_axis(),
            )
        if self.use_batch:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                need_graph_success=True,
                enable_graph_attempt=4,
                max_attempts=4,
                enable_finetune_trajopt=True,
                parallel_finetune=True,
                time_dilation_factor=1.0,
            )
        else:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=False,
                enable_graph_attempt=7,
                max_attempts=10,
                pose_cost_metric=pose_metric,
                enable_finetune_trajopt=True,
                time_dilation_factor=1.0,
            )
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.01,
            collision_activation_distance=self.collision_activation_distance,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache=self._get_motion_gen_collision_cache(),
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            optimize_dt=True,
            trajopt_dt=None,
            trim_steps=None,
            project_pose_to_goal_frame=False,
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": 700, "mesh": 700},
        )
        self.ik_solver = IKSolver(ik_config)
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        if self.use_batch:
            self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        else:
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.world_model = self.motion_gen.world_collision
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.motion_gen.update_world(self.world_cfg)
        self._world_update_signature = self._make_world_update_signature(self.world_cfg)

    def update_pose_cost_metric(self, hold_vec_weight: Optional[List[float]] = None) -> None:
        # reference: https://curobo.org/advanced_examples/3_constrained_planning.html
        # [angular-x, angular-y, angular-z, linear-x, linear-y, linear-z]
        # For example,
        # when hold_vec_weight is None, the corresponding list is [0, 0, 0, 0, 0, 0],
        # there is no cost added in any directions.
        # When hold_vec_weight = [1, 1, 1, 0, 0, 0], the tool orientation is holed.
        # assert hold_vec_weight is None or len(hold_vec_weight) == 6
        if hold_vec_weight:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(hold_vec_weight),
            )
        else:
            pose_cost_metric = None
        self.plan_config.pose_cost_metric = pose_cost_metric

    @staticmethod
    def _debug_json_ready(value: Any):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.ndarray):
            return TemplateController._debug_json_ready(value.tolist())
        if isinstance(value, torch.Tensor):
            return TemplateController._debug_json_ready(value.detach().cpu().numpy())
        if isinstance(value, (list, tuple)):
            return [TemplateController._debug_json_ready(v) for v in value]
        if isinstance(value, dict):
            return {str(k): TemplateController._debug_json_ready(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _debug_joint_state_to_dict(joint_state):
        if joint_state is None:
            return None
        payload = {}
        for field in ("position", "velocity", "acceleration", "jerk"):
            if hasattr(joint_state, field):
                payload[field] = TemplateController._debug_json_ready(getattr(joint_state, field))
        if hasattr(joint_state, "joint_names"):
            payload["joint_names"] = TemplateController._debug_json_ready(getattr(joint_state, "joint_names"))
        if hasattr(joint_state, "shape"):
            payload["shape"] = TemplateController._debug_json_ready(getattr(joint_state, "shape"))
        return payload

    @staticmethod
    def _debug_norm_delta(a, b):
        if a is None or b is None:
            return None
        a_arr = np.asarray(a, dtype=float).reshape(-1)
        b_arr = np.asarray(b, dtype=float).reshape(-1)
        if a_arr.shape != b_arr.shape:
            return None
        return float(np.linalg.norm(a_arr - b_arr))

    def _write_curobo_plan_debug(
        self,
        *,
        result,
        sim_js,
        js_names,
        ee_trans,
        ee_ori,
        raw_plan,
        ordered_cmd_plan,
        branch: str,
        selected_path_index=None,
        selected_path_source: str = "",
    ) -> None:
        try:
            os.makedirs(self._curobo_plan_debug_dir, exist_ok=True)
            self._curobo_plan_debug_counter += 1
            current_full = np.asarray(sim_js.positions, dtype=float).copy()
            current_arm = current_full[self.arm_indices].copy()
            first_ordered = None
            last_ordered = None
            if ordered_cmd_plan is not None and len(ordered_cmd_plan) > 0:
                first_ordered = ordered_cmd_plan[0].position.detach().cpu().numpy()
                last_ordered = ordered_cmd_plan[-1].position.detach().cpu().numpy()

            cu_js = JointState(
                position=self.tensor_args.to_device(sim_js.positions),
                velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=js_names,
            )
            cu_js_ordered = cu_js.get_ordered_joint_state(self.cmd_js_names)

            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = (
                f"{timestamp}_{int((time.time() % 1) * 1000000):06d}_"
                f"{self.name}_{self.lr_name}_plan_{self._curobo_plan_debug_counter:04d}.json"
            )
            output_path = os.path.join(self._curobo_plan_debug_dir, filename)

            payload = {
                "schema_version": 1,
                "controller": {
                    "name": self.name,
                    "lr_name": self.lr_name,
                    "robot_file": self.robot_file,
                    "use_batch": bool(self.use_batch),
                    "branch": branch,
                    "selected_path_index": self._debug_json_ready(selected_path_index),
                    "selected_path_source": selected_path_source,
                },
                "joint_mapping": {
                    "robot_dof_names": self._debug_json_ready(js_names),
                    "cmd_js_names": self._debug_json_ready(self.cmd_js_names),
                    "raw_js_names": self._debug_json_ready(self.raw_js_names),
                    "arm_indices": self._debug_json_ready(self.arm_indices),
                    "gripper_indices": self._debug_json_ready(self.gripper_indices),
                    "idx_list": self._debug_json_ready(self.idx_list),
                },
                "goal": {
                    "ee_translation": self._debug_json_ready(ee_trans),
                    "ee_orientation": self._debug_json_ready(ee_ori),
                },
                "input_current_state": {
                    "sim_js": self._debug_joint_state_to_dict(sim_js),
                    "current_arm_sim_order": self._debug_json_ready(current_arm),
                    "curobo_input_ordered_cmd_js_names": self._debug_joint_state_to_dict(cu_js_ordered),
                },
                "result_summary": {
                    "success": self._debug_json_ready(getattr(result, "success", None)),
                    "status": self._debug_json_ready(getattr(result, "status", None)),
                    "valid_query": self._debug_json_ready(getattr(result, "valid_query", None)),
                    "position_error": self._debug_json_ready(getattr(result, "position_error", None)),
                    "rotation_error": self._debug_json_ready(getattr(result, "rotation_error", None)),
                    "cspace_error": self._debug_json_ready(getattr(result, "cspace_error", None)),
                    "optimized_dt": self._debug_json_ready(getattr(result, "optimized_dt", None)),
                    "interpolation_dt": self._debug_json_ready(getattr(result, "interpolation_dt", None)),
                    "path_buffer_last_tstep": self._debug_json_ready(
                        getattr(result, "path_buffer_last_tstep", None)
                    ),
                    "used_graph": self._debug_json_ready(getattr(result, "used_graph", None)),
                    "attempts": self._debug_json_ready(getattr(result, "attempts", None)),
                    "trajopt_attempts": self._debug_json_ready(getattr(result, "trajopt_attempts", None)),
                },
                "continuity": {
                    "first_ordered_minus_current_arm_norm": self._debug_norm_delta(first_ordered, current_arm),
                    "last_ordered_minus_current_arm_norm": self._debug_norm_delta(last_ordered, current_arm),
                    "first_ordered_position": self._debug_json_ready(first_ordered),
                    "last_ordered_position": self._debug_json_ready(last_ordered),
                },
                "raw_plan": self._debug_joint_state_to_dict(raw_plan),
                "ordered_cmd_plan": self._debug_joint_state_to_dict(ordered_cmd_plan),
            }
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(self._debug_json_ready(payload), handle, indent=2, ensure_ascii=False)
            print(
                "[curobo-plan-debug] Wrote plan debug: "
                f"{output_path}; first_delta={payload['continuity']['first_ordered_minus_current_arm_norm']}"
            )
        except Exception as exc:  # Debug writing must never break an episode.
            print(f"[curobo-plan-debug] Failed to write plan debug: {exc}")

    @staticmethod
    def _signature_value(value: Any):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, numbers.Real):
            return round(float(value), 6)
        if isinstance(value, np.ndarray):
            return tuple(round(float(x), 6) for x in value.reshape(-1).tolist())
        if isinstance(value, torch.Tensor):
            return tuple(round(float(x), 6) for x in value.detach().cpu().reshape(-1).tolist())
        if isinstance(value, (list, tuple)):
            return tuple(TemplateController._signature_value(x) for x in value)
        return str(value)

    @classmethod
    def _make_world_update_signature(cls, world_cfg: WorldConfig):
        objects = getattr(world_cfg, "objects", None) or []
        signature = []
        for obj in objects:
            signature.append(
                (
                    type(obj).__name__,
                    getattr(obj, "name", None),
                    cls._signature_value(getattr(obj, "pose", None)),
                    cls._signature_value(getattr(obj, "dims", None)),
                    cls._signature_value(getattr(obj, "scale", None)),
                    getattr(obj, "file_path", None),
                    cls._signature_value(getattr(obj, "vertices", None)),
                    cls._signature_value(getattr(obj, "faces", None)),
                )
            )
        return tuple(signature)

    def _update_world_if_changed(self, obstacles: WorldConfig) -> None:
        signature = self._make_world_update_signature(obstacles)
        if self.motion_gen is not None and signature != getattr(self, "_world_update_signature", None):
            self.motion_gen.update_world(obstacles)
        self.world_cfg = obstacles
        self._world_update_signature = signature

    def update(self) -> None:
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=self.ignore_substring, reference_prim_path=self.reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)

    def _clear_attached_object_state(self) -> None:
        if self.motion_gen is None:
            return
        try:
            self.motion_gen.detach_object_from_robot()
            self.motion_gen.clear_world_cache()
            self._world_update_signature = None
        except Exception as exc:
            print(f"[curobo-controller] Failed to clear attached object state during reset: {exc}")

    def reset(self, ignore_substring: Optional[str] = None) -> None:
        if ignore_substring:
            self.ignore_substring = ignore_substring
        self._clear_attached_object_state()
        self.update()
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.num_last_cmd = 0
        self._last_arm_action = None
        self.num_plan_failed = 0
        if self.lr_name == "left":
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif self.lr_name == "right":
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        if self.lr_name == "left":
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
        else:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
        self.T_base_ee_init = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_world_base_init = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_world_ee_init = self.T_world_base_init @ self.T_base_ee_init
        self._ee_trans, self._ee_ori = self.get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(self._ee_trans)
        self._ee_ori = self.tensor_args.to_device(self._ee_ori)
        self.update_pose_cost_metric()

    def plan_batch(self, ee_translation_goal_batch, ee_orientation_goal_batch, sim_js, js_names):
        t1 = time.time()
        torch.cuda.synchronize()
        sim_js_positions = (sim_js.positions)[np.newaxis, :]
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal_batch),
            quaternion=self.tensor_args.to_device(ee_orientation_goal_batch),
            batch=CUROBO_BATCH_SIZE,
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))),
            velocity=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            acceleration=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            jerk=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        result = self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        t2 = time.time()
        torch.cuda.synchronize()
        print("plan batch duration :", t2 - t1)
        return result

    def plan(self, ee_translation_goal, ee_orientation_goal, sim_js: JointState, js_names: list):
        if self.use_batch:
            ik_goal = Pose(
                position=self.tensor_args.to_device(ee_translation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                quaternion=self.tensor_args.to_device(ee_orientation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                batch=CUROBO_BATCH_SIZE,
            )
            cu_js = JointState(
                position=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))),
                velocity=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                acceleration=self.tensor_args.to_device(
                    np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))
                )
                * 0.0,
                jerk=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                joint_names=js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
            return self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        return self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())

    def forward(self, manip_cmd, eps=5e-3):
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = dict(manip_cmd[3])
        skip_plan = bool(params.pop("skip_plan", False))
        gripper_action = params.pop("gripper_action", None)
        params.pop("t_eps", None)
        params.pop("o_eps", None)
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True, gripper_action=gripper_action)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps, skip_plan=skip_plan, gripper_action=gripper_action)

    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        gripper_action=None,
    ):
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        plan_flag = torch.logical_or(
            torch.norm(self._ee_trans - ee_trans) > eps,
            torch.norm(self._ee_ori - ee_ori) > eps,
        )
        if not skip_plan:
            if plan_flag:
                self.cmd_plan = None
                self.cmd_idx = 0
                self._step_idx = 0
                self.num_last_cmd = 0
                self._last_arm_action = None
                result = self.plan(ee_trans, ee_ori, sim_js, js_names)
                if self.use_batch:
                    if result.success.any():
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        paths = result.get_successful_paths()
                        position_filter_res = filter_paths_by_position_error(
                            paths, result.position_error[result.success]
                        )
                        rotation_filter_res = filter_paths_by_rotation_error(
                            paths, result.rotation_error[result.success]
                        )
                        filtered_paths = [
                            p for i, p in enumerate(paths) if position_filter_res[i] and rotation_filter_res[i]
                        ]
                        if len(filtered_paths) == 0:
                            filtered_paths = paths
                        sort_weights = self._get_sort_path_weights()  # pylint: disable=assignment-from-none
                        weights_arg = self.tensor_args.to_device(sort_weights) if sort_weights is not None else None
                        sorted_indices = sort_by_difference_js(filtered_paths, weights=weights_arg)
                        cmd_plan = self.motion_gen.get_full_js(paths[sorted_indices[0]])
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self._write_curobo_plan_debug(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=cmd_plan,
                            ordered_cmd_plan=self.cmd_plan,
                            branch="batch",
                            selected_path_index=sorted_indices[0],
                            selected_path_source="paths[sorted_indices[0]]",
                        )
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self.num_plan_failed += 1
                else:
                    succ = result.success.item()
                    if succ:
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        cmd_plan = result.get_interpolated_plan()
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self._write_curobo_plan_debug(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=cmd_plan,
                            ordered_cmd_plan=self.cmd_plan,
                            branch="single",
                            selected_path_index=0,
                            selected_path_source="result.get_interpolated_plan()",
                        )
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self.num_plan_failed += 1
            if self.cmd_plan and self._step_idx % 1 == 0:
                cmd_state = self.cmd_plan[self.cmd_idx]
                arm_action = cmd_state.position.cpu().numpy()
                self._last_arm_action = np.asarray(arm_action, dtype=float).copy()
                art_action = ArticulationAction(
                    arm_action,
                    cmd_state.velocity.cpu().numpy() * 0.0,
                    joint_indices=self.idx_list,
                )
                self.cmd_idx += self.ds_ratio
                if self.cmd_idx >= len(self.cmd_plan):
                    self.cmd_idx = 0
                    self.cmd_plan = None
            else:
                self.num_last_cmd += 1
                if self._last_arm_action is None:
                    arm_action = sim_js.positions[self.arm_indices]
                else:
                    arm_action = self._last_arm_action
                art_action = ArticulationAction(joint_positions=arm_action)
        else:
            arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float).copy()
            self._last_arm_action = arm_action.copy()
            art_action = ArticulationAction(joint_positions=arm_action)
            self.num_last_cmd += 1
        self._step_idx += 1
        arm_action = art_action.joint_positions
        if gripper_action is None:
            gripper_action = self.get_gripper_action()
        else:
            gripper_action = np.asarray(gripper_action, dtype=float)
        joint_positions = np.concatenate([arm_action, gripper_action])
        self._action = {
            "joint_positions": joint_positions,
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
        return self._action

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)

    def get_ee_pose(self):
        sim_js = self.robot.get_joints_state()
        q_state = torch.tensor(sim_js.positions[self.arm_indices], **self.tensor_args.as_torch_dict()).reshape(1, -1)
        ee_pose = self.kin_model.get_state(q_state)
        return ee_pose.ee_position[0].cpu().numpy(), ee_pose.ee_quaternion[0].cpu().numpy()

    def get_armbase_pose(self):
        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)

    def forward_kinematic(self, q_state: np.ndarray):
        q_state = q_state.reshape(1, -1)
        q_state = self.tensor_args.to_device(q_state)
        out = self.kin_model.get_state(q_state)
        return out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()

    def close_gripper(self):
        self._gripper_state = -1.0

    def open_gripper(self):
        self._gripper_state = 1.0

    def _get_curobo_world_object_names(self) -> List[str]:
        if self.motion_gen is not None:
            world_model = getattr(self.motion_gen, "world_model", None)
            objects = getattr(world_model, "objects", None)
            if objects is not None:
                return [obj.name for obj in objects]

        objects = getattr(self.world_cfg, "objects", None)
        if objects is not None:
            return [obj.name for obj in objects]
        return []

    @staticmethod
    def _select_attach_descendants(object_names: List[str]) -> List[str]:
        visual_names = [name for name in object_names if "/visual" in name or name.endswith("/visual")]
        if visual_names:
            return visual_names[:1]

        collision_names = [name for name in object_names if "/collisions/" in name or name.endswith("/collisions")]
        if collision_names:
            return collision_names[:1]

        return object_names

    def _resolve_attach_object_names(self, obj_prim_path) -> Tuple[List[str], List[str]]:
        requested_names = obj_prim_path if isinstance(obj_prim_path, (list, tuple)) else [obj_prim_path]
        world_object_names = self._get_curobo_world_object_names()
        world_object_set = set(world_object_names)

        resolved_names = []
        disabled_names = []
        for requested_name in requested_names:
            if requested_name in world_object_set:
                candidates = [requested_name]
                descendants = []
            else:
                prefix = requested_name.rstrip("/") + "/"
                descendants = [name for name in world_object_names if name.startswith(prefix)]
                candidates = self._select_attach_descendants(descendants)
                if not candidates:
                    candidates = [requested_name]

            for candidate in candidates:
                if candidate not in resolved_names:
                    resolved_names.append(candidate)
            for descendant in descendants:
                if descendant not in resolved_names and descendant not in disabled_names:
                    disabled_names.append(descendant)

        return resolved_names, disabled_names

    def attach_obj(self, obj_prim_path: str, link_name="attached_object"):
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        object_names, disabled_names = self._resolve_attach_object_names(obj_prim_path)
        self.motion_gen.attach_objects_to_robot(
            cu_js,
            object_names,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )
        for object_name in disabled_names:
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot()

    def update_specific(self, ignore_substring, reference_prim_path):
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)

    def test_single_ik(self, ee_trans, ee_ori):
        assert not self.use_batch
        ik_goal = Pose(position=self.tensor_args.to_device(ee_trans), quaternion=self.tensor_args.to_device(ee_ori))
        result = self.ik_solver.solve_single(ik_goal)
        succ = result.success.item()
        if succ:  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

    def test_batch_forward(self, ee_trans_batch_np, ee_ori_batch_np):
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan_batch(ee_trans_batch, ee_ori_batch, sim_js, js_names)

        return result

    def test_single_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray):
        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        succ = result.success.item()
        if succ:
            print("Success")
            return 1
        print("Plan did not converge to a solution.")
        return 0

    def test_forward_from_joint_positions(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_arm_positions: Optional[np.ndarray] = None,
    ):
        """Plan without changing runtime controller state and return the path endpoint."""
        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        sim_js.positions = np.asarray(sim_js.positions, dtype=float).copy()
        if start_arm_positions is not None:
            start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
            if len(start_arm_positions) != len(self.arm_indices):
                raise ValueError(
                    "start_arm_positions must match the controller arm joint count: "
                    f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
                )
            sim_js.positions[self.arm_indices] = start_arm_positions

        result = self.plan(ee_trans, ee_ori, sim_js, self.robot.dof_names)
        success_values = result.success.detach().cpu().numpy()
        if not bool(np.asarray(success_values).any()):
            return False, None, result

        if self.use_batch:
            paths = result.get_successful_paths()
            if not paths:
                return False, None, result
            position_filter_res = filter_paths_by_position_error(paths, result.position_error[result.success])
            rotation_filter_res = filter_paths_by_rotation_error(paths, result.rotation_error[result.success])
            filtered_paths = [
                path
                for path_index, path in enumerate(paths)
                if position_filter_res[path_index] and rotation_filter_res[path_index]
            ]
            if not filtered_paths:
                filtered_paths = paths
            sort_weights = self._get_sort_path_weights()  # pylint: disable=assignment-from-none
            weights_arg = self.tensor_args.to_device(sort_weights) if sort_weights is not None else None
            sorted_indices = sort_by_difference_js(filtered_paths, weights=weights_arg)
            cmd_plan = self.motion_gen.get_full_js(filtered_paths[sorted_indices[0]])
        else:
            cmd_plan = result.get_interpolated_plan()
        cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
        end_arm_positions = np.asarray(cmd_plan[-1].position.detach().cpu(), dtype=float)
        return True, end_arm_positions, result

    def pre_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray, expected_js=None, ds_ratio=1):
        assert ee_trans is not None and ee_ori is not None
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        if expected_js is not None:
            sim_js.positions[self.arm_indices] = expected_js
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        if self.use_batch:
            if result.success.any():
                print("Success")
                cmd_plans = result.get_successful_paths()
                cmd_plan = random.choice(cmd_plans)
                cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                N = cmd_plan.shape[0]
                dt = self.motion_gen.interpolation_dt
                self.ds_ratio = ds_ratio
                cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
                return cmd_time, np.array(cmd_plan[-1].position.cpu())
            print("Plan did not converge to a solution.")
            self.num_plan_failed = 1000
            return 0, expected_js
        succ = result.success.item()
        if succ:
            print("Success")
            cmd_plan = result.get_interpolated_plan()
            N = cmd_plan.shape[0]
            dt = self.motion_gen.interpolation_dt
            self.ds_ratio = ds_ratio
            cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
            return cmd_time, np.array(cmd_plan[-1].position.cpu())
        print("Plan did not converge to a solution.")
        self.num_plan_failed = 1000
        return 0, expected_js

    def in_plane_rotation(self, target_rotate: np.ndarray):
        action = deepcopy(self._action)
        last_arm = len(self.arm_indices) - 1
        action["joint_positions"][last_arm] -= target_rotate
        action["arm_action"][last_arm] -= target_rotate
        return action

    def mobile_move(self, target: np.ndarray, joint_indices: np.ndarray = None, initial_position: np.ndarray = None):
        return {
            "joint_positions": initial_position + target,
            "joint_indices": np.array(joint_indices),
            "lr_name": "whole",
        }

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        arm_action = np.asarray(arm_action, dtype=float).copy()
        self.cmd_plan = None
        self.cmd_idx = 0
        self._last_arm_action = arm_action.copy()
        if gripper_state == 1.0:
            self.open_gripper()
        elif gripper_state == -1.0:
            self.close_gripper()
        else:
            raise NotImplementedError
        gripper_action = self.get_gripper_action()
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
