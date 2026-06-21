"""Nav2 runtime configuration normalization and parameter generation."""

from __future__ import annotations

from copy import deepcopy
import os

import yaml

from workflows.simbox.core.mobile.platforms import get_mobile_base_platform

from .utils import footprint_inscribed_radius

NAV2_DEFAULT_MAX_ACKERMANN_STEER_RAD = 0.6981
NAV2_DEFAULT_POSITION_TOLERANCE_M = 0.10
NAV2_DEFAULT_YAW_TOLERANCE_RAD = 0.10

DEFAULT_NAV2_SKILL_FOOTPRINT_POINTS = [
    [0.36, 0.24],
    [0.32, 0.29],
    [-0.32, 0.29],
    [-0.36, 0.24],
    [-0.36, -0.24],
    [-0.32, -0.29],
    [0.32, -0.29],
    [0.36, -0.24],
]
DEFAULT_NAV2_SKILL_INFLATION_RADIUS_M = 0.34
DEFAULT_NAV2_SKILL_MIN_TURN_RADIUS_M = 0.47644


def format_nav2_footprint(points: list[list[float]]) -> str:
    return "[" + ", ".join(f"[{float(x):.3f}, {float(y):.3f}]" for x, y in points) + "]"


def nav2_skill_cfg(base_cfg: dict) -> dict:
    return dict(base_cfg.get("nav2_skill", {}))


def _deep_update_dict(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_dict(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _normalize_footprint_points(points) -> list[list[float]]:
    if not isinstance(points, (list, tuple)):
        return []
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append([float(point[0]), float(point[1])])
    if len(normalized) < 3:
        return []
    return normalized


def _build_mppi_follow_path_params(
    follow_path_cfg: dict,
    follow_path_plugin: str,
    *,
    max_velocity: tuple[float, float, float],
    min_velocity: tuple[float, float, float],
    max_accel: tuple[float, float, float],
    max_decel: tuple[float, float, float],
) -> dict:
    return {
        "plugin": follow_path_plugin,
        "time_steps": int(follow_path_cfg.get("time_steps", 40)),
        "model_dt": float(follow_path_cfg.get("model_dt", 0.05)),
        "batch_size": int(follow_path_cfg.get("batch_size", 1200)),
        "iteration_count": int(follow_path_cfg.get("iteration_count", 1)),
        "prune_distance": float(follow_path_cfg.get("prune_distance", 1.8)),
        "transform_tolerance": float(follow_path_cfg.get("transform_tolerance", 0.3)),
        "temperature": float(follow_path_cfg.get("temperature", 0.3)),
        "gamma": float(follow_path_cfg.get("gamma", 0.015)),
        "motion_model": str(follow_path_cfg.get("motion_model", "Omni")),
        "open_loop": bool(follow_path_cfg.get("open_loop", False)),
        "visualize": bool(follow_path_cfg.get("visualize", False)),
        "regenerate_noises": bool(follow_path_cfg.get("regenerate_noises", False)),
        "reset_period": float(follow_path_cfg.get("reset_period", 1.0)),
        "retry_attempt_limit": int(follow_path_cfg.get("retry_attempt_limit", 1)),
        "vx_max": float(max_velocity[0]),
        "vx_min": float(min_velocity[0]),
        "vy_max": float(max_velocity[1]),
        "vy_min": float(min_velocity[1]),
        "wz_max": float(max_velocity[2]),
        "ax_max": float(max_accel[0]),
        "ax_min": float(max_decel[0]),
        "ay_max": float(max_accel[1]),
        "ay_min": float(max_decel[1]),
        "az_max": float(max_accel[2]),
        "vx_std": float(follow_path_cfg.get("vx_std", 0.12)),
        "vy_std": float(follow_path_cfg.get("vy_std", 0.14)),
        "wz_std": float(follow_path_cfg.get("wz_std", 0.25)),
        "TrajectoryVisualizer": dict(
            follow_path_cfg.get(
                "TrajectoryVisualizer",
                {
                    "trajectory_step": 5,
                    "time_step": 3,
                },
            )
        ),
        "TrajectoryValidator": dict(
            follow_path_cfg.get(
                "TrajectoryValidator",
                {
                    "plugin": "mppi::DefaultOptimalTrajectoryValidator",
                    "collision_lookahead_time": 2.0,
                    "consider_footprint": True,
                },
            )
        ),
        "critics": list(
            follow_path_cfg.get(
                "critics",
                [
                    "ConstraintCritic",
                    "CostCritic",
                    "GoalCritic",
                    "GoalAngleCritic",
                    "PathAlignCritic",
                    "PathFollowCritic",
                    "PathAngleCritic",
                    "TwirlingCritic",
                ],
            )
        ),
        "ConstraintCritic": dict(
            follow_path_cfg.get(
                "ConstraintCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 4.0,
                },
            )
        ),
        "CostCritic": dict(
            follow_path_cfg.get(
                "CostCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 3.8,
                    "critical_cost": 300.0,
                    "consider_footprint": True,
                    "collision_cost": 1000000.0,
                    "near_goal_distance": 0.4,
                    "trajectory_point_step": 2,
                },
            )
        ),
        "GoalCritic": dict(
            follow_path_cfg.get(
                "GoalCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 5.0,
                    "threshold_to_consider": 1.4,
                },
            )
        ),
        "GoalAngleCritic": dict(
            follow_path_cfg.get(
                "GoalAngleCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 3.0,
                    "threshold_to_consider": 0.4,
                },
            )
        ),
        "PathAlignCritic": dict(
            follow_path_cfg.get(
                "PathAlignCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 10.0,
                    "threshold_to_consider": 0.8,
                    "offset_from_furthest": 10,
                    "max_path_occupancy_ratio": 0.2,
                    "use_path_orientations": True,
                },
            )
        ),
        "PathFollowCritic": dict(
            follow_path_cfg.get(
                "PathFollowCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 8.0,
                    "threshold_to_consider": 1.4,
                    "offset_from_furthest": 6,
                },
            )
        ),
        "PathAngleCritic": dict(
            follow_path_cfg.get(
                "PathAngleCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 3.0,
                    "threshold_to_consider": 0.8,
                    "offset_from_furthest": 10,
                    "max_angle_to_furthest": 0.78539816339,
                    "mode": 2,
                },
            )
        ),
        "TwirlingCritic": dict(
            follow_path_cfg.get(
                "TwirlingCritic",
                {
                    "enabled": True,
                    "cost_power": 1,
                    "cost_weight": 8.0,
                },
            )
        ),
    }


def configure_base_cfg_for_nav2_skill(
    base_cfg: dict,
    *,
    map_output_dir: str = "output/nav2_maps",
    map_resolution: float = 0.02,
    map_z_min: float = 0.0,
    map_z_max: float = 0.35,
    map_include_visual_wall_geometry: bool = True,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    nav2_skill_overrides: dict | None = None,
):
    """Normalize mobile-base config for external compose-managed Nav2 sessions."""

    base_cfg = deepcopy(base_cfg)
    platform = get_mobile_base_platform(base_cfg)
    base_cfg = platform.normalize_base_cfg(base_cfg)
    ros_cfg = base_cfg.setdefault("ros", {})

    ros_cfg["cmd_vel_topic"] = str(ros_cfg.get("cmd_vel_topic", "/cmd_vel"))
    ros_cfg.pop("command_topic", None)
    ros_cfg.pop("motion_mode_topic", None)
    ros_cfg.pop("command_type", None)
    ros_cfg.pop("internal_cmdvel_controller_enabled", None)
    ros_cfg["max_steer_angle_ackermann"] = float(platform.max_steer_angle_ackermann(base_cfg))
    virtual_odom_cfg = ros_cfg.setdefault("virtual_odom", {})
    virtual_odom_cfg["enabled"] = False
    virtual_odom_cfg["publish_twist"] = True
    virtual_odom_cfg["use_world_z"] = True
    virtual_odom_cfg["default_z"] = float(virtual_odom_cfg.get("default_z", 0.0))

    localization_cfg = ros_cfg.setdefault("localization", {})
    localization_cfg["enabled"] = True
    localization_cfg["mode"] = "static_map_truth_pose"
    localization_cfg["map_resolution"] = float(map_resolution)
    localization_cfg["map_output_dir"] = str(map_output_dir)
    localization_cfg["map_z_min"] = float(map_z_min)
    localization_cfg["map_z_max"] = float(map_z_max)
    localization_cfg["map_include_visual_wall_geometry"] = bool(map_include_visual_wall_geometry)
    localization_cfg["map_frame"] = str(localization_cfg.get("map_frame", "map"))
    localization_cfg["odom_frame"] = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    localization_cfg["base_frame"] = str(localization_cfg.get("base_frame", ros_cfg.get("base_frame", "base_link")))
    localization_cfg.setdefault(
        "clear_footprint_points",
        platform.default_nav2_footprint_points(base_cfg),
    )
    localization_cfg.setdefault("robot_clear_radius_m", 0.0)

    nav2_cfg = ros_cfg.setdefault("nav2", {})
    nav2_cfg["enabled"] = True
    nav2_cfg["global_frame"] = str(localization_cfg.get("map_frame", "map"))
    nav2_cfg["robot_base_frame"] = str(localization_cfg.get("base_frame", ros_cfg.get("base_frame", "base_link")))
    nav2_cfg["skill_managed"] = True
    nav2_cfg["runtime_mode"] = "external_compose"
    nav2_cfg["stack_request_root"] = str(nav2_cfg.get("stack_request_root", "output/ros_bridge/runtime_requests"))
    nav2_cfg["stack_status_root"] = str(nav2_cfg.get("stack_status_root", "output/ros_bridge/runtime_status"))
    nav2_cfg["goal_request_root"] = str(nav2_cfg.get("goal_request_root", "output/ros_bridge/goal_requests"))
    nav2_cfg["goal_status_root"] = str(nav2_cfg.get("goal_status_root", "output/ros_bridge/goal_status"))
    nav2_cfg["goal_result_root"] = str(nav2_cfg.get("goal_result_root", "output/ros_bridge/goal_result"))
    nav2_cfg["stack_reuse"] = bool(nav2_cfg.get("stack_reuse", True))
    nav2_cfg["goal_transport"] = str(nav2_cfg.get("goal_transport", "ros_topic_bridge"))
    nav2_cfg["load_map_service"] = str(nav2_cfg.get("load_map_service", "/map_server/load_map"))
    nav2_cfg["clear_global_costmap_service"] = str(
        nav2_cfg.get("clear_global_costmap_service", "/global_costmap/clear_entirely_global_costmap")
    )
    nav2_cfg["clear_local_costmap_service"] = str(
        nav2_cfg.get("clear_local_costmap_service", "/local_costmap/clear_entirely_local_costmap")
    )
    nav2_cfg["bridge_map_update_topic"] = str(nav2_cfg.get("bridge_map_update_topic", "/simbox/nav_bridge/map_update"))
    nav2_cfg["bridge_goal_topic"] = str(nav2_cfg.get("bridge_goal_topic", "/simbox/nav_bridge/goal"))
    nav2_cfg["bridge_plan_topic"] = str(nav2_cfg.get("bridge_plan_topic", "/simbox/nav_bridge/plan"))
    nav2_cfg["bridge_cancel_topic"] = str(nav2_cfg.get("bridge_cancel_topic", "/simbox/nav_bridge/cancel"))
    nav2_cfg["bridge_reset_topic"] = str(nav2_cfg.get("bridge_reset_topic", "/simbox/nav_bridge/reset"))
    nav2_cfg["bridge_status_topic"] = str(nav2_cfg.get("bridge_status_topic", "/simbox/nav_bridge/status"))
    nav2_cfg["bridge_result_topic"] = str(nav2_cfg.get("bridge_result_topic", "/simbox/nav_bridge/result"))
    nav2_cfg["bridge_plan_result_topic"] = str(
        nav2_cfg.get("bridge_plan_result_topic", "/simbox/nav_bridge/plan_result")
    )
    nav2_cfg["bridge_alive_timeout_sec"] = float(nav2_cfg.get("bridge_alive_timeout_sec", 3.0))

    base_cfg.setdefault("nav2_skill", {})
    if isinstance(nav2_skill_overrides, dict):
        _deep_update_dict(base_cfg["nav2_skill"], nav2_skill_overrides)
    base_cfg["nav2_skill"]["position_tolerance_m"] = float(position_tolerance_m)
    base_cfg["nav2_skill"]["yaw_tolerance_rad"] = float(yaw_tolerance_rad)
    return deepcopy(base_cfg)


def configure_robot_for_nav2_skill(
    robot,
    *,
    map_output_dir: str = "output/nav2_maps",
    map_resolution: float = 0.02,
    map_z_min: float = 0.0,
    map_z_max: float = 0.35,
    map_include_visual_wall_geometry: bool = True,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    nav2_skill_overrides: dict | None = None,
):
    base_cfg = configure_base_cfg_for_nav2_skill(
        getattr(robot, "base_cfg", {}),
        map_output_dir=map_output_dir,
        map_resolution=map_resolution,
        map_z_min=map_z_min,
        map_z_max=map_z_max,
        map_include_visual_wall_geometry=map_include_visual_wall_geometry,
        position_tolerance_m=position_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
        nav2_skill_overrides=nav2_skill_overrides,
    )
    robot.base_cfg = base_cfg
    return deepcopy(base_cfg)


def generate_nav2_bringup_artifacts(
    output_dir: str,
    *,
    base_cfg: dict,
    map_yaml_path: str,
    position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
    yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    params_filename: str = "split_aloha_nav2_skill_params.yaml",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    nav_to_pose_bt, nav_through_poses_bt = _write_nav2_bt_files(output_dir, base_cfg)
    params = _build_nav2_params(
        nav_to_pose_bt=nav_to_pose_bt,
        nav_through_poses_bt=nav_through_poses_bt,
        base_cfg=base_cfg,
        position_tolerance_m=position_tolerance_m,
        yaw_tolerance_rad=yaw_tolerance_rad,
    )
    params["map_server"]["ros__parameters"]["yaml_filename"] = str(map_yaml_path)
    params_path = os.path.join(output_dir, params_filename)
    with open(params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=False)
    return {
        "params_path": params_path,
        "params": params,
        "nav_to_pose_bt": nav_to_pose_bt,
        "nav_through_poses_bt": nav_through_poses_bt,
    }


def _write_nav2_bt_files(output_dir: str, base_cfg: dict) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    nav_to_pose_bt_path = os.path.join(output_dir, "navigate_to_pose_w_replanning_no_motion_recovery.xml")
    nav_through_poses_bt_path = os.path.join(
        output_dir,
        "navigate_through_poses_w_replanning_no_motion_recovery.xml",
    )
    skill_cfg = nav2_skill_cfg(base_cfg)
    bt_cfg = dict(skill_cfg.get("bt_navigator", {}))
    nav2_cfg = dict(base_cfg.get("ros", {}).get("nav2", {}))
    replanning_hz = float(bt_cfg.get("replanning_hz", 0.25))
    replan_retry_attempt_limit = int(bt_cfg.get("replan_retry_attempt_limit", 3))
    remove_passed_goals_radius = float(bt_cfg.get("remove_passed_goals_radius", 0.7))
    clear_global_costmap_service = str(
        nav2_cfg.get("clear_global_costmap_service", "/global_costmap/clear_entirely_global_costmap")
    )
    clear_local_costmap_service = str(
        nav2_cfg.get("clear_local_costmap_service", "/local_costmap/clear_entirely_local_costmap")
    )
    nav_to_pose_bt = f"""<!-- Holonomic Nav2 navigation without recovery/fallback behaviors. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithReplanning">
      <RateController hz="{replanning_hz}">
        <RecoveryNode number_of_retries="{replan_retry_attempt_limit}" name="ReplanToPose">
          <ComputePathToPose goal="{{goal}}" path="{{raw_path}}" planner_id="GridBased"/>
          <Sequence name="ReplanRecovery">
            <ClearEntireCostmap
              name="ClearGlobalCostmapForReplan"
              service_name="{clear_global_costmap_service}"/>
            <ClearEntireCostmap
              name="ClearLocalCostmapForReplan"
              service_name="{clear_local_costmap_service}"/>
          </Sequence>
        </RecoveryNode>
      </RateController>
      <SmoothPath unsmoothed_path="{{raw_path}}" smoothed_path="{{path}}" smoother_id="simple_smoother"/>
      <FollowPath path="{{path}}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
"""

    nav_through_poses_bt = f"""<!-- Holonomic Nav2 navigation through poses without recovery/fallback behaviors. -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateThroughPosesWithReplanning">
      <RateController hz="{replanning_hz}">
        <RecoveryNode number_of_retries="{replan_retry_attempt_limit}" name="ReplanThroughPoses">
          <Sequence name="ComputePathThroughRemainingGoals">
            <RemovePassedGoals input_goals="{{goals}}" output_goals="{{goals}}" radius="{remove_passed_goals_radius}"/>
            <ComputePathThroughPoses goals="{{goals}}" path="{{raw_path}}" planner_id="GridBased"/>
          </Sequence>
          <Sequence name="ReplanRecovery">
            <ClearEntireCostmap
              name="ClearGlobalCostmapForReplan"
              service_name="{clear_global_costmap_service}"/>
            <ClearEntireCostmap
              name="ClearLocalCostmapForReplan"
              service_name="{clear_local_costmap_service}"/>
          </Sequence>
        </RecoveryNode>
      </RateController>
      <SmoothPath unsmoothed_path="{{raw_path}}" smoothed_path="{{path}}" smoother_id="simple_smoother"/>
      <FollowPath path="{{path}}" controller_id="FollowPath"/>
    </PipelineSequence>
  </BehaviorTree>
</root>
"""

    with open(nav_to_pose_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_to_pose_bt)
    with open(nav_through_poses_bt_path, "w", encoding="utf-8") as file:
        file.write(nav_through_poses_bt)
    return nav_to_pose_bt_path, nav_through_poses_bt_path


def _build_nav2_params(
    nav_to_pose_bt: str,
    nav_through_poses_bt: str,
    base_cfg: dict,
    *,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
):
    ros_cfg = dict(base_cfg.get("ros", {}))
    platform = get_mobile_base_platform(base_cfg)
    controller_hard_limits = platform.nav2_controller_hard_limits(base_cfg)
    max_velocity = list(controller_hard_limits["max_velocity"])
    min_velocity = list(controller_hard_limits["min_velocity"])
    max_accel = list(controller_hard_limits["max_accel"])
    max_decel = list(controller_hard_limits["max_decel"])
    nav2_cfg = dict(ros_cfg.get("nav2", {}))
    localization_cfg = dict(ros_cfg.get("localization", {}))
    map_frame = str(localization_cfg.get("map_frame", nav2_cfg.get("global_frame", "map")))
    odom_frame = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    base_frame = str(
        localization_cfg.get("base_frame", nav2_cfg.get("robot_base_frame", ros_cfg.get("base_frame", "base_link")))
    )
    skill_cfg = nav2_skill_cfg(base_cfg)
    footprint_points = _normalize_footprint_points(skill_cfg.get("footprint_points"))
    if not footprint_points:
        footprint_points = platform.default_nav2_footprint_points(base_cfg)
    footprint = format_nav2_footprint(footprint_points)
    inflation_radius = float(skill_cfg.get("inflation_radius_m", platform.default_nav2_inflation_radius_m(base_cfg)))
    inscribed_radius = footprint_inscribed_radius(footprint_points)
    inflation_radius = max(inflation_radius, inscribed_radius)
    minimum_turning_radius = float(platform.default_nav2_minimum_turning_radius_m(base_cfg))
    bt_cfg = dict(skill_cfg.get("bt_navigator", {}))
    bt_plugins = list(
        bt_cfg.get(
            "plugin_lib_names",
            [
                "nav2_compute_path_to_pose_action_bt_node",
                "nav2_compute_path_through_poses_action_bt_node",
                "nav2_smooth_path_action_bt_node",
                "nav2_follow_path_action_bt_node",
                "nav2_clear_costmap_service_bt_node",
                "nav2_goal_updated_condition_bt_node",
                "nav2_remove_passed_goals_action_bt_node",
                "nav2_rate_controller_bt_node",
                "nav2_pipeline_sequence_bt_node",
                "nav2_recovery_node_bt_node",
                "nav2_navigate_to_pose_action_bt_node",
                "nav2_navigate_through_poses_action_bt_node",
            ],
        )
    )
    controller_cfg = dict(skill_cfg.get("controller_server", {}))
    progress_checker_cfg = dict(controller_cfg.get("progress_checker", {}))
    goal_checker_cfg = dict(controller_cfg.get("goal_checker", {}))
    follow_path_cfg = dict(controller_cfg.get("follow_path", {}))
    local_costmap_cfg = dict(skill_cfg.get("local_costmap", {}))
    global_costmap_cfg = dict(skill_cfg.get("global_costmap", {}))
    planner_cfg = dict(skill_cfg.get("planner_server", {}))
    smoother_cfg = dict(skill_cfg.get("smoother_server", {}))
    behavior_cfg = dict(skill_cfg.get("behavior_server", {}))
    waypoint_cfg = dict(skill_cfg.get("waypoint_follower", {}))
    velocity_smoother_cfg = dict(skill_cfg.get("velocity_smoother", {}))

    follow_path_plugin = str(follow_path_cfg.get("plugin", "nav2_mppi_controller::MPPIController"))
    rotation_shim_plugin = "nav2_rotation_shim_controller::RotationShimController"
    rotate_to_heading_enabled = bool(follow_path_cfg.get("rotate_to_heading_enabled", True))
    if follow_path_plugin == rotation_shim_plugin and not rotate_to_heading_enabled:
        follow_path_plugin = str(
            follow_path_cfg.get("primary_controller", "nav2_mppi_controller::MPPIController")
        )
    if follow_path_plugin == "dwb_core::DWBLocalPlanner":
        follow_path_params = {
            "plugin": follow_path_plugin,
            "debug_trajectory_details": bool(follow_path_cfg.get("debug_trajectory_details", False)),
            "short_circuit_trajectory_evaluation": bool(
                follow_path_cfg.get("short_circuit_trajectory_evaluation", True)
            ),
            "stateful": bool(follow_path_cfg.get("stateful", True)),
            "min_vel_x": float(follow_path_cfg.get("min_vel_x", -0.35)),
            "max_vel_x": float(follow_path_cfg.get("max_vel_x", 0.35)),
            "min_vel_y": float(follow_path_cfg.get("min_vel_y", -0.25)),
            "max_vel_y": float(follow_path_cfg.get("max_vel_y", 0.25)),
            "max_vel_theta": float(follow_path_cfg.get("max_vel_theta", 0.60)),
            "min_speed_xy": float(follow_path_cfg.get("min_speed_xy", 0.0)),
            "max_speed_xy": float(follow_path_cfg.get("max_speed_xy", 0.40)),
            "min_speed_theta": float(follow_path_cfg.get("min_speed_theta", 0.0)),
            "acc_lim_x": float(follow_path_cfg.get("acc_lim_x", 0.35)),
            "acc_lim_y": float(follow_path_cfg.get("acc_lim_y", 0.35)),
            "acc_lim_theta": float(follow_path_cfg.get("acc_lim_theta", 0.70)),
            "decel_lim_x": float(follow_path_cfg.get("decel_lim_x", -0.35)),
            "decel_lim_y": float(follow_path_cfg.get("decel_lim_y", -0.35)),
            "decel_lim_theta": float(follow_path_cfg.get("decel_lim_theta", -0.70)),
            "vx_samples": int(follow_path_cfg.get("vx_samples", 15)),
            "vy_samples": int(follow_path_cfg.get("vy_samples", 15)),
            "vtheta_samples": int(follow_path_cfg.get("vtheta_samples", 20)),
            "sim_time": float(follow_path_cfg.get("sim_time", 1.2)),
            "linear_granularity": float(follow_path_cfg.get("linear_granularity", 0.05)),
            "angular_granularity": float(follow_path_cfg.get("angular_granularity", 0.05)),
            "transform_tolerance": float(follow_path_cfg.get("transform_tolerance", 0.4)),
            "critics": list(
                follow_path_cfg.get(
                    "critics",
                    [
                        "BaseObstacle",
                        "GoalAlign",
                        "PathAlign",
                        "PathDist",
                        "GoalDist",
                        "Oscillation",
                        "RotateToGoal",
                    ],
                )
            ),
            "BaseObstacle.scale": float(follow_path_cfg.get("BaseObstacle.scale", 0.06)),
            "PathAlign.scale": float(follow_path_cfg.get("PathAlign.scale", 20.0)),
            "PathAlign.forward_point_distance": float(
                follow_path_cfg.get("PathAlign.forward_point_distance", 0.12)
            ),
            "GoalAlign.scale": float(follow_path_cfg.get("GoalAlign.scale", 16.0)),
            "GoalAlign.forward_point_distance": float(
                follow_path_cfg.get("GoalAlign.forward_point_distance", 0.12)
            ),
            "PathDist.scale": float(follow_path_cfg.get("PathDist.scale", 24.0)),
            "GoalDist.scale": float(follow_path_cfg.get("GoalDist.scale", 20.0)),
            "RotateToGoal.scale": float(follow_path_cfg.get("RotateToGoal.scale", 18.0)),
            "RotateToGoal.slowing_factor": float(follow_path_cfg.get("RotateToGoal.slowing_factor", 4.0)),
            "RotateToGoal.lookahead_time": float(follow_path_cfg.get("RotateToGoal.lookahead_time", -1.0)),
        }
    elif follow_path_plugin == "nav2_mppi_controller::MPPIController":
        follow_path_params = _build_mppi_follow_path_params(
            follow_path_cfg,
            follow_path_plugin,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            max_accel=max_accel,
            max_decel=max_decel,
        )
    elif follow_path_plugin == rotation_shim_plugin:
        primary_controller = str(
            follow_path_cfg.get("primary_controller", "nav2_mppi_controller::MPPIController")
        )
        follow_path_params = _build_mppi_follow_path_params(
            follow_path_cfg,
            primary_controller,
            max_velocity=max_velocity,
            min_velocity=min_velocity,
            max_accel=max_accel,
            max_decel=max_decel,
        )
        follow_path_params.update(
            {
                "plugin": follow_path_plugin,
                "primary_controller": primary_controller,
                "angular_dist_threshold": float(follow_path_cfg.get("angular_dist_threshold", 0.35)),
                "forward_sampling_distance": float(follow_path_cfg.get("forward_sampling_distance", 0.5)),
                "rotate_to_heading_angular_vel": float(
                    follow_path_cfg.get("rotate_to_heading_angular_vel", 0.3)
                ),
                "max_angular_accel": float(follow_path_cfg.get("max_angular_accel", max_accel[2])),
                "simulate_ahead_time": float(follow_path_cfg.get("simulate_ahead_time", 1.0)),
                "rotate_to_goal_heading": bool(follow_path_cfg.get("rotate_to_goal_heading", True)),
            }
        )
    else:
        follow_path_params = {"plugin": follow_path_plugin}
    hard_limit_override_keys = {
        "vx_max",
        "vx_min",
        "vy_max",
        "vy_min",
        "wz_max",
        "ax_max",
        "ax_min",
        "ay_max",
        "ay_min",
        "az_max",
    }
    rotation_shim_only_keys = {
        "primary_controller",
        "angular_dist_threshold",
        "forward_sampling_distance",
        "rotate_to_heading_angular_vel",
        "max_angular_accel",
        "simulate_ahead_time",
        "rotate_to_goal_heading",
    }
    internal_follow_path_keys = {"plugin", "rotate_to_heading_enabled"}
    if follow_path_plugin != rotation_shim_plugin:
        internal_follow_path_keys.update(rotation_shim_only_keys)
    for key, value in follow_path_cfg.items():
        if key not in internal_follow_path_keys and key not in hard_limit_override_keys:
            follow_path_params[key] = value

    local_costmap_frame = str(
        local_costmap_cfg.get(
            "global_frame",
            odom_frame if bool(local_costmap_cfg.get("rolling_window", True)) else map_frame,
        )
    )
    local_costmap_plugins = list(local_costmap_cfg.get("plugins", ["static_layer", "inflation_layer"]))
    global_costmap_frame = str(global_costmap_cfg.get("global_frame", map_frame))
    global_costmap_plugins = list(global_costmap_cfg.get("plugins", ["static_layer", "inflation_layer"]))

    planner_plugin = str(planner_cfg.get("plugin", "nav2_smac_planner/SmacPlannerLattice"))
    planner_params = {
        "plugin": planner_plugin,
        "tolerance": float(planner_cfg.get("tolerance", 0.10)),
        "allow_unknown": bool(planner_cfg.get("allow_unknown", False)),
    }
    planner_passthrough_keys = {"tolerance", "allow_unknown"}
    if planner_plugin == "nav2_navfn_planner/NavfnPlanner":
        planner_params["use_astar"] = bool(planner_cfg.get("use_astar", True))
        planner_passthrough_keys.update({"use_astar"})
    elif planner_plugin == "nav2_smac_planner/SmacPlanner2D":
        planner_params.update(
            {
                "downsample_costmap": bool(planner_cfg.get("downsample_costmap", False)),
                "downsampling_factor": int(planner_cfg.get("downsampling_factor", 1)),
                "max_iterations": int(planner_cfg.get("max_iterations", 1000000)),
                "max_on_approach_iterations": int(planner_cfg.get("max_on_approach_iterations", 1000)),
                "max_planning_time": float(planner_cfg.get("max_planning_time", 2.0)),
                "cost_travel_multiplier": float(planner_cfg.get("cost_travel_multiplier", 2.0)),
            }
        )
        planner_passthrough_keys.update(
            {
                "downsample_costmap",
                "downsampling_factor",
                "max_iterations",
                "max_on_approach_iterations",
                "max_planning_time",
                "cost_travel_multiplier",
            }
        )
    elif planner_plugin == "nav2_smac_planner/SmacPlannerLattice":
        planner_params.update(
            {
                "downsample_costmap": bool(planner_cfg.get("downsample_costmap", False)),
                "downsampling_factor": int(planner_cfg.get("downsampling_factor", 1)),
                "max_iterations": int(planner_cfg.get("max_iterations", 1000000)),
                "max_on_approach_iterations": int(planner_cfg.get("max_on_approach_iterations", 2000)),
                "max_planning_time": float(planner_cfg.get("max_planning_time", 3.0)),
                "smooth_path": bool(planner_cfg.get("smooth_path", True)),
                "minimum_turning_radius": float(planner_cfg.get("minimum_turning_radius", minimum_turning_radius)),
                "reverse_penalty": float(planner_cfg.get("reverse_penalty", 1.0)),
                "change_penalty": float(planner_cfg.get("change_penalty", 0.0)),
                "non_straight_penalty": float(planner_cfg.get("non_straight_penalty", 1.05)),
                "cost_penalty": float(planner_cfg.get("cost_penalty", 2.0)),
                "rotation_penalty": float(planner_cfg.get("rotation_penalty", 3.0)),
                "retrospective_penalty": float(planner_cfg.get("retrospective_penalty", 0.015)),
                "analytic_expansion_ratio": float(planner_cfg.get("analytic_expansion_ratio", 3.5)),
                "analytic_expansion_max_length": float(planner_cfg.get("analytic_expansion_max_length", 2.5)),
                "analytic_expansion_max_cost": float(planner_cfg.get("analytic_expansion_max_cost", 200.0)),
                "analytic_expansion_max_cost_override": bool(
                    planner_cfg.get("analytic_expansion_max_cost_override", False)
                ),
                "cache_obstacle_heuristic": bool(planner_cfg.get("cache_obstacle_heuristic", True)),
                "allow_reverse_expansion": bool(planner_cfg.get("allow_reverse_expansion", False)),
                "lattice_filepath": str(
                    planner_cfg.get(
                        "lattice_filepath",
                        "/opt/ros/humble/share/nav2_smac_planner/sample_primitives/5cm_resolution/0.5m_turning_radius/omni/output.json",
                    )
                ),
                "smoother": dict(
                    planner_cfg.get(
                        "smoother",
                        {
                            "tolerance": 1.0e-10,
                            "max_iterations": 1000,
                            "w_data": 0.2,
                            "w_smooth": 0.3,
                            "do_refinement": True,
                        },
                    )
                ),
            }
        )
        planner_passthrough_keys.update(
            {
                "downsample_costmap",
                "downsampling_factor",
                "max_iterations",
                "max_on_approach_iterations",
                "max_planning_time",
                "smooth_path",
                "minimum_turning_radius",
                "reverse_penalty",
                "change_penalty",
                "non_straight_penalty",
                "cost_penalty",
                "rotation_penalty",
                "retrospective_penalty",
                "analytic_expansion_ratio",
                "analytic_expansion_max_length",
                "analytic_expansion_max_cost",
                "analytic_expansion_max_cost_override",
                "cache_obstacle_heuristic",
                "allow_reverse_expansion",
                "lattice_filepath",
                "smoother",
            }
        )
    for key, value in planner_cfg.items():
        if key == "plugin":
            continue
        if planner_plugin in {
            "nav2_navfn_planner/NavfnPlanner",
            "nav2_smac_planner/SmacPlanner2D",
            "nav2_smac_planner/SmacPlannerLattice",
        }:
            if key in planner_passthrough_keys:
                planner_params[key] = value
        else:
            planner_params[key] = value

    return {
        "bt_navigator": {
            "ros__parameters": {
                "use_sim_time": True,
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "bt_loop_duration": int(bt_cfg.get("bt_loop_duration", 10)),
                "default_server_timeout": int(bt_cfg.get("default_server_timeout", 20)),
                "wait_for_service_timeout": int(bt_cfg.get("wait_for_service_timeout", 1000)),
                "default_bt_xml_filename": nav_to_pose_bt,
                "default_nav_to_pose_bt_xml": nav_to_pose_bt,
                "default_nav_through_poses_bt_xml": nav_through_poses_bt,
                "plugin_lib_names": bt_plugins,
            }
        },
        "bt_navigator_navigate_to_pose_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "controller_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "controller_frequency": float(controller_cfg.get("controller_frequency", 20.0)),
                "min_x_velocity_threshold": float(controller_cfg.get("min_x_velocity_threshold", 0.001)),
                "min_y_velocity_threshold": float(controller_cfg.get("min_y_velocity_threshold", 0.001)),
                "min_theta_velocity_threshold": float(controller_cfg.get("min_theta_velocity_threshold", 0.001)),
                "failure_tolerance": float(controller_cfg.get("failure_tolerance", 1.20)),
                "progress_checker_plugin": "progress_checker",
                "goal_checker_plugins": ["general_goal_checker"],
                "controller_plugins": ["FollowPath"],
                "progress_checker": {
                    "plugin": str(progress_checker_cfg.get("plugin", "nav2_controller::SimpleProgressChecker")),
                    "required_movement_radius": float(progress_checker_cfg.get("required_movement_radius", 0.05)),
                    "movement_time_allowance": float(progress_checker_cfg.get("movement_time_allowance", 90.0)),
                },
                "general_goal_checker": {
                    "stateful": bool(goal_checker_cfg.get("stateful", False)),
                    "plugin": str(goal_checker_cfg.get("plugin", "nav2_controller::SimpleGoalChecker")),
                    "xy_goal_tolerance": float(goal_checker_cfg.get("xy_goal_tolerance", position_tolerance_m)),
                    "yaw_goal_tolerance": float(goal_checker_cfg.get("yaw_goal_tolerance", yaw_tolerance_rad)),
                },
                "FollowPath": follow_path_params,
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": float(local_costmap_cfg.get("update_frequency", 10.0)),
                    "publish_frequency": float(local_costmap_cfg.get("publish_frequency", 4.0)),
                    "global_frame": local_costmap_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": bool(local_costmap_cfg.get("rolling_window", True)),
                    "width": int(local_costmap_cfg.get("width", 6)),
                    "height": int(local_costmap_cfg.get("height", 6)),
                    "resolution": float(local_costmap_cfg.get("resolution", 0.05)),
                    "footprint": footprint,
                    "footprint_padding": float(local_costmap_cfg.get("footprint_padding", 0.0)),
                    "plugins": local_costmap_plugins,
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": float(local_costmap_cfg.get("cost_scaling_factor", 3.0)),
                        "inflation_radius": inflation_radius,
                    },
                    "always_send_full_costmap": bool(local_costmap_cfg.get("always_send_full_costmap", True)),
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "use_sim_time": True,
                    "update_frequency": float(global_costmap_cfg.get("update_frequency", 4.0)),
                    "publish_frequency": float(global_costmap_cfg.get("publish_frequency", 2.0)),
                    "global_frame": global_costmap_frame,
                    "robot_base_frame": base_frame,
                    "rolling_window": bool(global_costmap_cfg.get("rolling_window", False)),
                    "resolution": float(global_costmap_cfg.get("resolution", 0.05)),
                    "track_unknown_space": bool(global_costmap_cfg.get("track_unknown_space", False)),
                    "footprint": footprint,
                    "footprint_padding": float(global_costmap_cfg.get("footprint_padding", 0.0)),
                    "plugins": global_costmap_plugins,
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "cost_scaling_factor": float(global_costmap_cfg.get("cost_scaling_factor", 3.0)),
                        "inflation_radius": inflation_radius,
                    },
                    "always_send_full_costmap": bool(global_costmap_cfg.get("always_send_full_costmap", True)),
                }
            }
        },
        "planner_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "expected_planner_frequency": float(planner_cfg.get("expected_planner_frequency", 10.0)),
                "planner_plugins": ["GridBased"],
                "GridBased": planner_params,
            }
        },
        "smoother_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoother_plugins": ["simple_smoother"],
                "simple_smoother": {
                    "plugin": str(smoother_cfg.get("plugin", "nav2_smoother::SimpleSmoother")),
                    "tolerance": float(smoother_cfg.get("tolerance", 1.0e-10)),
                    "max_its": int(smoother_cfg.get("max_its", 1000)),
                    "do_refinement": bool(smoother_cfg.get("do_refinement", True)),
                },
            }
        },
        "behavior_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "costmap_topic": "local_costmap/costmap_raw",
                "footprint_topic": "local_costmap/published_footprint",
                "cycle_frequency": float(behavior_cfg.get("cycle_frequency", 10.0)),
                "behavior_plugins": list(behavior_cfg.get("behavior_plugins", ["wait"])),
                "spin": dict(behavior_cfg.get("spin", {"plugin": "nav2_behaviors/Spin"})),
                "backup": dict(behavior_cfg.get("backup", {"plugin": "nav2_behaviors/BackUp"})),
                "drive_on_heading": dict(
                    behavior_cfg.get("drive_on_heading", {"plugin": "nav2_behaviors/DriveOnHeading"})
                ),
                "wait": dict(behavior_cfg.get("wait", {"plugin": "nav2_behaviors/Wait"})),
                "global_frame": map_frame,
                "robot_base_frame": base_frame,
                "transform_tolerance": float(behavior_cfg.get("transform_tolerance", 0.2)),
                "simulate_ahead_time": float(behavior_cfg.get("simulate_ahead_time", 2.0)),
                "max_rotational_vel": float(behavior_cfg.get("max_rotational_vel", 0.35)),
                "min_rotational_vel": float(behavior_cfg.get("min_rotational_vel", 0.1)),
                "rotational_acc_lim": float(behavior_cfg.get("rotational_acc_lim", 1.0)),
            }
        },
        "waypoint_follower": {
            "ros__parameters": {
                "use_sim_time": True,
                "loop_rate": int(waypoint_cfg.get("loop_rate", 20)),
                "stop_on_failure": bool(waypoint_cfg.get("stop_on_failure", False)),
                "waypoint_task_executor_plugin": str(waypoint_cfg.get("waypoint_task_executor_plugin", "wait_at_waypoint")),
                "wait_at_waypoint": dict(
                    waypoint_cfg.get(
                        "wait_at_waypoint",
                        {
                            "plugin": "nav2_waypoint_follower::WaitAtWaypoint",
                            "enabled": True,
                            "waypoint_pause_duration": 0,
                        },
                    )
                ),
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "use_sim_time": True,
                "smoothing_frequency": float(velocity_smoother_cfg.get("smoothing_frequency", 20.0)),
                "scale_velocities": bool(velocity_smoother_cfg.get("scale_velocities", False)),
                "feedback": str(velocity_smoother_cfg.get("feedback", "OPEN_LOOP")),
                "max_velocity": list(max_velocity),
                "min_velocity": list(min_velocity),
                "max_accel": list(max_accel),
                "max_decel": list(max_decel),
                "odom_topic": str(ros_cfg.get("odom_topic", "/odom")),
                "odom_duration": float(velocity_smoother_cfg.get("odom_duration", 0.1)),
                "deadband_velocity": list(velocity_smoother_cfg.get("deadband_velocity", [0.0, 0.0, 0.0])),
                "velocity_timeout": float(velocity_smoother_cfg.get("velocity_timeout", 1.0)),
            }
        },
        "map_server": {
            "ros__parameters": {
                "use_sim_time": True,
                "yaml_filename": str(localization_cfg.get("map_yaml_path", "")),
            }
        },
        "global_costmap_client": {"ros__parameters": {"use_sim_time": True}},
        "local_costmap_client": {"ros__parameters": {"use_sim_time": True}},
        "planner_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "controller_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "behavior_server_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "waypoint_follower_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
        "amcl": {"ros__parameters": {"enabled": False}},
        "amcl_rclcpp_node": {"ros__parameters": {"use_sim_time": True}},
    }
