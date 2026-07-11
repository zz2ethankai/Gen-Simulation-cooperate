"""Nav2 runtime configuration normalization and artifact generation."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from string import Template

import yaml

from workflows.simbox.core.mobile.platforms import get_mobile_base_platform

from .utils import footprint_inscribed_radius


NAV2_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
NAV2_PARAMS_CONFIG_PATH = NAV2_CONFIG_DIR / "nav2_params.yaml"
NAV2_DEFAULT_CONFIG_PATH = NAV2_CONFIG_DIR / "default_nav.yaml"
NAV2_BT_CONFIG_DIR = NAV2_CONFIG_DIR / "behavior_trees"


def _load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


_NAV2_RUNTIME_DEFAULTS = _load_yaml_config(NAV2_DEFAULT_CONFIG_PATH)["nav2_runtime_defaults"]
NAV2_DEFAULT_POSITION_TOLERANCE_M = float(_NAV2_RUNTIME_DEFAULTS["position_tolerance_m"])
NAV2_DEFAULT_YAW_TOLERANCE_RAD = float(_NAV2_RUNTIME_DEFAULTS["yaw_tolerance_rad"])


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
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            normalized.append([float(point[0]), float(point[1])])
    return normalized if len(normalized) >= 3 else []


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
    virtual_odom_cfg.update({"enabled": False, "publish_twist": True, "use_world_z": True})
    virtual_odom_cfg["default_z"] = float(virtual_odom_cfg.get("default_z", 0.0))

    localization_cfg = ros_cfg.setdefault("localization", {})
    localization_cfg.update(
        {
            "enabled": True,
            "mode": "static_map_truth_pose",
            "map_resolution": float(map_resolution),
            "map_output_dir": str(map_output_dir),
            "map_include_visual_wall_geometry": bool(map_include_visual_wall_geometry),
        }
    )
    localization_cfg["map_z_min"] = float(localization_cfg.get("map_z_min", map_z_min))
    localization_cfg["map_z_max"] = float(localization_cfg.get("map_z_max", map_z_max))
    localization_cfg.setdefault("clear_footprint_points", platform.default_nav2_footprint_points(base_cfg))
    localization_cfg.setdefault("robot_clear_radius_m", 0.0)

    map_frame = str(localization_cfg.get("map_frame", "map"))
    odom_frame = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    base_frame = str(localization_cfg.get("base_frame", ros_cfg.get("base_frame", "base_link")))
    localization_cfg.update({"map_frame": map_frame, "odom_frame": odom_frame, "base_frame": base_frame})

    nav2_cfg = ros_cfg.setdefault("nav2", {})
    nav2_cfg.update(
        {
            "enabled": True,
            "global_frame": map_frame,
            "robot_base_frame": base_frame,
            "skill_managed": True,
            "runtime_mode": "external_compose",
        }
    )

    skill_cfg = base_cfg.setdefault("nav2_skill", {})
    if isinstance(nav2_skill_overrides, dict):
        _deep_update_dict(skill_cfg, nav2_skill_overrides)
    skill_cfg["position_tolerance_m"] = float(position_tolerance_m)
    skill_cfg["yaw_tolerance_rad"] = float(yaw_tolerance_rad)
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
    params_filename: str = "nav2_skill_params.yaml",
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


def _behavior_tree_cfg(base_cfg: dict) -> dict:
    skill_cfg = nav2_skill_cfg(base_cfg)
    config = deepcopy(skill_cfg.get("bt_navigator", {}))
    _deep_update_dict(config, skill_cfg.get("behavior_tree", {}))
    return config


def _write_nav2_bt_files(output_dir: str, base_cfg: dict) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    bt_cfg = _behavior_tree_cfg(base_cfg)
    nav2_cfg = base_cfg["ros"]["nav2"]
    substitutions = {
        **bt_cfg,
        "clear_global_costmap_service": nav2_cfg["clear_global_costmap_service"],
        "clear_local_costmap_service": nav2_cfg["clear_local_costmap_service"],
    }

    output_paths = []
    for filename in (
        "navigate_to_pose_w_replanning_no_motion_recovery.xml",
        "navigate_through_poses_w_replanning_no_motion_recovery.xml",
    ):
        template_path = NAV2_BT_CONFIG_DIR / filename
        rendered = Template(template_path.read_text(encoding="utf-8")).substitute(substitutions)
        output_path = Path(output_dir) / filename
        output_path.write_text(rendered, encoding="utf-8")
        output_paths.append(str(output_path))
    return output_paths[0], output_paths[1]


def _select_profile(profiles: dict, default_plugin: str, overrides: dict) -> dict:
    plugin = str(overrides.get("plugin", default_plugin))
    profile = deepcopy(profiles.get(plugin, {"plugin": plugin}))
    _deep_update_dict(profile, overrides)
    profile["plugin"] = plugin
    return profile


def _controller_params(template_cfg: dict, skill_cfg: dict, hard_limits: dict) -> tuple[dict, dict]:
    controller_cfg = deepcopy(skill_cfg.get("controller_server", {}))
    follow_path_cfg = controller_cfg.pop("follow_path", {})
    progress_checker_cfg = controller_cfg.pop("progress_checker", {})
    goal_checker_cfg = controller_cfg.pop("goal_checker", {})

    profiles = template_cfg["controller_profiles"]
    default_plugin = template_cfg["defaults"]["controller_plugin"]
    rotation_shim_plugin = "nav2_rotation_shim_controller::RotationShimController"
    rotate_to_heading = bool(follow_path_cfg.pop("rotate_to_heading_enabled", True))
    if follow_path_cfg.get("plugin", default_plugin) == rotation_shim_plugin and not rotate_to_heading:
        follow_path_cfg["plugin"] = follow_path_cfg.pop(
            "primary_controller",
            profiles[rotation_shim_plugin]["primary_controller"],
        )
        for key in (
            "angular_dist_threshold",
            "forward_sampling_distance",
            "rotate_to_heading_angular_vel",
            "max_angular_accel",
            "simulate_ahead_time",
            "rotate_to_goal_heading",
        ):
            follow_path_cfg.pop(key, None)

    follow_path = _select_profile(profiles, default_plugin, follow_path_cfg)
    dynamic_limits = {
        "vx_max": hard_limits["max_velocity"][0],
        "vx_min": hard_limits["min_velocity"][0],
        "vy_max": hard_limits["max_velocity"][1],
        "vy_min": hard_limits["min_velocity"][1],
        "wz_max": hard_limits["max_velocity"][2],
        "ax_max": hard_limits["max_accel"][0],
        "ax_min": hard_limits["max_decel"][0],
        "ay_max": hard_limits["max_accel"][1],
        "ay_min": hard_limits["max_decel"][1],
        "az_max": hard_limits["max_accel"][2],
    }
    for key, value in dynamic_limits.items():
        if key in follow_path and follow_path[key] is None:
            follow_path[key] = float(value)
    if follow_path.get("max_angular_accel", False) is None:
        follow_path["max_angular_accel"] = float(hard_limits["max_accel"][2])
    return controller_cfg, {
        "FollowPath": follow_path,
        "progress_checker": progress_checker_cfg,
        "general_goal_checker": goal_checker_cfg,
    }


def _planner_params(template_cfg: dict, skill_cfg: dict, minimum_turning_radius: float) -> tuple[dict, dict]:
    planner_cfg = deepcopy(skill_cfg.get("planner_server", {}))
    server_cfg = {}
    if "expected_planner_frequency" in planner_cfg:
        server_cfg["expected_planner_frequency"] = planner_cfg.pop("expected_planner_frequency")
    planner = _select_profile(
        template_cfg["planner_profiles"],
        template_cfg["defaults"]["planner_plugin"],
        planner_cfg,
    )
    if planner.get("minimum_turning_radius", False) is None:
        planner["minimum_turning_radius"] = float(minimum_turning_radius)
    return server_cfg, planner


def _merge_costmap_config(target: dict, overrides: dict) -> None:
    overrides = deepcopy(overrides)
    if "cost_scaling_factor" in overrides:
        overrides.setdefault("inflation_layer", {})["cost_scaling_factor"] = overrides.pop(
            "cost_scaling_factor"
        )
    _deep_update_dict(target, overrides)


def _build_nav2_params(
    nav_to_pose_bt: str,
    nav_through_poses_bt: str,
    base_cfg: dict,
    *,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
):
    template_cfg = _load_yaml_config(NAV2_PARAMS_CONFIG_PATH)
    params = deepcopy(template_cfg["params"])
    skill_cfg = nav2_skill_cfg(base_cfg)
    ros_cfg = base_cfg["ros"]
    nav2_cfg = ros_cfg["nav2"]
    localization_cfg = ros_cfg["localization"]
    platform = get_mobile_base_platform(base_cfg)
    hard_limits = platform.nav2_controller_hard_limits(base_cfg)

    controller_overrides, nested_controller_overrides = _controller_params(
        template_cfg,
        skill_cfg,
        hard_limits,
    )
    controller_params = params["controller_server"]["ros__parameters"]
    _deep_update_dict(controller_params, controller_overrides)
    controller_params["general_goal_checker"].update(
        {
            "xy_goal_tolerance": float(position_tolerance_m),
            "yaw_goal_tolerance": float(yaw_tolerance_rad),
        }
    )
    for key, overrides in nested_controller_overrides.items():
        if key == "FollowPath":
            controller_params[key] = overrides
        else:
            _deep_update_dict(controller_params[key], overrides)

    planner_server_overrides, planner = _planner_params(
        template_cfg,
        skill_cfg,
        platform.default_nav2_minimum_turning_radius_m(base_cfg),
    )
    planner_params = params["planner_server"]["ros__parameters"]
    _deep_update_dict(planner_params, planner_server_overrides)
    planner_params["GridBased"] = planner

    local_costmap = params["local_costmap"]["local_costmap"]["ros__parameters"]
    global_costmap = params["global_costmap"]["global_costmap"]["ros__parameters"]
    local_costmap_cfg = skill_cfg.get("local_costmap", {})
    global_costmap_cfg = skill_cfg.get("global_costmap", {})
    _merge_costmap_config(local_costmap, local_costmap_cfg)
    _merge_costmap_config(global_costmap, global_costmap_cfg)

    _deep_update_dict(
        params["smoother_server"]["ros__parameters"]["simple_smoother"],
        skill_cfg.get("smoother_server", {}),
    )
    for section in ("behavior_server", "waypoint_follower", "velocity_smoother"):
        _deep_update_dict(params[section]["ros__parameters"], skill_cfg.get(section, {}))

    bt_params = params["bt_navigator"]["ros__parameters"]
    for key, value in skill_cfg.get("bt_navigator", {}).items():
        if key in bt_params:
            bt_params[key] = deepcopy(value)

    map_frame = str(localization_cfg.get("map_frame", nav2_cfg["global_frame"]))
    odom_frame = str(localization_cfg.get("odom_frame", ros_cfg.get("odom_frame", "odom")))
    base_frame = str(localization_cfg.get("base_frame", nav2_cfg["robot_base_frame"]))
    odom_topic = str(ros_cfg.get("odom_topic", "/odom"))
    bt_params.update(
        {
            "global_frame": map_frame,
            "robot_base_frame": base_frame,
            "odom_topic": odom_topic,
            "default_bt_xml_filename": nav_to_pose_bt,
            "default_nav_to_pose_bt_xml": nav_to_pose_bt,
            "default_nav_through_poses_bt_xml": nav_through_poses_bt,
        }
    )

    footprint_points = _normalize_footprint_points(skill_cfg.get("footprint_points"))
    if not footprint_points:
        footprint_points = platform.default_nav2_footprint_points(base_cfg)
    footprint = format_nav2_footprint(footprint_points)
    inflation_radius = max(
        float(skill_cfg.get("inflation_radius_m", platform.default_nav2_inflation_radius_m(base_cfg))),
        footprint_inscribed_radius(footprint_points),
    )
    local_costmap.update(
        {
            "global_frame": str(
                local_costmap_cfg.get(
                    "global_frame",
                    odom_frame if local_costmap["rolling_window"] else map_frame,
                )
            ),
            "robot_base_frame": base_frame,
            "footprint": footprint,
        }
    )
    global_costmap.update(
        {
            "global_frame": str(global_costmap_cfg.get("global_frame", map_frame)),
            "robot_base_frame": base_frame,
            "footprint": footprint,
        }
    )
    local_costmap["inflation_layer"]["inflation_radius"] = inflation_radius
    global_costmap["inflation_layer"]["inflation_radius"] = inflation_radius

    behavior_params = params["behavior_server"]["ros__parameters"]
    behavior_params.update({"global_frame": map_frame, "robot_base_frame": base_frame})
    smoother_params = params["velocity_smoother"]["ros__parameters"]
    smoother_params.update(
        {
            "max_velocity": list(hard_limits["max_velocity"]),
            "min_velocity": list(hard_limits["min_velocity"]),
            "max_accel": list(hard_limits["max_accel"]),
            "max_decel": list(hard_limits["max_decel"]),
            "odom_topic": odom_topic,
        }
    )
    return params
