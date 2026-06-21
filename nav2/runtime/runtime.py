"""Workflow-side split Nav2 session manager."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
import math
import os
import shutil
from typing import Optional
import uuid

from .config import (
    NAV2_DEFAULT_POSITION_TOLERANCE_M,
    NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    generate_nav2_bringup_artifacts,
)
from .debug import Nav2SkillResult, TaskShim, runtime_control_debug_snapshot
from .dynamic_goal import (
    ApproachConfig,
    check_footprint_static_collision,
    load_static_map,
    resolve_approach_footprint_padding_m,
    resolve_nav2_footprint_points,
    sample_approach_candidates,
    sort_candidates_for_preflight,
    write_candidates_debug,
)
from .utils import angle_diff_rad, safe_name, time_monotonic, yaw_from_wxyz

LOGGER = logging.getLogger("simbox.nav2_skill")


class PersistentNav2RuntimeManager:
    """Workflow-side manager that drives Nav2 through a standard-message ROS bridge."""

    STATE_IDLE = "idle"
    STATE_WAITING_FOR_STACK_READY = "waiting_for_stack_ready"
    STATE_WAITING_FOR_MAP_READY = "waiting_for_map_ready"
    STATE_WAITING_FOR_DYNAMIC_PLAN = "waiting_for_dynamic_plan"
    STATE_WAITING_FOR_GOAL_ACCEPTED = "waiting_for_goal_accepted"
    STATE_RUNNING = "running"
    STATE_POST_SUCCESS_SETTLING = "post_success_settling"
    STATE_SUCCEEDED = "succeeded"
    STATE_FAILED = "failed"

    def __init__(
        self,
        *,
        world,
        task,
        robot,
        output_root: str = "output/ros_bridge/skills",
        scene_name: str = "nav2_skill_scene",
    ):
        self.world = world
        self.task = task
        self.robot = robot
        self.output_root = str(output_root)
        self.scene_name = str(scene_name)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.approach_config: Optional[ApproachConfig] = None
        self._approach_target_pose: dict = {}
        self._dynamic_goal_selected: dict = {}
        self._dynamic_goal_candidates: list[dict] = []
        self._dynamic_goal_plan_index = 0
        self._dynamic_goal_active_plan_request_id = ""
        self._dynamic_goal_plan_deadline = None
        self._dynamic_goal_debug_path = ""
        self.nav2_position_tolerance_m = NAV2_DEFAULT_POSITION_TOLERANCE_M
        self.nav2_yaw_tolerance_rad = NAV2_DEFAULT_YAW_TOLERANCE_RAD
        self.position_tolerance_m = NAV2_DEFAULT_POSITION_TOLERANCE_M
        self.yaw_tolerance_rad = NAV2_DEFAULT_YAW_TOLERANCE_RAD
        self.startup_timeout_sec = 60.0
        self.runtime_timeout_sec = 240.0

        self.state = self.STATE_IDLE
        self.result = Nav2SkillResult()
        self._base_cfg = deepcopy(getattr(robot, "base_cfg", {}))
        self._map_info = None
        self._params_path = ""
        self._stack_output_dir = ""
        self._goal_output_dir = ""
        self._stack_id = ""
        self._request_id = ""
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._bridge_client = None
        self._bridge_robot_name = ""
        self._goal_output_tag = ""
        self._goal_debug_map_info = None
        self._goal_params_path = ""
        self._restore_after_nav_started = False
        self._post_success_started_at = None
        self._post_success_settle_started_at = None
        self._local_goal_reached_started_at = None
        self._post_success_trigger = ""
        self._cleaned_up = False
        self._session_uuid = uuid.uuid4().hex

    @property
    def done(self) -> bool:
        return bool(self.result.done)

    @property
    def success(self) -> bool:
        return bool(self.result.success)

    def bind(self, *, world, task, robot, scene_name: Optional[str] = None):
        self.world = world
        self.task = task
        self.robot = robot
        self._base_cfg = deepcopy(getattr(robot, "base_cfg", {}))
        if scene_name is not None:
            self.scene_name = str(scene_name)
        current_robot_name = safe_name(getattr(self.robot, "name", "robot"))
        if self._bridge_client is not None and current_robot_name != self._bridge_robot_name:
            try:
                self._bridge_client.destroy()
            finally:
                self._bridge_client = None
                self._bridge_robot_name = ""

    def _ensure_bridge_client(self):
        from nav2.bridge.client import Nav2BridgeClient

        robot_name = safe_name(getattr(self.robot, "name", "robot"))
        if self._bridge_client is not None and robot_name == self._bridge_robot_name:
            return self._bridge_client

        if self._bridge_client is not None:
            try:
                self._bridge_client.destroy()
            finally:
                self._bridge_client = None

        self._bridge_client = Nav2BridgeClient(
            self.robot,
            self._base_cfg,
            node_name=f"nav2_bridge_client_{robot_name}",
        )
        self._bridge_robot_name = robot_name
        return self._bridge_client

    def _stack_signature(self, params: dict) -> str:
        payload = {
            "scene_name": str(self.scene_name),
            "params": params,
            "map_info": self._map_info,
            "nav2_cfg": dict(self._base_cfg.get("ros", {}).get("nav2", {})),
            "nav2_skill_cfg": dict(self._base_cfg.get("nav2_skill", {})),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    def _session_scoped_path(self, path_value: str) -> str:
        path = str(path_value or "").strip()
        if not path:
            return path
        normalized = os.path.normpath(path)
        if os.path.basename(normalized) == self._session_uuid:
            return normalized
        return os.path.join(normalized, self._session_uuid)

    def _apply_session_output_namespace(self):
        ros_cfg = self._base_cfg.setdefault("ros", {})
        localization_cfg = ros_cfg.setdefault("localization", {})
        localization_cfg["map_output_dir"] = self._session_scoped_path(
            str(localization_cfg.get("map_output_dir", "output/nav2_maps"))
        )

        nav2_cfg = ros_cfg.setdefault("nav2", {})
        for key, default in (
            ("stack_request_root", "output/ros_bridge/runtime_requests"),
            ("stack_status_root", "output/ros_bridge/runtime_status"),
            ("goal_request_root", "output/ros_bridge/goal_requests"),
            ("goal_status_root", "output/ros_bridge/goal_status"),
            ("goal_result_root", "output/ros_bridge/goal_result"),
        ):
            nav2_cfg[key] = self._session_scoped_path(str(nav2_cfg.get(key, default)))

    def _prepare_stack_artifacts(self):
        from nav2.mapgen.exporter import IsaacStaticMapExporter

        self._apply_session_output_namespace()
        robot_tag = safe_name(getattr(self.robot, "name", "robot"))
        scene_tag = safe_name(self.scene_name)
        stack_tag = f"{robot_tag}_{scene_tag}"
        self._stack_output_dir = os.path.join(self.output_root, self._session_uuid, stack_tag, "stack")
        os.makedirs(self._stack_output_dir, exist_ok=True)

        exporter = IsaacStaticMapExporter(
            workflow=TaskShim(self.task),
            robot=self.robot,
            base_cfg=self._base_cfg,
            scene_name=self.scene_name,
        )
        translation, orientation = self.robot.get_mobile_base_pose()
        self._map_info = exporter.export_map(
            output_dir=self._base_cfg["ros"]["localization"]["map_output_dir"],
            clear_center_xy=(float(translation[0]), float(translation[1]), float(yaw_from_wxyz(orientation))),
        )
        self._base_cfg.setdefault("ros", {}).setdefault("localization", {})["map_yaml_path"] = self._map_info["yaml_path"]
        artifacts = generate_nav2_bringup_artifacts(
            self._stack_output_dir,
            base_cfg=self._base_cfg,
            map_yaml_path=str(self._map_info["yaml_path"]),
            position_tolerance_m=self.nav2_position_tolerance_m,
            yaw_tolerance_rad=self.nav2_yaw_tolerance_rad,
        )
        self._params_path = str(artifacts["params_path"])
        self._stack_id = f"{robot_tag}::{scene_tag}::{self._session_uuid}::{self._stack_signature(artifacts['params'])}"

    def _new_goal_output_dir(self) -> str:
        robot_tag = safe_name(getattr(self.robot, "name", "robot"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_root, f"{robot_tag}_nav2_goal_{stamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _freeze_goal_debug_artifacts(self):
        self._goal_debug_map_info = dict(self._map_info or {})
        self._goal_params_path = str(self._params_path)
        if not self._goal_output_dir:
            return

        debug_dir = os.path.join(self._goal_output_dir, "debug_inputs")
        os.makedirs(debug_dir, exist_ok=True)

        map_info = dict(self._goal_debug_map_info or {})
        yaml_path = str(map_info.get("yaml_path", "")).strip()
        pgm_path = str(map_info.get("pgm_path", "")).strip()
        if yaml_path:
            yaml_src = yaml_path if os.path.isabs(yaml_path) else os.path.abspath(yaml_path)
            yaml_dst = os.path.join(debug_dir, "map.yaml")
            shutil.copy2(yaml_src, yaml_dst)
            map_info["yaml_path"] = yaml_dst
        if pgm_path:
            pgm_src = pgm_path if os.path.isabs(pgm_path) else os.path.abspath(pgm_path)
            pgm_dst = os.path.join(debug_dir, "map.pgm")
            shutil.copy2(pgm_src, pgm_dst)
            map_info["pgm_path"] = pgm_dst
        self._goal_debug_map_info = map_info

        if self._params_path:
            params_src = self._params_path if os.path.isabs(self._params_path) else os.path.abspath(self._params_path)
            params_dst = os.path.join(debug_dir, os.path.basename(params_src) or "nav2_skill_params.yaml")
            shutil.copy2(params_src, params_dst)
            self._goal_params_path = params_dst

    def begin_goal(
        self,
        *,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        approach_config: Optional[ApproachConfig] = None,
        nav2_position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
        nav2_yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
        position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
        yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
        startup_timeout_sec: float = 60.0,
        runtime_timeout_sec: float = 240.0,
    ):
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_yaw = float(goal_yaw)
        self.approach_config = approach_config
        self.nav2_position_tolerance_m = float(nav2_position_tolerance_m)
        self.nav2_yaw_tolerance_rad = float(nav2_yaw_tolerance_rad)
        self.position_tolerance_m = float(position_tolerance_m)
        self.yaw_tolerance_rad = float(yaw_tolerance_rad)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.runtime_timeout_sec = float(runtime_timeout_sec)
        previous_request_id = str(self._request_id)
        self.result = Nav2SkillResult()
        self.state = self.STATE_IDLE
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._goal_output_dir = self._new_goal_output_dir()
        self._goal_output_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._request_id = self._goal_output_tag
        self._goal_debug_map_info = None
        self._goal_params_path = ""
        self._restore_after_nav_started = False
        self._post_success_started_at = None
        self._post_success_settle_started_at = None
        self._local_goal_reached_started_at = None
        self._post_success_trigger = ""
        self._approach_target_pose = {}
        self._dynamic_goal_selected = {}
        self._dynamic_goal_candidates = []
        self._dynamic_goal_plan_index = 0
        self._dynamic_goal_active_plan_request_id = ""
        self._dynamic_goal_plan_deadline = None
        self._dynamic_goal_debug_path = ""

        bridge_client = self._ensure_bridge_client()
        bridge_client.reset_debug_trace()
        if previous_request_id:
            bridge_client.cancel_request(previous_request_id)
        self._reset_robot_bridge_state(clear_debug_history=True)
        self._prepare_robot_bridge_for_navigation()
        self._prepare_stack_artifacts()
        self._freeze_goal_debug_artifacts()
        self.state = self.STATE_WAITING_FOR_STACK_READY
        self._startup_deadline = time_monotonic() + self.startup_timeout_sec

    def prepare_for_reset(self):
        self._finalize_robot_bridge_after_navigation()
        if self._bridge_client is not None:
            try:
                self._bridge_client.publish_reset()
            except Exception:
                LOGGER.exception("failed to reset nav2 bridge during reset")
        self.result = Nav2SkillResult()
        self.state = self.STATE_IDLE
        self._startup_deadline = None
        self._goal_accept_deadline = None
        self._runtime_deadline = None
        self._post_success_deadline = None
        self._request_id = ""
        self._restore_after_nav_started = False
        self._post_success_started_at = None
        self._post_success_settle_started_at = None
        self._local_goal_reached_started_at = None
        self._post_success_trigger = ""
        self._approach_target_pose = {}
        self._dynamic_goal_selected = {}
        self._dynamic_goal_candidates = []
        self._dynamic_goal_plan_index = 0
        self._dynamic_goal_active_plan_request_id = ""
        self._dynamic_goal_plan_deadline = None
        self._dynamic_goal_debug_path = ""

    def step(self):
        if self.done or self.state == self.STATE_IDLE:
            return

        if self._robot_base_state_is_invalid():
            reason = self._robot_base_invalid_reason()
            self._fail(
                "robot_base_state_invalid",
                f"Navigation stopped because the mobile base state became non-finite: {reason}",
            )
            return

        bridge_client = self._ensure_bridge_client()
        bridge_client.step(step_dt=self._step_dt())

        if self._robot_base_state_is_invalid():
            reason = self._robot_base_invalid_reason()
            self._fail(
                "robot_base_state_invalid",
                f"Navigation stopped because the mobile base state became non-finite: {reason}",
            )
            return

        if self.state == self.STATE_WAITING_FOR_STACK_READY:
            if bridge_client.bridge_online and bridge_client.nav_stack_ready:
                bridge_client.publish_map_update(
                    request_id=self._request_id,
                    stack_id=self._stack_id,
                    map_yaml_path=str(self._map_info["yaml_path"]),
                    scene_name=self.scene_name,
                )
                self.state = self.STATE_WAITING_FOR_MAP_READY
                return
            if time_monotonic() >= float(self._startup_deadline):
                self._fail("stack_not_ready", "Timed out waiting for Nav2 bridge heartbeat/action readiness.")
            return

        status = bridge_client.request_status(self._request_id)
        result = bridge_client.request_result(self._request_id)
        bridge_state = self._bridge_state_name(status=status, result=result)

        if self.state == self.STATE_WAITING_FOR_MAP_READY:
            if bridge_state == "ready":
                if self.approach_config is not None:
                    self._initialize_dynamic_goal_candidates()
                    if self.done:
                        return
                    self.state = self.STATE_WAITING_FOR_DYNAMIC_PLAN
                    self._publish_next_dynamic_plan_request(bridge_client)
                else:
                    self._publish_navigation_goal(bridge_client)
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if time_monotonic() >= float(self._startup_deadline):
                self._fail("map_update_timeout", "Timed out waiting for bridge adapter to load the map.")
            return

        if self.state == self.STATE_WAITING_FOR_DYNAMIC_PLAN:
            self._consume_dynamic_plan_result(bridge_client)
            if self.done:
                return
            if self.state == self.STATE_WAITING_FOR_GOAL_ACCEPTED:
                return
            if self._dynamic_goal_plan_deadline is not None and time_monotonic() >= float(self._dynamic_goal_plan_deadline):
                current = self._active_dynamic_candidate()
                if current is not None:
                    current["path_ok"] = False
                    current["path_state"] = "timeout"
                    current["path_detail"] = "timed out waiting for ComputePathToPose plan result"
                    self._write_dynamic_goal_candidates_debug()
                    self._dynamic_goal_plan_index += 1
                    self._dynamic_goal_active_plan_request_id = ""
                    self._dynamic_goal_plan_deadline = None
                    self._publish_next_dynamic_plan_request(bridge_client)
            return

        if self.state == self.STATE_WAITING_FOR_GOAL_ACCEPTED:
            if bridge_state in {"accepted", "running"}:
                self.state = self.STATE_RUNNING
                self._runtime_deadline = time_monotonic() + self.runtime_timeout_sec
                return
            if bridge_state == "succeeded":
                self._enter_post_success_settling()
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if time_monotonic() >= float(self._goal_accept_deadline):
                bridge_client.cancel_request(self._request_id)
                self._fail("goal_not_accepted", "Timed out waiting for bridge adapter to accept the goal.")
            return

        if self.state == self.STATE_RUNNING:
            self._update_pose_result_fields()
            if bridge_state == "succeeded":
                self._enter_post_success_settling(trigger="nav2_result")
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if self._local_goal_reached_hold_elapsed():
                bridge_client.cancel_request(self._request_id)
                self._enter_post_success_settling(trigger="local_goal_reached")
                return
            if time_monotonic() >= float(self._runtime_deadline):
                bridge_client.cancel_request(self._request_id)
                self._fail("runtime_timeout", "Timed out while waiting for the navigation goal to finish.")
            return

        if self.state == self.STATE_POST_SUCCESS_SETTLING:
            self._update_pose_result_fields()
            restore_done = self._robot_bridge_restore_after_navigation_done()
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                if self._post_success_trigger == "local_goal_reached" and bridge_state == "canceled":
                    pass
                else:
                    self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                    return
            now = self._sim_time()
            if restore_done and self._post_success_settle_started_at is None:
                self._post_success_settle_started_at = now
            elif not restore_done:
                self._post_success_settle_started_at = None
            settle_done = (
                self._post_success_settle_started_at is not None
                and now - float(self._post_success_settle_started_at) >= self._post_success_settle_sec()
            )
            if self._goal_within_tolerance() and restore_done and settle_done:
                self.result.done = True
                self.result.success = True
                self.state = self.STATE_SUCCEEDED
                if self._post_success_trigger == "local_goal_reached":
                    success_reason = "local_goal_reached"
                    success_message = (
                        "Local goal tolerance held before Nav2 action result; "
                        "navigation settled successfully."
                    )
                else:
                    success_reason = "goal_succeeded"
                    success_message = "Nav2 goal reached within skill tolerances after post-success settling."
                success_snapshot = self._write_debug_snapshot(
                    "success_snapshot.json",
                    success_reason,
                    success_message,
                )
                self._log_result_summary(
                    level=logging.INFO,
                    reason=success_reason,
                    message=success_message,
                    control_snapshot=success_snapshot.get("control", {}),
                )
                self._finalize_robot_bridge_after_navigation()
                return
            if now >= float(self._post_success_deadline):
                if self._post_success_trigger == "local_goal_reached":
                    success_source = "Local goal tolerance was held"
                else:
                    success_source = "Nav2 reported success"
                if not restore_done:
                    self._fail(
                        "restore_after_navigation_timeout",
                        f"{success_source} but wheel restore did not finish within post-success simulation timeout.",
                    )
                else:
                    self._fail(
                        "goal_tolerance_not_met",
                        f"{success_source} but skill tolerances were not met after post-success settling.",
                    )

    def shutdown(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if self._bridge_client is not None and not self.result.done:
            try:
                self._bridge_client.cancel_request(self._request_id)
            except Exception:
                LOGGER.exception("failed to cancel nav2 goal during shutdown")
            try:
                self._write_debug_snapshot(
                    "shutdown_snapshot.json",
                    "runtime_shutdown",
                    "Nav2 runtime manager was shut down before the goal finished",
                )
            except Exception:
                LOGGER.exception("failed to write nav2 shutdown snapshot")
        self._finalize_robot_bridge_after_navigation()
        if self._bridge_client is not None:
            try:
                self._bridge_client.destroy()
            except Exception:
                LOGGER.exception("failed to destroy nav2 bridge client")
            finally:
                self._bridge_client = None
                self._bridge_robot_name = ""

    def _bridge_state_name(self, *, status: dict, result: dict) -> str:
        status_state = str((status or {}).get("state", "")).strip().lower()
        if status_state:
            return status_state
        return str((result or {}).get("state", "")).strip().lower()

    @staticmethod
    def _bridge_detail(*, status: dict, result: dict) -> str:
        status_detail = str((status or {}).get("detail", "")).strip()
        if status_detail:
            return status_detail
        return str((result or {}).get("detail", "")).strip()

    def _goal_within_tolerance(self) -> bool:
        return (
            self.result.final_distance_to_goal <= self.position_tolerance_m
            and self.result.final_nav_distance_to_goal <= self.position_tolerance_m
            and self.result.final_yaw_error_rad <= self.yaw_tolerance_rad
        )

    def _local_goal_reached_hold_elapsed(self) -> bool:
        now = self._sim_time()
        if not self._goal_within_tolerance():
            self._local_goal_reached_started_at = None
            return False
        if self._local_goal_reached_started_at is None:
            self._local_goal_reached_started_at = now
        return now - float(self._local_goal_reached_started_at) >= self._local_goal_reached_hold_sec()

    def _step_dt(self) -> float:
        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        if callable(get_physics_dt):
            return float(get_physics_dt())
        return float(getattr(self.world, "physics_dt", 1.0 / 60.0))

    def _sim_time(self) -> float:
        value = getattr(self.world, "current_time", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return value
        return time_monotonic()

    def _post_success_timeout_sec(self) -> float:
        nav2_skill_cfg = self._base_cfg.get("nav2_skill", {}) if isinstance(self._base_cfg, dict) else {}
        return max(float(nav2_skill_cfg.get("post_success_timeout_sec", 3.0)), 0.1)

    def _post_success_settle_sec(self) -> float:
        nav2_skill_cfg = self._base_cfg.get("nav2_skill", {}) if isinstance(self._base_cfg, dict) else {}
        return max(float(nav2_skill_cfg.get("post_success_settle_sec", 0.25)), 0.0)

    def _local_goal_reached_hold_sec(self) -> float:
        nav2_skill_cfg = self._base_cfg.get("nav2_skill", {}) if isinstance(self._base_cfg, dict) else {}
        return max(float(nav2_skill_cfg.get("local_goal_reached_hold_sec", 0.5)), 0.0)

    def _enter_post_success_settling(self, *, trigger: str = "nav2_result"):
        self.state = self.STATE_POST_SUCCESS_SETTLING
        self._post_success_trigger = str(trigger)
        self._start_robot_bridge_restore_after_navigation()
        now = self._sim_time()
        self._post_success_started_at = now
        self._post_success_deadline = now + self._post_success_timeout_sec()
        self._post_success_settle_started_at = None

    def _robot_base_state_is_invalid(self) -> bool:
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        has_bad_state = getattr(bridge, "has_non_finite_state", None)
        if callable(has_bad_state) and bool(has_bad_state()):
            return True
        try:
            translation, orientation = self.robot.get_mobile_base_pose()
            values = [float(translation[0]), float(translation[1]), float(translation[2])]
            values.extend(float(v) for v in list(orientation)[:4])
            return not all(math.isfinite(value) for value in values)
        except Exception:
            return True

    def _robot_base_invalid_reason(self) -> str:
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        reason_fn = getattr(bridge, "non_finite_state_reason", None)
        if callable(reason_fn):
            reason = str(reason_fn()).strip()
            if reason:
                return reason
        return "non_finite_mobile_base_pose"

    def _start_robot_bridge_heading_alignment(self):
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        if bridge is None or not hasattr(bridge, "start_heading_alignment"):
            return
        follow_path_cfg = (
            self._base_cfg.get("nav2_skill", {})
            .get("controller_server", {})
            .get("follow_path", {})
        )
        if not bool(follow_path_cfg.get("rotate_to_heading_enabled", False)):
            return
        try:
            bridge.start_heading_alignment(
                target_x=float(self.goal_x),
                target_y=float(self.goal_y),
                tolerance_rad=float(follow_path_cfg.get("angular_dist_threshold", 0.12)),
                rotate_vel=float(follow_path_cfg.get("rotate_to_heading_angular_vel", 0.3)),
            )
        except Exception:
            LOGGER.exception("failed to start base heading alignment gate")

    def _publish_navigation_goal(self, bridge_client):
        self._start_robot_bridge_heading_alignment()
        bridge_client.publish_goal(
            request_id=self._request_id,
            goal_x=self.goal_x,
            goal_y=self.goal_y,
            goal_yaw=self.goal_yaw,
        )
        self.state = self.STATE_WAITING_FOR_GOAL_ACCEPTED
        self._goal_accept_deadline = time_monotonic() + min(self.startup_timeout_sec, 30.0)

    def _initialize_dynamic_goal_candidates(self):
        assert self.approach_config is not None
        try:
            target = self._resolve_approach_target_pose()
        except Exception as exc:  # pylint: disable=broad-except
            self._fail("approach_target_unavailable", str(exc))
            return
        self._approach_target_pose = dict(target)
        try:
            static_map = load_static_map(str(self._map_info["yaml_path"]))
            footprint_points = resolve_nav2_footprint_points(self._base_cfg)
            footprint_padding_m = resolve_approach_footprint_padding_m(self._base_cfg, self.approach_config)
        except Exception as exc:  # pylint: disable=broad-except
            self._fail("approach_preflight_unavailable", f"Dynamic approach preflight unavailable: {exc}")
            return

        raw_candidates = sample_approach_candidates(
            self.approach_config,
            (float(target["x"]), float(target["y"])),
        )
        candidates = []
        for candidate in raw_candidates:
            static_result = check_footprint_static_collision(
                static_map=static_map,
                footprint_points=footprint_points,
                x=float(candidate["x"]),
                y=float(candidate["y"]),
                yaw=float(candidate["yaw"]),
                free_value_min=int(self.approach_config.static_free_value_min),
                footprint_padding_m=float(footprint_padding_m),
            )
            candidate = {
                **candidate,
                "static_ok": bool(static_result.get("ok", False)),
                "static_reason": str(static_result.get("reason", "")),
                "static_check": static_result,
                "path_ok": False,
                "path_state": "not_requested",
                "path_detail": "",
                "path_length_m": float("inf"),
            }
            candidates.append(candidate)

        self._dynamic_goal_candidates = sort_candidates_for_preflight(candidates)
        self._dynamic_goal_plan_index = 0
        self._dynamic_goal_active_plan_request_id = ""
        self._dynamic_goal_plan_deadline = None
        self._dynamic_goal_debug_path = os.path.join(self._goal_output_dir, "dynamic_goal_candidates.json")
        self._write_dynamic_goal_candidates_debug()

    def _resolve_approach_target_pose(self) -> dict:
        assert self.approach_config is not None
        target_name = str(self.approach_config.target_name)
        task_objects = getattr(self.task, "_task_objects", {}) or {}
        target = task_objects.get(target_name) if isinstance(task_objects, dict) else None
        if target is None:
            objects = getattr(self.task, "objects", {}) or {}
            if isinstance(objects, dict):
                target = objects.get(target_name)
        if target is None:
            fixtures = getattr(self.task, "fixtures", {}) or {}
            if isinstance(fixtures, dict):
                target = fixtures.get(target_name)
        if target is None or not hasattr(target, "get_world_pose"):
            raise KeyError(f"navigate approach target '{target_name}' was not found in task._task_objects")
        translation, orientation = target.get_world_pose()
        return {
            "name": target_name,
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]) if len(translation) > 2 else 0.0,
            "orientation_wxyz": [float(value) for value in list(orientation)[:4]],
        }

    def _publish_next_dynamic_plan_request(self, bridge_client):
        while self._dynamic_goal_plan_index < len(self._dynamic_goal_candidates):
            candidate = self._dynamic_goal_candidates[self._dynamic_goal_plan_index]
            if not bool(candidate.get("static_ok", False)):
                candidate["path_state"] = "skipped_static_rejected"
                self._dynamic_goal_plan_index += 1
                continue

            plan_request_id = f"{self._request_id}_approach_{int(candidate['index'])}"
            candidate["plan_request_id"] = plan_request_id
            candidate["path_state"] = "pending"
            self._dynamic_goal_active_plan_request_id = plan_request_id
            self._dynamic_goal_plan_deadline = time_monotonic() + min(max(self.startup_timeout_sec, 1.0), 10.0)
            self._write_dynamic_goal_candidates_debug()
            bridge_client.publish_plan_request(
                request_id=self._request_id,
                plan_request_id=plan_request_id,
                goal_x=float(candidate["x"]),
                goal_y=float(candidate["y"]),
                goal_yaw=float(candidate["yaw"]),
            )
            return

        self._write_dynamic_goal_candidates_debug()
        self._fail(
            "approach_no_reachable_candidate",
            "Dynamic approach found no candidate that passed static footprint and Nav2 ComputePathToPose checks.",
        )
        return

    def _consume_dynamic_plan_result(self, bridge_client):
        if not self._dynamic_goal_active_plan_request_id:
            self._publish_next_dynamic_plan_request(bridge_client)
            return
        candidate = self._active_dynamic_candidate()
        if candidate is None:
            self._dynamic_goal_active_plan_request_id = ""
            self._publish_next_dynamic_plan_request(bridge_client)
            return

        payload = bridge_client.request_plan_result(
            request_id=self._request_id,
            plan_request_id=self._dynamic_goal_active_plan_request_id,
        )
        if not payload:
            return

        planning = dict(payload.get("planning", {}))
        path = dict(planning.get("path", {}))
        path_length_m = float(path.get("path_length_m", float("inf")))
        path_ok = str(payload.get("state", "")).strip().lower() == "succeeded" and math.isfinite(path_length_m)
        candidate["path_ok"] = bool(path_ok)
        candidate["path_state"] = str(payload.get("state", ""))
        candidate["path_detail"] = str(payload.get("detail", ""))
        candidate["path_status_code"] = int(payload.get("status_code", -1))
        candidate["path_length_m"] = path_length_m
        candidate["path_num_poses"] = int(path.get("num_poses", 0))
        candidate["planning"] = planning
        if path_ok:
            self._dynamic_goal_selected = dict(candidate)
            self.goal_x = float(candidate["x"])
            self.goal_y = float(candidate["y"])
            self.goal_yaw = float(candidate["yaw"])
            self._dynamic_goal_active_plan_request_id = ""
            self._dynamic_goal_plan_deadline = None
            self._write_dynamic_goal_candidates_debug()
            bridge_client.clear_cached_bridge_state()
            self._publish_navigation_goal(bridge_client)
            return

        self._dynamic_goal_plan_index += 1
        self._dynamic_goal_active_plan_request_id = ""
        self._dynamic_goal_plan_deadline = None
        self._write_dynamic_goal_candidates_debug()
        self._publish_next_dynamic_plan_request(bridge_client)

    def _active_dynamic_candidate(self) -> Optional[dict]:
        if 0 <= self._dynamic_goal_plan_index < len(self._dynamic_goal_candidates):
            return self._dynamic_goal_candidates[self._dynamic_goal_plan_index]
        return None

    def _write_dynamic_goal_candidates_debug(self):
        if not self._dynamic_goal_debug_path:
            return
        payload = {
            "approach": self._approach_debug_payload(),
            "selected": dict(self._dynamic_goal_selected),
            "candidates": self._json_safe(self._dynamic_goal_candidates),
        }
        try:
            write_candidates_debug(self._dynamic_goal_debug_path, payload)
        except Exception:
            LOGGER.exception("failed to write dynamic goal candidates debug")

    def _approach_debug_payload(self) -> dict:
        if self.approach_config is None:
            return {}
        return {
            "target_name": str(self.approach_config.target_name),
            "target_pose": dict(self._approach_target_pose),
            "min_distance": float(self.approach_config.min_distance),
            "max_distance": float(self.approach_config.max_distance),
            "sample_count": int(self.approach_config.sample_count),
            "footprint_padding_m": float(resolve_approach_footprint_padding_m(self._base_cfg, self.approach_config)),
            "sampling_random": bool(self.approach_config.sampling_random),
            "sampling_seed": self.approach_config.sampling_seed,
            "sampling_strategy": (
                "uniform_random_radius_random_angle"
                if bool(self.approach_config.sampling_random)
                else "linear_radius_golden_angle"
            ),
            "debug_path": str(self._dynamic_goal_debug_path),
        }

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(number):
            return number
        return None

    def _write_debug_snapshot(self, filename: str, reason: str, message: str):
        self._update_pose_result_fields()
        control_snapshot = runtime_control_debug_snapshot(self.robot)
        bridge_client = self._bridge_client
        planning_payload = dict((bridge_client.latest_result if bridge_client is not None else {}).get("planning", {}))
        trajectory_payload = bridge_client.odom_trace if bridge_client is not None else []
        artifacts = self._write_navigation_artifacts(
            planning_payload=planning_payload,
            trajectory_payload=trajectory_payload,
            control_snapshot=control_snapshot,
        )
        debug_snapshot = {
            "robot": getattr(self.robot, "name", "robot"),
            "state": str(self.state),
            "reason": str(reason),
            "message": str(message),
            "goal": {"x": float(self.goal_x), "y": float(self.goal_y), "yaw": float(self.goal_yaw)},
            "goal_source": "approach" if self.approach_config is not None else "fixed",
            "approach": self._approach_debug_payload(),
            "dynamic_goal": dict(self._dynamic_goal_selected),
            "world_xy": list(self.result.final_world_xy),
            "world_yaw": float(self.result.final_world_yaw),
            "nav_xy": list(self.result.final_nav_xy),
            "nav_yaw": float(self.result.final_nav_yaw),
            "world_dist": float(self.result.final_distance_to_goal),
            "nav_dist": float(self.result.final_nav_distance_to_goal),
            "yaw_err": float(self.result.final_yaw_error_rad),
            "post_success_trigger": str(self._post_success_trigger),
            "local_goal_reached_started_at": self._local_goal_reached_started_at,
            "local_goal_reached_hold_sec": self._local_goal_reached_hold_sec(),
            "control": control_snapshot,
            "planning": planning_payload,
            "map_info": dict(self._goal_debug_map_info or self._map_info or {}),
            "params_path": str(self._goal_params_path or self._params_path),
            "stack_id": str(self._stack_id),
            "nav2_runtime": {
                "bridge_online": bool(bridge_client.bridge_online) if bridge_client is not None else False,
                "nav_stack_ready": bool(bridge_client.nav_stack_ready) if bridge_client is not None else False,
                "latest_status": bridge_client.latest_status if bridge_client is not None else {},
                "latest_result": bridge_client.latest_result if bridge_client is not None else {},
            },
            "artifacts": artifacts,
        }
        if self._goal_output_dir:
            snapshot_path = os.path.join(self._goal_output_dir, filename)
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                json.dump(debug_snapshot, handle, indent=2, ensure_ascii=False)
        return debug_snapshot

    def _log_result_summary(self, *, level: int, reason: str, message: str, control_snapshot: dict | None = None):
        if control_snapshot is None:
            control_snapshot = runtime_control_debug_snapshot(self.robot)
        LOGGER.log(
            level,
            (
                "nav2 skill finished: robot=%s success=%s reason=%s message=%s "
                "goal=(%.3f, %.3f, %.3f) world_xy=%s nav_xy=%s "
                "world_dist=%.3f nav_dist=%.3f yaw_err=%.3f "
                "nav2_tol_xy=%.3f nav2_tol_yaw=%.3f "
                "skill_tol_xy=%.3f skill_tol_yaw=%.3f control=%s"
            ),
            getattr(self.robot, "name", "robot"),
            bool(self.result.success),
            str(reason),
            str(message),
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_yaw),
            self.result.final_world_xy,
            self.result.final_nav_xy,
            float(self.result.final_distance_to_goal),
            float(self.result.final_nav_distance_to_goal),
            float(self.result.final_yaw_error_rad),
            float(self.nav2_position_tolerance_m),
            float(self.nav2_yaw_tolerance_rad),
            float(self.position_tolerance_m),
            float(self.yaw_tolerance_rad),
            control_snapshot,
        )

    def _write_navigation_artifacts(self, *, planning_payload: dict, trajectory_payload: list[dict], control_snapshot: dict) -> dict:
        artifacts = {}
        if not self._goal_output_dir:
            return artifacts

        planning_summary = {}
        if planning_payload:
            planning_summary = {
                "state": str(planning_payload.get("state", "")),
                "source": str(planning_payload.get("source", "")),
                "status_code": planning_payload.get("status_code"),
                "planning_time_sec": planning_payload.get("planning_time_sec"),
            }
            path_payload = dict(planning_payload.get("path", {}))
            if path_payload:
                planning_summary["frame_id"] = str(path_payload.get("frame_id", ""))
                planning_summary["num_poses"] = int(path_payload.get("num_poses", 0))
                planning_summary["path_length_m"] = float(path_payload.get("path_length_m", 0.0))
            planned_path_path = os.path.join(self._goal_output_dir, "planned_path.json")
            with open(planned_path_path, "w", encoding="utf-8") as handle:
                json.dump(planning_payload, handle, indent=2, ensure_ascii=False)
            artifacts["planned_path"] = planned_path_path
        if planning_summary:
            artifacts["planned_path_summary"] = planning_summary

        if trajectory_payload:
            actual_trajectory_path = os.path.join(self._goal_output_dir, "actual_trajectory.json")
            with open(actual_trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(trajectory_payload, handle, indent=2, ensure_ascii=False)
            artifacts["actual_trajectory"] = actual_trajectory_path
            artifacts["actual_trajectory_summary"] = {
                "num_samples": len(trajectory_payload),
                "start_xy": [float(trajectory_payload[0]["x"]), float(trajectory_payload[0]["y"])],
                "end_xy": [float(trajectory_payload[-1]["x"]), float(trajectory_payload[-1]["y"])],
            }

        bridge_snapshot = dict(control_snapshot.get("bridge", {}) or {})
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        bridge_cmd_vel_history = list(getattr(bridge, "_debug_cmd_vel_history", [])) if bridge is not None else []
        if bridge_cmd_vel_history:
            cmd_vel_history_path = os.path.join(self._goal_output_dir, "cmd_vel_history.json")
            with open(cmd_vel_history_path, "w", encoding="utf-8") as handle:
                json.dump(bridge_cmd_vel_history, handle, indent=2, ensure_ascii=False)
            artifacts["cmd_vel_history"] = cmd_vel_history_path
            artifacts["cmd_vel_history_summary"] = {
                "num_samples": len(bridge_cmd_vel_history),
                "last_received_cmd_vel": bridge_snapshot.get("last_received_cmd_vel"),
            }

        bridge_command_history = list(getattr(bridge, "_debug_command_history", [])) if bridge is not None else []
        if bridge_command_history:
            bridge_command_history_path = os.path.join(self._goal_output_dir, "bridge_command_history.json")
            with open(bridge_command_history_path, "w", encoding="utf-8") as handle:
                json.dump(bridge_command_history, handle, indent=2, ensure_ascii=False)
            artifacts["bridge_command_history"] = bridge_command_history_path
            artifacts["bridge_command_history_summary"] = {
                "num_samples": len(bridge_command_history),
                "steering_command_sign": bridge_snapshot.get("steering_command_sign"),
            }
        return artifacts

    def _fail(self, reason: str, message: str):
        self._update_pose_result_fields()
        self.result.done = True
        self.result.success = False
        self.result.failure_reason = str(reason)
        self.result.error_message = str(message)
        control_snapshot = runtime_control_debug_snapshot(self.robot)
        self._write_debug_snapshot("failure_snapshot.json", reason, message)
        self._log_result_summary(
            level=logging.ERROR,
            reason=self.result.failure_reason,
            message=self.result.error_message,
            control_snapshot=control_snapshot,
        )
        self.state = self.STATE_FAILED
        self._finalize_robot_bridge_after_navigation()

    def _reset_robot_bridge_state(self, *, clear_debug_history: bool):
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        if bridge is None or not hasattr(bridge, "reset"):
            return
        try:
            bridge.reset(clear_debug_history=clear_debug_history)
        except Exception:
            LOGGER.exception(
                "failed to reset base bridge for robot=%s clear_debug_history=%s",
                getattr(self.robot, "name", "robot"),
                clear_debug_history,
            )

    def _prepare_robot_bridge_for_navigation(self):
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        prepare_fn = getattr(bridge, "prepare_for_navigation", None)
        if not callable(prepare_fn):
            return
        try:
            prepare_fn()
        except Exception:
            LOGGER.exception("failed to prepare base bridge for navigation")

    def _start_robot_bridge_restore_after_navigation(self):
        if self._restore_after_nav_started:
            return
        self._restore_after_nav_started = True
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        finalize_fn = getattr(bridge, "finalize_after_navigation", None)
        if not callable(finalize_fn):
            return
        try:
            finalize_fn()
        except Exception:
            LOGGER.exception("failed to start base bridge restore after navigation")

    def _robot_bridge_restore_after_navigation_done(self) -> bool:
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        done_fn = getattr(bridge, "restore_after_navigation_done", None)
        if callable(done_fn):
            try:
                return bool(done_fn())
            except Exception:
                LOGGER.exception("failed to query base bridge restore after navigation state")
                return True
        if bridge is None:
            return True
        return not bool(getattr(bridge, "_restore_after_navigation", False))

    def _finalize_robot_bridge_after_navigation(self):
        bridge = getattr(self.robot, "_simbox_ros_base_bridge", None)
        if bridge is None:
            return
        self._start_robot_bridge_restore_after_navigation()
        if self._restore_after_nav_started:
            return
        finalize_fn = getattr(bridge, "finalize_after_navigation", None)
        if callable(finalize_fn):
            try:
                finalize_fn()
                return
            except Exception:
                LOGGER.exception("failed to finalize base bridge after navigation")
        self._reset_robot_bridge_state(clear_debug_history=False)

    def _update_pose_result_fields(self):
        try:
            world_translation, world_orientation = self.robot.get_mobile_base_pose()
            world_xy = (float(world_translation[0]), float(world_translation[1]))
            world_yaw = float(yaw_from_wxyz(world_orientation))
        except Exception:
            LOGGER.exception("failed to read mobile base pose for nav2 result")
            return
        if not all(math.isfinite(value) for value in (world_xy[0], world_xy[1], world_yaw)):
            LOGGER.error(
                "mobile base pose is non-finite during nav2 result update: xy=%s yaw=%s",
                world_xy,
                world_yaw,
            )
            return

        bridge_client = self._bridge_client
        result_payload = bridge_client.request_result(self._request_id) if bridge_client is not None else {}
        status_payload = bridge_client.request_status(self._request_id) if bridge_client is not None else {}
        reported_pose = dict((result_payload or status_payload).get("reported_pose", {}))

        if bridge_client is not None:
            nav_x, nav_y, nav_yaw = bridge_client.get_current_pose_xy_yaw()
            nav_xy = (float(nav_x), float(nav_y))
            nav_yaw = float(nav_yaw)
        elif {"x", "y", "yaw"} <= set(reported_pose.keys()):
            nav_xy = (float(reported_pose["x"]), float(reported_pose["y"]))
            nav_yaw = float(reported_pose["yaw"])
        else:
            nav_xy = world_xy
            nav_yaw = world_yaw
        if not all(math.isfinite(value) for value in (nav_xy[0], nav_xy[1], nav_yaw)):
            nav_xy = world_xy
            nav_yaw = world_yaw

        self.result.final_world_xy = world_xy
        self.result.final_world_yaw = world_yaw
        self.result.final_nav_xy = nav_xy
        self.result.final_nav_yaw = nav_yaw
        self.result.final_distance_to_goal = math.hypot(self.goal_x - world_xy[0], self.goal_y - world_xy[1])
        self.result.final_nav_distance_to_goal = math.hypot(self.goal_x - nav_xy[0], self.goal_y - nav_xy[1])
        self.result.final_yaw_error_rad = abs(angle_diff_rad(self.goal_yaw, world_yaw))

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


SkillManagedNav2Session = PersistentNav2RuntimeManager
