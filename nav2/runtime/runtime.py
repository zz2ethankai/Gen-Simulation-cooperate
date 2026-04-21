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

from .config import (
    NAV2_DEFAULT_POSITION_TOLERANCE_M,
    NAV2_DEFAULT_YAW_TOLERANCE_RAD,
    generate_nav2_bringup_artifacts,
)
from .debug import Nav2SkillResult, TaskShim, runtime_control_debug_snapshot
from .utils import angle_diff_rad, safe_name, time_monotonic, yaw_from_wxyz

LOGGER = logging.getLogger("simbox.nav2_skill")


class PersistentNav2RuntimeManager:
    """Workflow-side manager that drives Nav2 through a standard-message ROS bridge."""

    STATE_IDLE = "idle"
    STATE_WAITING_FOR_STACK_READY = "waiting_for_stack_ready"
    STATE_WAITING_FOR_MAP_READY = "waiting_for_map_ready"
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
        self._cleaned_up = False

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

    def _prepare_stack_artifacts(self):
        from nav2.mapgen.exporter import IsaacStaticMapExporter

        robot_tag = safe_name(getattr(self.robot, "name", "robot"))
        scene_tag = safe_name(self.scene_name)
        stack_tag = f"{robot_tag}_{scene_tag}"
        self._stack_output_dir = os.path.join(self.output_root, stack_tag, "stack")
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
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
        )
        self._params_path = str(artifacts["params_path"])
        self._stack_id = f"{robot_tag}::{scene_tag}::{self._stack_signature(artifacts['params'])}"

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
        position_tolerance_m: float = NAV2_DEFAULT_POSITION_TOLERANCE_M,
        yaw_tolerance_rad: float = NAV2_DEFAULT_YAW_TOLERANCE_RAD,
        startup_timeout_sec: float = 60.0,
        runtime_timeout_sec: float = 240.0,
    ):
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)
        self.goal_yaw = float(goal_yaw)
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

        bridge_client = self._ensure_bridge_client()
        bridge_client.reset_debug_trace()
        if previous_request_id:
            bridge_client.cancel_request(previous_request_id)
        self._reset_robot_bridge_state(clear_debug_history=True)
        self._prepare_stack_artifacts()
        self._freeze_goal_debug_artifacts()
        self.state = self.STATE_WAITING_FOR_STACK_READY
        self._startup_deadline = time_monotonic() + self.startup_timeout_sec

    def prepare_for_reset(self):
        self._reset_robot_bridge_state(clear_debug_history=False)
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

    def step(self):
        if self.done or self.state == self.STATE_IDLE:
            return

        bridge_client = self._ensure_bridge_client()
        bridge_client.step(step_dt=self._step_dt())

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
                bridge_client.publish_goal(
                    request_id=self._request_id,
                    goal_x=self.goal_x,
                    goal_y=self.goal_y,
                    goal_yaw=self.goal_yaw,
                )
                self.state = self.STATE_WAITING_FOR_GOAL_ACCEPTED
                self._goal_accept_deadline = time_monotonic() + min(self.startup_timeout_sec, 30.0)
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if time_monotonic() >= float(self._startup_deadline):
                self._fail("map_update_timeout", "Timed out waiting for bridge adapter to load the map.")
            return

        if self.state == self.STATE_WAITING_FOR_GOAL_ACCEPTED:
            if bridge_state in {"accepted", "running"}:
                self.state = self.STATE_RUNNING
                self._runtime_deadline = time_monotonic() + self.runtime_timeout_sec
                return
            if bridge_state == "succeeded":
                self.state = self.STATE_POST_SUCCESS_SETTLING
                self._post_success_deadline = time_monotonic() + 2.0
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
                self.state = self.STATE_POST_SUCCESS_SETTLING
                self._post_success_deadline = time_monotonic() + 2.0
                return
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if time_monotonic() >= float(self._runtime_deadline):
                bridge_client.cancel_request(self._request_id)
                self._fail("runtime_timeout", "Timed out while waiting for the navigation goal to finish.")
            return

        if self.state == self.STATE_POST_SUCCESS_SETTLING:
            self._update_pose_result_fields()
            if bridge_state in {"failed", "rejected", "aborted", "canceled"}:
                self._fail("bridge_" + bridge_state, self._bridge_detail(status=status, result=result) or f"Bridge ended with {bridge_state}")
                return
            if self._goal_within_tolerance():
                self.result.done = True
                self.result.success = True
                self.state = self.STATE_SUCCEEDED
                self._write_debug_snapshot(
                    "success_snapshot.json",
                    "goal_succeeded",
                    "Nav2 goal reached within skill tolerances after post-success settling.",
                )
                self._reset_robot_bridge_state(clear_debug_history=False)
                return
            if time_monotonic() >= float(self._post_success_deadline):
                self._fail(
                    "goal_tolerance_not_met",
                    "Nav2 reported success but skill tolerances were not met after post-success settling.",
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
        self._reset_robot_bridge_state(clear_debug_history=False)
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

    def _step_dt(self) -> float:
        get_physics_dt = getattr(self.world, "get_physics_dt", None)
        if callable(get_physics_dt):
            return float(get_physics_dt())
        return float(getattr(self.world, "physics_dt", 1.0 / 60.0))

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
            "world_xy": list(self.result.final_world_xy),
            "world_yaw": float(self.result.final_world_yaw),
            "nav_xy": list(self.result.final_nav_xy),
            "nav_yaw": float(self.result.final_nav_yaw),
            "world_dist": float(self.result.final_distance_to_goal),
            "nav_dist": float(self.result.final_nav_distance_to_goal),
            "yaw_err": float(self.result.final_yaw_error_rad),
            "control": control_snapshot,
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
        failure_snapshot = self._write_debug_snapshot("failure_snapshot.json", reason, message)
        LOGGER.error(
            "nav2 skill failed: robot=%s reason=%s message=%s world_xy=%s nav_xy=%s world_dist=%.3f nav_dist=%.3f yaw_err=%.3f control=%s",
            failure_snapshot["robot"],
            self.result.failure_reason,
            self.result.error_message,
            self.result.final_world_xy,
            self.result.final_nav_xy,
            self.result.final_distance_to_goal,
            self.result.final_nav_distance_to_goal,
            self.result.final_yaw_error_rad,
            control_snapshot,
        )
        self.state = self.STATE_FAILED
        self._reset_robot_bridge_state(clear_debug_history=False)

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

    def _update_pose_result_fields(self):
        world_translation, world_orientation = self.robot.get_mobile_base_pose()
        world_xy = (float(world_translation[0]), float(world_translation[1]))
        world_yaw = float(yaw_from_wxyz(world_orientation))

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
