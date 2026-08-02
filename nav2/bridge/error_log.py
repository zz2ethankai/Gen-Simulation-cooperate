"""Compact goal-scoped Nav2 warning and error collection."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


_NAV2_LOGGER_NAMES = {
    "amcl",
    "behavior_server",
    "behavior_tree_engine",
    "bt_navigator",
    "collision_monitor",
    "controller_server",
    "docking_server",
    "global_costmap",
    "lifecycle_manager_navigation",
    "local_costmap",
    "map_server",
    "planner_server",
    "recoveries_server",
    "route_server",
    "smoother_server",
    "velocity_smoother",
    "waypoint_follower",
}


def is_nav2_logger(name: str) -> bool:
    normalized = str(name).strip().strip("/")
    leaf = normalized.rsplit("/", 1)[-1]
    root_logger = leaf.split(".", 1)[0]
    return root_logger in _NAV2_LOGGER_NAMES or root_logger.startswith("nav2_")


class Nav2ErrorLogBuffer:
    """Aggregate Nav2 failure messages without storing motion traces."""

    WARN_LEVEL = 30

    _INFO_FAILURE_MARKERS = (
        "abort",
        "cannot",
        "collision",
        "exception",
        "fail",
        "invalid",
        "lethal",
        "no valid",
        "optimizer",
        "timeout",
        "timed out",
        "unable",
    )

    def __init__(self):
        self.request_id = ""
        self.started_wall_time_sec = -1.0
        self._entries: OrderedDict[tuple[str, int, str, str, str], dict[str, Any]] = OrderedDict()
        self.dropped_unique_entries = 0

    def reset(self, *, request_id: str, started_wall_time_sec: float):
        self.request_id = str(request_id)
        self.started_wall_time_sec = float(started_wall_time_sec)
        self._entries.clear()
        self.dropped_unique_entries = 0

    def append(
        self,
        *,
        level: int,
        name: str,
        message: str,
        function: str = "",
        file: str = "",
        line: int = 0,
        ros_time_sec: float | None = None,
        wall_time_sec: float,
    ) -> None:
        if not is_nav2_logger(name):
            return
        message = str(message).strip()
        if not message:
            return
        if int(level) < self.WARN_LEVEL and not (
            int(level) >= 20 and any(marker in message.lower() for marker in self._INFO_FAILURE_MARKERS)
        ):
            return
        key = (str(name), int(level), message, str(function), str(file))
        existing = self._entries.get(key)
        if existing is not None:
            existing["count"] += 1
            existing["last_ros_time_sec"] = ros_time_sec
            existing["last_wall_time_sec"] = float(wall_time_sec)
            existing["line"] = int(line)
            return
        self._entries[key] = {
            "level": level_name(level),
            "level_value": int(level),
            "node": str(name),
            "message": message,
            "function": str(function),
            "file": str(file),
            "line": int(line),
            "count": 1,
            "first_ros_time_sec": ros_time_sec,
            "last_ros_time_sec": ros_time_sec,
            "first_wall_time_sec": float(wall_time_sec),
            "last_wall_time_sec": float(wall_time_sec),
        }

    def snapshot(self) -> dict[str, Any]:
        entries = [dict(entry) for entry in self._entries.values()]
        return {
            "request_id": self.request_id,
            "started_wall_time_sec": self.started_wall_time_sec,
            "unique_entry_count": len(entries),
            "message_count": sum(int(entry["count"]) for entry in entries),
            "dropped_unique_entries": int(self.dropped_unique_entries),
            "entries": entries,
        }


def level_name(level: int) -> str:
    value = int(level)
    if value >= 50:
        return "FATAL"
    if value >= 40:
        return "ERROR"
    if value >= 30:
        return "WARN"
    if value >= 20:
        return "INFO"
    return "DEBUG"
