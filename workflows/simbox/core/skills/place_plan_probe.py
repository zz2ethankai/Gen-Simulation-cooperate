"""Planning-only Place probe gated by a real Pick attachment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.skills.base_skill import register_skill
from core.skills.place import Place


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@register_skill
class PlacePlanProbe(Place):
    """Plan Place after Pick has physically attached the object, without placing it."""

    def _attachment_evidence(self) -> tuple[dict[str, Any] | None, str | None]:
        runtime = self._require_skill_runtime()
        if getattr(runtime.robot_port, "collision_world_mode", "") != "physics_schema":
            return None, "PLACE_PROBE_REQUIRES_PHYSICS_SCHEMA"
        manager = runtime.robot_port.collision_scene_manager
        if manager is None:
            return None, "PLACE_PROBE_COLLISION_MANAGER_MISSING"
        object_name = self.pick_obj.name
        record = manager.records.get(object_name)
        raw_state = getattr(record, "state", None)
        state = getattr(raw_state, "value", raw_state)
        if record is None or state != "attached":
            return None, "PLACE_PROBE_OBJECT_NOT_ATTACHED"
        robot = str(runtime.name)
        arm = str(runtime.arm_name)
        if (record.owner_robot, record.owner_arm) != (robot, arm):
            return None, "PLACE_PROBE_ATTACHMENT_OWNER_MISMATCH"
        events = [
            event
            for event in manager.object_state_events
            if event.get("entity") == object_name
        ]
        if not events:
            return None, "PLACE_PROBE_ATTACHMENT_EVIDENCE_MISSING"
        event = events[-1]
        if not (
            event.get("from") == "active_target_approach"
            and event.get("to") == "attached"
            and event.get("reason") == "attach"
            and event.get("owner_robot") == robot
            and event.get("owner_arm") == arm
        ):
            return None, "PLACE_PROBE_ATTACHMENT_EVIDENCE_MISSING"
        return dict(event), None

    def _write_result(self, result: dict[str, Any]) -> None:
        result_path = Path(str(self.skill_cfg["result_path"])).expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result.update(
            {
                "schema_version": 1,
                "probe": "place_planning",
                "candidate_id": str(self.skill_cfg["candidate_id"]),
                "arm": str(self._require_skill_runtime().arm_name),
                "objects": [str(value) for value in self.skill_cfg["objects"]],
            }
        )
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_value(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(result_path)

    def _hold_current_pose(self) -> None:
        self.manip_list = [self.measured_hold_command()]

    def generate_manip_cmds(self):
        attachment, failure_code = self._attachment_evidence()
        if failure_code is not None:
            self.failure_reason = failure_code
            self._write_result(
                {
                    "feasible": False,
                    "failure_code": failure_code,
                    "attachment": attachment,
                    "planned_phases": [],
                }
            )
            self._hold_current_pose()
            return

        super().generate_manip_cmds()
        planned_phases = [
            command.phase.value
            for command in self.manip_list
            if hasattr(command, "phase")
        ]
        required_phases = {"transit_preplace", "terminal_place_descent"}
        feasible = not self.failure_reason and required_phases.issubset(planned_phases)
        failure_code = None if feasible else self.failure_reason or "PLACE_PROBE_DID_NOT_PLAN"
        target = _json_value(self._target_intent or {})
        self._write_result(
            {
                "feasible": feasible,
                "failure_code": failure_code,
                "attachment": attachment,
                "planned_phases": planned_phases,
                "selected_target": target,
                "planning": _json_value(self._selected_plan),
            }
        )
        self._hold_current_pose()

    def is_success(self):
        return True

    def is_terminal_success(self):
        return True

    def is_feasible(self, th=5):
        return True

    def is_record(self):
        return False
