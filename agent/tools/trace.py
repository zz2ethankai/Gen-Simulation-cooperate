"""Append-only structured trace records for Agent runs and attempts."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceContext:
    run_id: str
    variant_id: str = ""
    attempt_id: str = ""
    parent_variant_id: str = ""
    seed: int | None = None
    profile_id: str = ""
    profile_hash: str = ""
    source_hash: str = ""
    scene_revision: str = "source"
    world_revision: int | None = None


@dataclass(frozen=True)
class TraceEvent:
    context: TraceContext
    stage: str
    status: str
    skill: str = ""
    failure_code: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        context = value.pop("context")
        return {"event_id": self.event_id, "time": self.time, **context, **value}


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            data = payload.encode("utf-8")
            written = 0
            while written < len(data):
                written += os.write(descriptor, data[written:])
        finally:
            os.close(descriptor)
