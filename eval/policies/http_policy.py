from __future__ import annotations

import json
from typing import Any
from urllib import request


class HttpJsonPolicyClient:
    """Simple JSON-over-HTTP policy client.

    Expected request: observation dict.
    Expected response: JSON with either `actions`, `action`, or `action_dict`.
    """

    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def reset(self, task_meta: dict[str, Any]) -> None:
        return None

    def infer(self, observation: dict[str, Any]) -> Any:
        payload = json.dumps(observation).encode("utf-8")
        req = request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def close(self) -> None:
        return None

