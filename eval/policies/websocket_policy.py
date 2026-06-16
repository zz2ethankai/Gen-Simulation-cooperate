from __future__ import annotations

import json
from typing import Any


class JsonWebSocketPolicyClient:
    """JSON WebSocket client for StarVLA-style serving endpoints."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._conn = None

    def reset(self, task_meta: dict[str, Any]) -> None:
        if self._conn is None:
            import websocket

            self._conn = websocket.create_connection(self.endpoint, timeout=self.timeout_s)

    def infer(self, observation: dict[str, Any]) -> Any:
        if self._conn is None:
            self.reset({})
        self._conn.send(json.dumps(observation))
        return json.loads(self._conn.recv())

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class OpenPIWebSocketPolicyClient:
    """Adapter for OpenPI's lightweight websocket client package."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._client = None

    def reset(self, task_meta: dict[str, Any]) -> None:
        if self._client is None:
            from openpi_client import websocket_client_policy

            self._client = websocket_client_policy.WebsocketClientPolicy(self.host, self.port)

    def infer(self, observation: dict[str, Any]) -> Any:
        if self._client is None:
            self.reset({})
        return self._client.infer(observation)

    def close(self) -> None:
        self._client = None

