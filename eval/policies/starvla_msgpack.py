from __future__ import annotations

import functools
import os
import time
from typing import Any

import msgpack
import numpy as np


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported numpy dtype for msgpack transport: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
_msgpack_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class StarVLAMsgpackPolicyClient:
    """Client for StarVLA's native msgpack websocket policy server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10093,
        timeout_s: float = 300.0,
        unnorm_key: str | None = None,
        image_keys: list[str] | None = None,
        prompt_key: str = "detailed_prompt",
        request_args: dict[str, Any] | None = None,
        action_slice: list[int] | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.unnorm_key = unnorm_key
        self.image_keys = image_keys or []
        self.prompt_key = prompt_key
        self.request_args = request_args or {}
        self.action_slice = action_slice
        self._conn = None
        self._server_metadata: dict[str, Any] = {}
        self._packer = _MsgpackPacker()

    def reset(self, task_meta: dict[str, Any]) -> None:
        if self._conn is None:
            self._connect()

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._conn is None:
            self._connect()

        request = self._build_request(observation)
        self._conn.send(self._packer.pack(request))
        response = self._conn.recv()
        if isinstance(response, str):
            raise RuntimeError(f"StarVLA server returned text error:\n{response}")

        result = _msgpack_unpackb(response)
        if result.get("status") == "error":
            raise RuntimeError(f"StarVLA inference error: {result.get('error')}")

        actions = np.asarray(result.get("data", result)["actions"])
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if self.action_slice is not None:
            start, end = self.action_slice
            actions = actions[:, start:end]
        return {"actions": actions.tolist()}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_server_metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def get_metadata(self) -> dict[str, Any]:
        return self.get_server_metadata()

    def _connect(self) -> None:
        import websockets.sync.client

        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(key, None)

        uri = f"ws://{self.host}:{self.port}"
        started_at = time.time()
        while True:
            try:
                self._conn = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=None,
                    open_timeout=self.timeout_s,
                    close_timeout=self.timeout_s,
                )
                metadata = self._conn.recv()
                if isinstance(metadata, str):
                    raise RuntimeError(f"StarVLA server returned text during handshake:\n{metadata}")
                self._server_metadata = _msgpack_unpackb(metadata)
                return
            except OSError:
                if time.time() - started_at > self.timeout_s:
                    raise TimeoutError(f"Timed out connecting to StarVLA server at {uri}")
                time.sleep(2)

    def _build_request(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = self._extract_images(observation)
        prompt = _coerce_prompt(_get_by_path(observation, self.prompt_key))
        if not prompt:
            prompt = _coerce_prompt(observation.get("prompt", ""))

        request = {
            "examples": [
                {
                    "image": images,
                    "lang": prompt,
                }
            ],
            **self.request_args,
        }
        if self.unnorm_key:
            request["unnorm_key"] = self.unnorm_key
        return request

    def _extract_images(self, observation: dict[str, Any]) -> list[np.ndarray]:
        if self.image_keys:
            return [np.asarray(_get_by_path(observation, key), dtype=np.uint8) for key in self.image_keys]

        cameras = observation.get("raw", {}).get("cameras", {})
        images = []
        for camera_name in sorted(cameras):
            camera_obs = cameras[camera_name]
            if "color_image" in camera_obs:
                images.append(np.asarray(camera_obs["color_image"], dtype=np.uint8))
        if not images:
            raise ValueError("No camera images found. Configure policy.policy_args.image_keys for StarVLA.")
        return images


def _get_by_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        value = value[part]
    return value


def _coerce_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)
