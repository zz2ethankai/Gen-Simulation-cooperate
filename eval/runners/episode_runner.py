from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from eval.specs import EvalSpec


class EpisodeRunner:
    def __init__(self, env, policy, spec: EvalSpec, artifact_dir: Path | None = None):
        self.env = env
        self.policy = policy
        self.spec = spec
        self.artifact_dir = artifact_dir

    def run(self, seed: int) -> dict[str, Any]:
        started_at = time.time()
        print("[eval] policy.reset start", flush=True)
        self.policy.reset({"task": self.spec.task.name, "seed": seed})
        print("[eval] policy.reset done", flush=True)
        print("[eval] env.reset start", flush=True)
        obs = self.env.reset(seed)
        print("[eval] env.reset done", flush=True)

        frames_by_camera: dict[str, list[Any]] = {}
        if self.spec.run_args.get("record_video", False):
            print("[eval] capture initial video frame", flush=True)
            _append_video_frames(frames_by_camera, obs, self.spec.run_args)

        pending_actions: list[Any] = []
        while not self.env.is_done():
            if not pending_actions:
                print(f"[eval] policy.infer step={getattr(self.env, 'step_count', 'unknown')}", flush=True)
                prediction = self.policy.infer(obs)
                pending_actions = _unpack_actions(prediction, self.spec.policy.open_loop_horizon)
                print(f"[eval] got {len(pending_actions)} action(s)", flush=True)
            print(f"[eval] env.step step={getattr(self.env, 'step_count', 'unknown')}", flush=True)
            obs = self.env.step(pending_actions.pop(0))
            if self.spec.run_args.get("record_video", False):
                _append_video_frames(frames_by_camera, obs, self.spec.run_args)

        print("[eval] episode done, collecting metrics", flush=True)
        metrics = self.env.get_metrics()
        episode = {
            "eval_name": self.spec.name,
            "task": self.spec.task.name,
            "policy": self.spec.policy.name,
            "seed": seed,
            "metrics": metrics,
            "duration_s": round(time.time() - started_at, 4),
        }
        artifacts = _write_videos(frames_by_camera, self.artifact_dir, seed, self.spec.run_args)
        if artifacts:
            episode["artifacts"] = artifacts
        return episode


def _unpack_actions(prediction: Any, horizon: int) -> list[Any]:
    if isinstance(prediction, dict) and "action_dict" in prediction:
        return [prediction]
    if isinstance(prediction, dict):
        actions = prediction.get("actions", prediction.get("action"))
    else:
        actions = prediction

    if actions is None:
        raise ValueError("Policy response must contain `actions`, `action`, or `action_dict`.")
    if _is_action_chunk(actions):
        return list(actions[:horizon])
    return [actions]


def _is_action_chunk(actions: Any) -> bool:
    return (
        isinstance(actions, list)
        and len(actions) > 0
        and isinstance(actions[0], (list, tuple, dict))
    )


def _append_video_frames(
    frames_by_camera: dict[str, list[Any]],
    observation: dict[str, Any],
    run_args: dict[str, Any],
) -> None:
    for camera_name, frame in _capture_video_frames(observation, run_args).items():
        frames_by_camera.setdefault(camera_name, []).append(frame)


def _capture_video_frames(observation: dict[str, Any], run_args: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    video_cameras = run_args.get("video_cameras")
    if video_cameras:
        frames = {}
        for index, camera in enumerate(video_cameras):
            camera_name, camera_path = _parse_camera_spec(camera, index)
            frames[camera_name] = _normalize_video_frame(_get_by_path(observation, camera_path))
        return frames

    camera_path = run_args.get("video_camera")
    if camera_path:
        return {"primary": _normalize_video_frame(_get_by_path(observation, camera_path))}

    cameras = observation.get("raw", {}).get("cameras", {})
    for camera_name in sorted(cameras):
        if "color_image" in cameras[camera_name]:
            return {camera_name: _normalize_video_frame(cameras[camera_name]["color_image"])}

    raise ValueError("record_video is enabled but no camera color_image was found.")


def _parse_camera_spec(camera: Any, index: int) -> tuple[str, str]:
    if isinstance(camera, str):
        return f"camera_{index}", camera
    return str(camera.get("name", f"camera_{index}")), str(camera["path"])


def _normalize_video_frame(frame: Any):
    import numpy as np

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected video frame shape HxWx3/4, got {frame.shape}.")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _write_videos(
    frames_by_camera: dict[str, list[Any]],
    artifact_dir: Path | None,
    seed: int,
    run_args: dict[str, Any],
) -> dict[str, Any] | None:
    if not frames_by_camera or artifact_dir is None:
        return None

    import imageio.v2 as imageio

    video_dir = artifact_dir / "videos" / f"seed_{seed}" / str(run_args.get("video_robot", "robot"))

    videos = {}
    for camera_name, frames in frames_by_camera.items():
        key = f"images.rgb.{_safe_filename(camera_name)}"
        camera_dir = video_dir / key
        camera_dir.mkdir(parents=True, exist_ok=True)
        video_path = camera_dir / "demo.mp4"
        imageio.mimsave(video_path, frames, fps=int(run_args.get("video_fps", 15)))
        videos[camera_name] = str(video_path)

    primary_video = next(iter(videos.values()))
    return {"video": primary_video, "videos": videos}


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return safe.strip("_") or "camera"


def _get_by_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    parts = path.split(".")
    index = 0
    while index < len(parts):
        if not isinstance(value, dict):
            traversed = ".".join(parts[:index])
            raise KeyError(f"Cannot resolve {path!r}: {traversed!r} is not a mapping.")

        remaining = ".".join(parts[index:])
        if remaining in value:
            return value[remaining]

        part = parts[index]
        if part in value:
            value = value[part]
            index += 1
            continue

        for end in range(len(parts), index + 1, -1):
            flat_key = ".".join(parts[index:end])
            if flat_key in value:
                value = value[flat_key]
                index = end
                break
        else:
            available = ", ".join(map(str, list(value.keys())[:12]))
            traversed = ".".join(parts[:index]) or "<root>"
            raise KeyError(f"Cannot resolve {path!r} at {traversed!r}. Available keys: {available}")
    return value
