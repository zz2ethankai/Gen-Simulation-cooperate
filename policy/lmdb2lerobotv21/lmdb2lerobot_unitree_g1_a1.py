"""Convert Unitree G1 SimBox LMDB episodes to LeRobot v2.1."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_FPS = 50
DEFAULT_ROBOT_TYPE = "Unitree G1 29-DOF"

UNITREE_G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_BASE_POSITION_NAMES = ("x", "y", "z")
_QUATERNION_NAMES = ("w", "x", "y", "z")
_QVEL_NAMES = (
    "base_angular_velocity.x",
    "base_angular_velocity.y",
    "base_angular_velocity.z",
    *(f"{name}.velocity" for name in UNITREE_G1_BODY_JOINT_NAMES),
)
_TRANSFORM_NAMES = tuple(f"m{row}{column}" for row in range(4) for column in range(4))

NUMERIC_FEATURE_SHAPES = {
    "states.body_joint.position": (29,),
    "states.body_joint.velocity": (29,),
    "states.base.position": (3,),
    "states.base.orientation": (4,),
    "qvel": (32,),
    "base_actions.vx_body": (1,),
    "base_actions.vy_body": (1,),
    "base_actions.wz_body": (1,),
    "base_actions.locomotion_mode": (1,),
    "master_actions.body_joint.position": (29,),
    "master_actions.body_joint.velocity": (29,),
    "master_actions.body_joint.effort": (29,),
    "actions.body_joint.position": (29,),
    "actions.body_joint.velocity": (29,),
    "actions.base.position": (3,),
    "actions.base.orientation": (4,),
    "camera2env_pose.ego": (16,),
    "camera2env_pose.global": (16,),
}


def _float_feature(shape: tuple[int, ...], names: Sequence[str]) -> dict[str, Any]:
    return {"dtype": "float32", "shape": shape, "names": list(names)}


FEATURES = {
    "images.rgb.ego": {
        "dtype": "video",
        "shape": (720, 1280, 3),
        "names": ["height", "width", "channel"],
    },
    "images.rgb.global": {
        "dtype": "video",
        "shape": (720, 1280, 3),
        "names": ["height", "width", "channel"],
    },
    "states.body_joint.position": _float_feature((29,), UNITREE_G1_BODY_JOINT_NAMES),
    "states.body_joint.velocity": _float_feature((29,), UNITREE_G1_BODY_JOINT_NAMES),
    "states.base.position": _float_feature((3,), _BASE_POSITION_NAMES),
    "states.base.orientation": _float_feature((4,), _QUATERNION_NAMES),
    "qvel": _float_feature((32,), _QVEL_NAMES),
    "base_actions.vx_body": _float_feature((1,), ("vx_body",)),
    "base_actions.vy_body": _float_feature((1,), ("vy_body",)),
    "base_actions.wz_body": _float_feature((1,), ("wz_body",)),
    "base_actions.locomotion_mode": _float_feature((1,), ("locomotion_mode",)),
    "master_actions.body_joint.position": _float_feature(
        (29,), UNITREE_G1_BODY_JOINT_NAMES
    ),
    "master_actions.body_joint.velocity": _float_feature(
        (29,), UNITREE_G1_BODY_JOINT_NAMES
    ),
    "master_actions.body_joint.effort": _float_feature(
        (29,), UNITREE_G1_BODY_JOINT_NAMES
    ),
    "actions.body_joint.position": _float_feature((29,), UNITREE_G1_BODY_JOINT_NAMES),
    "actions.body_joint.velocity": _float_feature((29,), UNITREE_G1_BODY_JOINT_NAMES),
    "actions.base.position": _float_feature((3,), _BASE_POSITION_NAMES),
    "actions.base.orientation": _float_feature((4,), _QUATERNION_NAMES),
    "camera2env_pose.ego": _float_feature((16,), _TRANSFORM_NAMES),
    "camera2env_pose.global": _float_feature((16,), _TRANSFORM_NAMES),
}

_VIDEO_KEYS = ("images.rgb.ego", "images.rgb.global")


def discover_episode_dirs(source_root: Path) -> list[Path]:
    """Return SimBox episode directories that contain LMDB data and metadata."""
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_root}")
    candidates = [source_root] if (source_root / "meta_info.pkl").is_file() else []
    candidates.extend(path.parent for path in source_root.rglob("meta_info.pkl"))
    return sorted(
        {
            episode_dir
            for episode_dir in candidates
            if (episode_dir / "lmdb" / "data.mdb").is_file()
        }
    )


def _normalize_feature_array(key: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if key.startswith("camera2env_pose.") and array.ndim == 3 and array.shape[1:] == (4, 4):
        array = array.reshape(array.shape[0], 16)
    if NUMERIC_FEATURE_SHAPES[key] == (1,) and array.ndim == 1:
        array = array[:, np.newaxis]
    expected_shape = NUMERIC_FEATURE_SHAPES[key]
    if array.ndim < 2 or array.shape[1:] != expected_shape:
        raise ValueError(
            f"Feature {key} must have per-step shape {expected_shape}, got {array.shape}"
        )
    return array.astype(np.float32, copy=False)


def build_episode_frames(
    arrays: Mapping[str, Any],
    *,
    task: str,
) -> list[dict[str, Any]]:
    """Build LeRobot frames while preserving every G1 supervision layer."""
    if not task.strip():
        raise ValueError("Episode language instruction must not be empty")
    missing = [key for key in NUMERIC_FEATURE_SHAPES if key not in arrays]
    if missing:
        raise KeyError(f"Episode is missing required G1 features: {missing}")

    normalized = {
        key: _normalize_feature_array(key, arrays[key]) for key in NUMERIC_FEATURE_SHAPES
    }
    lengths = {value.shape[0] for value in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("All G1 features must contain the same number of steps")
    episode_length = lengths.pop()
    if episode_length <= 0:
        raise ValueError("Episode must contain at least one step")

    return [
        {
            **{key: value[step].copy() for key, value in normalized.items()},
            "task": task,
        }
        for step in range(episode_length)
    ]


def _load_episode_arrays(episode_dir: Path) -> dict[str, np.ndarray]:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("The G1 converter requires the 'lmdb' Python package") from exc

    environment = lmdb.open(
        str(episode_dir / "lmdb"),
        readonly=True,
        lock=False,
        max_readers=32,
        readahead=False,
    )
    try:
        with environment.begin(write=False) as transaction:
            arrays = {}
            for key in NUMERIC_FEATURE_SHAPES:
                payload = transaction.get(key.encode("utf-8"))
                if payload is None:
                    raise KeyError(f"LMDB episode is missing required key: {key}")
                arrays[key] = pickle.loads(payload)
            return arrays
    finally:
        environment.close()


def _resolve_episode_videos(episode_dir: Path) -> dict[str, Path]:
    videos = {key: episode_dir / key / "demo.mp4" for key in _VIDEO_KEYS}
    missing = [str(path) for path in videos.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Episode is missing required RGB videos: {missing}")
    return videos


def load_episode(episode_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Load and validate one successful G1 SimBox episode."""
    episode_dir = Path(episode_dir)
    with (episode_dir / "meta_info.pkl").open("rb") as stream:
        metadata = pickle.load(stream)
    frames = build_episode_frames(
        _load_episode_arrays(episode_dir),
        task=str(metadata.get("language_instruction", "")),
    )
    if int(metadata.get("num_steps", -1)) != len(frames):
        raise ValueError(
            f"Metadata num_steps does not match LMDB arrays in {episode_dir}: "
            f"{metadata.get('num_steps')} != {len(frames)}"
        )
    image_step_ids = metadata.get("image_valid_step_ids", {})
    for key in _VIDEO_KEYS:
        step_ids = [int(value) for value in image_step_ids.get(key, [])]
        if step_ids != list(range(len(frames))):
            raise ValueError(
                f"Image steps for {key} must be ordered 0..{len(frames) - 1}, got "
                f"{step_ids[:3]}...{step_ids[-3:]}"
            )
    return frames, _resolve_episode_videos(episode_dir)


def _create_dataset_class():
    try:
        import torch
        import torchvision
        from lerobot.common.datasets.compute_stats import (
            auto_downsample_height_width,
            get_feature_stats,
            sample_indices,
        )
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.utils import (
            check_timestamps_sync,
            get_episode_data_index,
            validate_episode_buffer,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot v2.1 dependencies are unavailable. Run this converter with the "
            "pinned policy/openpi-InternData-A1 environment."
        ) from exc

    def sample_video(video_path: str) -> np.ndarray:
        reader = torchvision.io.VideoReader(video_path, stream="video")
        frames = torch.stack([frame["data"] for frame in reader]).numpy()
        selected = sample_indices(len(frames))
        sampled = [auto_downsample_height_width(frames[index]) for index in selected]
        return np.stack(sampled)

    def compute_episode_stats(episode_data: Mapping[str, Any], features: Mapping[str, Any]):
        stats = {}
        for key, data in episode_data.items():
            feature = features[key]
            if feature["dtype"] == "string":
                continue
            if feature["dtype"] == "video":
                feature_array = sample_video(data)
                axes_to_reduce = (0, 2, 3)
                keepdims = True
            else:
                feature_array = data
                axes_to_reduce = 0
                keepdims = data.ndim == 1
            feature_stats = get_feature_stats(
                feature_array,
                axis=axes_to_reduce,
                keepdims=keepdims,
            )
            if feature["dtype"] == "video":
                feature_stats = {
                    name: value if name == "count" else np.squeeze(value / 255.0, axis=0)
                    for name, value in feature_stats.items()
                }
            stats[key] = feature_stats
        return stats

    class UnitreeG1Dataset(LeRobotDataset):
        def add_frame(self, frame: Mapping[str, Any]) -> None:
            normalized_frame = {
                key: value.numpy() if isinstance(value, torch.Tensor) else value
                for key, value in frame.items()
            }
            if self.episode_buffer is None:
                self.episode_buffer = self.create_episode_buffer()
            frame_index = self.episode_buffer["size"]
            timestamp = normalized_frame.get("timestamp", frame_index / self.fps)
            self.episode_buffer["frame_index"].append(frame_index)
            self.episode_buffer["timestamp"].append(timestamp)
            for key, value in normalized_frame.items():
                if key == "timestamp":
                    continue
                if key == "task":
                    self.episode_buffer["task"].append(value)
                    continue
                if key not in self.features:
                    raise ValueError(f"Frame feature is not declared: {key}")
                self.episode_buffer[key].append(value)
            self.episode_buffer["size"] += 1

        def save_episode(
            self,
            episode_data: Mapping[str, Any] | None = None,
            videos: Mapping[str, Path] | None = None,
        ) -> None:
            if videos is None:
                raise ValueError("G1 episode videos are required")
            source_buffer = self.episode_buffer if episode_data is None else episode_data
            episode_buffer = dict(source_buffer)
            validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
            episode_length = episode_buffer.pop("size")
            tasks = episode_buffer.pop("task")
            episode_tasks = list(dict.fromkeys(tasks))
            episode_index = episode_buffer["episode_index"]
            episode_buffer["index"] = np.arange(
                self.meta.total_frames,
                self.meta.total_frames + episode_length,
            )
            episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

            for task in episode_tasks:
                if self.meta.get_task_index(task) is None:
                    self.meta.add_task(task)
            episode_buffer["task_index"] = np.array(
                [self.meta.get_task_index(task) for task in tasks]
            )
            for key, feature in self.features.items():
                if key in {"index", "episode_index", "task_index"}:
                    continue
                if feature["dtype"] == "video":
                    continue
                episode_buffer[key] = np.stack(episode_buffer[key])

            for key in self.meta.video_keys:
                if key not in videos:
                    raise KeyError(f"Missing video mapping for {key}")
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(videos[key], video_path)
                episode_buffer[key] = str(video_path)

            episode_stats = compute_episode_stats(episode_buffer, self.features)
            self._save_episode_table(episode_buffer, episode_index)
            self.meta.save_episode(episode_index, episode_length, episode_tasks, episode_stats)
            episode_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
            check_timestamps_sync(
                episode_buffer["timestamp"],
                episode_buffer["episode_index"],
                {key: value.numpy() for key, value in episode_data_index.items()},
                self.fps,
                self.tolerance_s,
            )
            if episode_data is None:
                self.episode_buffer = self.create_episode_buffer()

    return UnitreeG1Dataset


def convert_dataset(
    source_root: Path,
    save_root: Path,
    *,
    repo_id: str,
    fps: int = DEFAULT_FPS,
    num_episodes: int | None = None,
    overwrite: bool = False,
) -> int:
    """Convert G1 episodes and return the number successfully written."""
    if fps != DEFAULT_FPS:
        raise ValueError("G1 navigation episodes are currently recorded at exactly 50 FPS")
    episodes = discover_episode_dirs(source_root)
    if num_episodes is not None:
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        episodes = episodes[:num_episodes]
    if not episodes:
        raise ValueError(f"No valid G1 episodes found under {source_root}")

    save_root = Path(save_root)
    if save_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {save_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(save_root)

    dataset_class = _create_dataset_class()
    dataset = dataset_class.create(
        repo_id=repo_id,
        root=save_root,
        fps=fps,
        robot_type=DEFAULT_ROBOT_TYPE,
        features=FEATURES,
    )
    for episode_dir in episodes:
        frames, videos = load_episode(episode_dir)
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode(videos=videos)
    return len(episodes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Unitree G1 SimBox LMDB episodes to LeRobot v2.1."
    )
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/unitree-g1-navigation")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    converted = convert_dataset(
        args.src_path,
        args.save_path,
        repo_id=args.repo_id,
        fps=args.fps,
        num_episodes=args.num_episodes,
        overwrite=args.overwrite,
    )
    print(f"Converted {converted} Unitree G1 episode(s) to {args.save_path}")


if __name__ == "__main__":
    main()
