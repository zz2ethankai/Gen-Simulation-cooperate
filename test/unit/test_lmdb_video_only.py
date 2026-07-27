"""Streaming MP4-only logger tests."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import imageio
import lmdb
import numpy as np
import pytest


pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = ROOT / "workflows/simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.loggers.lmdb_logger import LmdbLogger  # noqa: E402


def _logger():
    logger = LmdbLogger(
        task_dir="task",
        language_instruction=["pick"],
        detailed_language_instruction=["pick the object"],
        collect_info="debug",
        max_size=0.001,
    )
    robot = "test_robot"
    logger.proprio_data_logger[robot] = {}
    logger.scalar_data_logger[robot] = {}
    logger.json_data_logger[robot] = {}
    logger.action_data_logger[robot] = {}
    logger.object_data_logger[robot] = {}
    return logger, robot


def test_video_only_stream_saves_mp4_without_lmdb_image_keys(tmp_path):
    logger, robot = _logger()
    key = "images.rgb.debug_global"
    for index in range(5):
        frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        logger.add_video_frame(robot, key, frame, step_idx=index)
    assert logger.color_image_logger == {}

    saved_dir = logger.save(tmp_path, "episode", save_img=True)[0]
    video_path = saved_dir / key / "demo.mp4"
    assert video_path.is_file()
    reader = imageio.get_reader(video_path)
    try:
        assert reader.count_frames() == 5
        assert tuple(reader.get_data(0).shape[:2]) == (48, 64)
        assert float(reader.get_meta_data()["fps"]) == pytest.approx(15.0)
    finally:
        reader.close()

    meta = pickle.loads((saved_dir / "meta_info.pkl").read_bytes())
    assert key not in meta["keys"]
    env = lmdb.open(str(saved_dir / "lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            keys = [raw.decode("utf-8", errors="replace") for raw, _ in txn.cursor()]
    finally:
        env.close()
    assert not any("debug_global" in item for item in keys)
    assert logger._video_temp_dir is None


def test_video_only_stream_saves_when_no_camera_json_was_recorded(tmp_path):
    logger, robot = _logger()
    logger.json_data_logger.pop(robot)
    logger.add_video_frame(
        robot,
        "images.rgb.debug_high_head",
        np.zeros((32, 32, 3), dtype=np.uint8),
    )

    saved_dir = logger.save(tmp_path, "episode", save_img=True)[0]

    assert (saved_dir / "images.rgb.debug_high_head/demo.mp4").is_file()
    assert (saved_dir / "lmdb/info.json").read_text(encoding="utf-8") == "{}"


def test_clear_closes_writer_and_removes_unsaved_temporary_mp4():
    logger, robot = _logger()
    logger.add_video_frame(
        robot, "images.rgb.debug_rear_top", np.zeros((32, 32, 3), dtype=np.uint8)
    )
    temp_dir = logger._video_temp_dir
    assert temp_dir is not None and temp_dir.is_dir()
    logger.clear(["pick"], ["pick the object"])
    assert not temp_dir.exists()
    assert logger._video_temp_dir is None
    assert logger._video_writers == {}


def test_async_save_clone_transfers_video_ownership_without_copying_ffmpeg_writer(
    tmp_path,
):
    logger, robot = _logger()
    key = "images.rgb.debug_global"
    for index in range(3):
        logger.add_video_frame(robot, key, np.full((32, 32, 3), index, dtype=np.uint8))
    temp_dir = logger._video_temp_dir
    snapshot = logger.clone_for_save()

    assert logger._video_temp_dir is None
    assert snapshot._video_temp_dir == temp_dir
    logger.clear(["next"], ["next episode"])
    assert temp_dir is not None and temp_dir.is_dir()

    saved_dir = snapshot.save(tmp_path, "episode", save_img=True)[0]
    assert (saved_dir / key / "demo.mp4").is_file()
    assert not temp_dir.exists()
