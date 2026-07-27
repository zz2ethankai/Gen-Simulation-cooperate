# pylint: skip-file
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import cv2
import imageio
import lmdb
import numpy as np
from core.loggers import BaseLogger
from tqdm import tqdm

DEFAULT_RGB_SCALE_FACTOR = 256000.0

def float_array_to_uint16_png(float_array):
    array = np.nan_to_num(float_array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.round(array * 10000.0)
    array = np.clip(array, 0, 65535)
    return array.astype(np.uint16)

def seg_array_to_uint16_png(seg_array):
    array = np.nan_to_num(seg_array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0, 65535)
    return array.astype(np.uint16)

# pylint: disable=line-too-long,unused-argument
class LmdbLogger(BaseLogger):
    def __init__(
        self,
        task_dir="Pick_up_the_object",
        language_instruction="Pick up the object.",
        detailed_language_instruction="Pick up the object with right gripper.",
        collect_info: str = "set1-1_collector1_20250715",
        version: str = "v1.0",
        tpi_initial_info: dict = {},
        max_size: int = 1,
        image_quality: int = 100,
        save_depth: bool = True,
        min_inttype: int = 0,
        max_inttype: int = 2**24 - 1,
    ):
        super().__init__(
            task_dir=task_dir,
            language_instruction=language_instruction,
            detailed_language_instruction=detailed_language_instruction,
            collect_info=collect_info,
            version=version,
            tpi_initial_info=tpi_initial_info,
        )
        self.max_size = int(max_size * 1024**4)
        self.image_quality = image_quality
        self.min_inttype = min_inttype
        self.max_inttype = max_inttype
        self._video_temp_dir = None
        self._video_writers = {}
        self._video_temp_paths = {}
        self._video_frame_counts = {}

    def close(self):
        self._discard_video_streams()

    @staticmethod
    def _video_stream_id(robot, key):
        return str(robot), str(key)

    def add_video_frame(self, robot, key, value, step_idx=None):
        """Stream one RGB frame to FFmpeg; frames never enter color_image_logger."""
        stream_id = self._video_stream_id(robot, key)
        if stream_id not in self._video_writers:
            if self._video_temp_dir is None:
                self._video_temp_dir = Path(tempfile.mkdtemp(prefix="simbox-video-only-"))
            safe_name = "__".join(stream_id).replace("/", "_").replace(os.sep, "_")
            temp_path = self._video_temp_dir / f"{safe_name}.mp4"
            self._video_temp_paths[stream_id] = temp_path
            self._video_writers[stream_id] = imageio.get_writer(
                str(temp_path),
                fps=15,
                codec="libx264",
                macro_block_size=None,
            )
            self._video_frame_counts[stream_id] = 0

        frame = np.asarray(value)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"video-only RGB frame must have shape HxWx3/4, got {frame.shape}")
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._video_writers[stream_id].append_data(frame)
        self._video_frame_counts[stream_id] += 1

    def _close_video_writers(self):
        first_error = None
        for writer in self._video_writers.values():
            try:
                writer.close()
            except Exception as exc:  # keep closing the remaining FFmpeg processes
                first_error = first_error or exc
        self._video_writers = {}
        if first_error is not None:
            raise first_error

    def _reset_video_stream_state(self):
        self._video_temp_dir = None
        self._video_writers = {}
        self._video_temp_paths = {}
        self._video_frame_counts = {}

    def _discard_video_streams(self):
        close_error = None
        try:
            self._close_video_writers()
        except Exception as exc:
            close_error = exc
        if self._video_temp_dir is not None:
            shutil.rmtree(self._video_temp_dir, ignore_errors=True)
        self._reset_video_stream_state()
        if close_error is not None:
            raise close_error

    def _finalize_video_streams(self, save_dirs_by_robot):
        self._close_video_writers()
        for stream_id, temp_path in self._video_temp_paths.items():
            robot_name, key = stream_id
            save_dir = save_dirs_by_robot.get(robot_name)
            if save_dir is None:
                raise ValueError(f"video-only stream has no episode directory for robot {robot_name}")
            output_dir = save_dir / key
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(output_dir / "demo.mp4"))
        if self._video_temp_dir is not None:
            shutil.rmtree(self._video_temp_dir, ignore_errors=True)
        self._reset_video_stream_state()

    def clone_for_save(self):
        """Deep-copy buffered data while transferring closed MP4 streams to the save snapshot."""
        from copy import deepcopy

        self._close_video_writers()
        snapshot = deepcopy(self)
        snapshot._video_temp_dir = self._video_temp_dir
        snapshot._video_temp_paths = dict(self._video_temp_paths)
        snapshot._video_frame_counts = dict(self._video_frame_counts)
        snapshot._video_writers = {}
        self._reset_video_stream_state()
        return snapshot

    def clear(
        self,
        language_instruction="Pick up the object.",
        detailed_language_instruction="Pick up the object with right gripper.",
    ):
        self._discard_video_streams()
        super().clear(language_instruction, detailed_language_instruction)

    def save(self, this_save_dir, timestamp: str, save_img: bool = True):
        saved_dirs = []
        save_dirs_by_robot = {}
        self._close_video_writers()
        for robot_idx, robot_name in enumerate(self.proprio_data_logger.keys()):
            save_dir = Path(this_save_dir)
            save_dir = save_dir / f"{robot_name}" / f"{self.task_dir}" / f"{self.collect_info}" / f"{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_dirs.append(save_dir)
            save_dirs_by_robot[robot_name] = save_dir

            # Meta info
            meta_info = {}
            meta_info["keys"] = {}
            meta_info["max_size"] = self.max_size
            meta_info["language_instruction"] = self.language_instruction[robot_idx]
            meta_info["detailed_language_instruction"] = self.detailed_language_instruction[robot_idx]
            print("language_instruction :", meta_info["language_instruction"])
            print("detailed_language_instruction :", meta_info["detailed_language_instruction"])
            meta_info["tpi_initial_info"] = self.tpi_initial_info
            meta_info["collect_info"] = self.collect_info
            meta_info["version"] = self.version
            meta_info["image_valid_step_ids"] = {}

            # Lmdb
            log_path_lmdb = save_dir / "lmdb"
            lmdb_env = lmdb.open(str(log_path_lmdb), map_size=self.max_size)
            txn = lmdb_env.begin(write=True)

            # Save json data
            robot_json_data = self.json_data_logger.get(robot_name, {})
            with open(log_path_lmdb / "info.json", "w") as f:
                json.dump(robot_json_data, f)
            txn.put("json_data".encode("utf-8"), pickle.dumps(robot_json_data))
            meta_info["keys"]["json_data"] = ["json_data".encode("utf-8")]

            # Save scalar data
            meta_info["keys"]["scalar_data"] = []
            for key, value in self.scalar_data_logger.get(robot_name, {}).items():
                txn.put(key.encode("utf-8"), pickle.dumps(value))
                meta_info["keys"]["scalar_data"].append(key.encode("utf-8"))

            # Save proprios
            meta_info["keys"]["proprio_data"] = []
            for key, value in self.proprio_data_logger[robot_name].items():
                txn.put(key.encode("utf-8"), pickle.dumps(value))
                meta_info["keys"]["proprio_data"].append(key.encode("utf-8"))

            # Save objects
            meta_info["keys"]["object_data"] = []
            if robot_name in self.object_data_logger:
                for key, value in self.object_data_logger[robot_name].items():
                    if "robotiq" in robot_name and key == "states.gripper.position":
                        value = [self.action_data_logger[robot_name]["master_actions.gripper.position"][0]] + (
                            self.action_data_logger[robot_name]["master_actions.gripper.position"]
                        )[:-1]
                    txn.put(key.encode("utf-8"), pickle.dumps(value))
                    meta_info["keys"]["object_data"].append(key.encode("utf-8"))

            # Save master actions
            meta_info["keys"]["action_data"] = []
            for key, value in self.action_data_logger.get(robot_name, {}).items():
                # Update gripper action
                if "gripper.position" in key:
                    value.pop(0)
                    value.append(value[-1])  # Use next gripper width as gripper action label

                txn.put(key.encode("utf-8"), pickle.dumps(value))
                meta_info["keys"]["action_data"].append(key.encode("utf-8"))

            # Save actions
            # Here we use next robot state as action
            for key, value in self.proprio_data_logger[robot_name].items():
                if "states." in key:
                    new_key = key.replace("states.", "actions.")
                    value.pop(0)
                    value.append(value[-1])  # Use next robot state as action
                    txn.put(new_key.encode("utf-8"), pickle.dumps(value))
                    meta_info["keys"]["action_data"].append(new_key.encode("utf-8"))

            if (
                "split_aloha" in robot_name
                or "lift2" in robot_name
                or "azure_loong" in robot_name
                or "genie" in robot_name
            ):
                left_gripper_openness = self.action_data_logger[robot_name]["master_actions.left_gripper.openness"]
                right_gripper_openness = self.action_data_logger[robot_name]["master_actions.right_gripper.openness"]

                txn.put("actions.left_gripper.openness".encode("utf-8"), pickle.dumps(left_gripper_openness))
                meta_info["keys"]["action_data"].append("actions.left_gripper.openness".encode("utf-8"))
                txn.put("actions.right_gripper.openness".encode("utf-8"), pickle.dumps(right_gripper_openness))
                meta_info["keys"]["action_data"].append("actions.right_gripper.openness".encode("utf-8"))
            elif "franka" in robot_name:
                gripper_openness = self.action_data_logger[robot_name]["master_actions.gripper.openness"]
                txn.put("actions.gripper.openness".encode("utf-8"), pickle.dumps(gripper_openness))
                meta_info["keys"]["action_data"].append("actions.gripper.openness".encode("utf-8"))

            # Save color images
            if save_img:
                for key, value in self.color_image_logger.get(robot_name, {}).items():
                    root_img_path = save_dir / f"{key}"
                    root_img_path.mkdir(parents=True, exist_ok=True)

                    step_ids = self.color_image_step_logger.get(robot_name, {}).get(key, [])
                    if len(step_ids) != len(value):
                        step_ids = list(range(len(value)))
                    else:
                        step_ids = [int(x) for x in step_ids]
                    meta_info["image_valid_step_ids"][key] = step_ids

                    meta_info["keys"][key] = []
                    for i, image in enumerate(tqdm(value)):
                        step_id = str(step_ids[i]).zfill(4)
                        txn.put(
                            f"{key}/{step_id}".encode("utf-8"),
                            pickle.dumps(cv2.imencode(".jpg", image.astype(np.uint8))[1]),
                        )
                        meta_info["keys"][key].append(f"{key}/{step_id}".encode("utf-8"))

                    imageio.mimsave(os.path.join(root_img_path, "demo.mp4"), value, fps=15)

                for key, value in self.depth_image_logger.get(robot_name, {}).items():
                    root_img_path = save_dir / f"{key}"
                    root_img_path.mkdir(parents=True, exist_ok=True)

                    step_ids = self.depth_image_step_logger.get(robot_name, {}).get(key, [])
                    if len(step_ids) != len(value):
                        step_ids = list(range(len(value)))
                    else:
                        step_ids = [int(x) for x in step_ids]
                    meta_info["image_valid_step_ids"][key] = step_ids

                    meta_info["keys"][key] = []
                    for i, image in enumerate(tqdm(value)):
                        step_id = str(step_ids[i]).zfill(4)
                        depth_image = float_array_to_uint16_png(np.asarray(image))
                        txn.put(
                            f"{key}/{step_id}".encode('utf-8'),
                            pickle.dumps(cv2.imencode('.png', depth_image)[1])
                        )
                        meta_info["keys"][key].append(f"{key}/{step_id}".encode('utf-8'))

                for key, value in self.seg_image_logger.get(robot_name, {}).items():
                    root_img_path = save_dir / f"{key}"
                    root_img_path.mkdir(parents=True, exist_ok=True)

                    step_ids = self.seg_image_step_logger.get(robot_name, {}).get(key, [])
                    if len(step_ids) != len(value):
                        step_ids = list(range(len(value)))
                    else:
                        step_ids = [int(x) for x in step_ids]
                    meta_info["image_valid_step_ids"][key] = step_ids

                    meta_info["keys"][key] = []
                    for i, image in enumerate(tqdm(value)):
                        step_id = str(step_ids[i]).zfill(4)
                        seg_image = seg_array_to_uint16_png(np.asarray(image))
                        txn.put(
                            f"{key}/{step_id}".encode('utf-8'),
                            pickle.dumps(cv2.imencode('.png', seg_image)[1])
                        )
                        meta_info["keys"][key].append(f"{key}/{step_id}".encode('utf-8'))

            meta_info["num_steps"] = self.log_num_steps
            txn.commit()
            lmdb_env.close()
            pickle.dump(meta_info, open(os.path.join(save_dir, "meta_info.pkl"), "wb"))
        if save_img:
            self._finalize_video_streams(save_dirs_by_robot)
        else:
            self._discard_video_streams()
        return saved_dirs

    def dump(self):
        logger_info = {}
        logger_info["proprio_data_logger"] = self.proprio_data_logger
        logger_info["max_size"] = self.max_size
        logger_info["language_instruction"] = self.language_instruction
        logger_info["detailed_language_instruction"] = self.detailed_language_instruction
        logger_info["tpi_initial_info"] = self.tpi_initial_info
        logger_info["collect_info"] = self.collect_info
        logger_info["version"] = self.version
        logger_info["json_data_logger"] = self.json_data_logger
        logger_info["scalar_data_logger"] = self.scalar_data_logger
        logger_info["object_data_logger"] = self.object_data_logger
        logger_info["action_data_logger"] = self.action_data_logger
        logger_info["log_num_steps"] = self.log_num_steps

        return pickle.dumps(logger_info)

    def dedump(self, ser):
        logger_info = pickle.loads(ser)
        self.proprio_data_logger = logger_info["proprio_data_logger"]
        self.max_size = logger_info["max_size"]
        self.language_instruction = logger_info["language_instruction"]
        self.detailed_language_instruction = logger_info["detailed_language_instruction"]
        self.tpi_initial_info = logger_info["tpi_initial_info"]
        self.collect_info = logger_info["collect_info"]
        self.version = logger_info["version"]
        self.json_data_logger = logger_info["json_data_logger"]
        self.scalar_data_logger = logger_info["scalar_data_logger"]
        self.object_data_logger = logger_info["object_data_logger"]
        self.action_data_logger = logger_info["action_data_logger"]
        self.log_num_steps = logger_info["log_num_steps"]

        return True
