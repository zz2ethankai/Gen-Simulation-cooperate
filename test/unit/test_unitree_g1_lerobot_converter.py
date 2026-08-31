from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "policy"
    / "lmdb2lerobotv21"
    / "lmdb2lerobot_unitree_g1_a1.py"
)


def _load_converter_module():
    spec = importlib.util.spec_from_file_location("unitree_g1_lerobot_converter", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load converter module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnitreeG1LeRobotConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = _load_converter_module()

    def _episode_arrays(self, length: int = 3):
        arrays = {}
        for index, (key, shape) in enumerate(self.converter.NUMERIC_FEATURE_SHAPES.items()):
            value_shape = (length, *shape) if shape else (length,)
            arrays[key] = np.full(value_shape, index + 0.25, dtype=np.float64)
        return arrays

    def test_feature_schema_preserves_all_control_layers(self):
        expected_keys = {
            "images.rgb.ego",
            "images.rgb.global",
            "states.body_joint.position",
            "states.body_joint.velocity",
            "states.base.position",
            "states.base.orientation",
            "qvel",
            "base_actions.vx_body",
            "base_actions.vy_body",
            "base_actions.wz_body",
            "base_actions.locomotion_mode",
            "master_actions.body_joint.position",
            "master_actions.body_joint.velocity",
            "master_actions.body_joint.effort",
            "actions.body_joint.position",
            "actions.body_joint.velocity",
            "actions.base.position",
            "actions.base.orientation",
            "camera2env_pose.ego",
            "camera2env_pose.global",
        }

        self.assertEqual(set(self.converter.FEATURES), expected_keys)
        self.assertEqual(
            self.converter.FEATURES["master_actions.body_joint.position"]["shape"],
            (29,),
        )
        self.assertEqual(
            self.converter.FEATURES["camera2env_pose.ego"]["shape"],
            (16,),
        )

    def test_build_episode_frames_preserves_numeric_values(self):
        arrays = self._episode_arrays()

        frames = self.converter.build_episode_frames(
            arrays,
            task="Walk the Unitree G1 forward.",
        )

        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0]["task"], "Walk the Unitree G1 forward.")
        self.assertEqual(frames[0]["base_actions.vx_body"].shape, (1,))
        self.assertEqual(frames[0]["camera2env_pose.ego"].shape, (16,))
        self.assertEqual(frames[0]["states.body_joint.position"].dtype, np.float32)
        np.testing.assert_allclose(
            frames[2]["master_actions.body_joint.position"],
            arrays["master_actions.body_joint.position"][2],
        )

    def test_build_episode_frames_rejects_length_mismatch(self):
        arrays = self._episode_arrays()
        arrays["base_actions.vx_body"] = arrays["base_actions.vx_body"][:-1]

        with self.assertRaisesRegex(ValueError, "same number of steps"):
            self.converter.build_episode_frames(arrays, task="Walk forward.")

    def test_discover_episode_dirs_requires_lmdb_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_episode = root / "valid"
            (valid_episode / "lmdb").mkdir(parents=True)
            (valid_episode / "lmdb" / "data.mdb").touch()
            (valid_episode / "meta_info.pkl").touch()
            invalid_episode = root / "invalid"
            invalid_episode.mkdir()
            (invalid_episode / "meta_info.pkl").touch()

            episodes = self.converter.discover_episode_dirs(root)

        self.assertEqual(episodes, [valid_episode])


if __name__ == "__main__":
    unittest.main()
