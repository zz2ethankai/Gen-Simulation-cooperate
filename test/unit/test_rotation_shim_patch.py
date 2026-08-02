"""Regression checks for the RotationShim overlay built into the Nav2 image."""

from pathlib import Path
import unittest


class RotationShimPatchTests(unittest.TestCase):
    def test_patch_limits_terminal_speed_and_holds_within_yaw_tolerance(self):
        patch_path = Path(__file__).parents[2] / "docker/nav2/patches/0001-rotation-shim-stop-before-overshoot.patch"
        patch = patch_path.read_text(encoding="utf-8")

        self.assertIn("max_vel_to_stop", patch)
        self.assertIn("std::min(rotate_to_heading_angular_vel_, max_vel_to_stop)", patch)
        self.assertIn("plugin_name_ = name;", patch)
        self.assertIn('position_goal_checker_->initialize(parent, plugin_name_ + ".position_checker"', patch)
        self.assertIn("yaw_goal_tolerance", patch)
        self.assertIn("std::fabs(angular_distance_to_heading) <= yaw_goal_tolerance", patch)
        self.assertIn("last_angular_vel_ = 0.0", patch)


if __name__ == "__main__":
    unittest.main()
