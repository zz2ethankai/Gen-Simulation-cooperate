from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMBOX_ROOT = REPO_ROOT / "workflows" / "simbox"
if str(SIMBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBOX_ROOT))

from core.mobile.platforms import UnitreeG1Platform, get_mobile_base_platform  # noqa: E402


class UnitreeG1PlatformTest(unittest.TestCase):
    def setUp(self):
        path = REPO_ROOT / "workflows/simbox/core/configs/bases/unitree_g1_sonic.yaml"
        self.base_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_registry_selects_unitree_g1_profile(self):
        platform = get_mobile_base_platform(self.base_cfg)

        self.assertIsInstance(platform, UnitreeG1Platform)
        self.assertEqual(platform.profile_name, "unitree_g1_decoupled_wbc")

    def test_profile_exposes_configured_footprint_and_velocity_limits(self):
        platform = UnitreeG1Platform()

        footprint = platform.default_navigation_footprint_points(self.base_cfg)
        limits = platform.navigation_controller_hard_limits(self.base_cfg)

        self.assertEqual(len(footprint), 4)
        self.assertEqual(limits["max_velocity"], [0.35, 0.2, 0.5])
        self.assertEqual(limits["min_velocity"], [0.0, -0.2, -0.5])


if __name__ == "__main__":
    unittest.main()
