from __future__ import annotations

import math
import unittest

from nav2.bridge.clock import _MonotonicSimTime


class MonotonicSimTimeTests(unittest.TestCase):
    def test_preserves_simulation_time_before_a_reset(self):
        timeline = _MonotonicSimTime(reset_gap_sec=2.0)

        self.assertEqual(timeline.translate(0.25), 0.25)
        self.assertEqual(timeline.translate(0.75), 0.75)

    def test_inserts_forward_gap_when_simulation_time_resets(self):
        timeline = _MonotonicSimTime(reset_gap_sec=2.0)

        self.assertEqual(timeline.translate(10.0), 10.0)
        self.assertEqual(timeline.translate(10.5), 10.5)
        self.assertEqual(timeline.translate(0.25), 12.5)
        self.assertEqual(timeline.translate(0.75), 13.0)

    def test_clamps_small_floating_point_regressions(self):
        timeline = _MonotonicSimTime(reset_gap_sec=2.0)

        self.assertEqual(timeline.translate(1.0), 1.0)
        self.assertEqual(timeline.translate(1.0 - 1.0e-12), 1.0)
        self.assertEqual(timeline.translate(1.5), 1.5)

    def test_rejects_non_finite_simulation_time(self):
        timeline = _MonotonicSimTime()

        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    timeline.translate(value)


if __name__ == "__main__":
    unittest.main()
