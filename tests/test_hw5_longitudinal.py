"""Guardrails for the HW5 longitudinal continuation (no downloads required)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw5_longitudinal import better, check_update, marker

class LongitudinalAuditTest(unittest.TestCase):
    def test_best_checkpoint_requires_real_dev_improvement(self):
        self.assertFalse(better(2.80, 2.80))
        self.assertFalse(better(2.799995, 2.80))
        self.assertTrue(better(2.79, 2.80))
        self.assertFalse(better(float("nan"), 2.8))

    def test_accept_requires_loss_and_root_change(self):
        check_update(3.0, 2.9, True, 0, "a", "b")
        for args in [(3., 3., True, 0, "a", "b"),
                     (3., 2.9, True, 0, "a", "a"),
                     (3., 2.9, True, 2, "a", "b"),
                     (3., 2.9, False, 2, "a", "b")]:
            with self.assertRaises(ValueError):
                check_update(*args)
        check_update(3.0, 3.0, False, 2, "a", "a")

    def test_native_metric_marker_is_unique(self):
        x = marker("HW5_FULL_GEOMETRIC_PASS input_bytes=100 transitions=99", "HW5_FULL_GEOMETRIC_PASS")
        self.assertEqual(x["transitions"], "99")
        with self.assertRaises(ValueError):
            marker("OTHER no=1", "HW5_FULL_GEOMETRIC_PASS")
        with self.assertRaises(ValueError):
            marker("HW5_FULL_GEOMETRIC_PASS a=1\nHW5_FULL_GEOMETRIC_PASS a=2",
                   "HW5_FULL_GEOMETRIC_PASS")

if __name__ == "__main__":
    unittest.main()
