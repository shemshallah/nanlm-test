"""Proof that mathematical sensor recordings are aligned before comparison."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"tools"))
from matrix_probe_bank import run
from compare_math_profile_outputs import analyze


class MatrixReportTests(unittest.TestCase):
    def test_profiles_share_clock_and_are_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"five"
            run(BASE/"tests/fixtures/data.noun",out,epochs=2,every=4)
            report=analyze(out)
            self.assertTrue(report["all_aligned"])
            self.assertEqual(report["ticks_per_profile"],18)
            self.assertEqual(len(report["profiles"]),5)
            self.assertEqual(len(report["global_correlations"]),10)
            self.assertEqual(len(report["pair_metric_correlations"]),10)
            self.assertEqual(len({report["profiles"][p]["per_channel"]["global"]["mean"]
                                  for p in report["profiles"]})>1,True)
            self.assertTrue(all(report["profiles"][p]["pair_metric_samples"]>0
                                for p in report["profiles"]))
            self.assertGreater(report["profiles"]["information"]["per_channel"]["entropy"]["max"],0)

    def test_rejects_missing_profile(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                analyze(Path(td))


if __name__=="__main__":
    unittest.main()
