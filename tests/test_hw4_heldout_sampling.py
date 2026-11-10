"""Bounded HW4 heldout evaluation must not silently masquerade as full-test CE."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import wordnet31_50epochs as runner


class HW4HeldoutSamplingTests(unittest.TestCase):
    def test_explicit_sample_uses_exact_requested_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"test.txt"
            path.write_bytes(b"acbde\n"*500)
            seen=[]
            def fake_run(args,timeout):
                seen.append(Path(args[-1]).read_bytes())
                return f"UNIFIED_EVAL bytes={len(seen[-1])-1} ce_nats=3.1 accuracy=0.2", 0.01
            with patch.object(runner,"run",side_effect=fake_run):
                result=runner.evaluate(Path(td)/"other-universe",path,128)
            self.assertEqual(len(seen[0]),128)
            self.assertEqual(result["coverage"],"prefix_sample")
            self.assertEqual(result["sample_bytes"],128)
            self.assertEqual(result["heldout_full_bytes"],3000)
            self.assertEqual(result["ce_nats"],3.1)
            self.assertEqual(result["tokens"],127)

    def test_full_test_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"test.txt"
            path.write_bytes(b"full corpus reference\n")
            seen=[]
            def fake_run(args,timeout):
                seen.append(Path(args[-1]).read_bytes())
                return "UNIFIED_EVAL bytes=21 ce_nats=4.2 accuracy=0.1",0.02
            with patch.object(runner,"run",side_effect=fake_run):
                full=runner.evaluate(Path(td)/"other",path,0)
            self.assertEqual(full["coverage"],"full")
            self.assertEqual(full["heldout_full_bytes"],len(seen[0]))
            self.assertEqual(full["sample_bytes"],len(seen[0]))

    def test_out_of_bounds_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"test.txt"
            path.write_bytes(b"data\n")
            for bad in (-1,1000001):
                with self.assertRaises(ValueError):
                    runner.evaluate(Path(td)/"other",path,bad)

if __name__=="__main__":
    unittest.main()
