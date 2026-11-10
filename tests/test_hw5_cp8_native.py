"""Native HW5:0 geometric CP8 training invariants and reproducibility."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw5_cp8_experiment import check_root_preserves_memory, parse_scores, parse_regimes


def call(*args):
    return subprocess.run(list(map(str, args)), check=False, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


class GeometricCP8Tests(unittest.TestCase):
    def test_train_only_native_credit_is_reproducible_and_preserves_mirror(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            docs = [(b"abracadabra magic word. \n" * 18),
                    (b"language models predict next bytes. \n" * 12)]
            corpus = base / "train.txt"
            corpus.write_bytes(b"".join(docs))
            lengths = base / "lengths.txt"
            lengths.write_text("".join(f"{len(d)}\n" for d in docs))
            bank = base / "train-only.root"
            triples = base / "exact-triples.bin"
            b = call("./hw5-context", "build", corpus, lengths, bank, triples)
            self.assertEqual(b.returncode, 0, b.stdout)
            before_root = hashlib.sha256(bank.read_bytes()).digest()
            before_triples = hashlib.sha256(triples.read_bytes()).digest()
            a = call("./hw5-cp8", "fit-cp", bank, triples, base / "factors", 2, 1234, 64)
            self.assertEqual(a.returncode, 0, a.stdout)
            self.assertEqual(a.stdout.count("HW5_CP8_TRAIN epoch="), 2)
            root1 = base / "factors.e01.root"
            root2 = base / "factors.e02.root"
            self.assertEqual(root1.stat().st_size, 8388608)
            self.assertEqual(root2.stat().st_size, 8388608)
            self.assertEqual(len(check_root_preserves_memory(bank, root2)), 64)
            self.assertNotEqual(root1.read_bytes(), root2.read_bytes())
            self.assertEqual(before_root, hashlib.sha256(bank.read_bytes()).digest())
            self.assertEqual(before_triples, hashlib.sha256(triples.read_bytes()).digest())
            rerun = call("./hw5-cp8", "fit-cp", bank, triples, base / "again", 2, 1234, 64)
            self.assertEqual(rerun.returncode, 0, rerun.stdout)
            self.assertEqual(root2.read_bytes(), (base / "again.e02.root").read_bytes())

    def test_native_gate_preserves_seen_predictions_and_mass(self):
        result = call("./hw5-cp8", "gate-selftest")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("known_probability_immutable=1", result.stdout)
        self.assertIn("unknown_mass_conserved=1", result.stdout)

    def test_native_rejects_invalid_smoothing(self):
        p = call("./hw5-cp8", "fit-cp", "/nonexistent", "/nonexistent",
                 "/nonexistent", 50, 1, 0)
        self.assertNotEqual(p.returncode, 0)

    def test_frozen_score_parser_and_native_origin(self):
        line = ("HW5_CP8_SCORE mode=cp8 beta=16 rho=0.03 scale=1 gate=1 "
                "ce_nats=2.12 accuracy=0.37 transitions=654321 "
                "exact_model=HW5_train_only bptt_attempted=0")
        result = parse_scores(line)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["transitions"], 654321)
        self.assertAlmostEqual(result[0]["rho"], 0.03)
        self.assertEqual(result[0]["gate"], 1)
        with self.assertRaises(ValueError):
            parse_scores(line.replace("bptt_attempted=0", "bptt_attempted=1"))
        with self.assertRaises(ValueError):
            parse_scores("incomplete")
        regimes = " ".join([
            "HW5_CP8_REGIME regime=seen_exact_triple transitions=10 ce_nats=2.0 accuracy=0.5",
            "exact_model=HW5_train_only bptt_attempted=0"])
        with self.assertRaises(ValueError):
            parse_regimes(regimes)  # needs all three regimes

if __name__ == "__main__":
    unittest.main()
