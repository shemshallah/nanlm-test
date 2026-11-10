#!/usr/bin/env python3
"""Native full pair-plane + persistent root regression tests (no Python trainer)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT_BYTES = 8_388_608
N = 255
PAIR_COUNT = 32640


def pair_index(a: int, b: int) -> int:
    if a == b:
        raise ValueError("fixed points use the head plane")
    if a > b:
        a, b = b, a
    return a * (511 - a) // 2 + b - a - 1


def run(*args: str) -> dict[str, str]:
    p = subprocess.run(["./hw5-fullpair", *args],
                       text=True, capture_output=True)
    if p.returncode:
        raise AssertionError(f"native failed: {p.stdout}\n{p.stderr}")
    line = next((l for l in p.stdout.splitlines()
                 if l.startswith("HW5_FULLPAIR_")), None)
    if not line:
        raise AssertionError("missing native measurement")
    return dict(re.findall(r"([a-z_0-9]+)=([^\s]+)", line))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FullPairNative(unittest.TestCase):
    def test_bijection_fills_every_physical_pair(self) -> None:
        touched = set()
        for a in range(N):
            for b in range(N):
                i = pair_index(a, 255 if a == b else b)
                self.assertGreaterEqual(i, 0)
                self.assertLess(i, PAIR_COUNT)
                touched.add(i)
        self.assertEqual(len(touched), PAIR_COUNT)
        # Every unordered off-diagonal pair stores two independent lanes.
        self.assertNotEqual(pair_index(1, 2), pair_index(1, 255))
        self.assertEqual(pair_index(1, 2), pair_index(2, 1))

    def test_native_train_reopen_and_eval_does_not_change_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            root = directory / "student.root"
            train = directory / "gutenberg-fixture.txt"
            dev = directory / "new-author.txt"
            train.write_bytes((b"Alice sees a rabbit. The rabbit sees Alice.\n"
                               b"The library of strange doors is open.\n") * 25)
            dev.write_bytes((b"The rabbit reads a strange book today.\n") * 12)
            result = run("train", str(root), str(train), "64")
            self.assertEqual(int(result["physical_pairs"]), PAIR_COUNT)
            self.assertEqual(int(result["input_bytes"]), train.stat().st_size)
            self.assertGreater(int(result["observed_pair_cells"]), 0)
            self.assertLess(int(result["observed_pair_cells"]), PAIR_COUNT)
            self.assertLess(float(result["ce_after"]),
                            float(result["ce_before"]))
            self.assertEqual(int(result["root_bytes"]), ROOT_BYTES)
            self.assertEqual(root.stat().st_size, ROOT_BYTES)
            with root.open("rb") as stream:
                stream.seek(255 * 128 + 112)
                self.assertEqual(stream.read(8), b"PAIRC256")
                self.assertEqual(stream.read(8), b"HW0NAN01")
                stream.seek(32768 + pair_index(ord("A"), 255) * 256 + 31 * 4)
                value = struct.unpack("<f", stream.read(4))[0]
                self.assertTrue(abs(value) > 1e-7)
            before = sha(root)
            evaluation = run("eval", str(root), str(dev))
            self.assertEqual(int(evaluation["pair_plane"]), 1)
            self.assertEqual(int(evaluation["physical_pairs"]), PAIR_COUNT)
            self.assertEqual(before, sha(root), "reopen/eval destroyed pair weights")
            again = run("eval", str(root), str(train))
            self.assertAlmostEqual(float(result["ce_after"]),
                                   float(again["ce_nats"]), places=6)
            self.assertEqual(before, sha(root))
            error = subprocess.run(
                ["./hw5-fullpair", "train", str(root), str(train), "64"],
                capture_output=True, text=True)
            self.assertNotEqual(error.returncode, 0)
            self.assertEqual(before, sha(root))

    def test_reserved_ff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            t = Path(temp)
            sample = t / "invalid.txt"
            sample.write_bytes(b"a" * 100 + b"\xff")
            p = subprocess.run(
                ["./hw5-fullpair", "train", str(t / "root"), str(sample), "64"],
                capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("0xFF", p.stderr)


if __name__ == "__main__":
    unittest.main()
