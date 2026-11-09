"""Regression tests for the blank-root, tensor-only high-temporal-resolution experiment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "tools"))
import temporal_c256_sensor as sensor


def frame(payload, tick, request_id=1, source=4, dest=0, kind=b"wordnet-u32"):
    word = (0x7ff8).to_bytes(2, "big") + payload.to_bytes(4, "little") + (0xfff8).to_bytes(2, "little")
    body = (b"NPF1" + bytes((1, source, dest, len(kind))) +
            struct.pack(">QQ", request_id, tick) + bytes(32) + word + kind)
    return body + hashlib.sha256(body).digest()


def document(payloads):
    items = []
    for i, p in enumerate(payloads):
        raw = frame(p, i+1, i+10)
        items.append({"schema": "nanrouter.zero-point-frame.v1", "frame_hex": raw.hex()})
    return {"schema": "nanrouter.zero-point-drain.v1", "endpoint": "HW0:0",
            "count": len(items), "frames": items}


class TemporalTensorSensorTests(unittest.TestCase):
    def test_geometry_and_native_frame_verification(self):
        self.assertEqual(sensor.ROOT_BYTES, 8388608)
        self.assertEqual(sensor.pair_index(0, 1), 0)
        self.assertEqual(sensor.pair_index(254, 255), 32639)
        self.assertEqual(sensor.physical_pair(254, 255), 8388352)
        self.assertEqual(sensor.parse_frame(frame(1930, 7)), (7, 1930))
        self.assertEqual(sensor.parse_drain(document([1740, 1930])),
                         [(1, 1740), (2, 1930)])
        corrupted = bytearray(frame(1930, 7))
        corrupted[58] ^= 1
        with self.assertRaisesRegex(ValueError, "digest"):
            sensor.parse_frame(bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "not HW4"):
            sensor.parse_frame(frame(1930, 7, source=5))
        with self.assertRaisesRegex(ValueError, "not HW4"):
            sensor.parse_frame(frame(1930, 7, dest=4))

    def test_fresh_root_tensor_only_high_resolution(self):
        # These are numeric IDs from a small WordNet-format fixture, NOT a
        # live HW4 corpus or proof of access to the deployed WordNet matrix.
        payloads = [1740, 1930, 2137, 2345, 2500, 1740, 2137, 1930, 2500, 2345]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            feed = directory / "nannet_fixture.json"
            feed.write_text(json.dumps(document(payloads)))
            root = directory / "blank.root"
            tensor = directory / "slices.ntc256"
            cmd = [sys.executable, str(BASE/"tools"/"temporal_c256_sensor.py"),
                   "--nannet-json", str(feed), "--root", str(root),
                   "--tensor", str(tensor)]
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout)
            self.assertEqual(result["frames"], len(payloads))
            self.assertEqual(result["root_bytes"], 8388608)
            self.assertEqual(root.stat().st_size, 8388608)
            self.assertGreater(result["tensor_entries"], len(payloads)*600)
            self.assertGreater(result["active_mirror_pairs"], 0)
            self.assertEqual(result["source"], "nannet-verified-fixture")
            self.assertEqual(result["tensor_record_bytes"], sensor.RECORD.size)
            # Reserved network head remains blank; no WordNet data is stored.
            with root.open("rb") as f:
                f.seek(255*128)
                self.assertEqual(f.read(128), bytes(128))
            raw = tensor.read_bytes()
            self.assertEqual(raw[:8], sensor.MAGIC)
            self.assertEqual(struct.unpack_from("<I", raw, 8)[0], 1)
            entries = zlib.decompress(raw[12:])
            self.assertEqual(len(entries) % sensor.RECORD.size, 0)
            self.assertEqual(len(entries)//sensor.RECORD.size, result["tensor_entries"])
            decoded = [sensor.RECORD.unpack_from(entries, k)
                       for k in range(0, len(entries), sensor.RECORD.size)]
            self.assertEqual({x[1] for x in decoded}, set(range(1, 11)))
            self.assertEqual({x[0] for x in decoded}, set(range(1, 11)))
            self.assertEqual({x[2] for x in decoded}, set(range(len(sensor.CHANNELS))))
            pairs = {}
            for tick, seq, channel, a, b, value in decoded:
                self.assertGreaterEqual(a, 0)
                self.assertLessEqual(a, 255)
                self.assertTrue(math_isfinite(value))
                if channel in (sensor.CHANNEL["mirror_forward"], sensor.CHANNEL["mirror_reverse"]):
                    pairs.setdefault(seq, []).append((channel, a, b, value))
            self.assertGreater(len(pairs), 0)
            for pair_samples in pairs.values():
                self.assertEqual(len(pair_samples), 2)
                forward = next(v for c,a,b,v in pair_samples if c == sensor.CHANNEL["mirror_forward"])
                reverse = next(v for c,a,b,v in pair_samples if c == sensor.CHANNEL["mirror_reverse"])
                self.assertAlmostEqual(forward, -reverse, places=6)
            # A fresh-root requirement protects existing artifacts from overwrite.
            rerun = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("exists", rerun.stderr)

    def test_empty_hw4_drain_fails_without_fabricated_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            feed = directory / "empty.json"
            feed.write_text(json.dumps(document([])))
            cmd = [sys.executable, str(BASE/"tools"/"temporal_c256_sensor.py"),
                   "--nannet-json", str(feed), "--root", str(directory/"root"),
                   "--tensor", str(directory/"tensor")]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("no WordNet frames", run.stderr)
            self.assertFalse((directory/"root").exists())

    def test_live_path_requires_authorization(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TOKEN"):
                sensor.fetch_live_drain("https://example.org")
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                sensor.fetch_live_drain("http://example.org")


def math_isfinite(x):
    import math
    return math.isfinite(x)


if __name__ == "__main__":
    unittest.main()
