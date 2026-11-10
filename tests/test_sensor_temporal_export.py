#!/usr/bin/env python3
"""Contract tests for the native MATS/NSF1 temporal export format."""
import contextlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sensor_temporal_export as exp  # noqa: E402

SCRIPT = Path(exp.__file__)


def row(identifier=255, heads=(255, 255, 255), v=(0.0, 0.0, 0.0),
        tensor=None, support=0.0, refute=0.0):
    if tensor is None:
        tensor = (0.0,) * 9
    return struct.pack("<4B18f", identifier, *heads, *v, *tensor,
                       0.0, support, refute, 0.0, 0.0, 0.0)


def encoded_frame(tick, global_v=(0., 0., 0.), global_q=None,
                  heads=(), triads=(), hot=(), links=()):
    payload = bytearray(struct.pack("<4sHHQII4H", b"MATS", 1, 0,
                                    tick, 2, 3,
                                    len(heads), len(triads), len(hot), len(links)))
    for part in list(heads) + list(triads) + list(hot) + list(links):
        payload.extend(part)
    payload.extend(row())
    payload.extend(row(v=global_v, tensor=global_q))
    return bytes(payload)


def edge(frm, to, flux):
    return struct.pack("<4BHHIfffH", 4, 5, frm, to,
                       exp.address(frm, to), exp.address(to, frm),
                       32768 + 256 * exp.pair_index(frm, to),
                       .6, .5, flux, 2)


def make_sidecar(path, frames, size=65536, corrupt_sequence=None):
    data = bytearray(size)
    header = struct.pack("<8sIIII", b"NLSLICE1", 1, size, 64, 64)
    data[:len(header)] = header
    offset = 64
    for seq, payload in enumerate(frames, 1):
        tick = struct.unpack_from("<Q", payload, 8)[0]
        record = struct.pack("<4sIQQII", b"NSF1", 1, seq, tick,
                             len(payload), exp.checksum(payload, seq, tick)) + payload
        if offset + len(record) > size:
            offset = 64
        # Simulate the writer's actual behavior: stale aligned records
        # outside the overwritten byte span remain in the ring.
        data[offset:offset + len(record)] = record
        if seq == corrupt_sequence:
            data[offset + 36] ^= 0x55
        offset = (offset + len(record) + 63) & ~63
        if offset >= size:
            offset = 64
    path.write_bytes(data)


class TemporalExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "recording.slices"

    def test_export_exact_ieee_float_vectors_and_full_tensors(self):
        q = (1., 2., 3., 4., 5., 6., 7., 8., 9.)
        head = row(42, (42, 255, 255), (3., -2., 1.), q, 2., 1.)
        frame = encoded_frame(12, (1., 2., 3.), q, heads=(head,))
        make_sidecar(self.path, [frame])
        loaded, stats = exp.load(self.path)
        self.assertEqual(stats["rejected_candidate_frames"], 0)
        self.assertEqual(loaded[0]["nanclock"], 12)
        samples = exp.measurements(loaded)
        self.assertEqual(len(samples), 3)  # head, hotfield, global
        self.assertEqual(samples[0]["heads"], (42, 255, 255))
        self.assertEqual(samples[0]["tensor"], q)
        self.assertEqual(samples[0]["support"], 2.0)
        self.assertEqual(samples[0]["refute"], 1.0)
        self.assertEqual(samples[-1]["vector"], (1., 2., 3.))

    def test_finite_difference_uses_real_nanclock_gap(self):
        q0 = (0.,) * 9
        q1 = (1.,) * 9
        make_sidecar(self.path, [
            encoded_frame(1, (1., 0., 0.), q0),
            encoded_frame(5, (0., 1., 0.), q1),
        ])
        frames, _ = exp.load(self.path)
        changes = exp.delta_rows(frames)
        global_change = next(c for c in changes if c["level"] == "GLOBAL")
        self.assertEqual(global_change["delta_ticks"], 4)
        self.assertAlmostEqual(global_change["vector_delta_norm"], math.sqrt(2))
        self.assertAlmostEqual(global_change["tensor_delta_fro"], 3.)
        self.assertAlmostEqual(global_change["vector_speed_per_tick"], math.sqrt(2) / 4)
        self.assertAlmostEqual(global_change["angular_change_rad"], math.pi / 2)
        self.assertAlmostEqual(global_change["angular_rate_rad_per_tick"], math.pi / 8)
        self.assertEqual(global_change["tensor_trace_delta"], 3.)

    def test_same_tick_different_sequence_has_no_spurious_rate(self):
        make_sidecar(self.path, [
            encoded_frame(9, (1., 0., 0.)),
            encoded_frame(9, (0., 1., 0.)),
        ])
        frames, _ = exp.load(self.path)
        c = next(r for r in exp.delta_rows(frames) if r["level"] == "GLOBAL")
        self.assertEqual(c["delta_ticks"], 0)
        self.assertIsNone(c["angular_rate_rad_per_tick"])
        self.assertIsNone(c["vector_speed_per_tick"])
        self.assertGreater(c["vector_delta_norm"], 1)

    def test_remapped_triad_not_compared_and_missing_hot_not_zero(self):
        t0 = row(3, (1, 2, 3), (1., 0., 0.))
        t1 = row(3, (1, 2, 4), (9., 0., 0.))
        h = row(3, (1, 2, 3), (4., 0., 0.))
        make_sidecar(self.path, [
            encoded_frame(1, triads=(t0,), hot=(h,)),
            encoded_frame(2, triads=(t1,)),
        ])
        frames, _ = exp.load(self.path)
        changes = exp.delta_rows(frames)
        tr = next(c for c in changes if c["level"] == "TRIAD")
        self.assertEqual(tr["status"], "membership_changed")
        self.assertIsNone(tr["vector_delta_norm"])
        self.assertFalse(any(c["level"] == "HOT" for c in changes))
        self.assertEqual(exp.summary(frames, {}, changes)["membership_changes"], 1)

    def test_mirror_flux_is_canonically_oriented_on_reversal(self):
        h4 = row(4, (0, 1, 2))
        h5 = row(5, (3, 4, 5))
        make_sidecar(self.path, [
            encoded_frame(1, hot=(h4, h5), links=(edge(1, 3, 2.0),)),
            encoded_frame(3, hot=(h4, h5), links=(edge(3, 1, -5.0),)),
        ])
        frames, _ = exp.load(self.path)
        self.assertEqual(frames[0]["mirrors"][0]["physical_offset"],
                         frames[1]["mirrors"][0]["physical_offset"])
        c = next(c for c in exp.delta_rows(frames) if c["level"] == "MIRROR")
        self.assertEqual(c["canonical_flux_delta"], 3.0)
        self.assertEqual(c["flux_rate_per_tick"], 1.5)

    def test_corrupted_frame_rejected_without_false_measurement(self):
        make_sidecar(self.path, [
            encoded_frame(1), encoded_frame(2), encoded_frame(3)
        ], corrupt_sequence=3)
        frames, stats = exp.load(self.path)
        self.assertEqual([f["seq"] for f in frames], [1, 2])
        self.assertEqual(stats["rejected_candidate_frames"], 1)

    def test_wrap_replays_in_sequence_order_and_reports_gap(self):
        make_sidecar(self.path, [encoded_frame(i) for i in range(1, 401)])
        frames, stats = exp.load(self.path)
        self.assertEqual(stats["capacity_bytes"], 65536)
        self.assertEqual(frames[-1]["seq"], 400)
        self.assertLess(len(frames), 400)
        self.assertEqual([f["seq"] for f in frames], sorted(f["seq"] for f in frames))
        summary = exp.summary(frames, stats, exp.delta_rows(frames))
        self.assertGreaterEqual(summary["unobserved_sequence_slots"], 0)

    def test_invalid_capacity_or_header_rejected(self):
        self.path.write_bytes(bytes(65536))
        with self.assertRaisesRegex(ValueError, "header"):
            exp.load(self.path)
        self.path.write_bytes(bytes(123))
        with self.assertRaisesRegex(ValueError, "size"):
            exp.load(self.path)

    def test_bad_mirror_address_rejected_even_with_valid_checksum(self):
        h4, h5 = row(4, (0, 1, 2)), row(5, (3, 4, 5))
        link = bytearray(edge(1, 3, 2.))
        link[4] ^= 0x01  # corrupt address, then recompute record checksum
        make_sidecar(self.path, [encoded_frame(2, hot=(h4, h5), links=(link,))])
        frames, stats = exp.load(self.path)
        self.assertEqual(len(frames), 0)
        self.assertEqual(stats["rejected_candidate_frames"], 1)

    def test_cli_jsonl_and_csv_export(self):
        make_sidecar(self.path, [encoded_frame(1), encoded_frame(3, (1., 0., 0.))])
        def invoke(*args):
            return subprocess.run([sys.executable, str(SCRIPT), str(self.path), *args],
                                  capture_output=True, text=True, check=True).stdout
        out = invoke("summary")
        summary = json.loads(out)
        self.assertEqual(summary["valid_frames"], 2)
        self.assertEqual(summary["last_nanclock"], 3)
        self.assertIn("level,id,heads", invoke("samples", "--format", "csv"))
        self.assertIn('"delta_ticks":2', invoke("deltas"))
        self.assertEqual(len(invoke("frames", "--limit", "1").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
