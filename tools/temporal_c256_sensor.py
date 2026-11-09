#!/usr/bin/env python3
"""Isolated temporal C(256,2) sensor for HW4 WordNet NANNET frames.

NO language-model weights, WordNet synsets, glosses, tokens or corpus rows are
persisted. A new, zero-filled 8 MiB mirror universe holds ONLY current sensor
state. The only export is a compressed stream of tensor coordinates and values.
This is an experimental mathematical *analogy*, not a quantum simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import sys
from urllib.request import Request, urlopen
import zlib

HEADS = 255
NODES = 256
TRIADS = 85
PAIRS = 32640
ROOT_BYTES = 8388608
HEAD_BYTES = 128
PAIR_BYTES = 256
ROOT_PAIR_OFFSET = 32768
CANONICAL_WORDNET_ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW4:0"
# Binary tensor record: nanclock, sequence, channel, row, col, float32 value.
RECORD = struct.Struct("<QQBBBf")
MAGIC = b"NTC25601"
CHANNELS = (
    "point", "triad", "hot_triad", "global", "mirror_forward",
    "mirror_reverse", "standing_wave", "spectral",
    "born_analogue", "curvature", "coherence", "entropy"
)
CHANNEL = {name: i for i, name in enumerate(CHANNELS)}
FRAME_MAGIC = b"NPF1"


def pair_index(a: int, b: int) -> int:
    if not 0 <= a < 256 or not 0 <= b < 256 or a == b:
        raise ValueError("invalid mirror pair")
    if a > b:
        a, b = b, a
    return a * (511 - a) // 2 + b - a - 1


def physical_pair(a: int, b: int) -> int:
    return ROOT_PAIR_OFFSET + PAIR_BYTES * pair_index(a, b)


def parse_frame(raw: bytes) -> tuple[int, int]:
    """Verify the actual NaNROUTER NPF1 frame (checksum != authentication).

    Return (nanclock_tick, opaque WordNet 32-bit payload).
    """
    if len(raw) < 97 or len(raw) > 256 or raw[:4] != FRAME_MAGIC or raw[4] != 1:
        raise ValueError("unsupported NANNET point frame")
    source, dest, kind_length = raw[5], raw[6], raw[7]
    if source != 4 or dest != 0 or not 1 <= kind_length <= 64:
        raise ValueError("not HW4:0 -> point-zero WordNet intake")
    if len(raw) != 64 + kind_length + 32:
        raise ValueError("incorrect point-frame length")
    if hashlib.sha256(raw[:-32]).digest() != raw[-32:]:
        raise ValueError("invalid point-frame digest")
    kind = raw[64:-32]
    if not kind.isascii() or any(not (c.isalnum() or c in "-_.") for c in kind.decode("ascii")):
        raise ValueError("invalid point-frame kind")
    # NaN16DE is far BE, payload LE, near LE. Retain payload as opaque u32;
    # the source HW4 service must define what its payload codes mean.
    tick = struct.unpack_from(">Q", raw, 16)[0]
    payload = struct.unpack_from("<I", raw, 58)[0]
    return tick, payload


def parse_drain(document: dict) -> list[tuple[int, int]]:
    if document.get("schema") != "nanrouter.zero-point-drain.v1":
        raise ValueError("NANNET zero-point drain schema required")
    if document.get("endpoint") != "HW0:0":
        raise ValueError("not the point-zero drain")
    frames = document.get("frames")
    if not isinstance(frames, list) or document.get("count") != len(frames) or len(frames) > 64:
        raise ValueError("invalid bounded NANNET drain")
    result = []
    last_tick = -1
    for row in frames:
        if not isinstance(row, dict) or not isinstance(row.get("frame_hex"), str):
            raise ValueError("missing verified NPF1 frame bytes")
        tick, payload = parse_frame(bytes.fromhex(row["frame_hex"]))
        if tick < last_tick:
            raise ValueError("nonmonotonic nanclock")
        last_tick = tick
        result.append((tick, payload))
    return result


def fetch_live_drain(url: str, *, timeout: float = 15) -> dict:
    """Read only from the deployed/authorized NANNET fabric, never fake HW4 data."""
    if not url.startswith("https://"):
        raise ValueError("NANNET URL must use HTTPS")
    token = os.environ.get("NANROUTER_FABRIC_TOKEN")
    if not token:
        raise ValueError("NANROUTER_FABRIC_TOKEN is required for live intake")
    endpoint = url.rstrip("/") + "/v1/fabric/points/0/drain"
    body = json.dumps({"limit": 64}, separators=(",", ":")).encode()
    req = Request(endpoint, data=body, method="POST", headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
        "Accept": "application/json"
    })
    with urlopen(req, timeout=timeout) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("NANNET response too large")
    return json.loads(data)


class BinaryTensor:
    """Streaming compressed tensor entries; never writes source corpus."""
    def __init__(self, destination: Path):
        self.stream = open(destination, "xb")
        self.compressor = zlib.compressobj(level=6)
        self.stream.write(MAGIC + struct.pack("<I", 1))
        self.entries = 0
        self.first_values = []

    def emit(self, tick: int, seq: int, channel: str, a: int, b: int, value: float):
        if not math.isfinite(value) or not 0 <= a < 256 or not 0 <= b < 256:
            raise ValueError("nonfinite/out-of-range tensor")
        value = max(-1e6, min(1e6, value))
        data = RECORD.pack(tick, seq, CHANNEL[channel], a, b, value)
        self.stream.write(self.compressor.compress(data))
        self.entries += 1
        if len(self.first_values) < 12:
            self.first_values.append((tick, seq, channel, a, b, round(value, 6)))

    def close(self):
        self.stream.write(self.compressor.flush())
        self.stream.close()


class TensorSensor:
    """Fresh mapped NaN mirror pairs, one 255-head recurrent mathematical sensor."""
    def __init__(self, root_path: Path, tensor_path: Path):
        # Exclusive creation prevents touching any prior 8 MiB universe.
        fd = os.open(root_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, ROOT_BYTES)
            self.root = mmap.mmap(fd, ROOT_BYTES)
        finally:
            os.close(fd)
        self.out = BinaryTensor(tensor_path)
        self.wave = [0.] * HEADS
        self.old_wave = [0.] * HEADS
        self.triad = [0.] * TRIADS
        self.prev_head = None
        self.seq = 0
        self.active_pairs = set()
        self.max_hot = 0

    @staticmethod
    def head(payload: int) -> int:
        # A deterministic coordinate, not a WordNet semantic classifier.
        return ((payload * 0x9E3779B1 & 0xffffffff) >> 24) % HEADS

    def pair_value(self, a: int, b: int) -> float:
        start = physical_pair(a, b) + (0 if a < b else 128)
        return struct.unpack_from("<f", self.root, start)[0]

    def set_pair(self, a: int, b: int, signal: float):
        if a == b:
            return
        base = physical_pair(a, b)
        old = self.pair_value(a, b)
        val = float(max(-4, min(4, 0.94 * old + 0.06 * signal)))
        # Exact sign mirror; two temporal orientations, one physical cell.
        struct.pack_into("<f", self.root, base, val if a < b else -val)
        struct.pack_into("<f", self.root, base + 128, -val if a < b else val)
        self.active_pairs.add(pair_index(a, b))

    def tick(self, nanclock: int, payload: int):
        if not 0 <= nanclock < 2**64 or not 0 <= payload <= 0xffffffff:
            raise ValueError("out-of-range incoming frame")
        self.seq += 1
        h = self.head(payload)
        # Damped, causal second-order standing wave across the fixed 255 heads.
        # Neighbor excitation is local to 85 triads; no trained LM is involved.
        new = [0.] * HEADS
        for i in range(HEADS):
            g = i // 3
            neighbor = (self.wave[3*g] + self.wave[3*g+1] + self.wave[3*g+2]) / 3
            kick = 0.32 if i == h else 0.
            new[i] = max(-4., min(4., 1.52*self.wave[i] - 0.64*self.old_wave[i]
                                             + 0.03*neighbor + kick))
        self.old_wave, self.wave = self.wave, new
        for i, val in enumerate(self.wave):
            struct.pack_into("<f", self.root, i * HEAD_BYTES + 8, val)
            self.out.emit(nanclock, self.seq, "point", i, i, val)
            self.out.emit(nanclock, self.seq, "standing_wave", i, 255, val - self.old_wave[i])
        if self.prev_head is not None and h != self.prev_head:
            p = self.prev_head
            self.set_pair(p, h, self.wave[p] - self.old_wave[h])
            self.out.emit(nanclock, self.seq, "mirror_forward", p, h, self.pair_value(p, h))
            self.out.emit(nanclock, self.seq, "mirror_reverse", h, p, self.pair_value(h, p))
            self.out.emit(nanclock, self.seq, "coherence", p, h,
                          self.wave[p] * self.wave[h] / (1+self.wave[p]**2+self.wave[h]**2))
        self.prev_head = h
        norm = sum(v*v for v in self.wave) + 1e-15
        total = sum(abs(v) for v in self.wave) + 1e-15
        entropy = 0.
        for g in range(TRIADS):
            a = 3*g
            value = sum(self.wave[a+k] for k in range(3))/3
            self.triad[g] = value
            self.out.emit(nanclock, self.seq, "triad", a, a+2, value)
        hot = sorted(range(TRIADS), key=lambda g: (-abs(self.triad[g]), g))[:8]
        self.max_hot = max(self.max_hot, sum(abs(self.triad[g]) > 1e-4 for g in range(TRIADS)))
        for g in hot:
            a = 3*g
            self.out.emit(nanclock, self.seq, "hot_triad", a, a+2, self.triad[g])
            # Curvature from the actual oriented, mutable mirror-pair register.
            x, y, z = a, a+1, a+2
            phase_xy = math.atan(self.pair_value(x, y))
            phase_yz = math.atan(self.pair_value(y, z))
            phase_zx = math.atan(self.pair_value(z, x))
            flux = math.atan2(math.sin(phase_xy+phase_yz+phase_zx),
                              math.cos(phase_xy+phase_yz+phase_zx))
            self.out.emit(nanclock, self.seq, "curvature", x, z, flux)
        global_value = sum(self.triad) / TRIADS
        self.out.emit(nanclock, self.seq, "global", 255, 255, global_value)
        for k in range(4):
            real = sum(v * math.cos(2*math.pi*(k+1)*i/HEADS) for i,v in enumerate(self.wave))
            imag = sum(v * math.sin(2*math.pi*(k+1)*i/HEADS) for i,v in enumerate(self.wave))
            self.out.emit(nanclock, self.seq, "spectral", 255, k, math.hypot(real, imag)/HEADS)
        # Born-inspired positive normalized squared amplitudes, NOT quantum data.
        for i in sorted(range(HEADS), key=lambda j: (-self.wave[j]**2, j))[:8]:
            q = self.wave[i]**2/norm
            self.out.emit(nanclock, self.seq, "born_analogue", i, 255, q)
        for v in self.wave:
            prob = abs(v)/total
            if prob > 0:
                entropy -= prob*math.log(prob)
        self.out.emit(nanclock, self.seq, "entropy", 255, 255, entropy)

    def finish(self) -> dict:
        self.root.flush()
        self.root.close()
        self.out.close()
        return {
            "frames": self.seq, "tensor_entries": self.out.entries,
            "active_mirror_pairs": len(self.active_pairs), "possible_pairs": PAIRS,
            "max_active_triads": self.max_hot,
            "tensor_raw_bytes": 12 + self.out.entries * RECORD.size,
            "tensor_compressed_bytes": os.path.getsize(self.out.stream.name),
            "example_tensor_values": self.out.first_values,
        }


def run(args) -> dict:
    if args.nannet_json is not None:
        document = json.loads(Path(args.nannet_json).read_text(encoding="utf-8"))
        source = "nannet-verified-fixture"
    else:
        document = fetch_live_drain(args.nannet_url)
        source = "live-nannet-point0-drain"
    samples = parse_drain(document)
    if not samples:
        raise RuntimeError("HW4:0 returned no WordNet frames; no sensor data fabricated")
    # A live stream with repeating tick is permitted; each received word
    # has a monotonic sequence within the tick. Nonmonotonic frames are rejected.
    sensor = TensorSensor(Path(args.root), Path(args.tensor))
    for tick, payload in samples:
        sensor.tick(tick, payload)
    result = sensor.finish()
    result["source"] = source
    result["nannet_route"] = CANONICAL_WORDNET_ROUTE
    result["root_bytes"] = ROOT_BYTES
    result["tensor_record_bytes"] = RECORD.size
    result["channels"] = list(CHANNELS)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--nannet-json", help="saved NaNROUTER point0 drain JSON (fixture/replay)")
    source.add_argument("--nannet-url", help="authorized NANNET HTTPS portal base URL")
    ap.add_argument("--root", required=True, help="NEW, never-existing 8 MiB root")
    ap.add_argument("--tensor", required=True, help="NEW .ntc256 binary tensor output")
    args = ap.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True, indent=2))
    except Exception as exc:
        print(f"SENSOR ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
