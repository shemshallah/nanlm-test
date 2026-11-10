#!/usr/bin/env python3
"""Read-only NaNLM mathematical time-slice exporter and temporal diagnostics.

Parses the native MATS v1 / NSF1 little-endian sidecar, WITHOUT creating a
model, opening the 8 MiB native root, or rewriting the source graph journal.
Only verified, bounded, fully decoded records enter the analysis.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
import sys
from pathlib import Path

HEADER_BYTES = 64
RECORD_BYTES = 32
ALIGN = 64
SAMPLE = struct.Struct("<4B18f")  # ID, three head IDs, vec3, tensor9, scalar6
LINK = struct.Struct("<4BHHIfffH")
PAYLOAD_HEADER = struct.Struct("<4sHHQII4H")
LEVELS = ("HEAD", "TRIAD", "HOT", "HOTFIELD", "GLOBAL")


def address(a: int, b: int) -> int:
    return (a << 8) | int(f"{b:08b}"[::-1], 2)


def pair_index(a: int, b: int) -> int:
    if a > b:
        a, b = b, a
    return a * (511 - a) // 2 + b - a - 1


def finite(values: tuple, label: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite " + label)


def sample(data: bytes, pos: int) -> tuple[dict, int]:
    if pos + SAMPLE.size > len(data):
        raise ValueError("truncated measurement row")
    row = SAMPLE.unpack_from(data, pos)
    finite(row[4:], "matrix measurement")
    return {
        "id": row[0], "heads": tuple(row[1:4]),
        "vector": tuple(row[4:7]), "tensor": tuple(row[7:16]),
        "activity": row[16], "support": row[17], "refute": row[18],
        "curvature": row[19], "score": row[20], "prior": row[21],
    }, pos + SAMPLE.size


def decode(payload: bytes) -> dict:
    if len(payload) < PAYLOAD_HEADER.size:
        raise ValueError("truncated MATS header")
    magic, version, reserved, tick, sources, facts, hn, tn, kn, ln = (
        PAYLOAD_HEADER.unpack_from(payload))
    if (magic, version, reserved) != (b"MATS", 1, 0):
        raise ValueError("invalid MATS version")
    if hn > 255 or tn > 85 or kn > 12 or ln > 66:
        raise ValueError("invalid row capacity")
    expected = PAYLOAD_HEADER.size + (hn + tn + kn + 2) * SAMPLE.size + ln * LINK.size
    if expected != len(payload):
        raise ValueError("MATS length mismatch")
    frame = {"nanclock": tick, "sources": sources, "facts": facts, "rows": [],
             "mirrors": []}
    pos = PAYLOAD_HEADER.size
    hot_ids = set()
    for level, count in (("HEAD", hn), ("TRIAD", tn), ("HOT", kn),
                         ("HOTFIELD", 1), ("GLOBAL", 1)):
        if level in ("HOTFIELD", "GLOBAL"):
            # Field rows follow links; handled below.
            continue
        seen = set()
        for _ in range(count):
            row, pos = sample(payload, pos)
            rid, heads = row["id"], row["heads"]
            if rid in seen:
                raise ValueError("duplicate sparse row")
            seen.add(rid)
            if level == "HEAD":
                if rid >= 255 or heads != (rid, 255, 255):
                    raise ValueError("invalid head row")
            elif rid >= 85 or any(h >= 255 for h in heads) or len(set(heads)) != 3:
                raise ValueError("invalid triad membership")
            row["level"] = level
            frame["rows"].append(row)
            if level == "HOT":
                hot_ids.add(rid)
    for _ in range(ln):
        if pos + LINK.size > len(payload):
            raise ValueError("truncated pair row")
        a, b, frm, to, forward, reverse, offset, cos, phase, flux, shared = (
            LINK.unpack_from(payload, pos))
        pos += LINK.size
        finite((cos, phase, flux), "mirror link")
        if (a == b or a not in hot_ids or b not in hot_ids or
                frm >= 255 or to >= 255 or frm == to or
                forward != address(frm, to) or reverse != address(to, frm) or
                offset != 32768 + 256 * pair_index(frm, to)):
            raise ValueError("invalid canonical mirror pair")
        frame["mirrors"].append({
            "group_a": a, "group_b": b, "from": frm, "to": to,
            "forward": forward, "reverse": reverse, "physical_offset": offset,
            "tensor_cos": cos, "phase_lock": phase, "oriented_flux": flux,
            "canonical_flux": flux if frm < to else -flux, "shared_sources": shared,
        })
    for level in ("HOTFIELD", "GLOBAL"):
        row, pos = sample(payload, pos)
        if row["id"] != 255:
            raise ValueError("invalid field ID")
        row["level"] = level
        frame["rows"].append(row)
    if pos != len(payload):
        raise ValueError("trailing MATS payload")
    return frame


def checksum(payload: bytes, seq: int, tick: int) -> int:
    h = 2166136261
    for byte in payload + struct.pack("<QQ", seq, tick):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def load(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    if not path.name.endswith(".slices"):
        path = Path(str(path) + ".slices")
    data = path.read_bytes()
    if not (65536 <= len(data) <= 67108864 and len(data) % ALIGN == 0):
        raise ValueError("invalid bounded sidecar size")
    expected = struct.pack("<8sIIII", b"NLSLICE1", 1, len(data), 64, 64)
    if data[:HEADER_BYTES] != expected + bytes(HEADER_BYTES - len(expected)):
        raise ValueError("invalid sidecar header or declared capacity")
    frames: list[dict] = []
    rejected = 0
    for off in range(HEADER_BYTES, len(data) - RECORD_BYTES + 1, ALIGN):
        if data[off:off + 4] != b"NSF1":
            continue
        version, seq, tick, length, digest = struct.unpack_from("<IQQII", data, off + 4)
        if version != 1 or seq == 0 or not (32 <= length <= 65536) or (
                off + RECORD_BYTES + length > len(data)):
            rejected += 1
            continue
        payload = data[off + RECORD_BYTES:off + RECORD_BYTES + length]
        if checksum(payload, seq, tick) != digest:
            rejected += 1
            continue
        try:
            frame = decode(payload)
        except (ValueError, struct.error):
            rejected += 1
            continue
        if frame["nanclock"] != tick:
            rejected += 1
            continue
        frame["seq"] = seq
        frame["offset"] = off
        frames.append(frame)
    frames.sort(key=lambda f: (f["seq"], f["offset"]))
    if len({f["seq"] for f in frames}) != len(frames):
        raise ValueError("multiple valid frames share a sequence number")
    return frames, {"capacity_bytes": len(data), "valid_frames": len(frames),
                    "rejected_candidate_frames": rejected, "sidecar": str(path)}


def measurements(frames: list[dict]) -> list[dict]:
    out = []
    for frame in frames:
        for row in frame["rows"]:
            out.append({"seq": frame["seq"], "nanclock": frame["nanclock"],
                        "sources": frame["sources"], "facts": frame["facts"], **row})
    return out


def norm(xs) -> float:
    return math.sqrt(sum(x * x for x in xs))


def angle(a, b) -> float | None:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return None  # Direction is undefined for a zero vector.
    dot = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.acos(max(-1.0, min(1.0, dot)))


def delta_rows(frames: list[dict]) -> list[dict]:
    """Adjacent retained frames, matched by stable IDs and exact membership.

    A gap in recorded sequences is reported, never silently interpolated.
    Two samples at the same tick have a difference, but no time derivative.
    """
    out = []
    for before, after in zip(frames, frames[1:]):
        dt = after["nanclock"] - before["nanclock"]
        if dt < 0:
            raise ValueError("nanclock regressed across retained sequences")
        gap = after["seq"] - before["seq"] - 1
        first = {(r["level"], r["id"]): r for r in before["rows"]}
        second = {(r["level"], r["id"]): r for r in after["rows"]}
        for key in sorted(first.keys() & second.keys()):
            old, new = first[key], second[key]
            remapped = old["heads"] != new["heads"]
            # Sparse HEAD/HOT/TRIAD rows can appear or disappear. Missing
            # measurements are not observations of mathematical zero.
            base = {"from_seq": before["seq"], "to_seq": after["seq"],
                    "from_tick": before["nanclock"], "to_tick": after["nanclock"],
                    "delta_ticks": dt, "missing_frames": gap,
                    "level": key[0], "id": key[1],
                    "remapped": remapped, "heads": new["heads"]}
            if remapped:
                out.append({**base, "status": "membership_changed",
                            "vector_delta": None, "tensor_delta": None,
                            "vector_delta_norm": None, "tensor_delta_fro": None,
                            "angular_change_rad": None, "angular_rate_rad_per_tick": None,
                            "vector_speed_per_tick": None, "tensor_speed_per_tick": None,
                            "support_delta": None, "refute_delta": None})
                continue
            dv = [b - a for a, b in zip(old["vector"], new["vector"])]
            dq = [b - a for a, b in zip(old["tensor"], new["tensor"])]
            ang = angle(old["vector"], new["vector"])
            out.append({**base, "status": "compared",
                        "vector_delta": dv, "tensor_delta": dq,
                        "vector_delta_norm": norm(dv), "tensor_delta_fro": norm(dq),
                        "angular_change_rad": ang,
                        "angular_rate_rad_per_tick": ang / dt if ang is not None and dt else None,
                        "vector_speed_per_tick": norm(dv) / dt if dt else None,
                        "tensor_speed_per_tick": norm(dq) / dt if dt else None,
                        "vector_derivative": [v / dt for v in dv] if dt else None,
                        "tensor_derivative": [v / dt for v in dq] if dt else None,
                        "support_delta": new["support"] - old["support"],
                        "refute_delta": new["refute"] - old["refute"],
                        "tensor_trace_delta": sum(dq[i] for i in (0, 4, 8))})
        old_links = {r["physical_offset"]: r for r in before["mirrors"]}
        new_links = {r["physical_offset"]: r for r in after["mirrors"]}
        for offset in sorted(old_links.keys() & new_links.keys()):
            old, new = old_links[offset], new_links[offset]
            out.append({"from_seq": before["seq"], "to_seq": after["seq"],
                        "from_tick": before["nanclock"], "to_tick": after["nanclock"],
                        "delta_ticks": dt, "missing_frames": gap, "level": "MIRROR",
                        "id": offset, "status": "compared", "remapped": False,
                        "canonical_flux_delta": new["canonical_flux"] - old["canonical_flux"],
                        "tensor_cos_delta": new["tensor_cos"] - old["tensor_cos"],
                        "phase_lock_delta": new["phase_lock"] - old["phase_lock"],
                        "flux_rate_per_tick": (
                            (new["canonical_flux"] - old["canonical_flux"]) / dt if dt else None)})
    return out


def summary(frames: list[dict], stats: dict, changes: list[dict]) -> dict:
    result = dict(stats)
    result["first_sequence"] = frames[0]["seq"] if frames else None
    result["last_sequence"] = frames[-1]["seq"] if frames else None
    result["first_nanclock"] = frames[0]["nanclock"] if frames else None
    result["last_nanclock"] = frames[-1]["nanclock"] if frames else None
    result["unobserved_sequence_slots"] = sum(
        b["seq"] - a["seq"] - 1 for a, b in zip(frames, frames[1:]))
    result["same_tick_frame_pairs"] = sum(
        a["nanclock"] == b["nanclock"] for a, b in zip(frames, frames[1:]))
    result["membership_changes"] = sum(c.get("status") == "membership_changed"
                                       for c in changes)
    result["compared_rows"] = sum(c.get("status") == "compared" for c in changes)
    result["max_global_vector_step"] = max(
        (c["vector_delta_norm"] for c in changes
         if c["level"] == "GLOBAL" and c["status"] == "compared"), default=None)
    result["max_global_tensor_step"] = max(
        (c["tensor_delta_fro"] for c in changes
         if c["level"] == "GLOBAL" and c["status"] == "compared"), default=None)
    result["warning"] = ("Bounded rolling telemetry; missing sparse rows and overwritten "
                         "frames are not zeros. This is not a provenance journal or a "
                         "measured improvement in held-out language performance.")
    return result


def emit(rows: list[dict], fmt: str) -> None:
    if fmt == "jsonl":
        for row in rows:
            print(json.dumps(row, allow_nan=False, separators=(",", ":")))
        return
    if not rows:
        return
    # A tidy numerical CSV: separate coordinate columns rather than a
    # Python tuple literal that downstream analysis would need to parse.
    def flatten(row):
        result = {}
        for key, value in row.items():
            if isinstance(value, (tuple, list)):
                for i, scalar in enumerate(value):
                    result[f"{key}_{i}"] = scalar
            elif value is None and key in ("vector_delta", "tensor_delta",
                                             "vector_derivative", "tensor_derivative"):
                continue
            else:
                result[key] = value
        return result
    flat = [flatten(row) for row in rows]
    keys = list(dict.fromkeys(key for row in flat for key in row))
    writer = csv.DictWriter(sys.stdout, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in flat:
        writer.writerow(row)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", help="snapshot path or explicit .slices path")
    parser.add_argument("command", choices=("frames", "samples", "mirrors", "deltas", "summary"))
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all retained verified frames; otherwise last N")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        frames, stats = load(args.sidecar)
        if args.limit:
            frames = frames[-args.limit:]
        changes = delta_rows(frames) if args.command in ("deltas", "summary") else []
        if args.command == "summary":
            emit([summary(frames, stats, changes)], args.format)
        elif args.command == "samples":
            emit(measurements(frames), args.format)
        elif args.command == "deltas":
            emit(changes, args.format)
        elif args.command == "mirrors":
            emit([{"seq": f["seq"], "nanclock": f["nanclock"], **m}
                  for f in frames for m in f["mirrors"]], args.format)
        else:
            emit([{k: v for k, v in f.items() if k != "rows" and k != "mirrors"} |
                  {"heads": sum(r["level"] == "HEAD" for r in f["rows"]),
                   "triads": sum(r["level"] == "TRIAD" for r in f["rows"]),
                   "hot": sum(r["level"] == "HOT" for r in f["rows"]),
                   "mirror_pairs": len(f["mirrors"])} for f in frames], args.format)
        return 0
    except (OSError, ValueError, struct.error) as e:
        print("sensor export rejected: " + str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
