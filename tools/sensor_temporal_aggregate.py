#!/usr/bin/env python3
"""Experimental read-only aggregation and source-grounded text bridge for MATS v1.

The numerical stream has no invertible word coordinates. Text is taken only
from a separately supplied NANLM_SENSOR_V2 provenance journal, never inferred
from three-vectors or tensors. No writing to either 8 MiB address space.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import sensor_temporal_export as export


def _mean(values):
    return sum(values) / len(values) if values else None


def _clean(value):
    # Avoid non-finite outputs from long aggregations of finite f32 records.
    return value if value is None or math.isfinite(value) else None


def _longest_contiguous(series):
    """Select a dense tick segment; same-tick retransmissions count once.

    Sequence gaps and absent sparse rows split the segment. Caller supplies
    (frame_seq, tick, sample_value) in frame order.
    """
    if not series:
        return []
    segments = []
    current = []
    for seq, tick, value in series:
        if current and tick == current[-1][1] and seq == current[-1][0] + 1:
            current[-1] = (seq, tick, value)  # latest within-tick sample
        elif current and tick == current[-1][1] + 1 and seq == current[-1][0] + 1:
            current.append((seq, tick, value))
        else:
            if current:
                segments.append(current)
            current = [(seq, tick, value)]
    if current:
        segments.append(current)
    return max(segments, key=lambda x: (len(x), x[-1][1]))


def _periodicity(series):
    """Conservative autocorrelation flag; *not* a learned standing wave.

    Need >=12 consecutive NaNclock ticks and >=3 cycles; mean-centered,
    variance-aware 3-vector recurrence. Damped monotone traces are not waves.
    """
    run = _longest_contiguous(series)
    if len(run) < 12:
        return {"status": "insufficient_contiguous_ticks", "period_ticks": None,
                "autocorrelation": None, "contiguous_ticks": len(run)}
    xs = [v for _, _, v in run]
    center = [sum(v[i] for v in xs) / len(xs) for i in range(3)]
    centered = [tuple(v[i] - center[i] for i in range(3)) for v in xs]
    def corr(lag):
        aa = centered[:-lag]
        bb = centered[lag:]
        dot = sum(sum(a[i] * b[i] for i in range(3)) for a, b in zip(aa, bb))
        n1 = sum(sum(x * x for x in a) for a in aa)
        n2 = sum(sum(x * x for x in b) for b in bb)
        if n1 < 1e-16 or n2 < 1e-16:
            return None
        return dot / math.sqrt(n1 * n2)
    candidates = []
    for lag in range(2, len(run) // 3 + 1):
        autocorr = corr(lag)
        # Suppress slowly drifting vectors with strong adjacent autocorrelation.
        first = corr(1)
        if autocorr is not None and autocorr >= 0.8 and first is not None and first < 0.5:
            candidates.append((lag, autocorr))
    if not candidates:
        return {"status": "not_detected", "period_ticks": None,
                "autocorrelation": None, "contiguous_ticks": len(run)}
    period, strength = max(candidates, key=lambda item: (item[1], -item[0]))
    return {"status": "oscillation_candidate_unvalidated", "period_ticks": period,
            "autocorrelation": _clean(strength), "contiguous_ticks": len(run)}


def aggregate(frames):
    """Aggregate per-stable-coordinate and mirror-link telemetry, not text.

    Samples are distinguished from unique ticks. Missing sparse measurements
    are never treated as zero, and remapped triads are separate histories.
    """
    rows = defaultdict(list)
    mirrors = defaultdict(list)
    observed_ticks = {f["nanclock"] for f in frames}
    changes = export.delta_rows(frames)
    change_map = defaultdict(list)
    for c in changes:
        if c["status"] == "compared":
            change_map[(c["level"], c["id"], tuple(c.get("heads", ())))].append(c)
    for f in frames:
        for r in f["rows"]:
            rows[(r["level"], r["id"], tuple(r["heads"]))].append((f, r))
        for m in f["mirrors"]:
            mirrors[m["physical_offset"]].append((f, m))
    result = []
    for (level, identifier, heads), entries in sorted(rows.items()):
        seqs = [f["seq"] for f, _ in entries]
        ticks = [f["nanclock"] for f, _ in entries]
        vectors = [r["vector"] for _, r in entries]
        tensors = [r["tensor"] for _, r in entries]
        ds = change_map[(level, identifier, heads)]
        comparable = [d for d in ds if d["delta_ticks"] > 0 and d["missing_frames"] == 0]
        series = [(f["seq"], f["nanclock"], r["vector"]) for f, r in entries]
        periodic = _periodicity(series)
        result.append({
            "kind": "coordinate", "level": level, "id": identifier,
            "heads": heads, "samples": len(entries), "distinct_ticks": len(set(ticks)),
            "first_tick": min(ticks), "last_tick": max(ticks),
            "tick_occupancy": len(set(ticks)) / len(observed_ticks) if observed_ticks else None,
            "max_contiguous_ticks": len(_longest_contiguous(series)),
            "mean_vector": [_clean(_mean([v[i] for v in vectors])) for i in range(3)],
            "mean_tensor": [_clean(_mean([q[i] for q in tensors])) for i in range(9)],
            "mean_activity": _clean(_mean([r["activity"] for _, r in entries])),
            "mean_support": _clean(_mean([r["support"] for _, r in entries])),
            "mean_refute": _clean(_mean([r["refute"] for _, r in entries])),
            "coactive_samples": sum(r["support"] > 0 and r["refute"] > 0
                                     for _, r in entries),
            "comparable_positive_tick_steps": len(comparable),
            "mean_speed_per_tick": _clean(_mean([d["vector_speed_per_tick"]
                                                  for d in comparable])),
            "periodicity": periodic["status"], "candidate_period_ticks": periodic["period_ticks"],
            "candidate_autocorrelation": periodic["autocorrelation"],
            "period_test_ticks": periodic["contiguous_ticks"],
            "last_sequence": seqs[-1],
        })
    for physical_offset, entries in sorted(mirrors.items()):
        vals = [m["canonical_flux"] for _, m in entries]
        phases = [m["phase_lock"] for _, m in entries]
        ticks = [f["nanclock"] for f, _ in entries]
        result.append({
            "kind": "mirror_pair", "level": "MIRROR", "id": physical_offset,
            "samples": len(entries), "distinct_ticks": len(set(ticks)),
            "first_tick": min(ticks), "last_tick": max(ticks),
            "tick_occupancy": len(set(ticks)) / len(observed_ticks) if observed_ticks else None,
            "mean_canonical_flux": _clean(_mean(vals)),
            "mean_abs_canonical_flux": _clean(_mean([abs(v) for v in vals])),
            "mean_phase_lock": _clean(_mean(phases)),
            "positive_flux_samples": sum(x > 0 for x in vals),
            "negative_flux_samples": sum(x < 0 for x in vals),
            "last_sequence": entries[-1][0]["seq"],
        })
    return result


def read_journal(path):
    """Read the authoritative V2 source/evidence journal, with strict bounds."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n")
        marker = "NANLM_SENSOR_V2\t"
        if not header.startswith(marker):
            raise ValueError("requires NANLM_SENSOR_V2 journal with event ticks")
        tick_text = header[len(marker):]
        if not tick_text.isascii() or not tick_text.isdecimal():
            raise ValueError("bad journal end tick")
        end_tick = int(tick_text)
        events = []
        last_tick = 0
        seen = set()
        for number, line in enumerate(handle, 2):
            if number > 200002:
                raise ValueError("journal exceeds bounded event count")
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7 or fields[1] not in ("+", "-"):
                raise ValueError(f"invalid journal line {number}")
            source, polarity, subject, relation, obj, raw_conf, raw_tick = fields
            if any(not x or len(x) > 192 or "\r" in x or "\x00" in x
                   for x in (source, subject, relation, obj)):
                raise ValueError(f"invalid source text at line {number}")
            try:
                confidence = float.fromhex(raw_conf) if raw_conf.startswith(("0x", "-0x")) else float(raw_conf)
            except ValueError as exc:
                raise ValueError(f"invalid confidence line {number}") from exc
            if not math.isfinite(confidence) or not (0 < confidence <= 1):
                raise ValueError(f"invalid confidence line {number}")
            if not raw_tick.isascii() or not raw_tick.isdecimal():
                raise ValueError(f"invalid nanclock line {number}")
            tick = int(raw_tick)
            if tick <= last_tick or tick > end_tick:
                raise ValueError("nonmonotonic journal event ticks")
            signature = (source, subject, relation, obj, polarity)
            if signature in seen:
                raise ValueError("duplicate source assertion")
            seen.add(signature)
            last_tick = tick
            events.append({"source": source, "polarity": polarity, "subject": subject,
                           "relation": relation, "object": obj, "confidence": confidence,
                           "tick": tick})
    return end_tick, events


def concepts(frames, journal_tick, events):
    """Evidence words come ONLY from the journal, and are never a vector inverse."""
    if not frames:
        return []
    if frames[-1]["nanclock"] > journal_tick:
        raise ValueError("journal is older than last verified sensor frame")
    tick = frames[-1]["nanclock"]
    available = [e for e in events if e["tick"] <= tick]
    for frame in frames:
        prefix = [e for e in available if e["tick"] <= frame["nanclock"]]
        if (frame["sources"] != len({e["source"] for e in prefix}) or
                frame["facts"] != len({(e["subject"], e["relation"], e["object"])
                                      for e in prefix})):
            raise ValueError("source journal counts disagree with verified frame; do not align")
    by_key = defaultdict(list)
    for e in available:
        by_key[(e["subject"], e["relation"], e["object"])].append(e)
    frame_at = defaultdict(list)
    for frame in frames:
        frame_at[frame["nanclock"]].append(frame)
    out = []
    for (subject, relation, obj), group in sorted(by_key.items()):
        supporting = sorted({e["source"] for e in group if e["polarity"] == "+"})
        refuting = sorted({e["source"] for e in group if e["polarity"] == "-"})
        state = ("both" if supporting and refuting else "supports" if supporting
                 else "refutes" if refuting else "unknown")
        aligned = sorted({f["seq"] for e in group for f in frame_at[e["tick"]]})
        # English relation syntax only for explicit, literal ISA assertions.
        article = "an" if obj[:1].lower() in "aeiou" else "a"
        phrase = f"{subject} is {article} {obj}" if relation == "ISA" else f"{subject} [{relation}] {obj}"
        out.append({"kind": "source_grounded_concept", "subject": subject,
                    "relation": relation, "object": obj, "phrase": phrase,
                    "evidence_state": state, "support_sources": supporting,
                    "refute_sources": refuting,
                    "event_ticks": sorted(e["tick"] for e in group),
                    "aligned_sequences": aligned,
                    "last_verified_sensor_tick": tick,
                    "provenance": "NANLM_SENSOR_V2 journal; geometry not decoded as text",
                    "temporal_alignment": "same NaNclock only; not attribution or causation",
                    "integrity": "counts aligned; not cryptographically linked"})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar")
    parser.add_argument("command", choices=("aggregate", "concepts"))
    parser.add_argument("--journal", help="required for words; NANLM_SENSOR_V2 snapshot")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.command == "concepts" and not args.journal:
        parser.error("concepts requires --journal; vectors cannot be inverted into words")
    try:
        frames, _ = export.load(args.sidecar)
        if args.limit:
            frames = frames[-args.limit:]
        if args.command == "aggregate":
            rows = aggregate(frames)
        else:
            end_tick, events = read_journal(args.journal)
            rows = concepts(frames, end_tick, events)
        export.emit(rows, args.format)
        return 0
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        print("sensor aggregation rejected: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())