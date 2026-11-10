#!/usr/bin/env python3
"""Summarize all native seed receipts without suppressing negative outcomes."""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path

FIELDS = (
    "carol_early_byte_ce", "carol_early_joint_ce",
    "carol_early_shuffled_ce", "carol_early_post_recurrence_ce",
    "carol_late_byte_ce", "carol_late_joint_ce",
    "carol_late_shuffled_ce", "carol_late_post_recurrence_ce",
)
def parse(text: str) -> list[dict]:
    parsed = []
    for line in text.splitlines():
        if not line.startswith("JOINT_GUTENBERG_SEED "):
            continue
        row = {}
        for word in line.split()[1:]:
            name, value = word.split("=", 1)
            row[name] = float(value)
            if not math.isfinite(row[name]):
                raise ValueError(f"nonfinite metric: {name}")
        if any(field not in row for field in FIELDS):
            raise ValueError("missing cross-author metric")
        for token_name in ("early_tokens", "late_tokens", "dev_tokens"):
            if row[token_name] < 300:
                raise ValueError("insufficient evaluation bytes")
        parsed.append(row)
    return parsed

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: report_joint_gutenberg.py RECEIPT MANIFEST REPORT")
    receipt = Path(sys.argv[1]).read_text(encoding="utf-8")
    assert "JOINT_GUTENBERG_AUDIT PASS" in receipt, "missing audit success receipt"
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    rows = parse(receipt)
    if len(rows) != 3 or len({r["seed"] for r in rows}) != 3:
        raise ValueError("expected three different training-order seeds")
    for row in rows:
        if row["root_bytes"] != 8388608 or row["recurrent_accepted"] not in (0, 1):
            raise ValueError("physical universe or acceptance invalid")
        for key in ("early", "late"):
            split = "test_carol_" + key
            count = manifest["splits"][split]["bytes"] - 1
            if row[key + "_tokens"] != count:
                raise ValueError("mismatched heldout byte count")
        if row["dev_tokens"] != manifest["splits"]["dev_alice"]["bytes"] - 1:
            raise ValueError("mismatched development byte count")
        if row["dev_post_recurrence_ce"] > row["dev_joint_ce"] + 1e-9:
            raise ValueError("acceptance gate failed zero-regression policy")
    variants = ("byte", "joint", "shuffled", "post_recurrence")
    results = []
    for row in rows:
        n1, n2 = row["early_tokens"], row["late_tokens"]
        values = {
            variant: (
                n1 * row[f"carol_early_{variant}_ce"] +
                n2 * row[f"carol_late_{variant}_ce"]
            ) / (n1 + n2)
            for variant in variants
        }
        results.append({
            "seed": int(row["seed"]), "heldout_tokens": int(n1 + n2),
            "combined_ce": values,
            "paired_delta_joint_minus_byte": values["joint"] - values["byte"],
            "paired_delta_shuffled_minus_byte": values["shuffled"] - values["byte"],
            "paired_delta_post_recurrent_minus_byte": (
                values["post_recurrence"] - values["byte"]
            ),
            "paired_delta_post_recurrent_minus_joint": (
                values["post_recurrence"] - values["joint"]
            ),
            "raw": row,
        })
    deltas = [r["paired_delta_joint_minus_byte"] for r in results]
    mean = statistics.mean(deltas)
    stdev = statistics.stdev(deltas)
    # 3 correlated training-order seeds on one fixed model family do not
    # justify a conventional population confidence interval or a p-value.
    print("JOINT_GUTENBERG_CROSS_AUTHOR seeds=3 " +
          f"tokens_per_seed={results[0]['heldout_tokens']} " +
          f"joint_minus_byte_mean_nats={mean:+.9f} " +
          f"joint_minus_byte_std_nats={stdev:.9f} " +
          f"joint_minus_byte_min_nats={min(deltas):+.9f} " +
          f"joint_minus_byte_max_nats={max(deltas):+.9f} " +
          "significant=unassessed", flush=True)
    for r in results:
        values = r["combined_ce"]
        print(f"JOINT_GUTENBERG_PAIR seed={r['seed']} "
              f"byte_ce={values['byte']:.9f} "
              f"joint_ce={values['joint']:.9f} "
              f"shuffled_ce={values['shuffled']:.9f} "
              f"post_recurrence_ce={values['post_recurrence']:.9f} "
              f"joint_minus_byte={r['paired_delta_joint_minus_byte']:+.9f} "
              f"post_minus_joint={r['paired_delta_post_recurrent_minus_joint']:+.9f}",
              flush=True)
    report = {
        "sources_and_splits": manifest,
        "seed_results": results,
        "aggregate": {
            "joint_minus_byte_mean_nats": mean,
            "joint_minus_byte_stdev_nats": stdev,
            "joint_minus_byte_min_nats": min(deltas),
            "joint_minus_byte_max_nats": max(deltas),
        },
        "interpretation": (
            "This is a heldout cross-author byte-transfer and interference audit "
            "with synthetic semantic supervision, not unsupervised Gutenberg "
            "semantic acquisition or a population-level significance test. "
            "Test text was not included in the optimizer gate. "
            "Seeds vary training order, not model initialization."
        ),
    }
    Path(sys.argv[3]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("JOINT_GUTENBERG_REPORT PASS", flush=True)

if __name__ == "__main__":
    main()
