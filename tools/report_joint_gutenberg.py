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
    "carol_early_pareto_ce", "carol_late_pareto_ce",
    "carol_early_frozen_ce", "carol_late_frozen_ce",
    "carol_late_shuffled_ce", "carol_late_post_recurrence_ce",
    "pareto_batches_accepted", "pareto_batches_rejected",
    "pareto_head_only_batches", "pareto_shared_alpha_sum",
    "pareto_role_test_ce", "pareto_sense_test_ce",
    "pareto_role_test_accuracy", "pareto_sense_test_accuracy",
    "frozen_role_test_ce", "frozen_sense_test_ce",
    "frozen_role_test_accuracy", "frozen_sense_test_accuracy",
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
        if abs(row["dev_frozen_ce"]-row["dev_byte_ce"]) > 1e-10:
            raise ValueError("frozen semantic head moved the shared byte readout")
        if row["frozen_role_updates"] <= 0 or row["frozen_sense_updates"] <= 0:
            raise ValueError("frozen semantic head failed to train")
        if row["dev_pareto_ce"] > row["dev_byte_ce"] + 1e-9:
            raise ValueError("Pareto arm worsened heldout development CE")
        if row["dev_post_recurrence_ce"] > row["dev_joint_ce"] + 1e-9:
            raise ValueError("acceptance gate failed zero-regression policy")
        # With six batches of three epochs, a positive shared-projection
        # update and a head-only fallback are mutually exclusive outcomes.
        n_shared = row["pareto_batches_accepted"]
        n_rejected = row["pareto_batches_rejected"]
        n_head = row["pareto_head_only_batches"]
        if (n_shared != int(n_shared) or n_rejected != int(n_rejected)
                or n_shared + n_rejected != 6 or n_rejected != n_head):
            raise ValueError("Pareto shared/head-only accounting is invalid")
        alpha_sum = row["pareto_shared_alpha_sum"]
        if not 0 <= alpha_sum <= n_shared or (n_shared == 0 and alpha_sum != 0):
            raise ValueError("invalid nonzero shared-gradient line-search")
        if n_shared == 0 and abs(row["dev_pareto_ce"]-row["dev_byte_ce"]) > 1e-10:
            raise ValueError("head-only specialization altered byte development CE")
    variants = ("byte", "joint", "shuffled", "pareto", "frozen", "post_recurrence")
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
            "paired_delta_pareto_minus_byte": values["pareto"] - values["byte"],
            "paired_delta_frozen_minus_byte": values["frozen"] - values["byte"],
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
    pareto_deltas = [r["paired_delta_pareto_minus_byte"] for r in results]
    stdev = statistics.stdev(deltas)
    # 3 correlated training-order seeds on one fixed model family do not
    # justify a conventional population confidence interval or a p-value.
    print("JOINT_GUTENBERG_CROSS_AUTHOR seeds=3 " +
          f"tokens_per_seed={results[0]['heldout_tokens']} " +
          f"joint_minus_byte_mean_nats={mean:+.9f} " +
          f"joint_minus_byte_std_nats={stdev:.9f} " +
          f"joint_minus_byte_min_nats={min(deltas):+.9f} " +
          f"joint_minus_byte_max_nats={max(deltas):+.9f} " +
          f"pareto_minus_byte_mean_nats={statistics.mean(pareto_deltas):+.9f} " +
          f"frozen_minus_byte_mean_nats={statistics.mean(r['paired_delta_frozen_minus_byte'] for r in results):+.9f} " +
          "significant=unassessed", flush=True)
    for r in results:
        values = r["combined_ce"]
        print(f"JOINT_GUTENBERG_PARETO seed={r['seed']} "
              f"shared_batches={int(r['raw']['pareto_batches_accepted'])} "
              f"head_only_batches={int(r['raw']['pareto_head_only_batches'])} "
              f"shared_alpha_sum={r['raw']['pareto_shared_alpha_sum']:.8f} "
              f"role_ce={r['raw']['pareto_role_test_ce']:.8f} "
              f"sense_ce={r['raw']['pareto_sense_test_ce']:.8f} "
              f"role_accuracy={r['raw']['pareto_role_test_accuracy']:.6f} "
              f"sense_accuracy={r['raw']['pareto_sense_test_accuracy']:.6f} "
              f"frozen_role_ce={r['raw']['frozen_role_test_ce']:.8f} "
              f"frozen_sense_ce={r['raw']['frozen_sense_test_ce']:.8f} "
              f"frozen_role_accuracy={r['raw']['frozen_role_test_accuracy']:.6f} "
              f"frozen_sense_accuracy={r['raw']['frozen_sense_test_accuracy']:.6f}",
              flush=True)
        print(f"JOINT_GUTENBERG_PAIR seed={r['seed']} "
              f"byte_ce={values['byte']:.9f} "
              f"joint_ce={values['joint']:.9f} "
              f"shuffled_ce={values['shuffled']:.9f} "
              f"pareto_ce={values['pareto']:.9f} "
              f"frozen_ce={values['frozen']:.9f} "
              f"post_recurrence_ce={values['post_recurrence']:.9f} "
              f"joint_minus_byte={r['paired_delta_joint_minus_byte']:+.9f} "
              f"pareto_minus_byte={r['paired_delta_pareto_minus_byte']:+.9f} "
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
            "pareto_minus_byte_mean_nats": statistics.mean(pareto_deltas),
            "frozen_minus_byte_mean_nats": statistics.mean(r["paired_delta_frozen_minus_byte"] for r in results),
            "shared_batches_by_seed": [
                int(r["raw"]["pareto_batches_accepted"]) for r in results
            ],
            "head_only_batches_by_seed": [
                int(r["raw"]["pareto_head_only_batches"]) for r in results
            ],
            "shared_alpha_sum_by_seed": [
                r["raw"]["pareto_shared_alpha_sum"] for r in results
            ],
        },
        "interpretation": (
            "This is a heldout cross-author byte-transfer and interference audit "
            "with synthetic semantic supervision, not unsupervised Gutenberg "
            "semantic acquisition or a population-level significance test. "
            "Test text was not included in the optimizer gate. "
            "Seeds vary training order, not model initialization. "
            "A semantic-head-only fallback is not semantic-to-byte transfer; "
            "only a positive line-search alpha counts as shared credit."
        ),
    }
    Path(sys.argv[3]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("JOINT_GUTENBERG_REPORT PASS", flush=True)

if __name__ == "__main__":
    main()
