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
    "carol_early_surgery_ce", "carol_late_surgery_ce",
    "carol_early_sham_ce", "carol_late_sham_ce",
    "carol_late_shuffled_ce", "carol_late_post_recurrence_ce",
    "pareto_batches_accepted", "pareto_batches_rejected",
    "pareto_head_only_batches", "pareto_shared_alpha_sum",
    "pareto_role_test_ce", "pareto_sense_test_ce",
    "pareto_role_test_accuracy", "pareto_sense_test_accuracy",
    "frozen_role_test_ce", "frozen_sense_test_ce",
    "frozen_role_test_accuracy", "frozen_sense_test_accuracy",
    "dev_surgery_ce", "surgery_batches_accepted",
    "surgery_batches_rejected", "surgery_head_only_batches",
    "surgery_conflicting_batches", "surgery_shared_alpha_sum",
    "surgery_role_test_ce", "surgery_sense_test_ce",
    "surgery_role_test_accuracy", "surgery_sense_test_accuracy",
    "dev_sham_ce", "sham_batches_accepted", "sham_batches_rejected",
    "sham_head_only_batches", "sham_conflicting_batches",
    "sham_shared_alpha_sum", "sham_role_test_ce", "sham_sense_test_ce",
    "sham_role_test_accuracy", "sham_sense_test_accuracy",
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
        if row["dev_sham_ce"] > row["dev_byte_ce"] + 1e-9:
            raise ValueError("shuffled projected-gradient arm worsened DEV CE")
        ns = row["sham_batches_accepted"]
        rs = row["sham_batches_rejected"]
        hs = row["sham_head_only_batches"]
        cs = row["sham_conflicting_batches"]
        if (any(v != int(v) for v in (ns, rs, hs, cs))
                or ns + rs != 6 or rs != hs or not 0 <= cs <= 6):
            raise ValueError("shuffled gradient-surgery batch accounting invalid")
        if not 0 <= row["sham_shared_alpha_sum"] <= ns:
            raise ValueError("shuffled gradient-surgery alpha accounting invalid")
        if (ns == 0 and (row["sham_shared_alpha_sum"] != 0 or
                        abs(row["dev_sham_ce"]-row["dev_byte_ce"]) > 1e-10)):
            raise ValueError("shuffled head-only specialization changed byte CE")
        if row["dev_surgery_ce"] > row["dev_byte_ce"] + 1e-9:
            raise ValueError("projected-gradient arm worsened Alice DEV byte CE")
        na = row["surgery_batches_accepted"]
        nr = row["surgery_batches_rejected"]
        nh = row["surgery_head_only_batches"]
        nc = row["surgery_conflicting_batches"]
        if (any(v != int(v) for v in (na, nr, nh, nc))
                or na + nr != 6 or nr != nh or not 0 <= nc <= 6):
            raise ValueError("projected-gradient batch accounting invalid")
        if not 0 <= row["surgery_shared_alpha_sum"] <= na:
            raise ValueError("projected-gradient alpha accounting invalid")
        if (na == 0 and (row["surgery_shared_alpha_sum"] != 0
                        or abs(row["dev_surgery_ce"] - row["dev_byte_ce"]) > 1e-10)):
            raise ValueError("projected-gradient head-only update changed byte CE")
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
    variants = ("byte", "joint", "shuffled", "pareto", "frozen", "surgery", "sham", "post_recurrence")
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
            "paired_delta_surgery_minus_byte": values["surgery"] - values["byte"],
            "paired_delta_sham_minus_byte": values["sham"] - values["byte"],
            "paired_delta_surgery_minus_sham": values["surgery"] - values["sham"],
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
          f"surgery_minus_byte_mean_nats={statistics.mean(r['paired_delta_surgery_minus_byte'] for r in results):+.9f} " +
          f"sham_minus_byte_mean_nats={statistics.mean(r['paired_delta_sham_minus_byte'] for r in results):+.9f} " +
          f"surgery_minus_sham_mean_nats={statistics.mean(r['paired_delta_surgery_minus_sham'] for r in results):+.9f} " +
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
        print(f"JOINT_GUTENBERG_SURGERY seed={r['seed']} "
              f"shared_batches={int(r['raw']['surgery_batches_accepted'])} "
              f"rejected_batches={int(r['raw']['surgery_batches_rejected'])} "
              f"conflicting_batches={int(r['raw']['surgery_conflicting_batches'])} "
              f"alpha_sum={r['raw']['surgery_shared_alpha_sum']:.8f} "
              f"role_ce={r['raw']['surgery_role_test_ce']:.8f} "
              f"sense_ce={r['raw']['surgery_sense_test_ce']:.8f} "
              f"heldout_byte_delta={r['paired_delta_surgery_minus_byte']:+.9f}",
              flush=True)
        print(f"JOINT_GUTENBERG_SHAM seed={r['seed']} "
              f"shared_batches={int(r['raw']['sham_batches_accepted'])} "
              f"rejected_batches={int(r['raw']['sham_batches_rejected'])} "
              f"conflicting_batches={int(r['raw']['sham_conflicting_batches'])} "
              f"alpha_sum={r['raw']['sham_shared_alpha_sum']:.8f} "
              f"role_ce={r['raw']['sham_role_test_ce']:.8f} "
              f"sense_ce={r['raw']['sham_sense_test_ce']:.8f} "
              f"heldout_byte_delta={r['paired_delta_sham_minus_byte']:+.9f} "
              f"true_minus_sham={r['paired_delta_surgery_minus_sham']:+.9f}",
              flush=True)
        print(f"JOINT_GUTENBERG_PAIR seed={r['seed']} "
              f"byte_ce={values['byte']:.9f} "
              f"joint_ce={values['joint']:.9f} "
              f"shuffled_ce={values['shuffled']:.9f} "
              f"pareto_ce={values['pareto']:.9f} "
              f"frozen_ce={values['frozen']:.9f} "
              f"surgery_ce={values['surgery']:.9f} "
              f"sham_ce={values['sham']:.9f} "
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
            "surgery_minus_byte_mean_nats": statistics.mean(r["paired_delta_surgery_minus_byte"] for r in results),
            "sham_minus_byte_mean_nats": statistics.mean(r["paired_delta_sham_minus_byte"] for r in results),
            "surgery_minus_sham_mean_nats": statistics.mean(r["paired_delta_surgery_minus_sham"] for r in results),
            "sham_shared_batches_by_seed": [int(r["raw"]["sham_batches_accepted"]) for r in results],
            "surgery_shared_batches_by_seed": [int(r["raw"]["surgery_batches_accepted"]) for r in results],
            "surgery_conflicting_batches_by_seed": [int(r["raw"]["surgery_conflicting_batches"]) for r in results],
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
            "only a positive line-search alpha counts as shared credit. "
            "Projected semantic steps remove conflicting first-order byte DEV "
            "components without adding byte-only gradient steps; independent "
            "nonlinear DEV and cross-author Dickens CE are also reported. "
            "A shuffled-label projected-gradient arm is matched for alpha "
            "schedule, train order and byte DEV gate, so task label credit "
            "must be assessed against it rather than byte-only alone."
        ),
    }
    Path(sys.argv[3]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("JOINT_GUTENBERG_REPORT PASS", flush=True)

if __name__ == "__main__":
    main()
