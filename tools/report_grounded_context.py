#!/usr/bin/env python3
"""Report corpus-derived contextual-order controls, including negative transfer."""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path

FIELDS = (
    "seed", "readout_init_seed", "train_pairs", "test_pairs",
    "heldout_byte_tokens", "truth_shared_accepted", "truth_shared_rejected",
    "sham_shared_accepted", "sham_shared_rejected", "alice_dev_byte_ce",
    "alice_dev_truth_ce", "alice_dev_sham_ce",
    "dickens_byte_ce", "dickens_truth_ce", "dickens_sham_ce",
    "dickens_frozen_ce", "dickens_truth_minus_byte",
    "dickens_sham_minus_byte", "dickens_truth_minus_sham",
    "context_truth_ce", "context_sham_ce", "context_frozen_ce",
    "context_truth_acc", "context_sham_acc", "context_frozen_acc",
    "root_bytes", "journal_events",
)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: report_grounded_context.py RECEIPT MANIFEST REPORT")
    raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    if "GROUNDED_CONTEXT_AUDIT PASS" not in raw:
        raise RuntimeError("grounded audit did not complete successfully")
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    rows: list[dict[str, float]] = []
    for line in raw.splitlines():
        if not line.startswith("GROUNDED_CONTEXT_SEED "):
            continue
        fields = {}
        for word in line.split()[1:]:
            key, value = word.split("=", 1)
            fields[key] = float(value)
            if not math.isfinite(fields[key]):
                raise ValueError(f"non-finite {key}")
        if any(field not in fields for field in FIELDS):
            raise ValueError("missing numerical receipt field")
        rows.append(fields)
    if len(rows) != 3 or len({r["seed"] for r in rows}) != 3:
        raise ValueError("expected three independent training-order seeds")
    if len({r["readout_init_seed"] for r in rows}) != 3:
        raise ValueError("readout initialization seeds must differ")
    for row in rows:
        if row["root_bytes"] != 8388608 or row["journal_events"] <= 0:
            raise ValueError("physical root/journal was not preserved")
        if row["train_pairs"] < 12 or row["test_pairs"] < 6:
            raise ValueError("not enough source-derived contexts")
        expected = (
            manifest["splits"]["test_carol_early"]["bytes"] - 1
            + manifest["splits"]["test_carol_late"]["bytes"] - 1
        )
        if row["heldout_byte_tokens"] != expected:
            raise ValueError("source-pinned evaluation count changed")
        for variant in ("truth", "sham"):
            a = row[f"{variant}_shared_accepted"]
            b = row[f"{variant}_shared_rejected"]
            if a != int(a) or b != int(b) or a + b != 6:
                raise ValueError("inaccurate shared update accounting")
            if row[f"alice_dev_{variant}_ce"] > row["alice_dev_byte_ce"] + 1e-9:
                raise ValueError("projected context training damaged byte DEV")
        if abs(row["dickens_frozen_ce"] - row["dickens_byte_ce"]) > 1e-10:
            raise ValueError("semantic-only head changed language probabilities")
        for metric in ("context_truth_acc", "context_sham_acc", "context_frozen_acc"):
            if not 0 <= row[metric] <= 1:
                raise ValueError(f"invalid classification accuracy: {metric}")
    fields = (
        "dickens_truth_minus_byte", "dickens_sham_minus_byte",
        "dickens_truth_minus_sham", "context_truth_ce",
        "context_sham_ce", "context_frozen_ce",
        "context_truth_acc", "context_sham_acc", "context_frozen_acc",
    )
    means = {name: statistics.mean(row[name] for row in rows) for name in fields}
    for row in rows:
        print(
            f"GROUNDED_CONTEXT_PAIR seed={int(row['seed'])} "
            f"init_seed={int(row['readout_init_seed'])} "
            f"true_shared={int(row['truth_shared_accepted'])} "
            f"sham_shared={int(row['sham_shared_accepted'])} "
            f"true_minus_byte={row['dickens_truth_minus_byte']:+.9f} "
            f"sham_minus_byte={row['dickens_sham_minus_byte']:+.9f} "
            f"true_minus_sham={row['dickens_truth_minus_sham']:+.9f} "
            f"context_true_ce={row['context_truth_ce']:.9f} "
            f"context_sham_ce={row['context_sham_ce']:.9f} "
            f"context_true_acc={row['context_truth_acc']:.6f} "
            f"context_sham_acc={row['context_sham_acc']:.6f}",
            flush=True,
        )
    print(
        "GROUNDED_CONTEXT_AGGREGATE seeds=3 "
        f"truth_minus_byte_mean={means['dickens_truth_minus_byte']:+.9f} "
        f"sham_minus_byte_mean={means['dickens_sham_minus_byte']:+.9f} "
        f"truth_minus_sham_mean={means['dickens_truth_minus_sham']:+.9f} "
        f"context_truth_ce_mean={means['context_truth_ce']:.9f} "
        f"context_sham_ce_mean={means['context_sham_ce']:.9f} "
        f"context_truth_acc_mean={means['context_truth_acc']:.6f} "
        f"context_sham_acc_mean={means['context_sham_acc']:.6f} "
        "significant=unassessed", flush=True,
    )
    report = {
        "sources": manifest,
        "rows": rows,
        "mean_metrics": means,
        "notes": (
            "Causal Gutenberg original versus interior-token-swap discrimination; "
            "the two anchors retain identical order in both classes. "
            "The real book provides training examples, not gold WordNet senses. "
            "Alice DEV gates byte nonregression, Dickens labels and bytes are "
            "never optimizer inputs. Three seeds share one book pair and model "
            "family; no population significance, semantic competence or natural "
            "language understanding claim follows. Favor matched shuffled "
            "controls over byte-only comparisons."
        ),
    }
    Path(sys.argv[3]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("GROUNDED_CONTEXT_REPORT PASS", flush=True)


if __name__ == "__main__":
    main()
