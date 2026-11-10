#!/usr/bin/env python3
"""Audit exactly three further sequential HW4 full-corpus TRAIL sweeps."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sys

CORPUS_BYTES = 58_341_529
START_ROOT_SHA256 = "a95a89f2856264d190d996363d39dc11bb8bb5ca8f30eae98c3cc915a2433695"
SOURCE_SHA256 = "6c29dc5027012a8ef6ec50404a7bdbb50be81a4d018c1f327665f826d8784604"
STAGES = (1, 2, 3, 4)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fields(path: Path, marker: str) -> dict[str, str]:
    lines = [line for line in path.read_text().splitlines()
             if line.startswith(marker + " ")]
    if len(lines) != 1:
        raise AssertionError(f"expected exactly one {marker} at {path}")
    return dict(re.findall(r"([a-z_0-9]+)=([^\s]+)", lines[0]))

def main() -> None:
    output = Path(sys.argv[1])
    source = json.loads(Path(sys.argv[2]).read_text())
    root = output / "learner-root"
    assert source["corpus_sha256"] == SOURCE_SHA256
    assert source["corpus_bytes"] == CORPUS_BYTES
    assert root.stat().st_size == 8_388_608
    receipt = {
        "schema": "nanlm-test.hw4-three-additional-trail-passes.v1",
        "corpus_sha256": SOURCE_SHA256,
        "source_hw4_root_sha256": source["source_hw4_root_sha256"],
        "source_hw4_sqlite_sha256": source["source_hw4_sqlite_sha256"],
        "hw4_immutable": True,
        "bptt_attempted": 0,
        "batch_method": "frozen-value two-rail TD eligibility, one scratch-tested rank3 commit per complete byte pass; not online recurrent training",
        "initial_completed_passes": 1,
        "additional_full_passes": 3,
        "total_completed_passes": 4,
        "bytes_per_full_pass": CORPUS_BYTES,
        "additional_bytes_seen": CORPUS_BYTES * 3,
        "all_four_passes_bytes_seen": CORPUS_BYTES * 4,
        "starting_student_sha256": START_ROOT_SHA256,
        "passes": [],
    }
    previous_root = START_ROOT_SHA256
    initial_prose = fields(output / "prose-pass1.log", "UNIFIED_EVAL")
    assert int(initial_prose["bytes"]) == 974
    for epoch in (2, 3, 4):
        trail = fields(output / f"trail-pass{epoch}.log", "HW4_FULL_TRAIL")
        previous_prose = fields(
            output / f"prose-pass{epoch-1}.log", "UNIFIED_EVAL")
        prose = fields(output / f"prose-pass{epoch}.log", "UNIFIED_EVAL")
        model_hash = (output / f"sha-pass{epoch}.txt").read_text().strip()
        for flag in ("forward_rail", "inverse_rail"):
            assert trail[flag] == "1"
        assert trail["bptt_attempted"] == "0"
        assert trail["hw4_read_only"] == "1"
        assert int(trail["input_bytes"]) == CORPUS_BYTES
        assert int(trail["transition_updates"]) == CORPUS_BYTES - 1
        assert int(trail["root_bytes"]) == 8_388_608
        assert int(prose["bytes"]) == int(previous_prose["bytes"]) == 974
        before_ce = float(trail["initial_teacher_ce"])
        after_ce = float(trail["final_teacher_ce"])
        assert after_ce <= before_ce + 1e-8
        before_prose_ce = float(previous_prose["ce_nats"])
        after_prose_ce = float(prose["ce_nats"])
        item = {
            "epoch": epoch,
            "corpus_bytes_seen": CORPUS_BYTES,
            "starting_root_sha256": previous_root,
            "finished_root_sha256": model_hash,
            "accepted": trail["accepted"] == "1",
            "training_ce_before": before_ce,
            "training_ce_after": after_ce,
            "training_accuracy_before": float(trail["initial_teacher_acc"]),
            "training_accuracy_after": float(trail["final_teacher_acc"]),
            "prose_ce_before": before_prose_ce,
            "prose_ce_after": after_prose_ce,
            "prose_accuracy_before": float(previous_prose["accuracy"]),
            "prose_accuracy_after": float(prose["accuracy"]),
            "positive_rail_l1": float(trail["geom_credit_plus_l1"]),
            "inverse_rail_l1": float(trail["geom_credit_minus_l1"]),
            "scan_seconds": float(trail["full_scan_seconds"]),
            "native_total_seconds": float(trail["total_seconds"]),
            "bytes_per_second": float(trail["corpus_bytes_per_second"]),
            "step_size": float(trail["step"]),
        }
        if epoch == 2:
            assert abs(before_prose_ce - 5.54283413) < 1e-6, (
                "loaded student was not original passed TRAIL checkpoint")
            assert abs(before_ce - 5.5636531316) < 1e-6, (
                "starting WordNet teacher CE is not first-pass checkpoint")
        if not item["accepted"]:
            assert model_hash == previous_root, (
                "rejected update changed student root")
        receipt["passes"].append(item)
        previous_root = model_hash
        print("HW4_TRAIL_ADDITIONAL_PASS " + json.dumps(
            item, sort_keys=True), flush=True)
    assert previous_root == sha(root), (
        "final root hash does not match final epoch")
    receipt["finished_student_sha256"] = previous_root
    receipt["scan_seconds_three_passes"] = sum(
        item["scan_seconds"] for item in receipt["passes"])
    receipt["native_total_seconds_three_passes"] = sum(
        item["native_total_seconds"] for item in receipt["passes"])
    receipt["initial_prose_ce"] = float(initial_prose["ce_nats"])
    receipt["final_prose_ce"] = receipt["passes"][-1]["prose_ce_after"]
    receipt["initial_prose_accuracy"] = float(initial_prose["accuracy"])
    receipt["final_prose_accuracy"] = receipt["passes"][-1]["prose_accuracy_after"]
    target = output / "three-more-receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("HW4_TRAIL_THREE_MORE_FINAL " + json.dumps(receipt, sort_keys=True),
          flush=True)

if __name__ == "__main__":
    main()
