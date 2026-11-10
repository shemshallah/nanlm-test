#!/usr/bin/env python3
"""HW5:0 Gutenberg train-only CP8 geometric learning, strict dev/test freeze.

Predictor and optimizer are entirely native C++, never Python. This script
checks immutable evidence and orchestrates native training/evaluation. CP8
lives in the original 8MiB C256 head-register region and does not allocate
a second universe, overwrite the protected head 255, or mutate mirror pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from hw5_context_experiment import STUDENT_SHA, load_manifest, sha

EPOCHS = 8
EXPECTED_ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW5:0"
RANK = 8
REGISTERS_START = 256 * 128


def native(argv, log: Path):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=False)
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f"native failed exit={p.returncode}: {p.stdout[-2000:]}")
    return p.stdout


def parse_scores(output: str):
    results = []
    for line in output.splitlines():
        if not line.startswith("HW5_CP8_SCORE "):
            continue
        vals = dict(re.findall(r"([a-z0-9_]+)=([^\s]+)", line))
        if vals["mode"] != "cp8" or vals["exact_model"] != "HW5_train_only" or vals["bptt_attempted"] != "0":
            raise ValueError("native attribution contract violated")
        row = {k: float(vals[k]) for k in ("beta", "rho", "scale", "ce_nats", "accuracy")}
        row["transitions"] = int(vals["transitions"])
        if not (row["beta"] == 16.0 and 0 <= row["rho"] <= 1 and
                0 <= row["scale"] <= 4 and 0 < row["ce_nats"] < 10 and
                0 <= row["accuracy"] <= 1 and row["transitions"] > 0):
            raise ValueError("invalid native CP8 scores")
        results.append(row)
    if not results:
        raise ValueError("native scorer emitted no CP8 evaluation rows")
    return results


def check_root_preserves_memory(source: Path, candidate: Path):
    if source.stat().st_size != 8388608 or candidate.stat().st_size != 8388608:
        raise ValueError("canonical 8MiB root changed size")
    original = source.read_bytes()
    trained = candidate.read_bytes()
    if trained[REGISTERS_START:] != original[REGISTERS_START:]:
        raise ValueError("CP8 wrote into the physical mirror-pair bank")
    for a in range(256):
        start = a * 128
        if a == 255 and trained[start:start + 128] != bytes(128):
            raise ValueError("CP8 changed gateway head 255")
        if trained[start:start + 29] != original[start:start + 29]:
            raise ValueError("CP8 changed canonical metadata or counts")
    if trained[29:32] != b"CP8":
        raise ValueError("CP8 checkpoint lacks native head-register schema")
    if not any(trained[a * 128 + 32:a * 128 + 128] != original[a * 128 + 32:a * 128 + 128]
               for a in range(255)):
        raise ValueError("CP8 failed to learn persistent head factors")
    return sha(candidate)


def run(reference: Path, student: Path, hw5: Path, output: Path):
    if output.exists():
        raise FileExistsError("experiment destination must be new")
    gen, manifest, splits = load_manifest(reference)
    hw5_report = json.loads((hw5 / "report.json").read_text())
    if hw5_report["source_reference"] != EXPECTED_ROUTE:
        raise ValueError("cross-slot source contamination")
    if sha(student) != STUDENT_SHA or hw5_report["student_sha256"] != STUDENT_SHA:
        raise ValueError("wrong HW5:0 geometric student")
    if not hw5_report["source_reference_immutable"] or hw5_report["memory_trained_on_documents"] != "train_only":
        raise ValueError("HW5 trained memory source is not train-only")
    bank = hw5 / "train-only-c256.root"
    triples = hw5 / "train-only-triples.bin"
    if sha(bank) != hw5_report["train_mirror_sha256"] or sha(triples) != hw5_report["train_triple_sha256"]:
        raise ValueError("changed source memory bank or triple ledger")
    source_digests = {"student": sha(student), "bank": sha(bank), "triples": sha(triples)}
    output.mkdir(parents=True)
    for split in ("dev", "test"):
        (output / f"{split}.lengths").write_text(
            "".join(f"{int(d['bytes'])}\n" for d in splits[split]))
    logs = output / "native-logs"
    logs.mkdir()
    with tempfile.TemporaryDirectory(prefix="hw5-cp8-train-") as temp:
        prefix = Path(temp) / "checkpoint"
        trainlog = native(["./hw5-cp8", "fit-cp", str(bank), str(triples), str(prefix),
                           str(EPOCHS), "5315127", "64"],
                          logs / "train-only-geometric-credit.log")
        lines = [x for x in trainlog.splitlines() if x.startswith("HW5_CP8_TRAIN ")]
        if len(lines) != EPOCHS or any(f"epoch={i} " not in s for i, s in enumerate(lines, 1)):
            raise ValueError("CP8 native training epoch ledger incomplete")
        dev_scores = []
        expected_dev = sum(int(d["bytes"]) - 2 for d in splits["dev"])
        for epoch in range(1, EPOCHS + 1):
            checkpoint = Path(str(prefix) + f".e{epoch:02d}.root")
            check_root_preserves_memory(bank, checkpoint)
            text = native(["./hw5-cp8", "score-cp-sweep", str(student), str(checkpoint),
                           str(triples), str(gen / "dev.txt"), str(output / "dev.lengths")],
                          logs / f"dev-epoch-{epoch:02d}.log")
            candidates = parse_scores(text)
            if len(candidates) != 19:  # rho=0 once + six positive rhos * three scales
                raise ValueError("incomplete or modified dev-only search grid")
            for row in candidates:
                if row["transitions"] != expected_dev:
                    raise ValueError("dev boundary ledger invalid")
                row["epoch"] = epoch
                dev_scores.append(row)
        best = min(dev_scores, key=lambda r: (r["ce_nats"], r["epoch"], r["rho"], r["scale"]))
        selected_path = output / "selected-geometric-cp8.root"
        shutil.copyfile(Path(str(prefix) + f".e{best['epoch']:02d}.root"), selected_path)
        selected_sha = check_root_preserves_memory(bank, selected_path)
    # Everything above is TRAIN or DEV only. Freeze parameters and SHA BEFORE
    # reading or evaluating any test book. Never tune on the test readout.
    freeze = {"beta": best["beta"], "rho": best["rho"], "scale": best["scale"],
              "epoch": best["epoch"], "selected_cp8_root_sha256": selected_sha,
              "selected_exclusively_on": "development_books"}
    (output / "frozen-dev-choice.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    (output / "dev-score-ledger.json").write_text(json.dumps(dev_scores, indent=2, sort_keys=True) + "\n")
    expected_test = sum(int(d["bytes"]) - 2 for d in splits["test"])
    scored = native(["./hw5-cp8", "score-cp", str(student), str(selected_path),
                     str(triples), str(gen / "test.txt"), str(output / "test.lengths"),
                     str(freeze["beta"]), str(freeze["rho"]), str(freeze["scale"])],
                    logs / "test-single-frozen-choice.log")
    actual = parse_scores(scored)
    if len(actual) != 1 or actual[0]["transitions"] != expected_test:
        raise ValueError("native HW5:0 test coverage or output changed")
    for field in ("beta", "rho", "scale"):
        if actual[0][field] != freeze[field]:
            raise ValueError("native test configuration changed after freeze")
    if {k: sha(v) for k, v in
            {"student": student, "bank": bank, "triples": triples}.items()} != source_digests:
        raise ValueError("native training/evaluation mutated preexisting evidence")
    baseline = hw5_report["test"]["best_memory_only"]
    result = {
        "schema": "nanlm-test.hw5-cp8-geometry.v1",
        "route": EXPECTED_ROUTE,
        "one_8mib_logical_universe": True,
        "native_trained_rank": RANK,
        "geometry_train_data": "HW5:0 Gutenberg train-only exact trigrams",
        "development_candidates": len(dev_scores),
        "frozen_on_development": freeze,
        "training_epochs": EPOCHS,
        "train_memory_sha256": source_digests["bank"],
        "train_triple_sha256": source_digests["triples"],
        "original_student_sha256": source_digests["student"],
        "baseline_immutable_hw5_test": {
            "ce_nats": baseline["ce_nats"], "accuracy": baseline["accuracy"]},
        "test_native_cp8": actual[0],
        "test_delta_ce_memory_minus_cp8": baseline["ce_nats"] - actual[0]["ce_nats"],
        "test_delta_accuracy_cp8_minus_memory":
            actual[0]["accuracy"] - baseline["accuracy"],
        "caveat": "No demonstration of semantic understanding or advantage from geometry is asserted without positive independently verified held-out results."
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("HW5_CP8_FINAL", json.dumps({
        "route": EXPECTED_ROUTE,
        "epochs": EPOCHS,
        "dev_candidates": len(dev_scores),
        "frozen_dev": freeze,
        "memory_test_ce": baseline["ce_nats"],
        "trained_cp8_test_ce": actual[0]["ce_nats"],
        "trained_cp8_test_accuracy": actual[0]["accuracy"],
        "delta_ce": result["test_delta_ce_memory_minus_cp8"],
        "student_immutable": True, "mirror_immutable": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--hw5-artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    x = p.parse_args()
    run(x.reference, x.student, x.hw5_artifact, x.output)
