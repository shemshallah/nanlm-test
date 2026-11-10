#!/usr/bin/env python3
"""Continue the *same* verified HW5 geometric student from Gutenberg epoch 20.

Optimize only against the training books, select checkpoint only using development
books, and evaluate the book-disjoint test partition once after selection. This is
rank-three aggregate conditional-byte learning, not recurrent BPTT or semantics.
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
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw5_transfer_experiment import ensure_source

PRIOR_SHA = "83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c"
WORDNET_SHA = "e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1"
PRIOR_EPOCH = 20
PRIOR_TRAIN_CE = 2.928596716
ROOT_BYTES = 8_388_608

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def marker(output: str, tag: str) -> dict[str, str]:
    lines = [x for x in output.splitlines() if x.startswith(tag + " ")]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one {tag}, got {len(lines)}")
    d = dict(re.findall(r"([a-z_0-9]+)=([^\s]+)", lines[0]))
    if not d:
        raise ValueError("no native metric fields")
    return d

def run(args: list[str], output: Path) -> tuple[str, float, int]:
    start = time.perf_counter()
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output.write_text(p.stdout)
    return p.stdout, time.perf_counter() - start, p.returncode

def better(candidate: float, previous: float, tolerance: float = 1e-5) -> bool:
    return math.isfinite(candidate) and candidate < previous - tolerance

def check_update(before: float, after: float, accepted: bool,
                 exit_code: int, old_sha: str, new_sha: str) -> None:
    if not all(math.isfinite(z) and z > 0 for z in (before, after)):
        raise ValueError("nonfinite or invalid training cross-entropy")
    if accepted:
        if exit_code or not after < before - 1e-9 or old_sha == new_sha:
            raise ValueError("accepted update did not change model and reduce training CE")
    elif exit_code not in (0, 2) or old_sha != new_sha:
        raise ValueError("rejected update modified model or had unknown failure")

def evaluate(root: Path, split: Path, path: Path) -> dict:
    before = sha(root)
    out, seconds, code = run(["./hw5-onepass", "eval", str(root), str(split)], path)
    if code:
        raise RuntimeError("native evaluation failed: " + out[-800:])
    f = marker(out, "HW5_GEOMETRIC_EVAL")
    expected = split.stat().st_size
    if (int(f["input_bytes"]) != expected or int(f["transitions"]) != expected - 1
            or int(f["bptt_attempted"]) != 0 or sha(root) != before):
        raise ValueError("evaluation corrupted the student or violated byte contract")
    ce, accuracy = float(f["ce_nats"]), float(f["accuracy"])
    if not (math.isfinite(ce) and ce > 0 and math.isfinite(accuracy)
            and 0 <= accuracy <= 1):
        raise ValueError("nonfinite evaluation metric")
    return {"bytes": expected, "transitions": expected - 1,
            "ce_nats": ce, "accuracy": accuracy, "seconds": seconds}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--prior-root", required=True, type=Path)
    ap.add_argument("--prior-report", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--additional-epochs", type=int, default=180)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience-checks", type=int, default=12)
    a = ap.parse_args()
    if not 1 <= a.additional_epochs <= 500 or not 1 <= a.eval_every <= 50:
        raise ValueError("invalid epoch/checkpoint count")
    if not 1 <= a.patience_checks <= 100:
        raise ValueError("invalid early-stopping patience")
    if a.output.exists():
        raise FileExistsError("output directory already exists")
    prior = json.loads(a.prior_report.read_text())
    reference, generation, sizes = ensure_source(a.store)
    if (a.prior_root.stat().st_size != ROOT_BYTES or sha(a.prior_root) != PRIOR_SHA
        or prior["completed_gutenberg_epochs"] != PRIOR_EPOCH
        or prior["final_student_sha256"] != PRIOR_SHA
        or prior["initial_student_root_sha256"] != WORDNET_SHA
        or prior["reference_corpus_sha256"] != reference["artifacts"]["all.txt"]["sha256"]
        or prior["source_split_bytes"] != sizes
        or prior["source_book_count"] != len(reference["books"])
        or abs(prior["final"]["train"]["ce_nats"] - PRIOR_TRAIN_CE) > 1e-7):
        raise ValueError("unverified 20-epoch root, report or source lineage")

    a.output.mkdir(parents=True)
    root = a.output / "learner-root"
    best = a.output / "best-dev-root"
    shutil.copyfile(a.prior_root, root)
    shutil.copyfile(root, best)
    train_file, dev_file = generation / "train.txt", generation / "dev.txt"
    test_file = generation / "test.txt"
    base_dev = evaluate(root, dev_file, a.output / "dev-epoch020.log")
    if abs(base_dev["ce_nats"] - prior["final"]["dev"]["ce_nats"]) > 1e-7:
        raise ValueError("initial development score differs from verified epoch 20")
    initial_root_sha = sha(root)
    report = {
        "schema": "nanlm-test.hw5-longitudinal-bookheldout.v1",
        "method": "one native rank-three aggregate conditional-byte gradient step per full training scan; no BPTT",
        "source_route": prior["source_route"],
        "source_reference_immutable": True,
        "source_corpus_sha256": prior["reference_corpus_sha256"],
        "split_bytes": sizes,
        "book_count": len(reference["books"]),
        "starting_wordnet_root_sha256": WORDNET_SHA,
        "starting_hw5_epoch": PRIOR_EPOCH,
        "starting_hw5_root_sha256": PRIOR_SHA,
        "requested_additional_epochs": a.additional_epochs,
        "eval_every": a.eval_every,
        "patience_checks": a.patience_checks,
        "bptt_attempted": 0,
        "selection_split": "dev books by title; test excluded from checkpoint selection",
        "historical_test_already_reported": True,
        "limitation": "This is a byte bigram head-geometry model, not recurrent semantic learning. HW5 test books are held out from training but were evaluated by earlier PR #38.",
        "count_baselines": prior["count_control"],
        "initial_dev": base_dev,
        "initial_test_from_prior_receipt": prior["final"]["test"],
        "best_dev_epoch": PRIOR_EPOCH,
        "best_dev_ce": base_dev["ce_nats"],
        "best_dev_root_sha256": initial_root_sha,
        "epochs": [],
        "status": "running",
    }
    previous_ce = PRIOR_TRAIN_CE
    previous_sha = initial_root_sha
    best_ce = base_dev["ce_nats"]
    best_epoch = PRIOR_EPOCH
    checks_without_gain = 0
    accepted = 0
    native_seconds = 0.0
    started = time.perf_counter()
    status = "epoch_limit"
    for i in range(1, a.additional_epochs + 1):
        epoch = PRIOR_EPOCH + i
        output, seconds, code = run(
            ["./hw5-onepass", "train", str(root), str(train_file)],
            a.output / f"train-epoch{epoch:03d}.log")
        f = marker(output, "HW5_FULL_GEOMETRIC_PASS")
        before, after = float(f["train_before_ce"]), float(f["train_after_ce"])
        chosen = f["aggregate_step_accepted"] == "1"
        new_sha = sha(root)
        if (int(f["input_bytes"]) != sizes["train"]
            or int(f["transitions"]) != sizes["train"] - 1
            or int(f["bptt_attempted"]) != 0 or int(f["root_bytes"]) != ROOT_BYTES
            or abs(before - previous_ce) > 5e-6):
            raise ValueError(f"broken corpus or optimizer continuity at epoch {epoch}")
        check_update(before, after, chosen, code, previous_sha, new_sha)
        row = {"epoch": epoch, "train_ce_before": before, "train_ce_after": after,
               "train_accuracy_before": float(f["train_before_accuracy"]),
               "train_accuracy_after": float(f["train_after_accuracy"]),
               "accepted": chosen, "step": float(f["step"]),
               "gradient_max": float(f["gradient_max"]),
               "native_seconds": seconds, "root_sha256": new_sha}
        if not all(math.isfinite(float(row[k])) for k in
                   ("train_accuracy_before", "train_accuracy_after", "step", "gradient_max")):
            raise ValueError("nonfinite trainer diagnostics")
        report["epochs"].append(row)
        native_seconds += seconds
        if not chosen:
            status = "native_plateau"
            break
        accepted += 1
        previous_sha = new_sha
        previous_ce = after
        if i <= 2 or i % a.eval_every == 0 or i == a.additional_epochs:
            d = evaluate(root, dev_file, a.output / f"dev-epoch{epoch:03d}.log")
            row["dev"] = d
            if better(d["ce_nats"], best_ce):
                best_ce = d["ce_nats"]
                best_epoch = epoch
                shutil.copyfile(root, best)
                report["best_dev_root_sha256"] = sha(best)
                checks_without_gain = 0
            else:
                checks_without_gain += 1
            row["selected_as_best"] = best_epoch == epoch
            print("HW5_LONGITUDINAL_CHECK " + json.dumps({
                "epoch": epoch, "train_ce": after, "dev_ce": d["ce_nats"],
                "best_dev_epoch": best_epoch, "best_dev_ce": best_ce,
                "dev_top1": d["accuracy"]}, sort_keys=True), flush=True)
            if checks_without_gain >= a.patience_checks:
                status = "dev_early_stop"
                break
        report["completed_additional_epochs"] = accepted
        report["last_root_sha256"] = new_sha
        report["best_dev_epoch"] = best_epoch
        report["best_dev_ce"] = best_ce
        report["total_native_seconds"] = native_seconds
        (a.output / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n")

    if sha(best) != report["best_dev_root_sha256"]:
        raise ValueError("best-dev checkpoint hash continuity failed")
    if sha(root) != previous_sha:
        raise ValueError("final model hash continuity failed")
    # The only NEW test evaluation occurs AFTER dev-based model selection.
    selected_test = evaluate(best, test_file, a.output / "selected-test-final.log")
    selected_train = evaluate(best, train_file, a.output / "selected-train-final.log")
    selected_dev = evaluate(best, dev_file, a.output / "selected-dev-final.log")
    last_dev = evaluate(root, dev_file, a.output / "last-dev-final.log")
    if abs(selected_dev["ce_nats"] - best_ce) > 1e-7:
        raise ValueError("selected checkpoint mismatch")
    report.update({
        "status": status,
        "completed_additional_epochs": accepted,
        "last_hw5_epoch": PRIOR_EPOCH + accepted,
        "best_dev_epoch": best_epoch,
        "best_dev_ce": best_ce,
        "last_model_sha256": previous_sha,
        "total_native_seconds": native_seconds,
        "total_train_bytes_scanned": (accepted + (status == "native_plateau")) * sizes["train"],
        "wall_seconds": time.perf_counter() - started,
        "selected": {"train": selected_train, "dev": selected_dev, "test": selected_test},
        "last_dev": last_dev,
        "count_test_ce": prior["count_control"]["test"]["ce_nats"],
        "count_test_accuracy": prior["count_control"]["test"]["accuracy"],
        "generalization_delta_test_ce": (prior["final"]["test"]["ce_nats"]
                                        - selected_test["ce_nats"]),
    })
    (a.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "epochs"}
    (a.output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print("HW5_LONGITUDINAL_FINAL " + json.dumps({
        "status": status, "extra_accepted": accepted,
        "best_dev_epoch": best_epoch, "best_dev_ce": best_ce,
        "test_ce_before": prior["final"]["test"]["ce_nats"],
        "test_ce_selected": selected_test["ce_nats"],
        "test_top1_selected": selected_test["accuracy"],
        "count_test_ce": report["count_test_ce"],
        "best_sha256": report["best_dev_root_sha256"],
    }, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
