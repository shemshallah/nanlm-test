#!/usr/bin/env python3
"""Native geometric rank-three continuation from verified HW5 epoch 200 to <=1000.

The previous 8 MiB learner stays intact. One aggregate native gradient update per
complete training split scan. Only the disjoint dev books select a checkpoint;
test is run after checkpoint selection and is not an untouched virgin test set
(because earlier experimental runs reported its scores).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw5_longitudinal import (sha, marker, run, better, check_update, evaluate, ROOT_BYTES)
from hw5_transfer_experiment import ensure_source

PARENT_EPOCH = 200
PARENT_SHA = "a975fc792ec6b505d9afb41424c2ea7e52649e61eed43070d217e77ccc4d9df1"
PRE_PARENT_SHA = "83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c"
PARENT_TRAIN_CE = 2.73078818
PARENT_DEV_CE = 2.699913994

def verify_parent(prior: dict, learner: Path, manifest: dict, sizes: dict) -> None:
    if not (learner.stat().st_size == ROOT_BYTES and sha(learner) == PARENT_SHA
        and prior["schema"] == "nanlm-test.hw5-longitudinal-bookheldout.v1"
        and prior["status"] == "epoch_limit"
        and prior["last_hw5_epoch"] == PARENT_EPOCH
        and prior["best_dev_epoch"] == PARENT_EPOCH
        and prior["best_dev_root_sha256"] == PARENT_SHA
        and prior["last_model_sha256"] == PARENT_SHA
        and prior["starting_hw5_root_sha256"] == PRE_PARENT_SHA
        and prior["completed_additional_epochs"] == 180
        and prior["source_corpus_sha256"] == manifest["artifacts"]["all.txt"]["sha256"]
        and prior["split_bytes"] == sizes
        and abs(prior["selected"]["train"]["ce_nats"] - PARENT_TRAIN_CE) < 1e-7
        and abs(prior["selected"]["dev"]["ce_nats"] - PARENT_DEV_CE) < 1e-7):
        raise ValueError("parent epoch-200 receipt/root/source lineage invalid")

def experiment(store: Path, prior_root: Path, prior_report: Path,
               output: Path, epochs: int, eval_every: int, patience: int) -> dict:
    if output.exists():
        raise FileExistsError("output directory must not already exist")
    manifest, generation, sizes = ensure_source(store)
    previous = json.loads(prior_report.read_text())
    verify_parent(previous, prior_root, manifest, sizes)
    output.mkdir(parents=True)
    root, best = output / "learner-root", output / "best-dev-root"
    shutil.copyfile(prior_root, root)
    shutil.copyfile(prior_root, best)
    train, dev, test = (generation / (x + ".txt") for x in ("train", "dev", "test"))
    baseline = evaluate(root, dev, output / "dev-epoch200.log")
    if abs(baseline["ce_nats"] - PARENT_DEV_CE) > 1e-7:
        raise ValueError("checkpoint evaluation disagrees with previous audit")
    bce = PARENT_DEV_CE
    b_epoch = PARENT_EPOCH
    b_sha = PARENT_SHA
    p_sha = PARENT_SHA
    p_ce = PARENT_TRAIN_CE
    accepted = 0
    no_gain_checks = 0
    train_seconds = 0
    start = time.perf_counter()
    status = "epoch_limit"
    report = {
        "schema": "nanlm-test.hw5-200-to-1000.v1",
        "continuation_from_epoch": PARENT_EPOCH,
        "starting_root_sha256": PARENT_SHA,
        "source_route": previous["source_route"],
        "source_corpus_sha256": previous["source_corpus_sha256"],
        "source_reference_immutable": True,
        "split_bytes": sizes,
        "previous_run_test": previous["selected"]["test"],
        "previous_run_dev": previous["selected"]["dev"],
        "count_baselines": previous["count_baselines"],
        "requested_additional_epochs": epochs,
        "max_cumulative_epoch": PARENT_EPOCH + epochs,
        "selection_split": "dev books; no test-based checkpoint selection",
        "test_already_evaluated_in_prior_studies": True,
        "bptt_attempted": 0,
        "method": "single native rank-3 aggregate byte-pair geometric update per full scan; not token-sequential TRAIL",
        "epochs": [],
    }
    for step in range(1, epochs + 1):
        epoch = PARENT_EPOCH + step
        output_text, seconds, code = run(
            ["./hw5-onepass", "train", str(root), str(train)],
            output / f"train-epoch{epoch:04d}.log")
        f = marker(output_text, "HW5_FULL_GEOMETRIC_PASS")
        before, after = float(f["train_before_ce"]), float(f["train_after_ce"])
        is_accepted = f["aggregate_step_accepted"] == "1"
        changed_sha = sha(root)
        if (int(f["input_bytes"]) != sizes["train"]
            or int(f["transitions"]) != sizes["train"] - 1
            or int(f["root_bytes"]) != ROOT_BYTES
            or int(f["bptt_attempted"]) != 0 or abs(before - p_ce) > 5e-6):
            raise ValueError("trainer/continuation contract broken at epoch " + str(epoch))
        check_update(before, after, is_accepted, code, p_sha, changed_sha)
        row = {"epoch": epoch, "accepted": is_accepted,
               "train_ce_before": before, "train_ce_after": after,
               "top1_after": float(f["train_after_accuracy"]),
               "gradient_max": float(f["gradient_max"]),
               "step": float(f["step"]), "native_seconds": seconds,
               "model_sha256": changed_sha}
        if not all(math.isfinite(row[k]) for k in
                   ("top1_after", "gradient_max", "step")):
            raise ValueError("nonfinite diagnostics")
        report["epochs"].append(row)
        train_seconds += seconds
        if not is_accepted:
            status = "native_plateau"
            break
        accepted += 1
        p_ce = after
        p_sha = changed_sha
        if step <= 2 or step % eval_every == 0 or step == epochs:
            m = evaluate(root, dev, output / f"dev-epoch{epoch:04d}.log")
            row["dev"] = m
            if better(m["ce_nats"], bce):
                bce, b_epoch = m["ce_nats"], epoch
                shutil.copyfile(root, best)
                b_sha = sha(best)
                no_gain_checks = 0
            else:
                no_gain_checks += 1
            print("HW5_POST200_CHECK " + json.dumps({
                "epoch": epoch, "train_ce": p_ce, "dev_ce": m["ce_nats"],
                "dev_top1": m["accuracy"], "best_epoch": b_epoch,
                "best_dev_ce": bce}, sort_keys=True), flush=True)
            if no_gain_checks >= patience:
                status = "dev_early_stop"
                break
        report.update({"completed_additional_epochs": accepted,
                       "last_epoch": PARENT_EPOCH + accepted,
                       "last_root_sha256": p_sha,
                       "best_dev_ce": bce,
                       "best_dev_epoch": b_epoch,
                       "best_dev_root_sha256": b_sha,
                       "native_training_seconds": train_seconds})
        (output / "report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n")

    if sha(best) != b_sha or sha(root) != p_sha:
        raise ValueError("model root hash mismatch after training")
    chosen = {
        "dev": evaluate(best, dev, output / "selected-dev.log"),
        "test": evaluate(best, test, output / "selected-test.log"),
        "train": evaluate(best, train, output / "selected-train.log"),
    }
    if abs(chosen["dev"]["ce_nats"] - bce) > 1e-7:
        raise ValueError("selected dev score not reproducible")
    report.update({
        "status": status, "completed_additional_epochs": accepted,
        "last_epoch": PARENT_EPOCH + accepted,
        "last_root_sha256": p_sha, "best_dev_root_sha256": b_sha,
        "best_dev_ce": bce, "best_dev_epoch": b_epoch,
        "selected": chosen,
        "native_training_seconds": train_seconds,
        "training_scanned_bytes": len(report["epochs"]) * sizes["train"],
        "elapsed_seconds": time.perf_counter() - start,
        "count_test_ce": previous["count_baselines"]["test"]["ce_nats"],
        "count_test_accuracy": previous["count_baselines"]["test"]["accuracy"],
    })
    (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (output / "summary.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "epochs"}, sort_keys=True, indent=2) + "\n")
    print("HW5_POST200_FINAL " + json.dumps({
        "status": status, "last_epoch": report["last_epoch"],
        "accepted_updates": accepted, "best_dev_epoch": b_epoch,
        "best_dev_ce": bce, "test_ce": chosen["test"]["ce_nats"],
        "test_top1": chosen["test"]["accuracy"],
        "count_ce": report["count_test_ce"], "best_sha256": b_sha,
        "native_seconds": train_seconds}, sort_keys=True), flush=True)
    return report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--prior-root", required=True, type=Path)
    ap.add_argument("--prior-report", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--additional-epochs", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience-checks", type=int, default=20)
    a = ap.parse_args()
    if not 1 <= a.additional_epochs <= 2000 or not 1 <= a.eval_every <= 100:
        raise ValueError("invalid epoch/evaluation bounds")
    if not 1 <= a.patience_checks <= 100:
        raise ValueError("invalid patience")
    experiment(a.store, a.prior_root, a.prior_report,
               a.output, a.additional_epochs, a.eval_every, a.patience_checks)

if __name__ == "__main__":
    main()
