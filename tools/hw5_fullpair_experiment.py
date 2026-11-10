#!/usr/bin/env python3
"""Full-capacity native HW5 pair-bank experiment with honest book-heldout audit.

Uses ONLY one logical 8 MiB student at a time: temp hyperparameter candidates
are isolated copies; the final retained root is trained on the full HW5
Gutenberg 20,597,424-byte reference. The locked test is evaluated once on
the dev-selected *train-only* candidate BEFORE the final all-books refit.
The final all-books root has no fresh heldout test claim.
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

EXPECTED_PRIOR_SHA = "83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c"
EXPECTED_WORDNET_SHA = "e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1"
ROOT_BYTES = 8_388_608
GRID = (16, 64, 128, 256, 512, 1024, 4096, 16384)
ALGORITHM = "native full-pair aggregate Dirichlet-posterior with frozen geometric prior; no BPTT, no per-token TRAIL"

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fields(log: str, marker: str) -> dict[str, str]:
    lines = [line for line in log.splitlines() if line.startswith(marker + " ")]
    if len(lines) != 1:
        raise ValueError("expected one " + marker)
    result = dict(re.findall(r"([a-z_0-9]+)=([^\s]+)", lines[0]))
    if not result:
        raise ValueError("empty metrics from " + marker)
    return result

def run(args: list[str], log_file: Path) -> tuple[dict[str,str], float]:
    start = time.perf_counter()
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    seconds = time.perf_counter() - start
    log_file.write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{args[0]} failed {proc.returncode}: {proc.stdout[-1500:]}")
    label = "HW5_FULLPAIR_TRAIN" if args[1] == "train" else "HW5_FULLPAIR_EVAL"
    return fields(proc.stdout, label), seconds

def check_native(row: dict[str,str], expected_bytes: int, plane: int) -> None:
    if (int(row["input_bytes"]) != expected_bytes
        or int(row["transitions"]) != expected_bytes - 1
        or int(row["physical_pairs"]) != 32640
        or int(row["root_bytes"]) != ROOT_BYTES
        or int(row["bptt_attempted"]) != 0):
        raise ValueError("broken native full-pair geometry/bytes contract")
    if "pair_plane" in row and int(row["pair_plane"]) != plane:
        raise ValueError("pair-bank marker was not restored")

def cleanup_model(path: Path) -> None:
    for name in (str(path), str(path) + ".overlay",
                 str(path) + ".overlay.tmp", str(path) + ".so3"):
        Path(name).unlink(missing_ok=True)

def clone_model(source: Path, target: Path) -> None:
    shutil.copy2(source, target)
    for suffix in (".overlay", ".so3"):
        existing = Path(str(source) + suffix)
        if existing.exists():
            shutil.copy2(existing, Path(str(target) + suffix))

def experiment(store: Path, prior_root: Path, prior_report: Path,
               output: Path) -> dict:
    if output.exists():
        raise FileExistsError("experiment output must be an unused directory")
    manifest, generation, split_bytes = ensure_source(store)
    prior = json.loads(prior_report.read_text())
    if (prior_root.stat().st_size != ROOT_BYTES or digest(prior_root) != EXPECTED_PRIOR_SHA
        or prior["completed_gutenberg_epochs"] != 20
        or prior["final_student_sha256"] != EXPECTED_PRIOR_SHA
        or prior["initial_student_root_sha256"] != EXPECTED_WORDNET_SHA
        or prior["reference_corpus_sha256"] != manifest["artifacts"]["all.txt"]["sha256"]
        or prior["source_split_bytes"] != split_bytes):
        raise ValueError("HW5 20-epoch student or source lineage invalid")
    output.mkdir(parents=True)
    started = time.perf_counter()
    train = generation / "train.txt"
    dev = generation / "dev.txt"
    test = generation / "test.txt"
    all_books = generation / "all.txt"
    if all_books.stat().st_size != manifest["corpus_bytes"] or manifest["corpus_bytes"] != 20_597_424:
        raise ValueError("not the pinned complete HW5 Gutenberg corpus")
    record = {
        "schema": "nanlm-test.hw5-fullpair-gutenberg.v1",
        "method": ALGORITHM,
        "source_route": manifest["route"],
        "source_owner": manifest["ring_owner"],
        "source_reference_sha256": manifest["artifacts"]["all.txt"]["sha256"],
        "immutable_hw5_root_sha256": manifest["artifacts"]["gutenberg-c256.root"]["sha256"],
        "immutable_triple_ledger_sha256": manifest["artifacts"]["gutenberg-triples.sqlite3"]["sha256"],
        "source_book_count": len(manifest["books"]),
        "all_book_bytes": manifest["corpus_bytes"],
        "book_disjoint_split_bytes": split_bytes,
        "parent_model_sha256": EXPECTED_PRIOR_SHA,
        "parent_wordnet_sha256": EXPECTED_WORDNET_SHA,
        "capacity": {"physical_pair_records": 32640, "head_registers": 256,
                     "canonical_cells": 32896, "root_bytes": ROOT_BYTES,
                     "directed_vocab_logits": 255 * 255,
                     "gateway_head": 255, "diagonal_slots": 255},
        "geometry_only_historical": prior["final"],
        "independent_count_baseline": prior["count_control"],
        "concentrations": list(GRID),
        "test_books_evaluated_once_after_dev_selection": True,
        "earlier_pr38_evaluated_these_test_books": True,
        "final_all_books_root_not_heldout": True,
        "model_plane_isolated_from_relation_journal": True,
        "bptt_attempted": 0,
        "candidates": [],
    }
    best_ce = float("inf")
    best_lambda: int | None = None
    best = output / "dev-selected-root"
    candidates = record["candidates"]
    for concentration in GRID:
        candidate = output / f"candidate-{concentration}.root"
        clone_model(prior_root, candidate)
        f, seconds = run(["./hw5-fullpair", "train", str(candidate),
                          str(train), str(concentration)],
                         output / f"train-lambda-{concentration}.log")
        check_native(f, split_bytes["train"], 1)
        if not (math.isfinite(float(f["ce_before"]))
                and math.isfinite(float(f["ce_after"]))
                and float(f["ce_after"]) <= float(f["ce_before"]) + 1e-6):
            raise ValueError("native fitting worsened training CE")
        root_after = digest(candidate)
        if root_after == EXPECTED_PRIOR_SHA:
            raise ValueError("learned pair plane did not modify the student")
        d, dev_seconds = run(["./hw5-fullpair", "eval", str(candidate),
                              str(dev)],
                             output / f"dev-lambda-{concentration}.log")
        check_native(d, split_bytes["dev"], 1)
        if digest(candidate) != root_after:
            raise ValueError("native dev evaluation mutated the student")
        ce = float(d["ce_nats"])
        if not (math.isfinite(ce) and ce > 0.0):
            raise ValueError("invalid dev cross entropy")
        item = {
            "concentration": concentration, "model_sha256": root_after,
            "train_ce_before": float(f["ce_before"]),
            "train_ce_after": float(f["ce_after"]),
            "train_accuracy_after": float(f["accuracy_after"]),
            "dev_ce": ce, "dev_accuracy": float(d["accuracy"]),
            "observed_pair_cells_train": int(f["observed_pair_cells"]),
            "observed_directed_transitions_train": int(f["observed_directed_transitions"]),
            "active_train_contexts": int(f["active_contexts"]),
            "nontrivial_parameter_updates": int(f["learned_directional_logits"]),
            "seconds": seconds + dev_seconds,
        }
        candidates.append(item)
        if ce < best_ce - 1e-9:
            best_ce, best_lambda = ce, concentration
            cleanup_model(best)
            clone_model(candidate, best)
        print("HW5_FULLPAIR_DEV " + json.dumps({
            "lambda": concentration, "dev_ce": ce,
            "dev_accuracy": item["dev_accuracy"],
            "train_ce": item["train_ce_after"],
            "best_lambda": best_lambda}, sort_keys=True), flush=True)
        cleanup_model(candidate)
    if best_lambda is None or not best.exists():
        raise ValueError("no valid dev-selected full pair model")
    selected_sha = digest(best)
    test_metrics, test_seconds = run(
        ["./hw5-fullpair", "eval", str(best), str(test)], output / "selected-test.log")
    check_native(test_metrics, split_bytes["test"], 1)
    if digest(best) != selected_sha:
        raise ValueError("test evaluation changed the selected student")
    record["selected"] = {
        "concentration": best_lambda, "dev_ce": best_ce,
        "train_only_root_sha256": selected_sha,
        "test_ce_nats": float(test_metrics["ce_nats"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_bytes": int(test_metrics["input_bytes"]),
        "test_seconds": test_seconds,
        "geometry_test_ce_nats": prior["final"]["test"]["ce_nats"],
        "geometry_test_accuracy": prior["final"]["test"]["accuracy"],
        "count_test_ce_nats": prior["count_control"]["test"]["ce_nats"],
        "count_test_accuracy": prior["count_control"]["test"]["accuracy"],
    }
    # The retained root now receives EVERY Gutenberg book, but that makes all
    # books training data. NEVER report a fresh heldout CE on this final root.
    final_root = output / "learner-root"
    clone_model(prior_root, final_root)
    f, seconds = run(["./hw5-fullpair", "train", str(final_root),
                      str(all_books), str(best_lambda)],
                     output / "full-all-books-train.log")
    check_native(f, manifest["corpus_bytes"], 1)
    final_sha = digest(final_root)
    reopened, _ = run(["./hw5-fullpair", "eval", str(final_root),
                       str(all_books)], output / "full-all-books-reload.log")
    check_native(reopened, manifest["corpus_bytes"], 1)
    if final_sha != digest(final_root):
        raise ValueError("full-book checkpoint failed native reload")
    if abs(float(reopened["ce_nats"]) - float(f["ce_after"])) > 1e-6:
        raise ValueError("checkpoint reload changed full-book predictor")
    record["full_all_books_refit"] = {
        "model_root_sha256": final_sha,
        "bytes_ingested": manifest["corpus_bytes"],
        "train_ce_before": float(f["ce_before"]),
        "train_ce_after": float(f["ce_after"]),
        "train_accuracy_before": float(f["accuracy_before"]),
        "train_accuracy_after": float(f["accuracy_after"]),
        "observed_pair_cells": int(f["observed_pair_cells"]),
        "observed_directed_transitions": int(f["observed_directed_transitions"]),
        "active_byte_contexts": int(f["active_contexts"]),
        "pair_records_allocated": int(f["physical_pairs"]),
        "nontrivial_parameter_updates": int(f["learned_directional_logits"]),
        "native_seconds": seconds,
        "reloaded_train_ce": float(reopened["ce_nats"]),
        "evaluation": "TRAIN ONLY: all HW5 books participated in fitting",
    }
    cleanup_model(best)
    record["wall_seconds"] = time.perf_counter() - started
    (output / "report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("HW5_FULLPAIR_FINAL " + json.dumps({
        "all_bytes": record["all_book_bytes"],
        "canonical_pair_records": 32640,
        "selected_lambda": best_lambda,
        "selected_test_ce": record["selected"]["test_ce_nats"],
        "selected_test_accuracy": record["selected"]["test_accuracy"],
        "geometry_test_ce": record["selected"]["geometry_test_ce_nats"],
        "count_test_ce": record["selected"]["count_test_ce_nats"],
        "all_root_sha256": final_sha,
        "full_corpus_train_ce": record["full_all_books_refit"]["train_ce_after"],
        "observed_pair_cells": record["full_all_books_refit"]["observed_pair_cells"],
    }, sort_keys=True), flush=True)
    return record

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--prior-root", required=True, type=Path)
    ap.add_argument("--prior-report", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    experiment(a.store, a.prior_root, a.prior_report, a.output)

if __name__ == "__main__":
    main()
