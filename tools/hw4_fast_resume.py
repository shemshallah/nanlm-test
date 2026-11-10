#!/usr/bin/env python3
"""Resume the proven native HW4 geometric learner without resetting its 8 MiB root.

Each update invokes hw4-onepass on the complete, SHA-verified corpus. This is
aggregate byte-pair credit, NOT recurrent/TRAIL sequence training or BPTT.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

SOURCE_SHA = "6c29dc5027012a8ef6ec50404a7bdbb50be81a4d018c1f327665f826d8784604"
START_SHA = "e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1"
SOURCE_BYTES = 58_341_529
ROOT_BYTES = 8_388_608
START_EPOCH = 103
START_CE = 2.689338438

def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for piece in iter(lambda: file.read(1 << 20), b""):
            sha.update(piece)
    return sha.hexdigest()

def marker(text: str, prefix: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith(prefix + " ")]
    if len(lines) != 1:
        raise ValueError(f"expected one {prefix} line, got {len(lines)}: {text[-500:]}")
    return dict(re.findall(r"([a-z_0-9]+)=([^\s]+)", lines[0]))

def run(cmd: list[str], log: Path, allow_nonzero: bool = False) -> tuple[str, float, int]:
    begin = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - begin
    message = result.stdout + result.stderr
    log.write_text(message)
    if result.returncode and not allow_nonzero:
        raise RuntimeError(f"{cmd[0]} returned {result.returncode}: {message[-1000:]}")
    return message, elapsed, result.returncode

def eval_prose(root: Path, output: Path, number: int) -> dict:
    text, duration, _ = run(
        ["./nanlm", str(root), "eval", "tests/fixtures/unified_quality_heldout.txt"],
        output / f"prose-{number:03d}.log")
    f = marker(text, "UNIFIED_EVAL")
    metrics = {"bytes": int(f["bytes"]), "ce": float(f["ce_nats"]),
               "top1": float(f["accuracy"]), "seconds": duration}
    if metrics["bytes"] != 974 or not all(math.isfinite(metrics[key]) for key in ("ce", "top1")):
        raise ValueError("invalid separate prose evaluation")
    return metrics

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--additional-epochs", type=int, default=200)
    p.add_argument("--prose-every", type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.additional_epochs <= 1000 or not 1 <= args.prose_every <= args.additional_epochs:
        raise ValueError("invalid training/evaluation interval")
    if args.root.resolve() == args.corpus.resolve() or "wordnet-c256.root" in str(args.root):
        raise ValueError("the immutable HW4 teacher is not a learner target")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.root.stat().st_size != ROOT_BYTES or digest(args.root) != START_SHA:
        raise ValueError("wrong epoch-103 8 MiB checkpoint; refusing restart or fork")
    if args.corpus.stat().st_size != SOURCE_BYTES or digest(args.corpus) != SOURCE_SHA:
        raise ValueError("HW4:0 source does not match its immutable pinned digest")
    before_sha = digest(args.root)
    prose_before = eval_prose(args.root, args.output, 0)
    record = {
        "schema": "nanlm-test.hw4-fast-continuation.v1",
        "method": "full-corpus native 255x255 aggregate geometric credit (not online TRAIL/BPTT)",
        "source_role": "immutable HW4:0 WordNet 3.1 reference",
        "corpus_sha256": SOURCE_SHA,
        "source_bytes_per_update": SOURCE_BYTES,
        "start_epoch": START_EPOCH,
        "start_student_sha256": before_sha,
        "requested_additional_epochs": args.additional_epochs,
        "completed_additional_epochs": 0,
        "bptt_attempted": 0,
        "prose_before": prose_before,
        "epochs": [],
    }
    prev_ce = START_CE
    started = time.perf_counter()
    status = "completed"
    for epoch in range(1, args.additional_epochs + 1):
        text, seconds, exit_code = run(
            ["./hw4-onepass", str(args.root), str(args.corpus)],
            args.output / f"train-{epoch:03d}.log", allow_nonzero=True)
        f = marker(text, "HW4_FULL_GEOMETRIC_PASS")
        first = float(f["train_before_ce"])
        last = float(f["train_after_ce"])
        accepted = f["aggregate_step_accepted"] == "1"
        if int(f["input_bytes"]) != SOURCE_BYTES or int(f["transitions"]) != SOURCE_BYTES - 1:
            raise ValueError("not a complete corpus scan")
        if int(f["bptt_attempted"]) != 0 or int(f["root_bytes"]) != ROOT_BYTES:
            raise ValueError("not the expected no-BPTT 8 MiB model")
        if not all(math.isfinite(v) for v in (first, last)) or abs(first - prev_ce) > 0.000005:
            raise ValueError(f"broken epoch/CE continuity at {epoch}: {first} versus {prev_ce}")
        if accepted and (exit_code != 0 or last >= first - 1e-9):
            raise ValueError(f"invalid accepted update at {epoch}")
        if not accepted and exit_code not in (0, 2):
            raise ValueError(f"unexpected native trainer failure {exit_code}: {text[-500:]}")
        row = {
            "additional_epoch": epoch,
            "cumulative_epoch": START_EPOCH + epoch,
            "corpus_bytes_scanned": SOURCE_BYTES,
            "train_ce_before": first,
            "train_ce_after": last,
            "top1_before": float(f["train_before_accuracy"]),
            "top1_after": float(f["train_after_accuracy"]),
            "step": float(f["step"]),
            "gradient_max": float(f["gradient_max"]),
            "accepted": accepted,
            "native_seconds": seconds,
        }
        if not accepted:
            if digest(args.root) != record.get("last_sha256", before_sha):
                raise ValueError("rejected update changed the student root")
            status = "optimizer_plateau"
            record["epochs"].append(row)
            print("HW4_FAST_PLATEAU " + json.dumps(row, sort_keys=True), flush=True)
            break
        record["completed_additional_epochs"] = epoch
        prev_ce = last
        # Hash every accepted root; a later rejected proposal must leave this exact
        # preceding checkpoint intact, even when it lands between prose probes.
        row["sha256"] = digest(args.root)
        record["last_sha256"] = row["sha256"]
        if epoch % args.prose_every == 0 or epoch == args.additional_epochs:
            row["prose"] = eval_prose(args.root, args.output, epoch)
        record["epochs"].append(row)
        record["last_train_ce"] = last
        record["last_train_accuracy"] = row["top1_after"]
        record["total_input_bytes_scanned"] = epoch * SOURCE_BYTES
        record["total_native_seconds"] = sum(x["native_seconds"] for x in record["epochs"])
        (args.output / "report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print("HW4_FAST_RESUME_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
    record["status"] = status
    record["elapsed_seconds"] = time.perf_counter() - started
    record["final_student_sha256"] = digest(args.root)
    record["final_cumulative_epoch"] = START_EPOCH + record["completed_additional_epochs"]
    record["prose_after"] = eval_prose(args.root, args.output, record["completed_additional_epochs"])
    (args.output / "report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (args.output / "summary.json").write_text(json.dumps({
        k: v for k, v in record.items() if k != "epochs"}, indent=2, sort_keys=True) + "\n")
    print("HW4_FAST_RESUME_FINAL " + json.dumps({
        "status": status,
        "start_epoch": START_EPOCH,
        "final_cumulative_epoch": record["final_cumulative_epoch"],
        "completed_additional_epochs": record["completed_additional_epochs"],
        "train_ce_before": START_CE,
        "train_ce_after": record.get("last_train_ce", START_CE),
        "prose_before": prose_before,
        "prose_after": record["prose_after"],
        "source_bytes_scanned": record.get("total_input_bytes_scanned", 0),
        "native_seconds": record.get("total_native_seconds", 0),
        "final_student_sha256": record["final_student_sha256"],
    }, sort_keys=True), flush=True)
    if record["completed_additional_epochs"] == 0:
        raise RuntimeError("no accepted new learning steps")
if __name__ == "__main__":
    main()
