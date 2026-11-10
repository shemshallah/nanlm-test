#!/usr/bin/env python3
"""HW5 paired attribution audit of immutable HW5 held-out book predictions.

This is experiment ORCHESTRATION only. All predictive distributions and CE
remain native C++ via hw5-context; Python never trains or estimates a model.
The HW5 dev-chosen configurations are frozen before any test book is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile

from hw5_context_experiment import STUDENT_SHA, load_manifest, parse, sha

BOOTSTRAP_SEED = 601209
BOOTSTRAP_DRAWS = 10000
CASES = ("best_memory_only", "best_geometry_positive")
SUPPORT_STRATA = ("seen_exact_triple", "seen_context_unseen_continuation", "unseen_context")


def read_triplets(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != b"HW5TRI01":
        raise ValueError("invalid native triple evidence magic")
    count, = struct.unpack_from("<Q", raw, 8)
    if len(raw) != 16 + 8 * count:
        raise ValueError("invalid triple count or trailing bytes")
    triples = set()
    contexts = set()
    prev = -1
    for key, n in struct.iter_unpack("<II", raw[16:]):
        if n <= 0 or key <= prev or key >> 24 or any(x == 255 for x in
                ((key >> 16) & 255, (key >> 8) & 255, key & 255)):
            raise ValueError("malformed triple evidence")
        triples.add(key)
        contexts.add(key >> 8)
        prev = key
    return triples, contexts


def coverage(data: bytes, triples: set[int], contexts: set[int]):
    tokens = len(data) - 2
    if tokens < 100:
        raise ValueError("book too short for native independent evaluation")
    exact = seen_context = 0
    for a, b, c in zip(data, data[1:], data[2:]):
        ab = (a << 8) | b
        seen_context += ab in contexts
        exact += ((ab << 8) | c) in triples
    return {"transitions": tokens,
            "seen_exact_triple_count": exact,
            "seen_pair_context_count": seen_context,
            "exact_triple_fraction": exact / tokens,
            "seen_pair_context_fraction": seen_context / tokens}


def score_one(student: Path, root: Path, triplets: Path,
              document: Path, lengths: Path, cfg: dict):
    cmd = ["./hw5-context", "score", str(student), str(root), str(triplets),
           str(document), str(lengths), "fixed"]
    cmd += [str(cfg[k]) for k in ("beta", "lambda", "delta", "kappa")]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"native fixed score failed: {result.stdout[-1200:]}")
    rows = parse(result.stdout)
    if len(rows) != 3 or [x["mode"] for x in rows] != [
            "geometry_only", "bigram_add05", "context_hybrid"]:
        raise ValueError("native score controls changed")
    strata = {}
    for line in result.stdout.splitlines():
        if not line.startswith("HW5_CONTEXT_STRATUM "):
            continue
        fields = dict(x.split("=", 1) for x in line.split()[1:])
        group = fields.pop("regime", None)
        if group not in SUPPORT_STRATA or group in strata:
            raise ValueError("duplicate or unknown native support regime")
        if fields.get("exact_model") != "HW5_train_only" or fields.get("bptt_attempted") != "0":
            raise ValueError("invalid native support provenance")
        if any(abs(float(fields[k]) - float(cfg[k])) > 1e-12 for k in
               ("beta", "lambda", "delta", "kappa")):
            raise ValueError("native geometric stratum configuration changed")
        row = {k: float(fields[k]) for k in ("ce_nats", "accuracy")}
        row["transitions"] = int(fields["transitions"])
        if row["transitions"] < 0 or not 0 <= row["accuracy"] <= 1 or not math.isfinite(row["ce_nats"]):
            raise ValueError("invalid native stratum metrics")
        strata[group] = row
    if set(strata) != set(SUPPORT_STRATA):
        raise ValueError("missing native support strata")
    if sum(v["transitions"] for v in strata.values()) != rows[-1]["transitions"]:
        raise ValueError("support strata do not partition native evaluation")
    aggregate_nll = sum(v["transitions"] * v["ce_nats"] for v in strata.values())
    aggregate_hits = sum(v["transitions"] * v["accuracy"] for v in strata.values())
    if (abs(aggregate_nll / rows[-1]["transitions"] - rows[-1]["ce_nats"]) > 2e-9 or
            abs(aggregate_hits / rows[-1]["transitions"] - rows[-1]["accuracy"]) > 2e-9):
        raise ValueError("support-stratified native scores do not sum to frozen score")
    return {**{r["mode"]: {key: r[key] for key in
            ("ce_nats", "accuracy", "transitions")} for r in rows}, "support_strata": strata}


def weighted(rows, metric):
    n = sum(r["transitions"] for r in rows)
    if n <= 0:
        raise ValueError("empty book sample")
    return sum(r[metric] * r["transitions"] for r in rows) / n


def paired_statistics(rows, draws=BOOTSTRAP_DRAWS):
    """Books, not individual bytes, are independent resampling clusters."""
    if len(rows) < 2:
        raise ValueError("at least two independent test books required")
    deltas = [r["best_memory_only"]["ce_nats"] -
              r["best_geometry_positive"]["ce_nats"] for r in rows]
    sizes = [r["transitions"] for r in rows]
    observed = sum(n * d for n, d in zip(sizes, deltas)) / sum(sizes)
    rng = random.Random(BOOTSTRAP_SEED)
    sample = []
    for _ in range(draws):
        indices = [rng.randrange(len(rows)) for _ in rows]
        denominator = sum(sizes[i] for i in indices)
        sample.append(sum(sizes[i] * deltas[i] for i in indices) / denominator)
    sample.sort()
    low = sample[max(0, math.floor(0.025 * draws) - 1)]
    high = sample[min(draws - 1, math.ceil(0.975 * draws) - 1)]
    if len(rows) <= 18:
        tail = 0
        total = 0
        for signs in itertools.product((-1, 1), repeat=len(rows)):
            flipped = sum(s * n * d for s, n, d in zip(signs, sizes, deltas)) / sum(sizes)
            tail += abs(flipped) >= abs(observed) - 1e-14
            total += 1
        p = tail / total
        method = "exact_book_sign_flip"
    else:
        # Conservative randomized sign flips, deterministic and explicitly MC.
        tail = 0
        total = 20000
        for _ in range(total):
            flipped = sum((1 if rng.getrandbits(1) else -1) * n * d
                          for n, d in zip(sizes, deltas)) / sum(sizes)
            tail += abs(flipped) >= abs(observed) - 1e-14
        p = (tail + 1) / (total + 1)
        method = "monte_carlo_book_sign_flip"
    return {
        "unit": "book",
        "test_books": len(rows),
        "delta_ce_memory_minus_geometry": observed,
        "interpretation": "positive means development-selected geometry improves cross-entropy",
        "bootstrap_95_percentile_interval": [low, high],
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "two_sided_sign_flip_p": p,
        "sign_flip_method": method,
        "positive_book_count": sum(x > 0 for x in deltas),
        "negative_book_count": sum(x < 0 for x in deltas),
        "zero_book_count": sum(x == 0 for x in deltas),
    }


def run(reference: Path, student: Path, hw5: Path, output: Path):
    if output.exists():
        raise FileExistsError("HW5 must create a fresh output directory")
    gen, manifest, splits = load_manifest(reference)
    if len(splits["test"]) < 2:
        raise ValueError("require multiple independent held-out books")
    hw5_report = json.loads((hw5 / "report.json").read_text())
    frozen = json.loads((hw5 / "frozen-dev-choices.json").read_text())
    if frozen != hw5_report["test_precommitted_choices"]:
        raise ValueError("HW5 frozen dev choices do not match source report")
    if hw5_report["source_sha256"] != manifest["artifacts"]["all.txt"]["sha256"]:
        raise ValueError("HW5 memory corpus reference differs")
    if sha(student) != STUDENT_SHA:
        raise ValueError("student changed since HW5")
    if hw5_report["student_sha256"] != STUDENT_SHA:
        raise ValueError("original HW5 student digest differs")
    root = hw5 / "train-only-c256.root"
    tri = hw5 / "train-only-triples.bin"
    if root.stat().st_size != 8388608 or sha(root) != hw5_report["train_mirror_sha256"]:
        raise ValueError("train-only canonical matrix changed")
    if sha(tri) != hw5_report["train_triple_sha256"]:
        raise ValueError("train-only exact triples changed")
    if frozen["best_memory_only"]["lambda"] != 0:
        raise ValueError("memory control has positive geometric weight")
    if frozen["best_geometry_positive"]["lambda"] <= 0:
        raise ValueError("positive geometry control was not frozen on dev")
    if not hw5_report["source_reference_immutable"] or hw5_report["memory_trained_on_documents"] != "train_only":
        raise ValueError("source provenance violation")
    student_before = sha(student)
    root_before = sha(root)
    triple_before = sha(tri)
    triples, contexts = read_triplets(tri)
    output.mkdir(parents=True)
    books = []
    with (gen / "test.txt").open("rb") as stream, tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        docpath = tmp / "one-book.txt"
        lengths = tmp / "one-book.lengths"
        for i, entry in enumerate(splits["test"]):
            nbytes = int(entry["bytes"])
            data = stream.read(nbytes)
            if len(data) != nbytes or b"\xff" in data:
                raise ValueError("test split truncated or reserved gateway byte found")
            docpath.write_bytes(data)
            lengths.write_text(f"{nbytes}\n")
            case_rows = {}
            case_strata = {}
            for case in CASES:
                native = score_one(student, root, tri, docpath, lengths, frozen[case])
                case_rows[case] = native["context_hybrid"]
                case_strata[case] = native["support_strata"]
                if case == CASES[0]:
                    case_rows["bigram_add05"] = native["bigram_add05"]
                    case_rows["geometry_only"] = native["geometry_only"]
            predicted = nbytes - 2
            if any(r["transitions"] != predicted for r in case_rows.values()):
                raise ValueError("native cross-book transition or undercount")
            observed_coverage = coverage(data, triples, contexts)
            for case in CASES:
                support = case_strata[case]
                if (support["seen_exact_triple"]["transitions"] != observed_coverage["seen_exact_triple_count"] or
                    support["seen_exact_triple"]["transitions"] +
                    support["seen_context_unseen_continuation"]["transitions"] !=
                        observed_coverage["seen_pair_context_count"]):
                    raise ValueError("native stratum support differs from train-only exact-triple ledger")
            books.append({"index": i, "book_id": str(entry["id"]),
                          "document_sha256": hashlib.sha256(data).hexdigest(),
                          "bytes": nbytes, "transitions": predicted,
                          "coverage": observed_coverage,
                          "support_strata": case_strata,
                          **case_rows})
            print("HW5_BOOK", json.dumps({
                "id": books[-1]["book_id"], "transitions": predicted,
                "delta_ce": books[-1]["best_memory_only"]["ce_nats"] -
                            books[-1]["best_geometry_positive"]["ce_nats"],
                "seen_pair_fraction": books[-1]["coverage"]["seen_pair_context_fraction"]
            }, sort_keys=True), flush=True)
        if stream.read(1):
            raise ValueError("test file has bytes beyond its manifest book ledger")
    if sha(student) != student_before or sha(root) != root_before or sha(tri) != triple_before:
        raise ValueError("read-only evaluation modified source evidence")
    totals = {}
    for case in (*CASES, "bigram_add05", "geometry_only"):
        part = [b[case] for b in books]
        totals[case] = {"ce_nats": weighted(part, "ce_nats"),
                        "accuracy": weighted(part, "accuracy"),
                        "transitions": sum(r["transitions"] for r in part)}
        expected = hw5_report["test"]["pure_teacher" if case == "geometry_only" else case]
        for metric in ("ce_nats", "accuracy"):
            if abs(totals[case][metric] - expected[metric]) > 2e-9:
                raise ValueError(f"per-book aggregate does not reproduce HW5 {case} {metric}")
    significance = paired_statistics(books)
    total_transitions = sum(b["transitions"] for b in books)
    support_summary = {}
    for group in SUPPORT_STRATA:
        support_summary[group] = {}
        for case in CASES:
            group_rows = [b["support_strata"][case][group] for b in books]
            n_group = sum(r["transitions"] for r in group_rows)
            support_summary[group][case] = {
                "transitions": n_group,
                "ce_nats": weighted(group_rows, "ce_nats") if n_group else None,
                "accuracy": weighted(group_rows, "accuracy") if n_group else None,
            }
        memory = support_summary[group][CASES[0]]
        geom = support_summary[group][CASES[1]]
        if memory["transitions"] != geom["transitions"]:
            raise ValueError("support partition differs between frozen models")
        support_summary[group]["delta_memory_minus_geometry_ce"] = (
            memory["ce_nats"] - geom["ce_nats"] if memory["ce_nats"] is not None else None)
    result = {
        "schema": "nanlm-test.hw5-paired-attribution.v1",
        "source_run": 42746674921,
        "source_branch_sha": "5972a7e0b16909816a0b5f723784c88b6856f4a0",
        "reference_route": hw5_report["source_reference"],
        "train_only_memory": True,
        "test_tuning": False,
        "frozen_dev_choices": {c: frozen[c] for c in CASES},
        "student_sha256": STUDENT_SHA,
        "train_mirror_sha256": root_before,
        "train_triple_sha256": triple_before,
        "total_test_transitions": total_transitions,
        "train_distinct_triples": len(triples),
        "train_distinct_pair_contexts": len(contexts),
        "test_exact_triple_coverage": sum(b["coverage"]["seen_exact_triple_count"] for b in books) / total_transitions,
        "test_seen_pair_context_coverage": sum(b["coverage"]["seen_pair_context_count"] for b in books) / total_transitions,
        "test_aggregate": totals,
        "native_support_stratified_test": support_summary,
        "paired_book_statistics": significance,
        "books": books,
        "caveat": "Byte-level predictive attribution only; no semantic-understanding or geometric-training superiority claim.",
    }
    (output / "report.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("HW5_ATTRIBUTION_FINAL", json.dumps({
        "test_books": len(books),
        "test_transitions": total_transitions,
        "memory_ce": totals["best_memory_only"]["ce_nats"],
        "geometry_ce": totals["best_geometry_positive"]["ce_nats"],
        "delta_memory_minus_geometry": significance["delta_ce_memory_minus_geometry"],
        "cluster_bootstrap_95": significance["bootstrap_95_percentile_interval"],
        "book_signflip_p": significance["two_sided_sign_flip_p"],
        "triple_coverage": result["test_exact_triple_coverage"],
        "pair_context_coverage": result["test_seen_pair_context_coverage"],
        "native_support_stratified_test": support_summary
    }, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--hw5-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.student, args.hw5_artifact, args.output)


if __name__ == "__main__":
    main()
