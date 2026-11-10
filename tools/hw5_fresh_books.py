#!/usr/bin/env python3
"""HW5:0 fresh public-domain Gutenberg prose, independent book-disjoint audit.

The original CP8 model / hyperparameters are frozen on original train/dev
BEFORE acquiring any candidate text. No corpus update, no training, no
test-driven tuning. Each accepted title must be absent from the original
21-book NaNROUTER HW5:0 manifest. All predictions run in native C++.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

from hw5_context_experiment import STUDENT_SHA, load_manifest, parse as parse_memory, sha
from hw5_cp8_experiment import check_root_preserves_memory, parse_scores, parse_regimes
from hw5_paired_attribution import paired_statistics

ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW5:0"
MAX_BOOK_BYTES = 5_900_000
TARGET_BOOKS = 4
# PREDECLARED order. Source-ID disjointness and UTF-8 validity only determine
# whether candidates qualify. Neither dev nor test CE drives selection.
CANDIDATES = (
    1342, 1661, 345, 2701, 1400, 84, 74, 76, 608,
    1232, 5200, 158, 768, 174, 1260, 996, 2000,
    25344, 2814, 3207, 4517, 204, 209, 32032,
)
MIRROR_ROOT = "https://gutenberg.pglaf.org/cache/epub"
START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\r\n]*\*\*\*", re.I)


def clean_plaintext(raw: bytes) -> bytes:
    """Reject missing delimiters: license boilerplate is not unseen prose."""
    if not raw or len(raw) > MAX_BOOK_BYTES:
        raise ValueError("invalid raw Gutenberg source length")
    book = raw.decode("utf-8-sig", errors="strict")
    begin = START_RE.search(book)
    finish = END_RE.search(book)
    if not begin or not finish or finish.start() <= begin.end():
        raise ValueError("Gutenberg start/end sentinels missing or malformed")
    body = book[begin.end():finish.start()].replace("\r\n", "\n").replace("\r", "\n").strip()
    result = (body + "\n").encode("utf-8")
    if not (1000 <= len(result) <= MAX_BOOK_BYTES) or b"\xff" in result:
        raise ValueError("invalid UTF-8 body length or reserved gateway byte")
    return result


def fetch_book(book_id: int, mirror: str = MIRROR_ROOT) -> tuple[bytes, str]:
    """Only the public-domain Gutenberg text mirror; bounded one-book reads."""
    if book_id not in CANDIDATES:
        raise ValueError("unexpected, non-precommitted Gutenberg ID")
    url = f"{mirror}/{book_id}/pg{book_id}.txt"
    request = urllib.request.Request(url, headers={
        "User-Agent": "NaNLM-HW5-OpenCorpus-Validation/1.0 (read-only 4-book scientific benchmark)"})
    with urllib.request.urlopen(request, timeout=25) as response:
        if response.status != 200:
            raise ValueError(f"book mirror failed HTTP {response.status}")
        raw = response.read(MAX_BOOK_BYTES + 1)
    if len(raw) > MAX_BOOK_BYTES:
        raise ValueError("Gutenberg candidate exceeds bounded source size")
    return raw, url


def native(command: list[str], log: Path) -> str:
    p = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(f"native C++ evaluation exited {p.returncode}: {p.stdout[-1300:]}")
    return p.stdout


def run(reference: Path, student: Path, original_hw5: Path,
        cp8: Path, output: Path, mirror: str = MIRROR_ROOT):
    if output.exists():
        raise FileExistsError("fresh evaluation must use a new output directory")
    generation, manifest, _ = load_manifest(reference)
    if len(manifest["books"]) != 21:
        raise ValueError("unexpected source reference book count")
    all_ids = {str(book["id"]) for book in manifest["books"]}
    baseline_report = json.loads((original_hw5 / "report.json").read_text())
    choice = json.loads((cp8 / "frozen-dev-choice.json").read_text())
    cp8_report = json.loads((cp8 / "report.json").read_text())
    original_root = original_hw5 / "train-only-c256.root"
    triples = original_hw5 / "train-only-triples.bin"
    learned_root = cp8 / "selected-geometric-cp8.root"
    if baseline_report["source_reference"] != ROUTE or cp8_report["route"] != ROUTE:
        raise ValueError("source is not HW5:0 Gutenberg")
    if sha(student) != STUDENT_SHA or baseline_report["student_sha256"] != STUDENT_SHA:
        raise ValueError("geometric student source changed")
    if sha(original_root) != baseline_report["train_mirror_sha256"] or sha(triples) != baseline_report["train_triple_sha256"]:
        raise ValueError("train-only evidence changed")
    if choice != cp8_report["frozen_on_development"] or choice["selected_exclusively_on"] != "development_books":
        raise ValueError("CP8 experiment did not freeze parameters on development books")
    if sha(learned_root) != choice["selected_cp8_root_sha256"]:
        raise ValueError("wrong frozen CP8 checkpoint")
    check_root_preserves_memory(original_root, learned_root)
    if cp8_report["original_student_sha256"] != STUDENT_SHA:
        raise ValueError("CP8 source student provenance mismatch")
    for key in ("beta", "rho", "scale", "gate"):
        if key not in choice:
            raise ValueError("missing dev-frozen prediction parameter")
    if int(choice["gate"]) not in (0, 1):
        raise ValueError("invalid dev-selected geometric gate")
    # The frozen model is verified BEFORE even downloading the new texts.
    frozen_sources = {k: sha(path) for k, path in {
        "student": student, "memory": original_root, "triples": triples,
        "cp8": learned_root}.items()}
    output.mkdir(parents=True)
    texts = output / "fresh-books"
    texts.mkdir()
    logs = output / "native-logs"
    logs.mkdir()
    report = {
        "schema": "nanlm-test.hw5-fresh-gutenberg-holdout.v1",
        "route": ROUTE,
        "precommitted_candidate_ids": list(CANDIDATES),
        "target_book_count": TARGET_BOOKS,
        "excluded_source_ids": sorted(all_ids),
        "model_frozen_prior_to_download": True,
        "dev_frozen_parameters": choice,
        "model_source_sha256": frozen_sources,
        "mirror": mirror,
        "accepted_books": [],
        "rejections": [],
    }
    selected: list[dict] = []
    for book_id in CANDIDATES:
        if len(selected) == TARGET_BOOKS:
            break
        if str(book_id) in all_ids:
            report["rejections"].append({"id": book_id, "reason": "original_hw5_source_overlap"})
            continue
        try:
            raw, url = fetch_book(book_id, mirror=mirror)
            data = clean_plaintext(raw)
        except (ValueError, UnicodeError, OSError) as error:
            report["rejections"].append({"id": book_id,
                                         "reason": f"unavailable_or_invalid_{type(error).__name__}"})
            continue
        path = texts / f"pg{book_id}.txt"
        path.write_bytes(data)
        selected.append({
            "id": book_id, "source_url": url, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_sha256": sha(path), "bytes": len(data),
            "transitions": len(data) - 2, "source_overlap": False,
        })
        # Bound retrieval rate when multiple books are fetched from a mirror.
        if len(selected) < TARGET_BOOKS:
            time.sleep(1)
    if len(selected) != TARGET_BOOKS:
        (output / "download-diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"only {len(selected)} disjoint, well-formed Gutenberg books available")
    for book in selected:
        doc = texts / f"pg{book['id']}.txt"
        sizes = texts / f"pg{book['id']}.lengths"
        sizes.write_text(f"{book['bytes']}\n")
        baseline_log = native([
            "./hw5-context", "score", str(student), str(original_root), str(triples),
            str(doc), str(sizes), "fixed", "16", "0", "0", "0"],
            logs / f"baseline-pg{book['id']}.log")
        baseline_rows = parse_memory(baseline_log)
        if len(baseline_rows) != 3 or baseline_rows[-1]["mode"] != "context_hybrid":
            raise ValueError("not a native frozen memory-only reference score")
        baseline = baseline_rows[-1]
        if baseline["transitions"] != book["transitions"] or baseline["lambda"] != 0:
            raise ValueError("memory baseline incorrect or cross-book transitions")
        cp8_log = native([
            "./hw5-cp8", "score-cp", str(student), str(learned_root), str(triples),
            str(doc), str(sizes), str(choice["beta"]), str(choice["rho"]),
            str(choice["scale"]), str(choice["gate"])],
            logs / f"geometric-pg{book['id']}.log")
        cp8_rows = parse_scores(cp8_log)
        if len(cp8_rows) != 1 or cp8_rows[0]["transitions"] != book["transitions"]:
            raise ValueError("invalid native CP8 test scoring count")
        regimes = parse_regimes(cp8_log)
        if sum(x["transitions"] for x in regimes.values()) != book["transitions"]:
            raise ValueError("native CP8 support attribution incomplete")
        book["best_memory_only"] = {k: baseline[k] for k in ("ce_nats", "accuracy", "transitions")}
        book["best_geometry_positive"] = cp8_rows[0]
        book["support_strata_cp8"] = regimes
        report["accepted_books"].append(book)
        print("HW5_FRESH_BOOK", json.dumps({
            "id": book["id"], "bytes": book["bytes"],
            "ce_memory": baseline["ce_nats"], "ce_cp8": cp8_rows[0]["ce_nats"],
            "delta_ce": baseline["ce_nats"] - cp8_rows[0]["ce_nats"]
        }, sort_keys=True), flush=True)
    if {k: sha(path) for k, path in {
            "student": student, "memory": original_root, "triples": triples,
            "cp8": learned_root}.items()} != frozen_sources:
        raise RuntimeError("fresh challenge modified frozen model or train-only evidence")
    def aggregate(key: str, metric: str) -> float:
        count = sum(b["transitions"] for b in selected)
        return sum(b[key][metric] * b["transitions"] for b in selected) / count
    report["fresh_test_aggregate"] = {
        name: {metric: aggregate(name, metric) for metric in ("ce_nats", "accuracy")}
        for name in ("best_memory_only", "best_geometry_positive")
    }
    report["fresh_test_aggregate"]["transitions"] = sum(b["transitions"] for b in selected)
    report["book_cluster_attribution"] = paired_statistics(selected)
    report["adaptive_split_warning"] = (
        "Prior experiments inspected the original two-book HW5 test set; "
        "the external candidate-ID list was fixed before this new fresh challenge. "
        "No fresh-book results influence model parameters.")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("HW5_FRESH_HOLDOUT_FINAL", json.dumps({
        "books": [b["id"] for b in selected],
        "transitions": report["fresh_test_aggregate"]["transitions"],
        "memory_ce": report["fresh_test_aggregate"]["best_memory_only"]["ce_nats"],
        "cp8_ce": report["fresh_test_aggregate"]["best_geometry_positive"]["ce_nats"],
        "paired": report["book_cluster_attribution"],
        "frozen_checkpoint_immutable": True
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--hw5-artifact", required=True, type=Path)
    parser.add_argument("--cp8-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mirror", default=MIRROR_ROOT)
    args = parser.parse_args()
    run(args.reference, args.student, args.hw5_artifact,
        args.cp8_artifact, args.output, mirror=args.mirror)
