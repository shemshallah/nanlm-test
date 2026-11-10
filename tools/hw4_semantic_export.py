#!/usr/bin/env python3
"""Convert pinned HW4 WordNet nouns to an EXACT native synset graph and query audit.

This is data export plus an independent BFS query oracle, NOT Python training.
The native WordNetIndex/SemanticBridge performs every scored inference.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw4_reference_corpus import source_store, verify_hw4

def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            sha.update(chunk)
    return sha.hexdigest()

def fnv(s: str) -> int:
    value = 14695981039346656037
    for b in s.encode("utf-8"):
        value ^= b
        value = (value * 1099511628211) & 0xffffffffffffffff
    return value

def safe(s: str) -> str:
    return " ".join(str(s).replace("\t", " ").replace("\r", " ").split())

def ancestry(source: int, parent: dict[int, list[int]], hops: int = 128) -> dict[int, int]:
    reached = {source: 0}
    queue = deque([source])
    while queue:
        a = queue.popleft()
        if reached[a] >= hops:
            continue
        for b in parent.get(a, []):
            if b in reached:
                continue
            reached[b] = reached[a] + 1
            queue.append(b)
    return reached

def export(db_path: Path, target: Path, sample_size: int, seed: int) -> dict:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("nonempty semantic export directory")
    target.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect("file:" + str(db_path.resolve()) + "?mode=ro&immutable=1",
                         uri=True) as db:
        nouns = list(db.execute(
            "SELECT synset, lemmas_json, gloss FROM synsets WHERE pos='n' ORDER BY synset"))
        if len(nouns) < 5:
            raise ValueError("WordNet noun inventory unexpectedly small")
        ids = {name: i + 1 for i, (name, _, _) in enumerate(nouns)}
        parent: dict[int, list[int]] = {}
        total = 0
        # Reverse/inverse pointers are deliberately NOT used to manufacture hypernyms.
        for src, dst, symbol in db.execute(
            "SELECT src,dst,symbol FROM pointers WHERE symbol IN ('@','@i') ORDER BY src,dst"):
            if src not in ids or dst not in ids or src == dst:
                continue
            a, b = ids[src], ids[dst]
            parent.setdefault(a, []).append(b)
            total += 1
    for a in parent:
        parent[a] = sorted(set(parent[a]))
    ledger = target / "wordnet-noun.wordnet.tsv"
    with ledger.open("w", encoding="utf-8", newline="\n") as out:
        edges = sum(len(v) for v in parent.values())
        out.write(f"NANLM-WN1\t{len(nouns)}\t{edges}\n")
        # The FNV address is a physical head anchor, NOT the identity.
        for name, names_json, gloss in nouns:
            words = json.loads(names_json)
            if not words:
                raise ValueError("missing WordNet synset lexeme")
            synset_id = ids[name]
            leaf = fnv("n:" + name) & 0xffff
            out.write(f"N\t{synset_id}\t{leaf}\t{safe(words[0])}\t{safe(gloss)}\n")
        for name, names_json, _ in nouns:
            for alias in sorted(set(map(safe, json.loads(names_json)))):
                if not alias:
                    raise ValueError("empty WordNet synonym")
                out.write(f"L\t{alias.lower()}\t{ids[name]}\n")
        for a in sorted(parent):
            for b in parent[a]:
                out.write(f"E\t{a}\t{b}\t0\t{fnv(str(a)+':'+str(b))}\n")
    rng = random.Random(seed)
    all_ids = list(ids.values())
    positives: list[tuple[int,int,int]] = []
    negatives: list[tuple[int,int,int]] = []
    visited_pairs: set[tuple[int,int]] = set()
    sources = list(parent)
    rng.shuffle(sources)
    for src in sources:
        reached = ancestry(src, parent, hops=128)
        # A proof requires at least three edges and no direct edge.
        useful = [dst for dst, dist in reached.items() if 3 <= dist <= 5]
        if useful and len(positives) < sample_size:
            dst = rng.choice(useful)
            if (src, dst) not in visited_pairs:
                positives.append((src, dst, 1))
                visited_pairs.add((src,dst))
        if len(negatives) < sample_size:
            for _ in range(50):
                dst = rng.choice(all_ids)
                if dst not in reached and (src, dst) not in visited_pairs:
                    negatives.append((src, dst, 0))
                    visited_pairs.add((src, dst))
                    break
        if len(positives) == sample_size and len(negatives) == sample_size:
            break
    if len(positives) < sample_size or len(negatives) < sample_size:
        raise ValueError(f"insufficient disjoint proofs/controls {len(positives)} / {len(negatives)}")
    rows = positives + negatives
    rng.shuffle(rows)
    query = target / "semantic-queries.tsv"
    with query.open("w", encoding="ascii", newline="\n") as out:
        for src,dst,gold in rows:
            out.write(f"isa\tn:{src}\tn:{dst}\t{gold}\n")
    result = {
        "schema": "nanlm-test.hw4-native-semantic-eval.v1",
        "nouns": len(nouns),
        "exact_isa_edges": edges,
        "source_pointer_rows": total,
        "sample_seed": seed,
        "queries": len(rows),
        "positive_three_to_five_hop": len(positives),
        "negative_unreachable": len(negatives),
        "exact_index_sha256": digest(ledger),
        "query_sha256": digest(query),
        "source_role": "immutable HW4:0 WordNet 3.1 SQLite snapshot",
        "scope": "native symbolic ISA reasoning benchmark, NOT learned semantics or train/test generalization"
    }
    (target / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--positive-and-negative-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261109)
    a = ap.parse_args()
    if not 10 <= a.positive_and_negative_per_class <= 5000:
        raise ValueError("query sample size must be 10..5000")
    with source_store(store=None, archive=a.archive) as store:
        manifest, db = verify_hw4(store)
        result = export(db, a.output, a.positive_and_negative_per_class, a.seed)
        result["hw4_root_sha256"] = manifest["root_sha256"]
        result["hw4_sqlite_sha256"] = manifest["sqlite_sha256"]
        (a.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("HW4_SEMANTIC_EXPORT " + json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
