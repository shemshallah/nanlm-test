#!/usr/bin/env python3
"""Read the immutable HW4 WordNet reference and build learner-only corpora.

The ONLY source is the exact SQLite ledger from a digest-pinned HW4 snapshot.
The dense 32640-pair projection is validated but is not falsely treated as
32700 exact concepts. No HW4 writes, no BPTT, no new model, no ring mutation.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile
import tempfile
from urllib.request import urlopen

ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW4:0"
OWNER = "NaNROUTER/MC0:0"
SCHEMA = "nanrouter.hw4-wordnet31-complete.v1"
CORPUS_SCHEMA = "nanlm-test.hw4-corpus.v1"
RELEASE_URL = ("https://github.com/shemshallah/NaNROUTER/releases/download/"
               "hw4-wordnet31-full-g1/hw4-wordnet31-full-reference.tar.gz")
RELEASE_SHA256 = "87dbcdd2d209db95e8d15be68109ef3642e78ef540ea12ad71f3b04982876695"
ROOT_BYTES = 8388608
SPLITS = ("train", "dev", "test")
POS = {"n":"noun","v":"verb","a":"adjective","s":"adjective satellite","r":"adverb"}
REL = {
    "@":"has hypernym", "@i":"has instance hypernym",
    "~":"has hyponym", "~i":"has instance hyponym",
    "!":"has antonym", "+":"has derivational connection to",
    "=":"has attribute", "*":"entails", ">":"causes",
    "^":"also see", "&":"is similar to", "$":"is in the same verb group as",
    "#m":"has member holonym", "#s":"has substance holonym",
    "#p":"has part holonym", "%m":"has member meronym",
    "%s":"has substance meronym", "%p":"has part meronym",
    ";c":"belongs to topic domain", ";r":"belongs to region domain",
    ";u":"belongs to usage domain", "-c":"has member of topic domain",
    "-r":"has member of region domain", "-u":"has member of usage domain",
    "\\":"has pertainym",
}
GEN_NAME = re.compile(r"generation-[0-9]{8}\Z")
SHA_HEX = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_SPLIT = b"nanlm-test.hw4.synset-source-split.v1\x00"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for blob in iter(lambda: file.read(1024*1024), b""):
            h.update(blob)
    return h.hexdigest()


def split_synset(synset: str) -> str:
    rank = int.from_bytes(
        hashlib.sha256(SOURCE_SPLIT + synset.encode("ascii")).digest()[:8],
        "big"
    ) % 100
    return "train" if rank < 80 else ("dev" if rank < 90 else "test")


def safe(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split())


def _doc(file: Path) -> dict:
    return json.loads(file.read_text(encoding="utf-8"))


def verify_hw4(store: Path) -> tuple[dict, Path]:
    """Read-only validation of source generation and both content hashes."""
    pointer = _doc(store / "CURRENT.json")
    if set(pointer) != {"schema", "generation_name", "manifest_sha256"}:
        raise ValueError("HW4 CURRENT schema fields invalid")
    if pointer["schema"] != SCHEMA or not GEN_NAME.fullmatch(pointer["generation_name"]):
        raise ValueError("HW4 CURRENT schema or generation invalid")
    if not SHA_HEX.fullmatch(pointer["manifest_sha256"]):
        raise ValueError("HW4 pointer digest invalid")
    folder = store / pointer["generation_name"]
    manifest_path = folder / "manifest.json"
    if sha256(manifest_path) != pointer["manifest_sha256"]:
        raise ValueError("HW4 reference manifest digest mismatch")
    m = _doc(manifest_path)
    if (m.get("schema") != SCHEMA or m.get("route") != ROUTE or
        m.get("ring_owner") != OWNER or m.get("main_ring_writable") is not False or
        m.get("generation_name") != pointer["generation_name"] or
        m.get("root_bytes") != ROOT_BYTES or m.get("pair_count") != 32640 or
        m.get("heads") != 256):
        raise ValueError("HW4 reference format or ring ownership mismatch")
    for name, field in (("wordnet-c256.root", "root_sha256"),
                        ("wordnet-exact.sqlite3", "sqlite_sha256")):
        path = folder / name
        if path.is_symlink() or not SHA_HEX.fullmatch(m.get(field,"")) or sha256(path) != m[field]:
            raise ValueError("HW4 artifact integrity failure: " + name)
    if (folder / "wordnet-c256.root").stat().st_size != ROOT_BYTES:
        raise ValueError("HW4 dense matrix size mismatch")
    dbfile = folder / "wordnet-exact.sqlite3"
    with sqlite3.connect("file:" + str(dbfile.resolve()) + "?mode=ro&immutable=1", uri=True) as db:
        for table, field in (("synsets","synsets"),("senses","senses"),("pointers","pointers")):
            if db.execute("SELECT COUNT(*) FROM "+table).fetchone()[0] != m[field]:
                raise ValueError("HW4 exact ledger table count mismatch: " + table)
        if db.execute("SELECT count(*) FROM pointers WHERE resolved<>1").fetchone()[0] != m["unresolved_pointers"]:
            raise ValueError("HW4 unresolved pointer count mismatch")
    return m, dbfile


@contextmanager
def source_store(*, store: Path | None, archive: Path | None, download_release: bool = False):
    """Work from a verified local HW4 store or the SHA-pinned archived export."""
    selected = sum((store is not None, archive is not None, download_release))
    if selected != 1:
        raise ValueError("exactly one of --store, --archive, --download-release required")
    if store is not None:
        yield Path(store)
        return
    with tempfile.TemporaryDirectory(prefix="nanlm-hw4-source-") as td:
        tmp = Path(td)
        original = tmp / "pinned.tar.gz"
        if download_release:
            total = 0
            digest = hashlib.sha256()
            with urlopen(RELEASE_URL, timeout=60) as response, original.open("xb") as out:
                while True:
                    part = response.read(1024*1024)
                    if not part: break
                    total += len(part)
                    if total > 64*1024*1024:
                        raise ValueError("oversized HW4 export download")
                    digest.update(part)
                    out.write(part)
            if digest.hexdigest() != RELEASE_SHA256:
                raise ValueError("HW4 release archive SHA mismatch")
        else:
            if sha256(Path(archive)) != RELEASE_SHA256:
                raise ValueError("HW4 release archive SHA mismatch")
            original = Path(archive)
        folder = tmp / "reference"
        folder.mkdir()
        with tarfile.open(original, "r:gz") as tar:
            entries = {entry.name:entry for entry in tar.getmembers() if entry.isfile()}
            root = "hw4-full-reference/"
            prefix = root + "generation-00000001/"
            allowed = (root + "CURRENT.json", prefix + "manifest.json",
                       prefix + "wordnet-exact.sqlite3", prefix + "wordnet-c256.root")
            if any(name not in entries for name in allowed):
                raise ValueError("missing HW4 exact snapshot members")
            for name in allowed:
                member = entries[name]
                if member.size > 128*1024*1024 or member.size < 1:
                    raise ValueError("invalid HW4 release entry length")
                dest = folder / name.removeprefix(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError("missing HW4 tar member stream")
                with stream, dest.open("xb") as out:
                    shutil.copyfileobj(stream, out)
        yield folder


def _rows(db: sqlite3.Connection, query: str):
    return iter(db.execute(query))


def _next(stream):
    return next(stream, None)


def generate(store: Path, output: Path, *, include_typed: bool = True) -> dict:
    """Deterministically split by SOURCE synset, never by an individual line.

    Heldout sources have no training definitions/sense rows/outgoing edges.
    Target labels may appear as targets of other sources: this is a
    source-held-out test, not a fully entity-disjoint graph benchmark.
    """
    m, dbfile = verify_hw4(store)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("HW4 corpus output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    counts = {s: Counter() for s in SPLITS}
    writers = {s:(output / (s+".txt")).open("wb") for s in SPLITS}
    edge_writers = ({s:(output / (s+".edges.tsv")).open("w", encoding="utf-8")
                     for s in SPLITS} if include_typed else {})
    seen = set()
    try:
        with sqlite3.connect("file:" + str(dbfile.resolve()) + "?mode=ro&immutable=1", uri=True) as db:
            sources = _rows(db,"SELECT synset,pos,lemmas_json,gloss FROM synsets ORDER BY synset")
            senses = _rows(db,"SELECT synset,sense_key,lemma,sense_number FROM senses ORDER BY synset,sense_key")
            ptrs = _rows(db, """
                SELECT p.src,p.dst,p.symbol,p.src_word,p.dst_word,
                       d.pos,d.lemmas_json,p.resolved
                FROM pointers p LEFT JOIN synsets d ON d.synset=p.dst
                ORDER BY p.src,p.id
            """)
            next_sense = _next(senses)
            next_ptr = _next(ptrs)
            for synset,pos,names_raw,gloss in sources:
                if synset in seen: raise ValueError("duplicate HW4 source synset")
                seen.add(synset)
                split = split_synset(synset)
                names = json.loads(names_raw)
                if not names or pos not in POS:
                    raise ValueError("invalid HW4 source synset")
                subject = safe(names[0])
                kind = POS[pos]
                def emit(text: str, category: str):
                    raw = (text.rstrip() + "\n").encode("utf-8")
                    if b"\xff" in raw:
                        raise ValueError("0xFF reserved by native byte learner")
                    writers[split].write(raw)
                    counts[split][category] += 1
                    counts[split]["bytes"] += len(raw)
                emit(f"{subject} ({kind}): {safe(gloss)}", "definitions")
                if len(names)>1:
                    emit(f"{subject} ({kind}) is also named {', '.join(safe(v) for v in names[1:])}.", "synonyms")
                while next_sense is not None and next_sense[0] < synset:
                    raise ValueError("HW4 sense references absent source synset")
                while next_sense is not None and next_sense[0] == synset:
                    _, sense_key, lemma, number = next_sense
                    # The exact sense key survives in HW4. Avoid training
                    # arbitrary offsets/hash bytes as the primary language.
                    emit(f"{safe(lemma)} ({kind}), sense {number}: {safe(gloss)}", "senses")
                    next_sense = _next(senses)
                while next_ptr is not None and next_ptr[0] < synset:
                    raise ValueError("HW4 pointer references absent source synset")
                while next_ptr is not None and next_ptr[0] == synset:
                    src,dst,sym,sw,dw,dstpos,dest_raw,resolved = next_ptr
                    if resolved != 1 or dest_raw is None or dstpos not in POS:
                        raise ValueError("HW4 pointer target missing from exact ledger")
                    dest_names = json.loads(dest_raw)
                    # Lexical pointers index individual lexemes, not just
                    # the first synset lemma. WordNet indices are 1-based.
                    src_name = safe(names[sw-1] if sw else names[0])
                    dst_name = safe(dest_names[dw-1] if dw else dest_names[0])
                    if (sw and not 1<=sw<=len(names)) or (dw and not 1<=dw<=len(dest_names)):
                        raise ValueError("HW4 lexical pointer index outside synset")
                    relationship = REL.get(sym, "has WordNet pointer "+sym)
                    emit(f"{src_name} ({kind}) {relationship} {dst_name} ({POS[dstpos]}).",
                         "relations")
                    if include_typed:
                        edge_writers[split].write(
                            f"{src}\t{dst}\t{sym}\t{sw}\t{dw}\n")
                    next_ptr = _next(ptrs)
                counts[split]["synsets"] += 1
            if next_sense is not None or next_ptr is not None:
                raise ValueError("orphan HW4 sense or relation rows")
    finally:
        for f in list(writers.values())+list(edge_writers.values()):
            f.close()
    totals = Counter()
    for c in counts.values(): totals.update(c)
    for name, expected in (("synsets","synsets"),("senses","senses"),
                           ("relations","pointers")):
        if totals[name] != m[expected]:
            raise ValueError(f"HW4 {name} loss during stream: {totals[name]} != {m[expected]}")
    output_files = {s:{
        "corpus_file":s+".txt", "sha256":sha256(output/(s+".txt")),
        "bytes":(output/(s+".txt")).stat().st_size,
        "edges_file":s+".edges.tsv" if include_typed else None,
        "edges_sha256":sha256(output/(s+".edges.tsv")) if include_typed else None,
        "counts":dict(counts[s]),
    } for s in SPLITS}
    report = {
        "schema": CORPUS_SCHEMA,
        "source_route": ROUTE,
        "source_owner": OWNER,
        "hw4_generation":m["generation"],
        "hw4_root_sha256":m["root_sha256"],
        "hw4_sqlite_sha256":m["sqlite_sha256"],
        "hw4_source_archive_sha256":m["source_archive_sha256"],
        "split_rule":"sha256(nanlm-test.hw4.synset-source-split.v1\\0+synset) % 100: 80/10/10",
        "split_caveat":"source-held-out; shared target identities may cross splits",
        "teacher_immutable":True,
        "learner":"nanlm-test / independent one-root geometric credit",
        "no_bptt":True,
        "total":dict(totals),
        "files":output_files,
    }
    (output/"hw4-corpus.manifest.json").write_text(
        json.dumps(report, sort_keys=True, indent=2)+"\n",encoding="utf-8")
    # Existing 50-epoch native trainer accepts this exact hash contract.
    train = output_files["train"]
    train_manifest = {
        "archive_sha256":m["source_archive_sha256"],
        "corpus_sha256":train["sha256"],
        "corpus_bytes":train["bytes"],
        "source_hw4_root_sha256":m["root_sha256"],
        "source_hw4_sqlite_sha256":m["sqlite_sha256"],
        "source_generation":m["generation"],
        "split":"train",
        "heldout_sha256":output_files["test"]["sha256"],
    }
    (output/"train.manifest.json").write_text(json.dumps(
        train_manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    inputs = cli.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--store",type=Path)
    inputs.add_argument("--archive",type=Path)
    inputs.add_argument("--download-release",action="store_true")
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--no-typed",action="store_true")
    args = cli.parse_args()
    with source_store(store=args.store,archive=args.archive,
                      download_release=args.download_release) as base:
        report=generate(base,args.output,include_typed=not args.no_typed)
    print("HW4_CORPUS "+json.dumps(report,sort_keys=True))


if __name__ == "__main__":
    main()
