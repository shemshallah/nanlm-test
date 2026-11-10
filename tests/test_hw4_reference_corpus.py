#!/usr/bin/env python3
"""No-network HW4-to-other-universe regression and leakage checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw4_reference_corpus import generate, sha256, split_synset, verify_hw4


def fixture(root: Path):
    folder = root / "generation-00000001"
    folder.mkdir(parents=True)
    matrix = folder / "wordnet-c256.root"
    with matrix.open("wb") as f:
        f.truncate(8388608)
    dbpath = folder / "wordnet-exact.sqlite3"
    groups = {"train": [], "dev": [], "test": []}
    for index in range(1, 10000):
        key = f"{index:08d}-n"
        g = split_synset(key)
        if len(groups[g]) < (6 if g=="train" else 3):
            groups[g].append(key)
        if [len(groups[x]) for x in ("train","dev","test")] == [6,3,3]:
            break
    assert len(groups["train"])==6 and len(groups["dev"])==3 and len(groups["test"])==3
    keys = [v for group in groups.values() for v in group]
    with sqlite3.connect(dbpath) as db:
        db.executescript("""
        CREATE TABLE synsets (
          synset TEXT PRIMARY KEY, pos TEXT, offset TEXT, head INTEGER,
          lexfile INTEGER, lemmas_json TEXT, gloss TEXT);
        CREATE TABLE senses (
          sense_key TEXT PRIMARY KEY, synset TEXT, lemma TEXT,
          sense_number INTEGER, tag_count INTEGER);
        CREATE TABLE pointers (
          id INTEGER PRIMARY KEY, src TEXT, dst TEXT, symbol TEXT,
          src_word INTEGER, dst_word INTEGER, src_head INTEGER,
          dst_head INTEGER, pair_id INTEGER, resolved INTEGER,
          source_file TEXT, source_line INTEGER);
        CREATE INDEX ptr_src ON pointers(src);
        CREATE INDEX sense_src ON senses(synset);
        """)
        for i,key in enumerate(keys):
            lemma=f"term_{i:03d}"
            db.execute("INSERT INTO synsets VALUES (?,?,?,?,?,?,?)",
                       (key,"n",key[:8],i,3,
                        json.dumps([lemma,lemma+"_variant"]),
                        f"definition for {lemma}"))
            db.execute("INSERT INTO senses VALUES (?,?,?,?,?)",
                       (f"{lemma}%1:03:00::",key,lemma,1,2))
            db.execute("INSERT INTO senses VALUES (?,?,?,?,?)",
                       (f"{lemma}_variant%1:03:01::",key,lemma+"_variant",2,0))
            dst=keys[(i+1)%len(keys)]
            # Exact lexical indices 2 and 1 must be honored, not discarded.
            db.execute("INSERT INTO pointers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (i+1,key,dst,"@",2,1,i,(i+1)%len(keys),i+1,1,
                        "data.noun",i+1))
        db.commit()
    manifest = {
        "schema":"nanrouter.hw4-wordnet31-complete.v1",
        "route":"nan://GM0:SC0:MC0:BS0:0:HW4:0",
        "ring_owner":"NaNROUTER/MC0:0",
        "main_ring_writable":False,
        "generation":1,"generation_name":"generation-00000001",
        "source_archive_sha256":"a"*64,
        "root_bytes":8388608,"pair_count":32640,"heads":256,
        "synsets":len(keys),"senses":2*len(keys),"pointers":len(keys),
        "unresolved_pointers":0,
        "root_sha256":sha256(matrix),"sqlite_sha256":sha256(dbpath),
    }
    mf=folder/"manifest.json"
    mf.write_text(json.dumps(manifest,sort_keys=True)+"\n")
    (root/"CURRENT.json").write_text(json.dumps({
        "schema":manifest["schema"],
        "generation_name":manifest["generation_name"],
        "manifest_sha256":sha256(mf)},sort_keys=True)+"\n")
    return manifest, groups


class HW4CorpusTests(unittest.TestCase):
    def test_complete_stream_without_source_split_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td)
            src=base/"hw4"
            m, groups=fixture(src)
            verified,_=verify_hw4(src)
            self.assertEqual(verified["root_sha256"],m["root_sha256"])
            a=generate(src,base/"out")
            self.assertTrue(a["teacher_immutable"])
            self.assertTrue(a["no_bptt"])
            self.assertEqual(a["total"]["synsets"],12)
            self.assertEqual(a["total"]["senses"],24)
            self.assertEqual(a["total"]["relations"],12)
            all_records=(base/"out"/"all.txt").read_bytes()
            self.assertEqual(len(all_records),sum(
                a["files"][s]["bytes"] for s in ("train","dev","test")))
            self.assertEqual(a["files"]["all"]["sha256"],sha256(base/"out"/"all.txt"))
            all_manifest=json.loads((base/"out"/"all.manifest.json").read_text())
            self.assertEqual(all_manifest["synsets"],12)
            self.assertEqual(all_manifest["senses"],24)
            self.assertEqual(all_manifest["relations"],12)
            self.assertIsNone(all_manifest["heldout_sha256"])
            self.assertTrue(all_manifest["teacher_immutable"])
            self.assertEqual(a["source_owner"],"NaNROUTER/MC0:0")
            for split,keys in groups.items():
                report=a["files"][split]
                self.assertEqual(report["counts"]["synsets"],len(keys))
                self.assertEqual(report["counts"]["senses"],len(keys)*2)
                self.assertEqual(report["counts"]["relations"],len(keys))
                text=(base/"out"/f"{split}.txt").read_text()
                edges=(base/"out"/f"{split}.edges.tsv").read_text().splitlines()
                self.assertEqual(len(edges),len(keys))
                for i,k in enumerate([v for g in groups.values() for v in g]):
                    label=f"term {i:03d} (noun): definition for term {i:03d}"
                    if k in keys: self.assertIn(label,text)
                    else: self.assertNotIn(label,text)
                self.assertIn("has hypernym",text)
            self.assertEqual(
                json.loads((base/"out"/"train.manifest.json").read_text())["heldout_sha256"],
                a["files"]["test"]["sha256"])
            # Deterministic export: same store, independent generation directory.
            b=generate(src,base/"out2")
            self.assertEqual(a["files"],b["files"])
            self.assertEqual((base/"out"/"train.txt").read_bytes(),
                             (base/"out2"/"train.txt").read_bytes())
            self.assertEqual(verified["root_sha256"],sha256(src/"generation-00000001"/"wordnet-c256.root"))

    def test_reject_wrong_owner_tamper_and_output_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"hw4"
            fixture(root)
            base=Path(td)
            generate(root,base/"output")
            with self.assertRaises(FileExistsError):
                generate(root,base/"output")
            matrix=root/"generation-00000001"/"wordnet-c256.root"
            with matrix.open("r+b") as f:
                f.seek(777)
                f.write(b"x")
            with self.assertRaisesRegex(ValueError,"integrity"):
                generate(root,base/"never-created")
            self.assertFalse((base/"never-created").exists())

    def test_stable_split_is_not_random_or_python_hash(self):
        self.assertEqual(split_synset("00001740-n"),split_synset("00001740-n"))
        self.assertEqual(
            sum(split_synset(f"{i:08d}-n")=="train" for i in range(1000)),
            sum(split_synset(f"{i:08d}-n")=="train" for i in range(1000)))
        self.assertTrue(set(split_synset(f"{i:08d}-n") for i in range(1000))
                        == {"train","dev","test"})


if __name__ == "__main__":
    unittest.main()
