#!/usr/bin/env python3
"""Pinned, nonoverlapping author-disjoint corpus for joint semantic-wave audit.

Uses the existing exact GITenberg commit+SHA verifier. No book text is
committed to the repository; split bytes are temporary CI inputs only.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

from public_domain_benchmark import book

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare_joint_gutenberg.py OUTPUT_DIRECTORY")
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    alice, alice_meta = book("alice")
    carol, carol_meta = book("carol")
    # Character offsets avoid splitting a Unicode code point. The titles,
    # authors, and training/development/test positions are all recorded.
    fragments = {
        "train_alice": ("alice", 1000, 3600, alice),
        "dev_alice": ("alice", 50000, 50640, alice),
        "test_carol_early": ("carol", 1200, 2170, carol),
        "test_carol_late": ("carol", 28000, 28970, carol),
    }
    manifest = {
        "books": {"alice": alice_meta, "carol": carol_meta},
        "splits": {},
        "test_policy": "No Carol bytes in either supervised training or the root update acceptance gate",
    }
    for label, (name, start, end, corpus) in fragments.items():
        assert len(corpus) > end, (label, len(corpus))
        text = corpus[start:end]
        blob = text.encode("utf-8")
        assert b"\xff" not in blob and len(blob) >= 500, label
        path = target / (label + ".txt")
        path.write_bytes(blob)
        manifest["splits"][label] = {
            "author": name, "char_start": start, "char_end": end,
            "file": path.name, "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        print("JOINT_GUTENBERG_SPLIT " + " ".join(
            f"{key}={value}" for key, value in {
                "name": label, "author": name, "chars": len(text),
                "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()
            }.items()), flush=True)
    (target / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("JOINT_GUTENBERG_SOURCES alice_sha256=" +
          alice_meta["source_sha256"] + " carol_sha256=" +
          carol_meta["source_sha256"], flush=True)

if __name__ == "__main__":
    main()
