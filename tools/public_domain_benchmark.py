#!/usr/bin/env python3
"""Reproducible cross-author benchmark using transient public-domain books.

Downloads Project Gutenberg plaintext during CI. Never commits or republishes
the works. SHA-256 of original sources and exact UTF-8 split bytes are logged.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXE = HERE / "nanlm"
SOURCES = {
    "alice": ("https://www.gutenberg.org/cache/epub/11/pg11.txt", 11,
              "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"),
    "carol": ("https://www.gutenberg.org/cache/epub/46/pg46.txt", 46,
              "8c36b93723d454982df045ec0efdf2664e5a48e6b5e2f670e883575f75ad0cc8"),
}

def book(name: str) -> tuple[str, dict]:
    url, number, expected_sha256 = SOURCES[name]
    request = urllib.request.Request(url, headers={
        "User-Agent": "NaNLM-public-domain-research/1.0 (noncommercial evaluation)",
        "Accept": "text/plain",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read(3_000_000)
    assert 30_000 < len(raw) < 3_000_000, (name, len(raw))
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == expected_sha256, (
        f"Project Gutenberg text for {name} changed: expected {expected_sha256}, "
        f"received {digest}; review and deliberately repin the source"
    )
    text = raw.decode("utf-8-sig")
    assert f"eBook #{number}" in text or f"EBook #{number}" in text or (
        f"ebooks/{number}" in text), "unexpected Gutenberg source"
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK [^\n]*\*\*\*", text)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK [^\n]*\*\*", text)
    assert start and end and end.start() > start.end(), "missing Gutenberg boundaries"
    body = text[start.end():end.start()]
    assert len(body) > 40_000
    return body, {"source": url, "ebook": number, "source_bytes": len(raw),
                  "source_sha256": hashlib.sha256(raw).hexdigest()}

def run(root: Path, *parts: object) -> str:
    proc = subprocess.run(
        [str(EXE), str(root), *(str(p) for p in parts)],
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True, timeout=600,
    )
    return proc.stdout

def ce(output: str) -> dict:
    m = re.search(r"UNIFIED_EVAL bytes=(\d+) ce_nats=([\d.eE+-]+) accuracy=([\d.eE+-]+)", output)
    assert m is not None, output
    result = {"tokens": int(m[1]), "cross_entropy": float(m[2]),
              "accuracy": float(m[3])}
    assert result["tokens"] > 500 and math.isfinite(result["cross_entropy"])
    return result

def save(path: Path, data: str) -> dict:
    blob = data.encode("utf-8")
    path.write_bytes(blob)
    assert len(blob) > 1500 and b"\xff" not in blob
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}

def main() -> None:
    alice, a_info = book("alice")
    carol, c_info = book("carol")
    with tempfile.TemporaryDirectory(prefix="nanlm-public-domain-") as temp:
        folder = Path(temp)
        # Slices are in Unicode characters, so no multibyte character is torn.
        # Development is noncontiguous and does not overlap training positions.
        train = folder / "train.txt"
        dev = folder / "dev.txt"
        held_a = folder / "carol_a.txt"
        held_b = folder / "carol_b.txt"
        splits = {
            "training": save(train, alice[1000:11500]),
            "development": save(dev, alice[50000:53000]),
            "carol_early": save(held_a, carol[1200:4200]),
            "carol_later": save(held_b, carol[28000:31000]),
        }
        assert splits["training"]["bytes"] >= 10000
        root = folder / "mapped_hw0"
        before = {name: ce(run(root,"eval",file)) for name,file in
                  (("carol_early",held_a),("carol_later",held_b))}
        trained = run(root,"train",train,1)
        assert root.stat().st_size == 8388608
        development = ce(run(root,"eval",dev))
        uncalibrated = {name: ce(run(root,"eval",file)) for name,file in
                        (("carol_early",held_a),("carol_later",held_b))}
        calibrated = run(root,"calibrate",dev)
        after = {name: ce(run(root,"eval",file)) for name,file in
                 (("carol_early",held_a),("carol_later",held_b))}
        sample = run(root,"generate","Alice was ",120,29)
        assert sample.startswith("Alice was ")
        weights = sum(v["tokens"] for v in after.values())
        combined = lambda group: sum(v["tokens"]*v["cross_entropy"]
                            for v in group.values())/weights
        report = {
            "sources": {"alice": a_info, "carol": c_info},
            "splits": splits, "before": before, "uncalibrated": uncalibrated,
            "after": after, "training_log": trained.strip(),
            "development": development, "calibration_log": calibrated.strip(),
            "weighted_before": combined(before),
            "weighted_uncalibrated": combined(uncalibrated),
            "weighted_after": combined(after),
            "generation_sample": sample[:140],
            "disclaimer": "Cross-author public-domain diagnostic; tiny training sample; not fluency proof",
        }
        (HERE/"public-domain-report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("PUBLIC_DOMAIN_SOURCE alice_sha256="+a_info["source_sha256"]+
              " carol_sha256="+c_info["source_sha256"])
        print(f"PUBLIC_DOMAIN_TRANSFER bytes={weights} "
              f"before_ce={report['weighted_before']:.8f} "
              f"uncalibrated_ce={report['weighted_uncalibrated']:.8f} "
              f"calibrated_ce={report['weighted_after']:.8f}")
        for name,row in after.items():
            print(f"PUBLIC_DOMAIN_DOMAIN name={name} tokens={row['tokens']} "
                  f"ce={row['cross_entropy']:.8f} accuracy={row['accuracy']:.6f}")
        print("PUBLIC_DOMAIN_GENERATION sample="+repr(sample[:100]))

if __name__=="__main__":
    main()
