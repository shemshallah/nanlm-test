#!/usr/bin/env python3
"""Verified WordNet 3.1 full dictionary -> byte-exact corpus (no hidden truncation)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

URLS = (
    "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz",
    "https://distcache.FreeBSD.org/ports-distfiles/wn3.1.dict.tar.gz",
)
SHA256 = "3f7d8be8ef6ecc7167d39b10d66954ec734280b5bdcd57f7d9eafe429d11c22a"
OFFICIAL_ARCHIVE_BYTES = 16358468
POS = ("noun", "verb", "adj", "adv")
REQUIRED = {
    *(f"data.{part}" for part in POS),
    *(f"index.{part}" for part in POS),
    *(f"{part}.exc" for part in POS),
    "index.sense", "lexnames",
}
# Every lexical database text file, not just nouns or glosses.
EXTRA = {"cntlist", "cntlist.rev", "frames.vrb", "sents.vrb",
         "sentidx.vrb", "verb.Framestext", "log.grind.3.1",
         "README", "LICENSE"}
MAX_SOURCE_BYTES = 80_000_000
MIN_SYNS = 100_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def acquire(destination: Path, alternate: Path | None = None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if alternate is not None:
        raw = alternate.read_bytes()
        # User-supplied ZIP may have a different archival checksum; the
        # exact bytes and file inventory will be stored in the manifest.
        if alternate.suffix != ".zip" and digest(raw) != SHA256:
            raise RuntimeError("Supplied WordNet tarball digest differs from the Princeton release")
        destination.write_bytes(raw)
        return {"source":str(alternate),"sha256":digest(raw),"bytes":len(raw),
                "official":digest(raw)==SHA256}
    if destination.is_file():
        raw = destination.read_bytes()
        if digest(raw)==SHA256 and len(raw)==OFFICIAL_ARCHIVE_BYTES:
            return {"source":URLS[0],"sha256":digest(raw),"bytes":len(raw),
                    "official":True}
        destination.unlink()
    errors = []
    for url in URLS:
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={
                    "User-Agent":"NaNLM-reproducible-WordNet31/1.0",
                    "Accept":"application/octet-stream"})
                with urllib.request.urlopen(request,timeout=90) as response:
                    raw=response.read(MAX_SOURCE_BYTES+1)
                if len(raw)!=OFFICIAL_ARCHIVE_BYTES or digest(raw)!=SHA256:
                    raise RuntimeError(f"Princeton WordNet archive SHA mismatch from {url}")
                destination.write_bytes(raw)
                return {"source":url,"sha256":digest(raw),"bytes":len(raw),
                        "official":True}
            except (OSError,urllib.error.URLError,RuntimeError) as exc:
                errors.append(f"{url} attempt {attempt+1}: {exc}")
                time.sleep(2)
    raise RuntimeError("WordNet 3.1 pinned source unavailable: "+" | ".join(errors))


def extract(archive: Path) -> dict[str,bytes]:
    data:dict[str,bytes]={}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            entries=((p.filename,z.read(p)) for p in z.infolist() if not p.is_dir())
            for name,blob in entries:
                base=Path(name).name
                if base in REQUIRED or base in EXTRA:
                    if base in data and data[base]!=blob:
                        raise RuntimeError(f"ambiguous duplicate WordNet file {base}")
                    data[base]=blob
    else:
        with tarfile.open(archive,"r:*") as t:
            for member in t:
                if not member.isfile():
                    continue
                base=Path(member.name).name
                if base not in REQUIRED and base not in EXTRA:
                    continue
                if member.size>MAX_SOURCE_BYTES:
                    raise RuntimeError("oversized WordNet member")
                stream=t.extractfile(member)
                if stream is None:
                    raise RuntimeError("WordNet member extraction failed")
                blob=stream.read(MAX_SOURCE_BYTES+1)
                if base in data and data[base]!=blob:
                    raise RuntimeError(f"ambiguous duplicate WordNet file {base}")
                data[base]=blob
    missing=sorted(REQUIRED-data.keys())
    if missing:
        raise RuntimeError(f"WordNet 3.1 incomplete: missing {missing}")
    if any(len(blob)>MAX_SOURCE_BYTES or b"\xff" in blob for blob in data.values()):
        raise RuntimeError("WordNet file exceeds size bound or contains reserved 0xFF")
    return data


def prepare(archive: Path, out: Path, source: dict, strict:bool=True) -> dict:
    members=extract(archive)
    out.parent.mkdir(parents=True,exist_ok=True)
    order=sorted(members)
    inventory={}
    synsets=0
    written=0
    h=hashlib.sha256()
    with out.open("wb") as f:
        for name in order:
            blob=members[name]
            lines=blob.splitlines()
            valid=sum(bool(line) and 48<=line[0]<=57
                      for line in lines) if name.startswith("data.") else 0
            synsets+=valid
            inventory[name]={"bytes":len(blob),"lines":len(lines),
                             "synsets":valid,"sha256":digest(blob)}
            # Full member contents including original pointer IDs, glosses,
            # all lemmas and morphological exceptions. No sample/word cap.
            segment=(f"\nWORDNET_3_1_BEGIN {name}\n".encode("ascii") +
                     blob + (b"" if blob.endswith(b"\n") else b"\n") +
                     f"WORDNET_3_1_END {name}\n".encode("ascii"))
            f.write(segment)
            h.update(segment)
            written+=len(segment)
    if strict and (synsets<MIN_SYNS or written<10_000_000):
        raise RuntimeError(f"WordNet 3.1 corpus too small: synsets={synsets} bytes={written}")
    report={"source":source,"archive_sha256":source["sha256"],
            "corpus_sha256":h.hexdigest(),"corpus_bytes":written,
            "synsets":synsets,"members":inventory,"selected_files":order}
    manifest=out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--archive",type=Path,default=None,
                        help="optional user-provided wn3.1.dict.zip or verified tarball")
    parser.add_argument("--cache",type=Path,default=Path("wordnet31.tar.gz"))
    parser.add_argument("--output",type=Path,default=Path("wordnet31-full.txt"))
    parser.add_argument("--fixture",action="store_true")
    args=parser.parse_args()
    source=acquire(args.cache,args.archive)
    report=prepare(args.cache,args.output,source,strict=not args.fixture)
    print("WORDNET31_SOURCE "+json.dumps(source,sort_keys=True),flush=True)
    print("WORDNET31_CORPUS "+json.dumps({
        k:v for k,v in report.items() if k!="members"
    },sort_keys=True),flush=True)
    for name,counts in report["members"].items():
        print("WORDNET31_MEMBER "+json.dumps({"file":name,**counts}),flush=True)


if __name__=="__main__":
    main()
