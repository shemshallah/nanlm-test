#!/usr/bin/env python3
"""Paired heldout CE probe for the *other* NaNLM universe trained from HW4.

Uses exactly one throwaway learner root (never the HW4 matrix) and the
existing native train-chunk/geometric-credit path, BPTT disabled.
The result is a measurement, not a success gate: negative improvement is
reported verbatim rather than hidden. This deliberately avoids the 50-epoch
driver's persistent checkpoint and never contaminates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(1048576),b""):
            h.update(blob)
    return h.hexdigest()


def execute(command, timeout=240):
    started = time.monotonic()
    p = subprocess.run([str(v) for v in command], capture_output=True,
                       text=True, check=False, timeout=timeout)
    if p.returncode:
        raise RuntimeError("native HW4 credit probe failed: " + p.stderr[-1200:])
    return p.stdout.strip(),time.monotonic()-started


def fields(line):
    return {k:v for k,v in re.findall(r"([a-z_]+)=([^\s]+)",line)}


def measure(engine, root, evaluation):
    out,seconds = execute([engine,root,"eval",evaluation])
    data=fields(out)
    if not out.startswith("UNIFIED_EVAL ") or int(data["bytes"]) < 10:
        raise ValueError("invalid native heldout evaluation")
    score=float(data["ce_nats"])
    acc=float(data["accuracy"])
    if not math.isfinite(score) or not math.isfinite(acc):
        raise ValueError("non-finite heldout evaluation")
    return {"ce_nats":score,"accuracy":acc,"bytes":int(data["bytes"]),
            "seconds":seconds}


def benchmark(*, engine, corpus, manifest, heldout, budget=2048, eval_bytes=2048):
    m=json.loads(Path(manifest).read_text())
    corpus, heldout=Path(corpus),Path(heldout)
    if digest(corpus)!=m["corpus_sha256"] or corpus.stat().st_size!=m["corpus_bytes"]:
        raise ValueError("HW4 source corpus digest mismatch")
    if digest(heldout)!=m["heldout_sha256"]:
        raise ValueError("HW4 heldout digest mismatch")
    if not (1<=budget<=65536 and 64<=eval_bytes<=65536):
        raise ValueError("probe length outside native 64 KiB boundary")
    with tempfile.TemporaryDirectory(prefix="hw4-credit-probe-") as td:
        temp=Path(td)
        root=temp/"learner-root"
        sample=temp/"heldout-smoke.txt"
        with heldout.open("rb") as f, sample.open("wb") as out:
            out.write(f.read(eval_bytes))
        if sample.stat().st_size!=eval_bytes:
            raise ValueError("heldout sample is shorter than requested")
        before=measure(engine,root,sample)
        raw,seconds=execute([engine,root,"train-chunk",corpus,0,budget,0])
        record=fields(raw)
        if not raw.startswith("WORDNET_CHUNK ") or int(record["bytes"])!=budget:
            raise ValueError("native geometric training count mismatch")
        if int(record["bptt_attempted"]) != 0:
            raise ValueError("BPTT executed against HW4 geometric-credit contract")
        credits=int(record["temporal_credit_accepted"])
        after=measure(engine,root,sample)
        if after["bytes"]!=before["bytes"]:
            raise ValueError("heldout token counts changed")
        control,control_seconds=execute([engine,root,"eval-credit",sample])
        credit=fields(control)
        if not control.startswith("UNIFIED_CREDIT_EVAL ") or int(credit["bptt_attempted"])!=0:
            raise ValueError("native temporal credit ablation contract failed")
        result={
            "schema":"nanlm-test.hw4-geometric-credit-measurement.v1",
            "source_hw4_root_sha256":m["source_hw4_root_sha256"],
            "source_hw4_sqlite_sha256":m["source_hw4_sqlite_sha256"],
            "source_training_sha256":m["corpus_sha256"],
            "source_heldout_sha256":m["heldout_sha256"],
            "heldout_protocol":"first N bytes of unseen-source test corpus; not fully entity-disjoint",
            "train_bytes":budget, "train_seconds":seconds,
            "online_ce_nats":float(record["online_ce"]),
            "temporal_credits_accepted":credits,
            "bptt_attempted":0,
            "before":before, "after":after,
            "heldout_ce_delta_before_minus_after":before["ce_nats"]-after["ce_nats"],
            "credit_ablation":{
                "full_ce":float(credit["full_ce"]),
                "no_credit_ce":float(credit["no_credit_ce"]),
                "delta_no_credit_minus_full":float(credit["delta_no_credit_minus_full"]),
                "seconds":control_seconds,
            },
        }
        print("HW4_CREDIT_MEASURE "+json.dumps(result,sort_keys=True))
        return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--engine",type=Path,default=Path("./nanlm"))
    a.add_argument("--corpus",type=Path,required=True)
    a.add_argument("--manifest",type=Path,required=True)
    a.add_argument("--heldout",type=Path,required=True)
    a.add_argument("--train-bytes",type=int,default=2048)
    a.add_argument("--eval-bytes",type=int,default=2048)
    args=a.parse_args()
    benchmark(engine=args.engine.resolve(),corpus=args.corpus,
              manifest=args.manifest,heldout=args.heldout,
              budget=args.train_bytes,eval_bytes=args.eval_bytes)


if __name__=="__main__":
    main()
