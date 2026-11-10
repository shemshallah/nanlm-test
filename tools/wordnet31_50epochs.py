#!/usr/bin/env python3
"""50 full epochs on the native one-root geometric-credit trainer.

Training is byte-complete and checkpointed in bounded sequential chunks.
The script never calls BPTT. Run at a bounded wall-time budget and resume
from its state directory rather than silently claiming 50 completed epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

HERE=Path(__file__).resolve().parents[1]
EXECUTABLE=HERE/"nanlm"
MAX_STEP_BYTES=65536


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path:Path,obj:dict)->None:
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args:list[str],timeout:int=300)->tuple[str,float]:
    started=time.monotonic()
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     timeout=timeout,check=False)
    spent=time.monotonic()-started
    if p.returncode:
        raise RuntimeError(f"command {args[1:]} exit={p.returncode} "
                           f"stderr={p.stderr[-1800:]} stdout={p.stdout[-1800:]}")
    return p.stdout.strip(),spent


def parse_kv(line:str)->dict:
    result={}
    for name,val in re.findall(r"([A-Za-z_]+)=([^\s]+)",line):
        result[name]=val
    return result


def evaluate(root:Path)->dict:
    held=HERE/"tests/fixtures/unified_quality_heldout.txt"
    if not held.is_file():
        return {}
    output,seconds=run([str(EXECUTABLE),str(root),"eval",str(held)],timeout=300)
    data=parse_kv(output)
    return {"seconds":seconds,"tokens":int(data["bytes"]),
            "ce_nats":float(data["ce_nats"]),
            "accuracy":float(data["accuracy"])}


def checkpoint(args:argparse.Namespace,source:dict)->None:
    start=time.monotonic()
    corpus=args.corpus
    size=corpus.stat().st_size
    fingerprint=sha(corpus)
    state_dir=args.state_dir
    state_dir.mkdir(parents=True,exist_ok=True)
    root=state_dir/"unified-root"
    progress_file=state_dir/"progress.json"
    if progress_file.is_file():
        p=json.loads(progress_file.read_text())
        if (p["corpus_sha256"]!=fingerprint or p["corpus_bytes"]!=size or
            p["epochs_target"]!=args.epochs):
            raise RuntimeError("resumption is not using exactly the original full corpus")
    else:
        if root.exists():
            raise RuntimeError("root exists without a progress journal; refusing unsafe replay")
        p={"corpus_sha256":fingerprint,"corpus_bytes":size,"epochs_target":args.epochs,
           "epochs_completed":0,"offset":0,"total_bytes_seen":0,
           "total_scored":0,"chunks":0,"total_runtime_seconds":0.0,
           "epoch_nll":0.0,"epoch_scored":0,"epoch_credit_accepted":0,
           "epoch_history":[],"initial_eval":None,"recent_eval":None}
        atomic_json(progress_file,p)
    log=state_dir/"chunks.jsonl"
    print("WORDNET31_TRAIN_CONFIG "+json.dumps({
        "archive_sha256":source["archive_sha256"],"corpus_sha256":fingerprint,
        "full_corpus_bytes":size,"epochs":args.epochs,
        "total_requested_bytes":size*args.epochs,
        "current_completed_bytes":p["total_bytes_seen"],
        "time_budget_seconds":args.budget_seconds,
        "max_overlay_bytes":args.max_overlay_bytes},sort_keys=True),flush=True)
    if not root.exists() and p["total_bytes_seen"]>0:
        raise RuntimeError("checkpoint exists without 8 MiB model root")
    if not args.pilot_only and p["initial_eval"] is None:
        p["initial_eval"]=evaluate(root)
        atomic_json(progress_file,p)
    target_chunks=1 if args.pilot_only else 1000000000
    chunk_size=min(args.initial_chunk_bytes,args.max_chunk_bytes,MAX_STEP_BYTES)
    stats={"last_throughput":None,"recent_online_ce":None}
    status="COMPLETE" if p["epochs_completed"]>=args.epochs else "PARTIAL"
    while p["epochs_completed"]<args.epochs and target_chunks>0:
        if time.monotonic()-start>args.budget_seconds:
            status="TIME_BUDGET";break
        overlay=Path(str(root)+".overlay")
        if overlay.is_file() and overlay.stat().st_size>args.max_overlay_bytes:
            status="OVERLAY_CAPACITY";break
        remaining=size-p["offset"]
        if remaining<=0:
            raise RuntimeError("invalid epoch offset")
        n=min(chunk_size,remaining)
        absolute_step=p["epochs_completed"]*size+p["offset"]
        output,spent=run([str(EXECUTABLE),str(root),"train-chunk",str(corpus),
                          str(p["offset"]),str(n),str(absolute_step)],
                         timeout=args.chunk_timeout_seconds)
        if not output.startswith("WORDNET_CHUNK "):
            raise RuntimeError(f"unrecognized native training output: {output}")
        record=parse_kv(output)
        if int(record["bytes"])!=n or int(record["bptt_attempted"])!=0:
            raise RuntimeError("native trainer violated exact-byte or no-BPTT contract")
        tokens=int(record["scored"])
        ce=float(record["online_ce"])
        if not math.isfinite(ce) or not math.isfinite(spent):
            raise RuntimeError("nonfinite online training result")
        credit=int(record["temporal_credit_accepted"])
        p["offset"]+=n
        p["total_bytes_seen"]+=n
        p["total_scored"]+=tokens
        p["chunks"]+=1
        p["total_runtime_seconds"]+=spent
        p["epoch_nll"]+=tokens*ce
        p["epoch_scored"]+=tokens
        p["epoch_credit_accepted"]+=credit
        events={"epoch":p["epochs_completed"]+1,
                "position":p["offset"],"bytes":n,
                "global_trained_bytes":p["total_bytes_seen"],
                "seconds":spent,"bytes_per_second":n/spent,
                "online_ce":ce,"credit_accepted":credit,
                "events":int(record["events"]),
                "hyperedges":int(record["hyperedges"])}
        with log.open("a",encoding="utf-8") as f:
            f.write(json.dumps(events,sort_keys=True)+"\n")
        stats["last_throughput"]=n/spent
        stats["recent_online_ce"]=ce
        if p["offset"]==size:
            p["epoch_history"].append({
                "epoch":p["epochs_completed"]+1,
                "bytes":size,"tokens":p["epoch_scored"],
                "online_ce":p["epoch_nll"]/max(1,p["epoch_scored"]),
                "temporal_credit_accepted":p["epoch_credit_accepted"]})
            p["epochs_completed"]+=1
            p["offset"]=0
            p["epoch_nll"]=0.0
            p["epoch_scored"]=0
            p["epoch_credit_accepted"]=0
            p["recent_eval"]=evaluate(root)
            print("WORDNET31_EPOCH "+json.dumps({
                **p["epoch_history"][-1],"heldout":p["recent_eval"]
            },sort_keys=True),flush=True)
        atomic_json(progress_file,p)
        remaining_budget=args.budget_seconds-(time.monotonic()-start)
        if remaining_budget<2*spent:
            status="TIME_BUDGET";break
        # Aim for ~12s updates. For large roots the event journal can
        # dominate each process; do not assume linear token throughput.
        chunk_size=min(args.max_chunk_bytes,MAX_STEP_BYTES,
                       max(args.min_chunk_bytes,
                           int(n/max(0.001,spent)*12)))
        if p["chunks"]<=2 or p["chunks"]%10==0 or args.pilot_only:
            print("WORDNET31_CHUNK "+json.dumps(events,sort_keys=True),flush=True)
        target_chunks-=1
    if not args.pilot_only and root.is_file():
        try:
            p["recent_eval"]=evaluate(root)
            atomic_json(progress_file,p)
        except Exception as e:
            print(f"WORDNET31_EVAL_ERROR {type(e).__name__}: {e}",flush=True)
    throughput=(p["total_bytes_seen"]/max(0.001,p["total_runtime_seconds"]))
    estimated_total_hours=(size*args.epochs/throughput/3600) if throughput>0 else None
    file_bytes={suffix:Path(str(root)+suffix).stat().st_size
                for suffix in ("",".overlay",".so3",".episodes")
                if Path(str(root)+suffix).is_file()}
    if p["epochs_completed"]>=args.epochs:
        status="COMPLETE"
    elif args.pilot_only:
        status="PILOT_ONLY"
    summary={"status":status,"epochs_target":args.epochs,
             "epochs_completed":p["epochs_completed"],
             "offset_into_next_epoch":p["offset"],
             "full_corpus_bytes":size,"total_requested_bytes":size*args.epochs,
             "bytes_actually_seen":p["total_bytes_seen"],
             "progress_percent":100*p["total_bytes_seen"]/(size*args.epochs),
             "scored_tokens":p["total_scored"],"chunks":p["chunks"],
             "measured_bytes_per_second":throughput,
             "extrapolated_total_hours_at_measured_rate":estimated_total_hours,
             "root_and_sidecar_bytes":file_bytes,
             "initial_eval":p["initial_eval"],"recent_eval":p["recent_eval"],
             "last":stats,"source":source}
    atomic_json(state_dir/"summary.json",summary)
    print("WORDNET31_FINAL "+json.dumps(summary,sort_keys=True),flush=True)


def main()->None:
    a=argparse.ArgumentParser()
    a.add_argument("--corpus",type=Path,required=True)
    a.add_argument("--manifest",type=Path,required=True)
    a.add_argument("--state-dir",type=Path,required=True)
    a.add_argument("--epochs",type=int,default=50)
    a.add_argument("--budget-seconds",type=int,default=900)
    a.add_argument("--chunk-timeout-seconds",type=int,default=300)
    a.add_argument("--initial-chunk-bytes",type=int,default=256)
    a.add_argument("--min-chunk-bytes",type=int,default=128)
    a.add_argument("--max-chunk-bytes",type=int,default=8192)
    a.add_argument("--max-overlay-bytes",type=int,default=256*1024*1024)
    a.add_argument("--pilot-only",action="store_true")
    args=a.parse_args()
    if args.epochs!=50:
        raise RuntimeError("This is the explicit 50-epoch experiment; do not silently downsample")
    manifest=json.loads(args.manifest.read_text())
    if args.corpus.stat().st_size!=manifest["corpus_bytes"] or sha(args.corpus)!=manifest["corpus_sha256"]:
        raise RuntimeError("full-corpus source SHA mismatch before training")
    if args.max_chunk_bytes>MAX_STEP_BYTES or args.min_chunk_bytes>args.initial_chunk_bytes:
        raise RuntimeError("invalid chunk size")
    checkpoint(args,manifest)


if __name__=="__main__":
    main()
