#!/usr/bin/env python3
"""HW5:0 Gutenberg source-disjoint transfer on ONE existing HW0 student.

Restores the exact completed HW4:0 epoch-103 model; does NOT train reference
matrices, create another production model, use BPTT, or claim temporal TRAIL.
Reports a native conditional-byte (order-one) geometric-credit experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

STARTING_ROOT_SHA = "e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1"
ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW5:0"
FIXTURE = Path("tests/fixtures/unified_quality_heldout.txt")
CHECKPOINTS = {1,2,5,10,15,20}
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def ensure_source(store:Path):
    current=json.loads((store/"CURRENT.json").read_text())
    gen=store/current["generation_name"]
    if sha(gen/"manifest.json")!=current["manifest_sha256"]:
        raise ValueError("HW5 reference current pointer checksum mismatch")
    m=json.loads((gen/"manifest.json").read_text())
    if m["route"]!=ROUTE or m["root_bytes"]!=8388608 or m["mirror_pairs"]!=32640 or m["canonical_cells"]!=32896:
        raise ValueError("unexpected HW5 geometry/route")
    if m["main_ring_writable"] is not False or m["ring_owner"]!="NaNROUTER/MC0:0":
        raise ValueError("HW5 cannot share writable training ring")
    for name,entry in m["artifacts"].items():
        p=gen/name
        if p.is_symlink() or p.stat().st_size!=entry["bytes"] or sha(p)!=entry["sha256"]:
            raise ValueError("HW5 corrupt source artifact: "+name)
    if len({d["id"] for d in m["books"]})!=len(m["books"]):
        raise ValueError("duplicate Gutenberg source")
    parts={s:sum(d["bytes"] for d in m["books"] if d["split"]==s)
           for s in ("train","dev","test")}
    for split,expected in parts.items():
        if expected<1024 or (gen/f"{split}.txt").stat().st_size!=expected:
            raise ValueError("invalid document-disjoint split "+split)
    if sum(parts.values())!=m["corpus_bytes"]:
        raise ValueError("split sums do not reconstruct exact corpus")
    return m,gen,parts
def run_cmd(command, log_path:Path):
    start=time.perf_counter()
    proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    seconds=time.perf_counter()-start
    log_path.write_text(proc.stdout)
    if proc.returncode!=0:
        raise RuntimeError(f"native failed {proc.returncode}: {command[0]}: {proc.stdout[-1200:]}")
    return proc.stdout,seconds
def fields(log:str,tag:str):
    found=[line for line in log.splitlines() if line.startswith(tag+" ")]
    if len(found)!=1:raise ValueError("missing or duplicate "+tag)
    return dict(re.findall(r"([a-z_]+)=([^\s]+)",found[0]))
def metrics(source:Path, root:Path, output:Path, label:str):
    before=sha(root)
    output_data,seconds=run_cmd(["./hw5-onepass","eval",str(root),str(source)],
                                output/f"{label}.log")
    after=sha(root)
    if before!=after:raise ValueError("read-only eval changed learner root")
    x=fields(output_data,"HW5_GEOMETRIC_EVAL")
    if x["bptt_attempted"]!="0" or int(x["input_bytes"])!=source.stat().st_size:
        raise ValueError("HW5 byte evaluation contract violated")
    return dict(bytes=int(x["input_bytes"]),ce_nats=float(x["ce_nats"]),
                accuracy=float(x["accuracy"]),seconds=seconds)
def byte_counts(path:Path):
    rows=[[0]*255 for _ in range(255)]
    contexts=[0]*255
    ycounts=[0]*255
    prev=None
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            for b in chunk:
                if b>=255:raise ValueError("0xff ingress")
                ycounts[b]+=1
                if prev is not None:
                    rows[prev][b]+=1
                    contexts[prev]+=1
                prev=b
    return rows,contexts,ycounts
def baseline(train:Path, test:Path, alpha=0.5):
    tr,tc,yc=byte_counts(train)
    te,_,_=byte_counts(test)
    denom=max(1,sum(yc))
    tot=0; nll=0.0;correct=0
    for a in range(255):
        best=max(range(255),key=lambda b:tr[a][b])
        for b in range(255):
            n=te[a][b]
            if not n:continue
            tot+=n
            correct+=n*(b==best)
            nll-=n*math.log((tr[a][b]+alpha)/(tc[a]+alpha*255))
    return dict(ce_nats=nll/tot,accuracy=correct/tot,transitions=tot,
                method="add-0.5 full-corpus aggregate bigram counts, diagnosis only")
def experiment(store:Path, previous:Path, output:Path, epochs:int):
    if output.exists():raise FileExistsError("experiment output must be new")
    output.mkdir(parents=True)
    start=time.perf_counter()
    manifest,g,parts=ensure_source(store)
    if previous.stat().st_size!=8388608 or sha(previous)!=STARTING_ROOT_SHA:
        raise ValueError("not restored from exact HW4 103-epoch checkpoint")
    model=output/"learner-root"
    shutil.copyfile(previous,model)
    source_digest=manifest["artifacts"]["all.txt"]["sha256"]
    report={
        "schema":"nanlm-test.hw5-gutenberg-transfer.v1",
        "source_route":ROUTE,"source_owner":"NaNROUTER/MC0:0",
        "source_reference_immutable":True,
        "reference_corpus_sha256":source_digest,
        "reference_root_sha256":manifest["artifacts"]["gutenberg-c256.root"]["sha256"],
        "reference_triple_ledger_sha256":manifest["artifacts"]["gutenberg-triples.sqlite3"]["sha256"],
        "source_book_count":len(manifest["books"]),
        "source_bytes":manifest["corpus_bytes"],"source_split_bytes":parts,
        "initial_student_root_sha256":STARTING_ROOT_SHA,
        "initial_student_cumulative_wordnet_epochs":103,
        "method":"one corpus scan and ONE rank-three aggregate byte-transition geometric credit step per epoch; no BPTT, no per-byte sequential TRAIL",
        "per_byte_trail_steps_claimed":0,"bptt_attempted":0,
        "requested_gutenberg_epochs":epochs,
        "completed_gutenberg_epochs":0,
        "epochs":[]
    }
    def evaluate(split:str,label:str):
        return metrics(g/f"{split}.txt",model,output,label)
    report["initial"]={s:evaluate(s,"before-"+s) for s in ("train","dev","test")}
    report["initial"]["prose"]=metrics(FIXTURE,model,output,"before-prose")
    report["count_control"]={s:baseline(g/"train.txt",g/f"{s}.txt") for s in ("dev","test")}
    native_secs=0
    for i in range(1,epochs+1):
        log,secs=run_cmd(["./hw5-onepass","train",str(model),str(g/"train.txt")],
                         output/f"epoch-{i:03d}.log")
        x=fields(log,"HW5_FULL_GEOMETRIC_PASS")
        if int(x["input_bytes"])!=parts["train"] or x["bptt_attempted"]!="0":
            raise ValueError("not one full document-disjoint train split per epoch")
        before=float(x["train_before_ce"]);after=float(x["train_after_ce"])
        if x["aggregate_step_accepted"]!="1" or not after<before:
            raise ValueError("native scratch acceptance gate rejected update")
        row=dict(gutenberg_epoch=i,cumulative_updates_after_wordnet=103+i,
                 train_before_ce=before,train_after_ce=after,
                 train_before_accuracy=float(x["train_before_accuracy"]),
                 train_after_accuracy=float(x["train_after_accuracy"]),
                 native_seconds=secs,gradient_max=float(x["gradient_max"]),
                 step=float(x["step"]),accepted=True)
        native_secs+=secs
        if i in CHECKPOINTS or i==epochs:
            row["dev"]=evaluate("dev",f"dev-{i:03d}")
            row["prose"]=metrics(FIXTURE,model,output,f"prose-{i:03d}")
            row["student_sha256"]=sha(model)
        report["epochs"].append(row)
        report["completed_gutenberg_epochs"]=i
        report["last_student_sha256"]=sha(model)
        report["native_training_seconds"]=native_secs
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("HW5_TRANSFER_EPOCH",json.dumps({"epoch":i,"ce":after,"accuracy":row["train_after_accuracy"],"seconds":secs,"dev_ce":row.get("dev",{}).get("ce_nats")}),flush=True)
    report["final"]={s:evaluate(s,"after-"+s) for s in ("train","dev","test")}
    report["final"]["prose"]=metrics(FIXTURE,model,output,"after-prose")
    report["total_elapsed_seconds"]=time.perf_counter()-start
    report["final_student_sha256"]=sha(model)
    report["bytes_scanned_in_training"]=epochs*parts["train"]
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    summary={k:report[k] for k in ("source_bytes","source_book_count","source_split_bytes",
              "initial","final","count_control","native_training_seconds",
              "total_elapsed_seconds","final_student_sha256","bytes_scanned_in_training",
              "completed_gutenberg_epochs")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("HW5_GUTENBERG_FINAL",json.dumps(summary,sort_keys=True),flush=True)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store",type=Path,required=True)
    ap.add_argument("--previous",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=20)
    a=ap.parse_args()
    if not 1<=a.epochs<=100:raise ValueError("epochs out of bounds")
    experiment(a.store,a.previous,a.output,a.epochs)
if __name__=="__main__":main()
