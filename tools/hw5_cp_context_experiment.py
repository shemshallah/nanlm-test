#!/usr/bin/env python3
"""Native Gutenberg CP second-order context distillation into one NaNLM root.

Only TRAIN titles contribute to gradient targets. Development books select
checkpoints, and test is evaluated after selection. Historical held-out titles
were already evaluated previously, so this is NOT a virgin test.
No exact-trigram bank at inference, no BPTT or per-token TRAIL.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from hw5_context_experiment import load_manifest, sha, STUDENT_SHA, HW5_ROUTE

ROOT_BYTES=8388608
LINE=re.compile(r"([a-z0-9_]+)=([^\s]+)")
def run(args,log):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"native command failed {result.returncode}: {args}: {result.stdout[-1400:]}")
    return result.stdout
def marker(text,prefix):
    found=[x for x in text.splitlines() if x.startswith(prefix+" ")]
    if len(found)!=1:raise ValueError("unexpected native marker "+prefix)
    return dict(LINE.findall(found[0]))
def scored(root,split,gen,out,mode="correct"):
    text=run(["./hw5-geometric-context","score",str(root),str(gen/f"{split}.txt"),
              str(out/f"{split}.lengths"),mode],out/f"{split}-{mode}-score.log")
    d=marker(text,"HW5_CP_SCORE")
    assert d["bptt_attempted"]=="0" and d["inference_trigram_lookup"]=="0"
    return {k:float(d[k]) for k in ("ce_nats","accuracy")} | {
        "transitions":int(d["transitions"]),"books":int(d["books"]),
        "context_rows":int(d["context_rows"])}
def experiment(reference,student,out,epochs,eval_every):
    started=time.perf_counter()
    gen,manifest,splits=load_manifest(reference)
    if out.exists():raise FileExistsError("output directory already exists")
    if student.stat().st_size!=ROOT_BYTES or sha(student)!=STUDENT_SHA:
        raise ValueError("wrong pinned HW5 Gutenberg geometric root")
    out.mkdir(parents=True)
    for split in ("train","dev","test"):
        (out/f"{split}.lengths").write_text(
            "".join(str(d["bytes"])+"\n" for d in splits[split]))
    first_sha=sha(student);ref_sha=sha(gen/"gutenberg-c256.root")
    learner=out/"learner-root";best=out/"best-dev-root"
    shutil.copyfile(student,learner)
    initial_dev=scored(learner,"dev",gen,out)
    best_dev=initial_dev["ce_nats"];best_epoch=0
    shutil.copyfile(learner,best);best_sha=sha(best)
    previous_sha=sha(learner)
    checks=[]
    for epoch in range(1,epochs+1):
        result=run(["./hw5-geometric-context","train",str(learner),
             str(gen/"train.txt"),str(out/"train.lengths"),"16"],
             out/f"train-{epoch:04d}.log")
        d=marker(result,"HW5_CP_TRAIN")
        assert int(d["transitions"])==sum(x["bytes"]-2 for x in splits["train"])
        assert int(d["books"])==len(splits["train"])
        assert d["bptt_attempted"]=="0" and int(d["root_bytes"])==ROOT_BYTES
        accepted=d["accepted"]=="1"
        new_sha=sha(learner)
        if (new_sha!=previous_sha)!=accepted:raise RuntimeError("acceptance/root hash mismatch")
        if float(d["after_soft_ce"])>float(d["before_soft_ce"])+1e-10:
            raise RuntimeError("native training soft-target CE regressed")
        row={"epoch":epoch,"accepted":accepted,"train_ce":float(d["after_ce"]),
             "train_soft_ce":float(d["after_soft_ce"]),
             "train_accuracy":float(d["after_accuracy"]),"root_sha256":new_sha,
             "step":float(d["step"])}
        previous_sha=new_sha
        if epoch==1 or epoch%eval_every==0 or not accepted or epoch==epochs:
            dev=scored(learner,"dev",gen,out)
            expected=sum(x["bytes"]-2 for x in splits["dev"])
            if dev["transitions"]!=expected:raise RuntimeError("cross-book dev contamination")
            row["dev"]=dev
            if dev["ce_nats"]+1e-8<best_dev:
                best_epoch=epoch;best_dev=dev["ce_nats"]
                shutil.copyfile(learner,best);best_sha=sha(best)
            print("HW5_CP_DEV",json.dumps({"epoch":epoch,"dev_ce":dev["ce_nats"],
                "dev_top1":dev["accuracy"],"best_epoch":best_epoch,
                "best_dev_ce":best_dev,"accepted":accepted},sort_keys=True),flush=True)
        checks.append(row)
        if not accepted:break
    if sha(student)!=first_sha or sha(gen/"gutenberg-c256.root")!=ref_sha:
        raise RuntimeError("immutable source or initial student mutated")
    if sha(best)!=best_sha:raise RuntimeError("selected root digest mismatch")
    selected_test=scored(best,"test",gen,out)
    wrong=scored(best,"test",gen,out,"wrong")
    selected_dev=scored(best,"dev",gen,out)
    if abs(selected_dev["ce_nats"]-best_dev)>1e-8:
        raise RuntimeError("development checkpoint not reproducible")
    selected_train=scored(best,"train",gen,out)
    for split,m in (("train",selected_train),("dev",selected_dev),("test",selected_test)):
        if m["transitions"]!=sum(x["bytes"]-2 for x in splits[split]):
            raise RuntimeError("boundary violation: "+split)
    report={"schema":"nanlm-test.hw5-native-cp-context-distillation.v1",
       "source_route":HW5_ROUTE,"source_manifest_sha256":sha(gen/"manifest.json"),
       "immutable_all_books_reference_sha256":ref_sha,
       "initial_geometric_root_sha256":first_sha,"best_geometric_root_sha256":best_sha,
       "best_epoch":best_epoch,"best_dev_ce":best_dev,
       "epochs_requested":epochs,"epochs_executed":len(checks),
       "train_books":len(splits["train"]),"dev_books":len(splits["dev"]),
       "test_books":len(splits["test"]),
       "train_bytes":sum(x["bytes"] for x in splits["train"]),
       "selected_train":selected_train,"selected_dev":selected_dev,
       "selected_test":selected_test,"test_wrong_context":wrong,
       "initial_dev":initial_dev,"epochs":checks,
       "model":"one 8MiB NaNLM root, existing 3D head vectors with native rank-three CP contextual credit",
       "train_only_distillation_beta":16,"inference_exact_triple_lookup":False,
       "bptt_attempted":0,"test_titles_previously_evaluated":True,
       "elapsed_seconds":time.perf_counter()-started}
    (out/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("HW5_CP_FINAL",json.dumps({
        "epochs":len(checks),"best_epoch":best_epoch,"dev_ce":best_dev,
        "test_ce":selected_test["ce_nats"],"test_top1":selected_test["accuracy"],
        "wrong_context_ce":wrong["ce_nats"],"wrong_context_top1":wrong["accuracy"],
        "best_sha256":best_sha,"student_unchanged":first_sha==sha(student),
        "reference_unchanged":ref_sha==sha(gen/"gutenberg-c256.root"),
        "inference_trigram_lookup":False},sort_keys=True),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",required=True,type=Path)
    p.add_argument("--student",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--eval-every",type=int,default=5)
    a=p.parse_args()
    if not 1<=a.epochs<=500 or not 1<=a.eval_every<=50:
        raise ValueError("bad training schedule")
    experiment(a.reference,a.student,a.output,a.epochs,a.eval_every)
if __name__=="__main__":main()
