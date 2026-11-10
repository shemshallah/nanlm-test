#!/usr/bin/env python3
"""HW5 native hypermatrix: tune only on development books, freeze for test.

Model: one SHA-pinned existing 8MiB geometric student; native read-only
C(256,2) mirror projection + sparse exact ordered triples reconstructed
strictly from Gutenberg TRAIN documents, never the all-books reference.
No gradient updates occur in this phase; memory is an evidence readout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

HW5_ROUTE="nan://GM0:SC0:MC0:BS0:0:HW5:0"
FULL_CORPUS_SHA="044ba606977039729f4077e666ed50250a380d473ccdb7d20d1149f10e04566f"
FULL_REFERENCE_ROOT_SHA="5199b69b17cb70c19e142340dffe38d1a85ea2ffd307f4ed305d7d46664b23bc"
STUDENT_SHA="83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c"
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
def load_manifest(store:Path):
    current=json.loads((store/"CURRENT.json").read_text())
    assert current["schema"]=="nanrouter.hw5-gutenberg-c256.v1"
    assert current["generation_name"]=="generation-00000001"
    gen=store/current["generation_name"]
    assert sha(gen/"manifest.json")==current["manifest_sha256"]
    manifest=json.loads((gen/"manifest.json").read_text())
    assert manifest["route"]==HW5_ROUTE
    assert manifest["root_bytes"]==8388608
    assert manifest["mirror_pairs"]==32640
    assert manifest["canonical_cells"]==32896
    assert manifest["main_ring_writable"] is False
    assert manifest["artifacts"]["all.txt"]["sha256"]==FULL_CORPUS_SHA
    assert manifest["artifacts"]["gutenberg-c256.root"]["sha256"]==FULL_REFERENCE_ROOT_SHA
    for name,meta in manifest["artifacts"].items():
        file=gen/name
        assert file.stat().st_size==meta["bytes"] and sha(file)==meta["sha256"],name
    splits={}
    for split in ("train","dev","test"):
        docs=[d for d in manifest["books"] if d["split"]==split]
        splits[split]=docs
        assert len(docs)>=2 and sum(d["bytes"] for d in docs)==(gen/f"{split}.txt").stat().st_size
    ids=[{d["id"] for d in splits[s]} for s in splits]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    return gen,manifest,splits
def command(args,log:Path):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed {result.returncode}: {result.stdout[-1400:]}")
    return result.stdout
def parse(text:str):
    rows=[]
    for line in text.splitlines():
        if not line.startswith("HW5_CONTEXT_SCORE "):continue
        values=dict(re.findall(r"([a-z_]+)=([^\s]+)",line))
        for name in ("ce_nats","accuracy","beta","lambda","delta","kappa"):
            values[name]=float(values[name])
        values["transitions"]=int(values["transitions"])
        assert values["bptt_attempted"]=="0" and values["exact_model"]=="HW5_train_only"
        assert 0<=values["accuracy"]<=1 and 0<values["ce_nats"]<10
        rows.append(values)
    if not rows:raise ValueError("no native context metrics")
    return rows
def select(rows):
    choices=[r for r in rows if r["mode"]=="context_hybrid"]
    def best(predicate):
        possible=[r for r in choices if predicate(r)]
        if not possible:raise ValueError("no development candidates")
        return min(possible,key=lambda r:(r["ce_nats"],r["beta"],r["lambda"],r["delta"],r["kappa"]))
    # Selection is exclusively from development; never inspect test scores here.
    return {
       "best_any":best(lambda r:True),
       "best_memory_only":best(lambda r:r["lambda"]==0),
       "best_geometry_positive":best(lambda r:r["lambda"]>0),
       "best_skip_positive":best(lambda r:r["delta"]>0),
       "best_trigram_only":best(lambda r:r["lambda"]==0 and r["delta"]==0),
       "pure_teacher":next(r for r in rows if r["mode"]=="geometry_only"),
       "bigram_add05":next(r for r in rows if r["mode"]=="bigram_add05")
    }
def experiment(store:Path,student:Path,output:Path):
    if output.exists():raise FileExistsError("use new experiment directory")
    output.mkdir(parents=True)
    begin=time.perf_counter()
    gen,manifest,splits=load_manifest(store)
    if student.stat().st_size!=8388608 or sha(student)!=STUDENT_SHA:
        raise ValueError("wrong geometric student checkpoint")
    source_root_sha=sha(gen/"gutenberg-c256.root")
    for split in splits:
        (output/(split+".lengths")).write_text("".join(f"{d['bytes']}\n" for d in splits[split]))
    trained_root=output/"train-only-c256.root"
    trained_triples=output/"train-only-triples.bin"
    built=command(["./hw5-context","build",str(gen/"train.txt"),
                    str(output/"train.lengths"),str(trained_root),str(trained_triples)],
                    output/"build.log")
    assert trained_root.stat().st_size==8388608
    bank=dict(re.findall(r"([a-z_]+)=([^\s]+)",next(line for line in built.splitlines()
                         if line.startswith("HW5_CONTEXT_BANK_BUILT "))))
    assert int(bank["books"])==len(splits["train"])
    assert int(bank["bytes"])==sum(d["bytes"] for d in splits["train"])
    assert int(bank["adjacent"])==int(bank["bytes"])-int(bank["books"])
    assert int(bank["lag2"])==int(bank["bytes"])-2*int(bank["books"])
    assert int(bank["root_bytes"])==8388608
    # A train-only model must NOT be byte-identical to the all-books source.
    assert sha(trained_root)!=source_root_sha
    before_student=sha(student)
    before_full=sha(gen/"gutenberg-c256.root")
    def score(split:str,mode:str,label:str,config=None):
        argv=["./hw5-context","score",str(student),str(trained_root),str(trained_triples),
              str(gen/f"{split}.txt"),str(output/f"{split}.lengths"),mode]
        if config:
            argv.extend(str(config[x]) for x in ("beta","lambda","delta","kappa"))
        rows=parse(command(argv,output/(label+".log")))
        expected=sum(d["bytes"]-2 for d in splits[split])
        if any(r["transitions"]!=expected for r in rows):
            raise ValueError("cross-document transitions detected")
        return rows
    development=score("dev","sweep","dev-sweep")
    selected=select(development)
    # Explicitly freeze every hyperparameter chosen on unseen dev titles
    # BEFORE any test evaluations; source filenames and digests are logged.
    frozen={name:{k:r[k] for k in ("beta","lambda","delta","kappa")}
            for name,r in selected.items() if name not in ("pure_teacher","bigram_add05")}
    (output/"frozen-dev-choices.json").write_text(json.dumps(frozen,indent=2,sort_keys=True)+"\n")
    test_base=score("test","fixed","test-primary",frozen["best_any"])
    final={"best_any":test_base[-1],"pure_teacher":test_base[0],"bigram_add05":test_base[1]}
    for name,cfg in frozen.items():
        if name=="best_any":continue
        final[name]=score("test","fixed","test-"+name,cfg)[-1]
    if before_student!=sha(student) or before_full!=sha(gen/"gutenberg-c256.root"):
        raise RuntimeError("read-only experiment changed source or geometry student")
    report={
       "schema":"nanlm-test.hw5-hypermatrix-context.v1",
       "source_reference":HW5_ROUTE,
       "source_reference_immutable":True,
       "all_books_used_for_scoring_memory":False,
       "memory_trained_on_documents":"train_only",
       "source_books":len(manifest["books"]),
       "train_books":len(splits["train"]),
       "dev_books":len(splits["dev"]),
       "test_books":len(splits["test"]),
       "source_sha256":FULL_CORPUS_SHA,
       "source_matrix_sha256":FULL_REFERENCE_ROOT_SHA,
       "student_sha256":STUDENT_SHA,
       "train_mirror_sha256":sha(trained_root),
       "train_triple_sha256":sha(trained_triples),
       "train_memory_built":bank,
       "dev_candidates":len([r for r in development if r["mode"]=="context_hybrid"]),
       "dev_selected":selected,
       "test_precommitted_choices":frozen,
       "test":final,
       "native_readout":"existing HW0 geometric 3-vector head CP rank-three * train-only mirror directed adjacent, lag2 and sparse exact ordered triples; source-disjoint dev tuning",
       "bptt_attempted":0,
       "per_byte_sequential_trail_claimed":0,
       "test_ce_improvement_vs_bigram":final["bigram_add05"]["ce_nats"]-final["best_any"]["ce_nats"],
       "test_ce_improvement_geometry_over_memory":
            final["best_memory_only"]["ce_nats"]-final["best_geometry_positive"]["ce_nats"],
       "elapsed_seconds":time.perf_counter()-begin,
    }
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    compact={k:{"ce_nats":v["ce_nats"],"accuracy":v["accuracy"],"beta":v["beta"],
                 "lambda":v["lambda"],"delta":v["delta"],"kappa":v["kappa"]}
             for k,v in final.items()}
    print("HW5_HYPERMATRIX_CONTEXT_FINAL",json.dumps({
        "source_books":report["source_books"],"train_bytes":int(bank["bytes"]),
        "train_distinct_triples":int(bank["distinct_triples"]),
        "selected_dev":frozen["best_any"],"test":compact,
        "trained_mirror_sha256":report["train_mirror_sha256"],
        "student_immutable":True,"reference_immutable":True,
        "dev_candidates":report["dev_candidates"]},sort_keys=True),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--student",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    experiment(a.reference,a.student,a.output)
if __name__=="__main__":main()
