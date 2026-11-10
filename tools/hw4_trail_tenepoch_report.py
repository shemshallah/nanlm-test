#!/usr/bin/env python3
"""Audit a continuous HW4:0 full corpus ten-epoch frozen-value TRAIL run."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import sys

CORPUS_BYTES = 58_341_529
SOURCE_HASH = "6c29dc5027012a8ef6ec50404a7bdbb50be81a4d018c1f327665f826d8784604"
START_HASH = "e97549fecc783ea2551f14924e3aaf1ade2046af6e663aa44dc158f254a9e873"

def read_marker(path: Path, marker: str) -> dict:
    matches = [line for line in path.read_text().splitlines() if line.startswith(marker+" ")]
    if len(matches) != 1:
        raise ValueError(f"expected one {marker}: {path}")
    return dict(re.findall(r"([a-z_0-9]+)=([^\s]+)",matches[0]))

def digest(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

def main():
    if len(sys.argv)!=3: raise SystemExit("usage: report STATE_DIR HW4_MANIFEST")
    rootdir = Path(sys.argv[1])
    manifest = json.loads(Path(sys.argv[2]).read_text())
    prior = json.loads((rootdir/"previous-three-more-receipt.json").read_text())
    assert manifest["corpus_bytes"]==CORPUS_BYTES and manifest["corpus_sha256"]==SOURCE_HASH
    assert manifest["teacher_immutable"] and manifest["heldout_sha256"] is None
    assert (manifest["synsets"],manifest["senses"],manifest["relations"],manifest["synonyms"]) == (117791,207235,378203,53869)
    assert prior["total_completed_passes"]==4 and prior["finished_student_sha256"]==START_HASH
    assert prior["corpus_sha256"]==SOURCE_HASH and prior["hw4_immutable"]
    assert prior["source_hw4_root_sha256"]==manifest["source_hw4_root_sha256"]
    assert prior["source_hw4_sqlite_sha256"]==manifest["source_hw4_sqlite_sha256"]
    assert len(prior["passes"])==3 and [p["epoch"] for p in prior["passes"]]==[2,3,4]
    assert prior["bptt_attempted"]==0 and all(p["corpus_bytes_seen"]==CORPUS_BYTES for p in prior["passes"])
    previous_sha=START_HASH
    previous_ce=float(prior["passes"][-1]["training_ce_after"])
    previous_prose_ce=float(prior["passes"][-1]["prose_ce_after"])
    previous_prose_acc=float(prior["passes"][-1]["prose_accuracy_after"])
    baseline=read_marker(rootdir/"prose-pass4.log","UNIFIED_EVAL")
    assert int(baseline["bytes"])==974
    assert abs(float(baseline["ce_nats"])-previous_prose_ce)<1e-6
    assert abs(float(baseline["accuracy"])-previous_prose_acc)<1e-6
    items=[]
    for epoch in range(5,11):
        t=read_marker(rootdir/f"trail-pass{epoch}.log","HW4_FULL_TRAIL")
        p=read_marker(rootdir/f"prose-pass{epoch}.log","UNIFIED_EVAL")
        nextsha=(rootdir/f"sha-pass{epoch}.txt").read_text().strip()
        assert re.fullmatch(r"[a-f0-9]{64}",nextsha)
        for flag in ("forward_rail","inverse_rail","hw4_read_only"):
            assert t[flag]=="1",(epoch,flag)
        assert t["bptt_attempted"]=="0"
        assert int(t["input_bytes"])==CORPUS_BYTES and int(t["transition_updates"])==CORPUS_BYTES-1
        assert int(t["root_bytes"])==8388608
        assert int(p["bytes"])==974
        c0,c1=float(t["initial_teacher_ce"]),float(t["final_teacher_ce"])
        assert math.isfinite(c0) and math.isfinite(c1)
        assert abs(c0-previous_ce)<1e-7,(epoch,c0,previous_ce)
        assert c1<=c0+1e-8,(epoch,c0,c1)
        accepted=t["accepted"]=="1"
        if not accepted:
            assert nextsha==previous_sha,("rejected scratch update modified root",epoch)
        measured={ "epoch":epoch,"corpus_bytes_seen":CORPUS_BYTES,
            "root_sha_before":previous_sha,"root_sha_after":nextsha,
            "accepted":accepted,"training_ce_before":c0,"training_ce_after":c1,
            "training_accuracy_before":float(t["initial_teacher_acc"]),
            "training_accuracy_after":float(t["final_teacher_acc"]),
            "prose_ce_before":previous_prose_ce,"prose_ce_after":float(p["ce_nats"]),
            "prose_accuracy_before":previous_prose_acc,"prose_accuracy_after":float(p["accuracy"]),
            "plus_rail_l1":float(t["geom_credit_plus_l1"]),
            "minus_rail_l1":float(t["geom_credit_minus_l1"]),
            "scan_seconds":float(t["full_scan_seconds"]),
            "total_seconds":float(t["total_seconds"]),
            "bytes_per_second":float(t["corpus_bytes_per_second"]),
            "step":float(t["step"]) }
        assert all(math.isfinite(value) for value in measured.values() if isinstance(value,float))
        items.append(measured)
        previous_sha=nextsha
        previous_ce=c1
        previous_prose_ce=float(p["ce_nats"])
        previous_prose_acc=float(p["accuracy"])
        print("HW4_TRAIL_TEN_PASS "+json.dumps(measured,sort_keys=True),flush=True)
    root=rootdir/"learner-root"
    assert root.stat().st_size==8388608 and digest(root)==previous_sha
    receipt={
        "schema":"nanlm-test.hw4-full-ten-epoch-two-rail-trail.v1",
        "source_route":manifest["source_route"],
        "corpus_sha256":SOURCE_HASH,
        "source_hw4_root_sha256":manifest["source_hw4_root_sha256"],
        "source_hw4_sqlite_sha256":manifest["source_hw4_sqlite_sha256"],
        "hw4_immutable":True,"bptt_attempted":0,
        "method":"one frozen-value two-rail TD eligibility sweep and at most one accepted rank-three update per complete pass, not online full recurrent training",
        "verified_previous_epochs":4,"additional_epochs":6,"total_epochs":10,
        "corpus_bytes_per_epoch":CORPUS_BYTES,
        "new_bytes_seen":CORPUS_BYTES*6,"total_bytes_seen":CORPUS_BYTES*10,
        "total_byte_transitions":(CORPUS_BYTES-1)*10,
        "starting_root_sha256":START_HASH,"finished_root_sha256":previous_sha,
        "previous_receipt":prior,"additional_passes":items,
        "new_scan_seconds":sum(x["scan_seconds"] for x in items),
        "new_native_seconds":sum(x["total_seconds"] for x in items),
        "new_accepted_updates":sum(x["accepted"] for x in items),
        "initial_training_ce_epoch4":float(prior["passes"][-1]["training_ce_after"]),
        "final_training_ce_epoch10":previous_ce,
        "initial_prose_ce_epoch4":float(prior["passes"][-1]["prose_ce_after"]),
        "final_prose_ce_epoch10":previous_prose_ce,
        "initial_prose_accuracy_epoch4":float(prior["passes"][-1]["prose_accuracy_after"]),
        "final_prose_accuracy_epoch10":previous_prose_acc,
        "limits":"Full corpus was trained every epoch: no WordNet heldout. Byte bigram CE and tiny 974-byte other-text probe do not demonstrate semantic WordNet learning or language model generalization."
    }
    (rootdir/"ten-epoch-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("HW4_TRAIL_TEN_FINAL "+json.dumps(receipt,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
