#!/usr/bin/env python3
"""Matched counterfactual BPTT training audit, not multiple production LMs.

Two disposable scratch roots receive the same text and initialization;
one disables ONLY truncated recurrent gradient updates. Both use the
identical shared fabric, teacher, triad routing and readout features.
"""
from __future__ import annotations
import json
import math
import re
import tempfile
from pathlib import Path
from transfer_benchmark import (ROOT, TRAIN, DEV, TESTS, run, audit,
                                generation_metrics, verify_distinct)

def train_control(base: Path, use_bptt: bool) -> dict:
    root = base / ("bptt_on" if use_bptt else "bptt_off")
    before = {name: audit(root, fixture)["full"] for name, fixture in TESTS.items()}
    log = run(root, "train", str(TRAIN), "2", "1" if use_bptt else "0")
    match = re.search(r"bptt_enabled=(\d+) bptt_attempted=(\d+) bptt_accepted=(\d+)", log)
    assert match is not None, log
    enabled, attempts, accepted = map(int, match.groups())
    assert enabled == int(use_bptt)
    assert 0 <= accepted <= attempts
    if use_bptt:
        assert attempts > 0 and accepted > 0, "BPTT failed to exercise learning"
    else:
        assert attempts == accepted == 0, "disabled BPTT was still updating"
    assert root.stat().st_size == 8388608
    uncalibrated = {name: audit(root, fixture)["full"]
                    for name, fixture in TESTS.items()}
    calibration = run(root,"calibrate",str(DEV))
    after = {name: audit(root, fixture)["full"] for name, fixture in TESTS.items()}
    generation = run(root, "generate", "A distant ", "120", "29")
    metrics = generation_metrics(generation, TRAIN.read_bytes())
    count=sum(x["tokens"] for x in after.values())
    assert count > 2000
    def weighted(samples: dict) -> float:
        return sum(x["ce"]*x["tokens"] for x in samples.values())/count
    result = {
        "enabled": use_bptt, "attempts": attempts, "accepted": accepted,
        "before": before, "uncalibrated": uncalibrated, "after": after,
        "weighted_uncalibrated": weighted(uncalibrated),
        "weighted_calibrated": weighted(after),
        "calibration": calibration.strip(), "generation": metrics,
    }
    assert math.isfinite(result["weighted_calibrated"])
    return result

def main() -> None:
    sources = verify_distinct()
    with tempfile.TemporaryDirectory(prefix="nanlm-bptt-control-") as temp:
        base=Path(temp)
        off=train_control(base,False)
        on=train_control(base,True)
    report={"corpus":sources,"bptt_off":off,"bptt_on":on,
            "delta_nats":on["weighted_calibrated"]-off["weighted_calibrated"],
            "scope":"reproducible matched counterfactual roots; deployed runtime remains one model"}
    (ROOT/"bptt-control-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"BPTT_CONTROL off_ce={off['weighted_calibrated']:.8f} "
          f"on_ce={on['weighted_calibrated']:.8f} "
          f"delta_nats={report['delta_nats']:+.8f} "
          f"accepted={on['accepted']} attempted={on['attempts']}")
    for name in TESTS:
        print(f"BPTT_DOMAIN name={name} off_ce={off['after'][name]['ce']:.8f} "
              f"on_ce={on['after'][name]['ce']:.8f}")
    print("BPTT_GENERATION_OFF sample="+repr(off["generation"]["continuation"][:100]))
    print("BPTT_GENERATION_ON sample="+repr(on["generation"]["continuation"][:100]))
if __name__=="__main__":
    main()
