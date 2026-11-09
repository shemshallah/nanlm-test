#!/usr/bin/env python3
"""Cross-profile statistical analysis of five WordNet C256 tensor recordings.

This tool operates ONLY on exported mathematical measurements. It does not
make an overmind, language decoder or persistent training update.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct
import sys
import zlib

MAGIC=b"MXS2561\0"
RECORD=struct.Struct("<QIHbBf")
PROFILES=("topology","geometry","wave","information","triadic")


def records(path):
    with Path(path).open("rb") as f:
        if f.read(8)!=MAGIC:
            raise ValueError("not a C256 mathematical specialist tensor")
        header_length=struct.unpack("<I",f.read(4))[0]
        if not 10<=header_length<=65536:
            raise ValueError("invalid tensor metadata length")
        metadata=json.loads(f.read(header_length).decode("utf-8"))
        if metadata.get("schema")!="nanlm.math-specialist-tensor.v1":
            raise ValueError("unknown tensor schema")
        yield metadata
        d=zlib.decompressobj()
        pending=b""
        while chunk:=f.read(1024*1024):
            pending+=d.decompress(chunk)
            length=len(pending)//RECORD.size * RECORD.size
            for offset in range(0,length,RECORD.size):
                yield RECORD.unpack_from(pending,offset)
            pending=pending[length:]
        pending+=d.flush()
        if pending or not d.eof or d.unused_data:
            raise ValueError("truncated or extra tensor data")


class Stats:
    def __init__(self):
        self.n=0
        self.positive=0
        self.nonzero=0
        self.mean=0.
        self.m2=0.
        self.low=float("inf")
        self.high=-float("inf")
        self.deltas=0
        self.last=None
        self.lag_delta=0.

    def push(self, x):
        if not math.isfinite(x):
            raise ValueError("nonfinite tensor")
        self.n+=1
        self.positive+=x>0
        self.nonzero+=abs(x)>1e-8
        shift=x-self.mean
        self.mean+=shift/self.n
        self.m2+=shift*(x-self.mean)
        self.low=min(x,self.low)
        self.high=max(x,self.high)
        if self.last is not None:
            self.lag_delta+=abs(x-self.last)
            self.deltas+=1
        self.last=x

    def export(self):
        return {"entries":self.n,"nonzero":self.nonzero,
                "nonzero_ratio":round(self.nonzero/self.n,6) if self.n else None,
                "mean":round(self.mean,9) if self.n else None,
                "stddev":round(math.sqrt(self.m2/self.n),9) if self.n else None,
                "min":self.low if self.n else None,
                "max":self.high if self.n else None,
                "mean_abs_consecutive_change":round(self.lag_delta/self.deltas,9)
                          if self.deltas else None}


def pearson(a,b):
    if len(a)!=len(b):
        raise ValueError("not time-aligned")
    n=len(a)
    if n<2:return None
    ma=sum(a)/n; mb=sum(b)/n
    va=sum((v-ma)*(v-ma) for v in a)
    vb=sum((v-mb)*(v-mb) for v in b)
    if va<1e-24 or vb<1e-24:return None
    return round(sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb),6)


def analyze(folder):
    folder=Path(folder)
    result={"schema":"nanlm.math-specialist-statistics.v1",
            "source_sha256":None, "profiles":{},
            "global_correlations":{},
            "pair_metric_correlations":{}}
    metric_series={}
    global_series={}
    event_ids={}
    for profile in PROFILES:
        stream=records(folder/(profile+".c256tensor"))
        meta=next(stream)
        if meta.get("profile")!=profile:
            raise ValueError("tensor profile mismatch")
        sha=meta["source_sha256"]
        if result["source_sha256"] is None:result["source_sha256"]=sha
        elif result["source_sha256"]!=sha:
            raise ValueError("profiles are not from the same corpus")
        channels=meta["channels"]
        stats=defaultdict(Stats)
        clock=set()
        metric=[]
        global_field=[]
        metrics_idx=channels.index("metric")
        global_idx=channels.index("global")
        for tick,cell,channel,orientation,reserved,value in stream:
            if not 0<=cell<32896 or not 0<=channel<len(channels) or reserved:
                raise ValueError("invalid tensor coordinate")
            stats[channels[channel]].push(value)
            clock.add(tick)
            if channel==metrics_idx:
                metric.append((tick,cell,value))
            if channel==global_idx:
                global_field.append((tick,value))
        event_ids[profile]=clock
        metric_series[profile]=metric
        global_series[profile]=global_field
        result["profiles"][profile]={
            "ticks":len(clock),
            "pair_metric_samples":len(metric),
            "global_samples":len(global_field),
            "per_channel":{name:state.export() for name,state in stats.items()}
        }
    initial=event_ids[PROFILES[0]]
    for name,clock in event_ids.items():
        if clock!=initial or clock!=set(range(1,len(clock)+1)):
            raise ValueError("specialist clocks are not exactly aligned")
    for i,p in enumerate(PROFILES):
        for q in PROFILES[i+1:]:
            keys_a=[(t,c) for t,c,v in metric_series[p]]
            keys_b=[(t,c) for t,c,v in metric_series[q]]
            if keys_a!=keys_b:
                raise ValueError("pair metrics have incompatible event coordinates")
            if [t for t,v in global_series[p]]!=[t for t,v in global_series[q]]:
                raise ValueError("global fields have incompatible nanclock")
            result["pair_metric_correlations"][p+"__"+q]=pearson(
                [v for _,_,v in metric_series[p]],
                [v for _,_,v in metric_series[q]])
            result["global_correlations"][p+"__"+q]=pearson(
                [v for _,v in global_series[p]],
                [v for _,v in global_series[q]])
    result["ticks_per_profile"]=len(initial)
    result["all_aligned"]=True
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensors",type=Path,required=True)
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    try:
        report=analyze(args.tensors)
        text=json.dumps(report,indent=2,sort_keys=True)+"\n"
        if args.out:
            args.out.write_text(text)
        else:
            print(text)
    except Exception as e:
        print("tensor stats error: "+str(e),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
