#!/usr/bin/env python3
"""Five independent 8 MiB mathematical universes; same WordNet event stream.

The outputs are time-sliced C(256,2) mathematical tensor maps, not text,
language predictions, quantum states, or a combined upper universe.

Source: a pinned WordNet data.noun file. The live HW4 NANNET service has not
been established; --data-noun does NOT claim network access. A future HW4
authenticated reader must supply the same validated ordered event contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import zlib

ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW4:0"
SCHEMA = "nanlm.math-specialist-tensor.v1"
PROFILES = ("topology", "geometry", "wave", "information", "triadic")
DIM, ACTIVE, TRIADS = 256, 255, 85
PAIR_COUNT, ROOT_BYTES, HEAD_BYTES, PAIR_BYTES = 32640, 8388608, 128, 256
MAGIC = b"MXS2561\0"
RECORD = struct.Struct("<QIHBbf")  # tick, cell, channel, orientation, reserved, value
CHANNELS = (
    "point", "triad", "hottriad", "global", "mirror_now", "mirror_lag",
    "metric", "metric_aux", "mode_0", "mode_1", "mode_2", "mode_3",
    "curvature", "probability", "entropy", "tensor_product"
)
C = {name: i for i, name in enumerate(CHANNELS)}
WORDNET_DOMAIN = b"nanrouter.wordnet.concept/1\0"
VALID_OFFSET = re.compile(r"^[0-9]{8}$")


def key_head(offset: str) -> int:
    """Match NaNROUTER's WordNet adapter: SHA256 namespaced 32-bit ID, low byte.

    Head 255 is a *coordinate* for source alignment but remains an ingress
    boundary: no mathematical cognitive state is written to its head record.
    """
    key = "synset:" + offset + "-n"
    cid = int.from_bytes(hashlib.sha256(WORDNET_DOMAIN + key.encode()).digest()[:4], "big")
    return cid & 255


def pair_index(a: int, b: int) -> int:
    if not 0 <= a < DIM or not 0 <= b < DIM or a == b:
        raise ValueError("invalid mirror pair")
    if a > b:
        a, b = b, a
    return a * (511 - a) // 2 + b - a - 1


def pair_cell(a: int, b: int) -> int:
    return DIM + pair_index(a, b)


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def relation_code(symbol: str) -> int:
    return hashlib.sha256(b"nanlm.wn.pointer/v1\0" + symbol.encode("ascii")).digest()[0]


def parse_wordnet_nouns(path: Path):
    """Streaming graph events: one vertex and then its outgoing noun pointers.

    No gloss text, lemma strings, or synset identifiers appear in tensor output.
    Event order is deterministic and identical for every mathematical universe.
    """
    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line_number, line in enumerate(f, 1):
            if not line or not line[0].isdigit():
                continue
            if "|" not in line:
                raise ValueError(f"data.noun line {line_number}: gloss delimiter missing")
            left, gloss = line.split("|", 1)
            parts = left.split()
            if len(parts) < 5 or not VALID_OFFSET.fullmatch(parts[0]) or parts[2] != "n":
                raise ValueError(f"data.noun line {line_number}: invalid header")
            try:
                n_words = int(parts[3], 16)
            except ValueError as exc:
                raise ValueError(f"data.noun line {line_number}: invalid word count") from exc
            if not 1 <= n_words <= 1024:
                raise ValueError(f"data.noun line {line_number}: word count out of range")
            ptr_at = 4 + 2 * n_words
            if len(parts) <= ptr_at:
                raise ValueError(f"data.noun line {line_number}: truncated lexemes")
            try:
                n_ptrs = int(parts[ptr_at], 10)
            except ValueError as exc:
                raise ValueError(f"data.noun line {line_number}: invalid pointer count") from exc
            if not 0 <= n_ptrs <= 10000 or len(parts) < ptr_at + 1 + n_ptrs * 4:
                raise ValueError(f"data.noun line {line_number}: truncated pointers")
            a = key_head(parts[0])
            # Bounded lexical metadata is a numeric stimulus, not gloss storage.
            lexical = math.log1p(min(len(gloss.strip()), 16384)) / 12.
            yield ("vertex", a, a, 0, lexical)
            start = ptr_at + 1
            for j in range(n_ptrs):
                symbol, dst, pos, _ = parts[start + 4*j:start + 4*j + 4]
                if pos != "n" or not VALID_OFFSET.fullmatch(dst):
                    continue
                b = key_head(dst)
                polarity = -1. if symbol == "!" else 1.
                yield ("relation", a, b, relation_code(symbol), polarity)


class TensorWriter:
    """Compressed, timestamped sparse C(256,2) readout.

    Cell 0..255 denotes a head; cell 256..32895 denotes unordered mirror
    pair_index. The orientation distinguishes current/lag, not extra cells.
    All records are numeric. Each profile has its own independent stream.
    """
    def __init__(self, path: Path, profile: str, digest: str, every: int):
        self.path = path
        self.f = path.open("xb")
        h = {"schema": SCHEMA, "source": "local-wordnet-data.noun",
             "source_sha256": digest, "comparison_reference": ROUTE,
             "profile": profile, "record_struct": "<QIHBbf",
             "channels": list(CHANNELS), "sample_every_ticks": every,
             "active_heads": ACTIVE, "triads": TRIADS, "ingress_head": 255,
             "mirror_semantics": "forward_current_vs_same-cell_previous-tick",
             "coordinate_hash": "nanrouter.wordnet.concept/1 SHA256 u32 low byte",
             "input_alignment": "same full ordered stream for every profile",
             "not_live_hw4": True}
        hdr = json.dumps(h, sort_keys=True, separators=(",", ":")).encode()
        self.f.write(MAGIC + struct.pack("<I", len(hdr)) + hdr)
        self.z = zlib.compressobj(6)
        self.count = 0
        self.nonzero = 0
        self.values = []
        self.distinct = set()
        self.minval = float("inf")
        self.maxval = -float("inf")

    def emit(self, tick: int, cell: int, name: str, value: float, orientation: int = 0):
        if (not 0 <= cell < DIM + PAIR_COUNT or name not in C or
            orientation not in (-1, 0, 1) or not math.isfinite(value)):
            raise ValueError("invalid mathematical tensor entry")
        x = float(max(-1e5, min(1e5, value)))
        self.f.write(self.z.compress(RECORD.pack(tick, cell, C[name], orientation, 0, x)))
        self.count += 1
        self.nonzero += abs(x) > 1e-8
        self.distinct.add(cell)
        self.minval = min(self.minval, x)
        self.maxval = max(self.maxval, x)
        if len(self.values) < 7 and (abs(x)>1e-8 or len(self.values)==0):
            self.values.append({"tick": tick, "cell": cell, "layer": name, "value": round(x, 7)})

    def finish(self) -> dict:
        self.f.write(self.z.flush())
        self.f.close()
        return {"entries": self.count, "nonzero": self.nonzero,
                "active_cells_sampled": len(self.distinct),
                "value_min": self.minval if self.count else None,
                "value_max": self.maxval if self.count else None,
                "sample_values": self.values, "bytes_compressed": self.path.stat().st_size,
                "file_sha256": file_digest(self.path)}


class Specialist:
    """Distinct update rules over one blank, anonymous 8 MiB root per profile."""
    def __init__(self, profile: str, root: Path, writer: TensorWriter,
                 *, every: int, hotspots: int):
        self.profile, self.writer = profile, writer
        self.every, self.hotspots = every, hotspots
        fd = os.open(root, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, ROOT_BYTES)
            self.mm = mmap.mmap(fd, ROOT_BYTES)
        finally:
            os.close(fd)
        self.x, self.old = [0.] * ACTIVE, [0.] * ACTIVE
        self.counts = [0.] * ACTIVE
        self.trajectories = [0.] * ACTIVE
        self.pairs = set()
        self.clock = 0
        self.collisions = 0
        self.ingress_hits = 0
        self.pair_updates = 0
        self.total = 0

    def get(self, a: int, b: int):
        off = DIM*HEAD_BYTES + PAIR_BYTES*pair_index(a,b)
        now = struct.unpack_from("<f", self.mm, off)[0]
        lag = struct.unpack_from("<f", self.mm, off+128)[0]
        return now, lag

    def set(self, a: int, b: int, v: float):
        off = DIM*HEAD_BYTES + PAIR_BYTES*pair_index(a,b)
        prior, _ = self.get(a,b)
        value = float(max(-1024, min(1024, v)))
        # One canonical pair; reverse NaN register is its previous time slice.
        struct.pack_into("<f", self.mm, off+128, prior)
        struct.pack_into("<f", self.mm, off, value)
        self.pairs.add(pair_index(a,b))
        self.pair_updates += 1
        return value, prior

    def vertex_update(self, h: int, impulse: float):
        if h==255:
            self.ingress_hits += 1
        else:
            self.counts[h] += 1.
            if self.profile == "topology":
                self.x[h] += 0.1
            elif self.profile == "geometry":
                self.x[h] = math.atan2(math.sin(self.x[h]+0.08*impulse),
                                        math.cos(self.x[h]+0.08*impulse))
            elif self.profile == "wave":
                self.x[h] += 0.1*impulse
            elif self.profile == "information":
                self.x[h] = self.counts[h]
            else:
                self.x[h] += 0.07*impulse

    def process(self, event, tick: int):
        kind, a, b, typ, stimulus = event
        self.clock = tick
        self.total += 1
        active_a = a != 255
        active_b = b != 255
        if kind == "vertex":
            self.vertex_update(a, stimulus)
        else:
            if a == b:
                self.collisions += 1
            for h in set((a,b)):
                self.vertex_update(h, 0.05)
            if a != b:
                current, _ = self.get(a,b)
                polarity = stimulus * (1 if a < b else -1)
                self.trajectories[a if active_a else 0] += polarity*0.05
                self.trajectories[b if active_b else 0] -= polarity*0.05
                if self.profile == "topology":
                    value = current + polarity
                elif self.profile == "geometry":
                    angle = 2*math.pi*typ/256
                    value = math.atan2(math.sin(current + 0.10*polarity*angle),
                                       math.cos(current + 0.10*polarity*angle))
                elif self.profile == "wave":
                    value = 0.92*current + 0.16*polarity*(1. + self.x[a if active_a else 0]**2)**-1
                elif self.profile == "information":
                    value = current + abs(stimulus)
                else:
                    wa = self.x[a] if active_a else 0.
                    wb = self.x[b] if active_b else 0.
                    value = 0.96*current + polarity*(1+wa*wb)/8.
                now, lag = self.set(a,b,value)
                self.writer.emit(tick, pair_cell(a,b), "mirror_now", now, 1)
                self.writer.emit(tick, pair_cell(a,b), "mirror_lag", lag, -1)
                if self.profile == "topology":
                    metric = abs(now)-abs(lag)
                elif self.profile == "geometry":
                    metric = 1. - math.cos(now)
                elif self.profile == "wave":
                    metric = (now-lag)**2
                elif self.profile == "information":
                    count_a = self.counts[a] if active_a else 0.
                    count_b = self.counts[b] if active_b else 0.
                    metric = math.log((abs(now)+0.5)*(self.total+1)/
                                      ((count_a+0.5)*(count_b+0.5)))
                else:
                    metric = now*(self.x[a] if active_a else 0.)*(self.x[b] if active_b else 0.)
                self.writer.emit(tick, pair_cell(a,b), "metric", metric,
                                 1 if a < b else -1)
                self.writer.emit(tick, pair_cell(a,b), "tensor_product",
                                 (self.x[a] if active_a else 0.) *
                                 (self.x[b] if active_b else 0.) * now)
        if self.profile == "wave":
            # Causal second-order standing-wave mechanics with triad coupling;
            # all 255 heads update each WordNet event tick.
            nxt = [0.]*ACTIVE
            for i in range(ACTIVE):
                g = i//3
                s = 3*g
                local = (self.x[s]+self.x[s+1]+self.x[s+2])/3.
                injection = 0.20 if (kind=="relation" and i in (a,b)) else 0.
                nxt[i] = max(-6., min(6., 1.48*self.x[i] - 0.64*self.old[i]
                                       + 0.035*(local-self.x[i]) + injection))
            self.old, self.x = self.x, nxt
        # Independent head states are stored only in this specialist's own root.
        changed = set((a,b))
        for i in changed:
            if i != 255:
                struct.pack_into("<f", self.mm, i*HEAD_BYTES+8,
                                 float(max(-1e5, min(1e5, self.x[i]))))
                self.writer.emit(tick, i, "point", self.x[i])
        self.report(tick, changed)

    def report(self, tick: int, changed: set[int]):
        # Report at every input tick; full diagnostic anchors each N ticks.
        groups = [(self.x[3*g]+self.x[3*g+1]+self.x[3*g+2])/3.
                  for g in range(TRIADS)]
        energy = sum(x*x for x in self.x)
        active_groups = {h//3 for h in changed if h<ACTIVE}
        for g in active_groups:
            self.writer.emit(tick, 3*g+2, "triad", groups[g])
        hot = sorted(range(TRIADS), key=lambda g:(-abs(groups[g]),g))[:self.hotspots]
        for g in hot:
            self.writer.emit(tick, 3*g+2, "hottriad", groups[g])
        field = sum(self.x)/ACTIVE
        self.writer.emit(tick, 255, "global", field)
        if self.profile == "geometry":
            # Orientation-independent, gauge-like local triangle curvature.
            for g in hot:
                aa,bb,cc=3*g,3*g+1,3*g+2
                phases = [self.get(aa,bb)[0], self.get(bb,cc)[0],
                          -self.get(aa,cc)[0]]
                curvature = math.atan2(math.sin(sum(phases)), math.cos(sum(phases)))
                self.writer.emit(tick, pair_cell(aa,cc), "curvature", curvature)
        elif self.profile == "information":
            total = sum(self.counts)
            if total:
                entropy = -sum((c/total)*math.log(c/total) for c in self.counts if c)
                self.writer.emit(tick, 255, "entropy", entropy)
                for g in hot:
                    p = sum(self.counts[g*3:g*3+3])/total
                    self.writer.emit(tick, 3*g+2, "probability", p)
        elif self.profile == "wave":
            for k in range(4):
                real = sum(x*math.cos(2*math.pi*(k+1)*i/ACTIVE) for i,x in enumerate(self.x))
                imag = sum(x*math.sin(2*math.pi*(k+1)*i/ACTIVE) for i,x in enumerate(self.x))
                self.writer.emit(tick, 255, "mode_"+str(k), math.hypot(real,imag)/ACTIVE)
            self.writer.emit(tick, 255, "metric_aux", energy/ACTIVE)
        elif self.profile == "triadic":
            for g in hot:
                aa=3*g
                product=self.x[aa]*self.x[aa+1]*self.x[aa+2]
                self.writer.emit(tick, pair_cell(aa,aa+2), "tensor_product", product)
            self.writer.emit(tick, 255, "metric_aux", energy/ACTIVE)
        else:
            self.writer.emit(tick, 255, "metric_aux", float(len(self.pairs)))
        if tick%self.every == 0:
            for h in range(ACTIVE):
                self.writer.emit(tick, h, "point", self.x[h])
            for g in range(TRIADS):
                self.writer.emit(tick, 3*g+2, "triad", groups[g])

    def finish(self) -> dict:
        self.mm.flush()
        self.mm.close()
        return {"profile": self.profile, "events": self.total,
                "unique_mirror_pairs": len(self.pairs),
                "mirror_updates": self.pair_updates,
                "hash_collision_events": self.collisions,
                "ingress_coordinate_events": self.ingress_hits,
                **self.writer.finish()}


def run(data_noun: Path, directory: Path, *, epochs: int = 1,
        every: int = 64, hotspots: int = 4, profiles=PROFILES,
        retain_roots: bool = False) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("choose a new empty output directory")
    if not 1 <= epochs <= 10 or not 1 <= every <= 65535 or not 1 <= hotspots <= TRIADS:
        raise ValueError("invalid experiment settings")
    if len(set(profiles)) != len(profiles) or not profiles or any(p not in PROFILES for p in profiles):
        raise ValueError("unknown/duplicate specialist")
    digest = file_digest(data_noun)
    directory.mkdir(parents=True, exist_ok=True)
    # Root files are temporary model state, never published tensor artifacts.
    temp = tempfile.TemporaryDirectory(prefix="nanlm-specialist-",dir=directory)
    instances = []
    try:
        for p in profiles:
            writer = TensorWriter(directory/(p+".c256tensor"),p,digest,every)
            instances.append(Specialist(p, Path(temp.name)/(p+".root"),
                              writer, every=every,hotspots=hotspots))
        tick = 0
        for epoch in range(epochs):
            for event in parse_wordnet_nouns(data_noun):
                tick += 1
                for inst in instances:
                    inst.process(event,tick)
        if not tick:
            raise ValueError("WordNet data.noun has no noun synset events")
        measurements = [inst.finish() for inst in instances]
        if retain_roots:
            for p in profiles:
                shutil.copyfile(Path(temp.name)/(p+".root"), directory/(p+".root"))
        result = {"schema": SCHEMA, "source": "local-wordnet-data.noun",
                  "source_sha256": digest, "reference_route": ROUTE,
                  "live_wordnet_matrix_connected": False,
                  "root_per_specialist_bytes": ROOT_BYTES,
                  "distinct_universes": len(instances),
                  "total_input_events_each": tick,
                  "epochs": epochs, "anchor_interval": every,
                  "hotspot_count": hotspots, "record_bytes": RECORD.size,
                  "sensor_only": True, "upper_universe_built": False,
                  "profiles": measurements}
        (directory/"summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        return result
    finally:
        temp.cleanup()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-noun",type=Path,required=True,
                    help="pinned WordNet data.noun; never mislabel as live HW4 access")
    ap.add_argument("--out",type=Path,required=True,help="fresh empty output directory")
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--snapshot-every",type=int,default=64)
    ap.add_argument("--hotspots",type=int,default=4)
    ap.add_argument("--profiles",nargs="+",choices=PROFILES,default=PROFILES)
    ap.add_argument("--retain-roots",action="store_true",help="local inspection only")
    args=ap.parse_args()
    try:
        print(json.dumps(run(args.data_noun,args.out,epochs=args.epochs,
                 every=args.snapshot_every,hotspots=args.hotspots,
                 profiles=tuple(args.profiles),retain_roots=args.retain_roots),
                 indent=2,sort_keys=True))
    except Exception as exc:
        print(f"specialist experiment failed: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
