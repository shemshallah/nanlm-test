# Sparse shared deep trunk + 255 head adapters (reference)

Status: **PyTorch reference in NaNLM v1**. This is the shared-trunk design
chosen for the next native port. It is not a NaN16 claim and it is not yet in
NaNLMind v2.

## Why shared trunk

Making all 255 heads deep is prohibitive: 256 independent d=128, 12-layer
heads need ~2.26 GiB of 16-bit weights + 8-bit moments, plus up to 1.5 GiB of
KV at 1024 context. A shared deep trunk plus 255 low-rank adapters costs:

```text
d=512, 12 layers, MLP=2048 trunk: 37.77M params
16-bit weights: 72 MiB
8-bit moments:   72 MiB
KV @1024:        24 MiB
255 rank-4 adapters: a few MiB
```

The heads remain persistent routing/specialization interfaces; the trunk does
the deep computation.

## Sparse activation

Per token:

```text
score_h = router_h(x) + hotspot_h
indices = top_k(score)
active triad  iff >=2 of its 3 heads are selected
field cap     limits total active triads
```

`hotspot_h` is a persistent bias derived from the triad's recent adapter
gradient norm. It decays each step, so hot triads stay active briefly without
permanent hard routing. Inactive adapters are not evaluated and receive no
gradient.

## Measured reference

PyTorch CPU, batch 1, sequence 8:

| configuration | active heads | tokens/s | note |
|---|---:|---:|---|
| d=32, 2 layers, sparse | 8/255 | 38.0 | small model |
| d=32, 2 layers, random top-k | 8/255 | 29.7 | routing control |
| d=32, 2 layers, dense | 255/255 | 7.36 | 5.2x slower than sparse |
| d=64, 4 layers, sparse | 8/255 | 12.9 | reference default |
| d=64, 4 layers, dense | 255/255 | 1.93 | 6.7x slower than sparse |

Sparse is not a free 32x: the shared trunk, attention, router and Python/PyTorch
overhead are fixed costs. Measured speedup is 5–7x on this CPU harness.

## Throughput math for 1000 tokens/s

`4 tok/s` does **not** multiply by 255 heads. Token rate is set by FLOPs,
memory bandwidth, batch and hardware, not by head count alone.

At an effective 10 GFLOP/s:

- a 75 MFLOP/token shared 12-layer trunk tops out around 130 tok/s;
- 1000 tok/s needs <=10 MFLOP/token at 10 GFLOP/s;
- 8 active heads at d=128, 12 layers are ~75 MFLOP/token, so ~130 tok/s;
- 1 active head at that depth is ~9.4 MFLOP/token, close to 1000 tok/s but
  loses the ensemble/specialization;
- batching increases aggregate throughput but not single-stream latency;
- a 100 GFLOP/s target (GPU or a much smaller/quantized trunk) makes 1000
  tok/s plausible at ~100 MFLOP/token.

The realistic first target is **100–200 tok/s** on CPU with a small shared
trunk and top-k=8–16; 1000 tok/s requires a smaller trunk, batching, or
accelerator.

## Files

- `experiments/sparse_shared/model.py` — trunk, 255 adapters, router, hotspot.
- `experiments/sparse_shared/train.py` — dense/sparse/random benchmarks.
- `tests/test_sparse_shared.py` — budget, top-k, inactive-gradient isolation,
  hotspot promotion and fixed-batch loss decrease.

Run:

```sh
python3 -m unittest tests.test_sparse_shared -v
python3 -m experiments.sparse_shared.train --mode sparse --active 8 --steps 2
python3 -m experiments.sparse_shared.train --mode dense --steps 1
```

## Integration into NaNLMind

1. Keep the 8 MiB root as canonical field.
2. Put the shared trunk weights in the quantized sidecar (16-bit weights,
   8-bit moments), demand-paged.
3. Store 255 adapters and router/hotspot state in the extension; hot triads
   set the mapped hypermatrix rows.
4. Port the local CFT/triad/field vote path: 2/3 triad, 43/85 field.
5. Native NaN16 kernel for the trunk/adapters; host reference only as control.
6. Measure throughput, RSS/cache/energy and held-out loss on source-disjoint
   corpora before claiming a win.

## Limits

- The reference is a shallow synthetic next-byte model, not language quality.
- The trunk is shared, so heads are not fully independent models; they are
  persistent specialized interfaces and voters.
- Hotspot routing is not yet end-to-end differentiable; router gradients and
  local credit need separate validation.
- 1000 tok/s is not demonstrated.
