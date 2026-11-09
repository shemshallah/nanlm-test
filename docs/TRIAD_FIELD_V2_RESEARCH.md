# NaNLM: unified sparse triad-field architecture — audit and research directions

**Date:** 2026-11-09. **Status:** architectural proposal, not implemented production code.
Repository revisions: `main` at `14edacd03be7`, TriadWaveFormer feature branch at `ac5b8e6baff2`.
See `tools/audit_reference_architecture.py` and
`.github/workflows/reference-architecture-audit.yml` for reproducible probes.

## Reconciled landscape

| Track | Real code | Positive result | Limitation |
|---|---|---|---|
| Native unified wave | `include/nanlm/unified_wave.hpp` | Causal model, mapped 8 MiB root, T/T*, sparse evidence, 8-step wave BPTT | 3D token vectors, weak expressive attention |
| TriadWaveFormer PR #24 | `include/nanlm/triad_waveformer.hpp` | 85 groups/3 roles, shared QKV, causal 64D field, 8 active groups | Numerically probed QKV/adapter updates, no full cross-field reverse-mode; development often disables triad readout |
| 1 MiB independent heads | `experiments/independent_heads/*` | 16-bit weight bank, quantized moments, local gradient/voting | Separate models, 256th head erroneously participates in full vote; incomplete optimizer rollback; position-free |
| Sparse shared trunk | `experiments/sparse_shared/*` | Single transformer trunk, 255 rank-4 adapters, sparse training gradients | No position coding, no recurrent field, no actual triad-level routing, no cross-author training |

One production architecture should combine the **shared trunk's expressive learned representation**,
the **TriadWaveFormer's causal field and mirror structure**, and **independent-head transactional ideas**
without instantiating hundreds of independent language models.

## Verified reference findings

GitHub Actions on the audit branch, PyTorch CPU, 11/11 existing tests passed.
The additional audit reproduced:

- One-layer independent-head and shared-trunk last-token logits differ by
  ~2–3e-7 when earlier input symbols are permuted but the last input is unchanged.
  Neither reference provides positional encoding. This is a representational
  limitation, not a gradient failure.
- Restoring the model weights with `bank.store` after a rejected Adam update leaves
  **10,821** int8 moment elements changed in the small test. The optimizer
  trajectory is not transactionally restored.
- With `heads=256`, full-field voting sees **86 triad IDs** because physical head
  255 (ingress) is treated as a cognitive member. A 43/85 approval rule
  additionally requires `total == 85`, so **all 85** legitimate triads
  approving still rejects this batch.
- Calling sparse-model inference mutates `usage` (delta 64 in the test).
  Train/telemetry mutation must be separately opt-in.
- If top-8 heads are selected *uniformly* and later required to form 2/3
  triads, the exact expected number of eligible groups is **0.216986711**.
  This is a mismatch between the proposed 2-of-3 policy and a head-level
  top-k router; the current sparse adapter computation does not itself
  enforce this vote rule.

A small CPU **three-step training** throughput probe with d=32, two
transformer layers, 8-byte sequences, batch=1, torch intra-op threads=1:

- sparse top-8: 1,689 token-positions/s
- random top-8: 2,071 token-positions/s
- dense top-255: 167 token-positions/s

These short-step measurements include teacher-forced training and Python
overheads. They are **not** autoregressive single-stream generation speed,
and cannot be compared directly with older benchmarks using other configs.

The full quantized independent-head bank is 507.089 MiB for 255 candidates,
509.077 MiB for 256 physical slots. Including the unchanged mapped 8 MiB root
yields 515.089 or 517.077 MiB **before** activations, code, geometry caches,
model float32 dequantization, or optimizer snapshots. Therefore the
"under 512 MiB" claim applies to the standalone bank, not total system RSS.

## Proposed unified model: SharedTrunk × Triad Operators × Memory Field

```text
UTF-8 bytes (one stream) -> positions, boundary features, byte embeddings
  -> one shared 4-layer/8-layer transformer trunk (not 255 models)
  -> all 255 lightweight mirror-oscillator/head-state updates
  -> route entire 3-head triads: score 85 groups, activate k=4..8
       lane A: directed temporal
       lane B: inverse-path consistency (using existing evidence, never future)
       lane C: nonlinear motif/hyperedge composition
  -> one trainable gated 64/128-dimensional recurrent global field
  -> residual feedback through the shared trunk and heads at next step
  -> one shared byte decoder + 255-class softmax (0xFF is ingress)
```

The head is a **computational specialist**, not its own language model,
and ideally not permanently tied to a byte value. Token/byte identity
is an embedding lookup; head identity is a separate learned routing slot.
The root remains the canonical NaN16 memory and network identity; added
weights reside in a versioned, checksum-validated *sidecar of the same logical
model*, not in a second instance or a copy of the root.

At each causal time t:

```text
z_t  = Trunk(Embed(b_<=t, position, boundary), F_(t-1))
s_g  = Router(z_t, F_(t-1), wave_g, mirror_g, usage_g)
G_t  = top_k_triads(s_g)   [select groups, not isolated heads]
u_g  = TriadLaneA/B/C(z_<=t, inverse edges, sparse hyperedges; shared QKV + LoRA_g)
F_t  = GatedUpdate(F_(t-1), z_t, sum_{g in G_t} softmax(s)_g u_g, spectral_t)
p_(t+1) = softmax(Decoder(z_t + Project(F_t) + WaveReadout + MirrorReadout))
```

No independent decoding/voting distributions. There is **one** normalized
conditional probability distribution and **one** end-to-end next-byte loss.

### Two practical model sizes

Budget estimates count QKV/projection/MLP trunk, 255 rank-4
adapters, router, byte embeddings/readout and an approximately 64D GRU.
They assume fp16 weights + two int8 optimizer moments per parameter.
They exclude KV activations, sparse graph caches, journal, and process RSS.

| Variant | Trunk | Total parameters (approx) | Weights+moments | Root + weights+moments |
|---|---|---:|---:|---:|
| Android first | d=96, 4 layers, MLP=384 | 0.75M | 2.84 MiB | 10.84 MiB |
| Balanced | d=128, 4 layers, MLP=512 | 1.19M | 4.52 MiB | 12.52 MiB |
| Large test | d=256, 8 layers, MLP=1024 | 7.08M | 27.0 MiB | 35.0 MiB |
| Server | d=512, 12 layers, MLP=2048 | 39.3M | 150 MiB | 158 MiB |

These are *projected parameter storage only*, not measured memory/latency;
head-pair memory remains in the existing fixed root. No model-size target
guarantees coherent language at small data scale.

### Orthogonal experimental tracks (all measured against the same heldout)

1. **Representation**: add position-aware attention (RoPE/ALiBi or learned
   positions), reversible byte-span/word-boundary features, a much wider
   shared trunk. Begin with a tiny causal n-gram/small transformer control.
2. **Triad routing**: score 85 groups and activate whole triads. Warm up
   dense, then gradually anneal top-k with exploration/load balancing and
   route entropy. Measure specialization/active group utilization; do not
   rely on independent uniform head top-k.
3. **Differentiation**: implement exact reverse-mode truncated BPTT through
   QKV, triad attention and 64D field using a PyTorch oracle first, then
   native analytical gradients and per-coordinate finite differences.
   Freeze routing decisions during gradient checking.
4. **Long-range memory**: retain 4 timescales, add sparse retrieval of
   learned phrase/motif anchors from the canonical mirror graph. A 64D
   leaky recurrent vector is not an exact 4k-byte memory.
5. **Geometry/hypermatrix**: treat gauge cycles, antisymmetric direction
   and rank-3/4 hyperedges as separately ablatable conditioning terms,
   not as automatically semantic content. Preserve refutation and support
   independently. Confirm that reverse-address signals are not merely
   duplicated evidence by another name.
6. **Training protocol**: atomic commit of *weights AND moments*, 255
   cognitive heads only; CFT/triad votes may approve graph/memory claims,
   while gradient steps should optimize one global likelihood. Idle triads
   should not veto every minibatch. Use one checkpoint version and replay
   journal across distributed workers.
7. **Evaluation**: pinned source-disjoint authors, multiple seeds and
   training-size curves (1k/10k/100k/1M bytes), UTF-8 validity, heldout
   CE/accuracy, real generation, repetition, energy, peak RSS and latency.
   Compare **full**, no-triad, no-global, no-wave, random/dense routing,
   same-budget GRU, and same-budget vanilla transformer.

## Critical acceptance gates before integration

- Fix ingress head 255 / 85-triad cardinality and impossible full vote.
- Validate rollback restores all optimizer moments, weights, scales, and
  global field/graph proposal state; use fault-injection/restart tests.
- Add causal positional signals; verify correct permutation sensitivity.
- Make inference observationally pure with optional telemetry.
- Require each added component to survive development-only calibration with
  **nonzero** learned contribution AND improve unseen-author CE over the
  exact same training budget, not merely lower training loss.
- Treat tests that pass on synthetic 8-token tasks as implementation
  smoke checks, not a publishable language-quality finding.

No integration claim is made by this research document.
