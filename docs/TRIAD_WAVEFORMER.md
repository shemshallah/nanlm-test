# NaNLM TriadWaveFormer: one root, staged generalization audit

## Invariants
This branch extends **one** production `UnifiedWaveLanguage` and its **one** mapped
8,388,608-byte `Universe`. All 255 cognitive head registers are simultaneously
available; head 255 retains the network-ingress identity and is excluded from
language computation. The 32,640 canonical mirror pairs remain directional typed
evidence, not a second model or a transformer-weight heap.

The 85 groups contain three dynamically assignable heads each. All are scored
every input step. The 8 highest-scoring groups perform three separate causal
attention lanes (forward, reverse-address-consistency, and multiplicative
composition) with one shared 6x6 Q/K/V weight bank. The 64D global field has
four 16-channel recurrence bands with distinct decay constants. Each step
feeds routed triad responses, wave activations, and the spectral modes into
the **same** field, which then biases the next triad routing round.

Important: the reverse-address attention lane is causal. It queries the
transposed orientation of evidence already stored for prior bytes; it never
has access to future ground-truth tokens.

### Exact physical register accounting

Existing namespace and 8MiB root size are unchanged. The new persistent
weights occupy unused locations in existing head records:

| Component | Location | Bytes | Constraints |
|---|---|---:|---|
| 255 output coefficients | head 0..254 [124..127] | 1020 | Excludes gateway 255 |
| 108 shared Q/K/V floats | heads 0..5,8..109 [105..108] | 432 | Skip 6 and 7, whose rule quaternions can use [96..111] |
| 85 triad adapter floats | heads 2,5,8,..,254 [100..103] | 340 | No head 0 calibration collision |
| Layout schema marker | head 0 byte 123 | 1 | Version D6 |
| **Total additional learned storage** | within existing head records | **1793** | No root expansion |

The Universe owns head bytes [8..11] (observation state), [16..19]
(counter), and [20..23] (accepted trials). Those locations are **not** used
by this kernel; an earlier draft attempted to overlap them and was corrected
before the corpus-stage results shown below. Caches, routed attention weights,
and the 64D evolving field are transient activations, not new persistent
language models. A live `State` now includes current and previous field
workspaces to support exact causal local replay.

The actual resident working set is greater than 8MiB, because transient
state, replay tape, graph caches, quaternion field, and overlay are separate
resources. The 8MiB claim applies only to the physical NaN16 root.

## Gradient and optimization status

**Currently implemented:** exact full-softmax gradients with improving
scratch proposals for the 255 triad output coefficients; proposal/rollback
of shared QKV/triad adapters by local centered-difference probing of the
causal field recomputation; existing native analytical standing-wave T/T*
BPTT; and a finite-difference correction for tape-observed input and
frequency registers when the newly coupled field contributes non-legacy
derivatives. True complete reverse-mode BPTT through shared projections,
routing, triad attention, and multi-timescale field remains **open**.

Sparse top-8 routing is deterministic with score/tie ordering. It can
starve unselected experts; load-balancing and learned semantic tokenization
are not solved. QKV and adapters are now trainable through one-step
accept/reject gradient probing, **not** full multistep backpropagation.
Do not infer that a passed local gradient test proves that the entire
new architecture is differentiable end-to-end.

## Measured corpus expansion: 2026-11-09

Same root throughout, evaluation strictly on a separate 974-byte held-out
fixture `unified_quality_heldout.txt`:

| Stage | Cumulative training bytes | Held-out CE nats/byte | Top-1 accuracy |
|---|---:|---:|---:|
| Initialized | 0 | 5.71983175 | 0.01334702 |
| Small | 420 | 2.99754809 | 0.19199179 |
| Expanded | 4,504 | 2.68615763 | 0.28850103 |

Second stage is the remaining portion of `unified_quality_train.txt`
plus `transfer_train.txt`, processed once on the *same* model/root.
Compared with the small stage, additional data reduced held-out CE by
**0.31139046** and raised accuracy by **0.09650924** (9.65 percentage points).

**Critical ablations after expansion (same stored root):**
`full=2.68615763`; `no_triad=2.67751472`;
`no_global_field=2.68603290`; `no_hyperedge=2.87725622`;
`teacher_only=2.84218975` nats/byte.
Thus **the extra training improved the unified root but the triad
readout did not yet improve the held-out score**; ablating it was better
by 0.00864291 nats/byte. This is a negative architectural finding
worth retaining. Do not claim TriadWaveFormer outperforms a matched
conventional transformer/recurrent control from these numbers.

This was a tiny authored text experiment. To claim corpus scaling or
general language competence, repeat with source-disjoint books/authors,
larger held-out samples, multiple seeds, frozen thresholds, matched
parameter-budget controls, and inspect real free generation for coherence.

## Reproducing

```sh
make nanlm test_triad_waveformer test_adaptive_triads
./test_triad_waveformer
./test_adaptive_triads
# CI: .github/workflows/triad-waveformer.yml
./nanlm /tmp/nanlm-stage eval tests/fixtures/unified_quality_heldout.txt
head -c 420 tests/fixtures/unified_quality_train.txt > /tmp/triad-small.txt
tail -c +421 tests/fixtures/unified_quality_train.txt > /tmp/triad-rest.txt
cat tests/fixtures/transfer_train.txt >> /tmp/triad-rest.txt
./nanlm /tmp/nanlm-stage train /tmp/triad-small.txt 1
./nanlm /tmp/nanlm-stage eval tests/fixtures/unified_quality_heldout.txt
./nanlm /tmp/nanlm-stage train /tmp/triad-rest.txt 1
./nanlm /tmp/nanlm-stage eval tests/fixtures/unified_quality_heldout.txt
./nanlm /tmp/nanlm-stage audit tests/fixtures/unified_quality_heldout.txt
```

Corpora and implementation change over time. The numbers above come
from a specific passing staged CI run **before** local QKV/adapter
optimization was added. Do not mix them with later commits without
recording the revision.
