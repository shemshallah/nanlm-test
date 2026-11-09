# NaNLM HW0:0 — one palindromic mirror-wave language system

**Architecture status: experimental, native, unified.**
This branch has one production text predictor and generator: `nanlm`.
It operates over the *same* 8,388,608-byte memory-mapped NaN16 root,
whose 255 cognitive palindrome heads interact through the entire
addressable 32,640-pair mirror fabric. Head 255 is the reserved gateway.
There are **not** 255 separate language models, instances, or roots.

The former n-gram, flat geometric and RNN32 systems are preserved only
as diagnostic control experiments in the separately named
`nanlm-diagnostics` binary. They are not loaded into or composed by the
production model.

## Run the single unified model

```sh
make nanlm CXX=clang++
./nanlm /tmp/hw0 init
./nanlm /tmp/hw0 train corpus.txt 2
./nanlm /tmp/hw0 spectrum
./nanlm /tmp/hw0 eval unseen.txt
./nanlm /tmp/hw0 audit unseen.txt
./nanlm /tmp/hw0 calibrate development.txt  # do not calibrate on the held-out test file
./nanlm /tmp/hw0 generate 'the future of ' 120 11 0.85 0.92
```

For baseline-only research:

```sh
make nanlm-diagnostics CXX=clang++
make test CXX=clang++
make validate CXX=clang++
```

## The 255-head computational substrate

Each cognitive head retains an existing trainable 3D source and target
vector, Boolean gate mask, temporal evidence and metadata. Four
additional head-local readout gains occupy previously unused bytes
**80–95** of its existing 128-byte palindromic head register, with the
format marker at byte 60. These weights gate four simultaneous experts:

- **Standing wave:** damped second-order oscillator at all 255 heads,
  coupling temporal forward and inverse wavefronts through opposite
  directed mirror lanes.
- **Head-local attention:** every candidate head is an attention query
  that softmax-attends over the past 16 observed bytes using trainable
  head vectors, trigonometric phases and temporal offsets.
- **Spectral graph modes:** an orthogonal four-vector power-iteration
  approximation to the real symmetric mirror-coupling operator, with
  eigenvalues and residuals reported for auditing. Local oscillator
  frequencies are modulated by its mode loadings.
- **Gauge/cycle geometry:** U(1) triangle holonomy and composed,
  order-sensitive SO(3) quaternion transport. Typed sparse hyperedge
  evidence contributes to the shared prediction.

The 16 typed relational channels per directed lane retain independent
support and refutation, along with inverse relation mapping.
Unrecognized, supported, refuted and contradictory states remain distinct.
This logical evidence is not interchangeable with token probability.

### Explicit temporal operators

For a directed transition matrix T obtained from mirror evidence,
both T and its transpose T* participate in causal, forward-time state
updates. The "inverse" is an adjoint relational view, **not a
retrocausal read of future test tokens**. The symmetric component of T
defines the current spectral graph. Its antisymmetric/directed component
affects the difference between the two wavefronts.

The ordinary word "transformer" here refers to **head-local attention
mechanisms**; this is not yet equivalent to 255 complete Transformer
blocks with full Q/K/V/MLP layers and end-to-end BPTT. That remains a
specific implementation goal, rather than a claimed capability.

## Learning, inference and limitations

One output distribution over 255 ordinary byte symbols is normalized by
a single softmax. At each step it uses all 255 attention readouts, the
current forward/inverse oscillator state, a small number of spectral
modes, native pair evidence, cycles and sparse hyperedges.

The existing FabricTeacher updates 3D head vectors, pair evidence,
phases and SO(3) transports. A second stage updates the four expert
gains **in the same 255 palindrome registers**, using scratch-accepted
next-byte cross-entropy. There are no standalone production weight
files for a separate byte model, RNN or hybrid.

Spectral eigenvectors are currently recomputed, not optimized by
backpropagating all the way through the eigensolver. The wave state is
a bounded ephemeral activation, not an additional persisted universe.
The full physical root stays at exactly 8 MiB; runtime adjacency,
wavefronts, address mapping, and existing phase/journal/quaternion
overlays consume extra memory. Checkpoint writes across these files
are **not yet jointly atomic**.

Generation feeds each chosen byte through the full 255-head update and
uses strict UTF-8 prefix-constrained sampling. This guarantees a
well-formed UTF-8 continuation for a valid prefix and requested length,
**not** grammatical, semantically correct or open-domain fluent text.
Raw symbol 255 remains reserved and is never sampled.

### Read-only contribution audit

`nanlm ROOT audit unseen.txt` replays the **same held-out file** eight times
against the **same mapped universe**. It reports `ce_nats` and top-1 accuracy
for the full predictor, each of its six removed feature families, and a
`teacher_only` readout with all six disabled. The teacher-only control is
**not** a separately trained language model: it is the existing native
head/pair teacher projection from that exact same root and checkpoint.
No audit pass updates trainable state. A change in cross-entropy is evidence
of a feature's contribution on this file, **not** proof of generalization.

A separate `calibrate development.txt` command can search four Q8.8
shared gain scales inside unused bytes 64–71 of palindrome head 0,
without overwriting its gauge potential at bytes 72–79. It minimizes
development-only CE and freezes the result for later test evaluation;
calibration does not instantiate another model or another root.

### Research and correctness gates

`tests/test_unified_wave.cpp` checks one root, 255 head excitation,
inverse temporal pairs, finite spectral modes, causal head attention,
probability normalization, native persistence, and deterministic
Unicode-safe generation. Two compilers are configured in
`.github/workflows/unified-wave.yml`.

We have **not** demonstrated that the unified architecture beats a
competitive count, recurrent or attention baseline on larger independent
corpora. Such comparisons must remain evaluation-only, with separate
train/development/test documents, multiple seeds and controlled model
sizes. No free-running fluency or AGI claim is warranted.

For the precise training/architecture roadmap, see
`docs/UNIFIED_WAVE_ARCHITECTURE.md`.

## Addressing invariants

- 256 palindrome fixed points: 255 cognitive, 1 gateway.
- 32,640 physical mirror pairs, each containing two directed lanes.
- 32,896 total canonical physical cells.
- 65,536 logical NaN16 leaves; mirror reversal is exact transposition.
- 8,388,608-byte root; four NaN16 sign/quiet classes are represented
  as opaque integer encodings, never executed as floating-point NaNs.
- Exactly one logical universe even when multiple machines later
  compute separate work on leased, synchronized head partitions.
