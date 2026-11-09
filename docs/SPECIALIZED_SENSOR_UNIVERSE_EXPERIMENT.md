# Specialized mathematical sensor universes — experiment protocol

**Status:** design/protocol, not trained results. **No upper universe or overmind implementation.**

## Goal

Run *independent*, initially blank 8,388,608-byte C(256,2) universes on one
identical ordered WordNet corpus stream. Each universe specializes in a distinct
mathematical observable, not in reconstructing WordNet words. Export only
timestamped matrix/tensor measurements. Preserve HW4:0 WordNet as an independent
frozen reference for *later* alignment/evaluation, never as a training label
or feature in these experiments.

## Controlled inputs

- Freeze exact corpus snapshot ID, source digest, license/provenance, record
  ordering, split assignment, and sequence IDs before any training.
- Use the same 64-bit nanclock tick and sub-tick sample sequence for every
  sensor. Report dropped, reordered, and duplicate frames.
- A 32-bit NaN16DE payload is **not** by itself a complete WordNet corpus.
  Require a verified source-to-corpus manifest and authenticated HW4 serving
  contract before calling an experiment a real WordNet training run.
- Freeze per-universe hyperparameters, initialization seed, total update
  count, and compute budget. Fit all normalization on training split only.
- Geometric credit assignment, **not BPTT**, is the baseline update mechanism.
- Every universe starts from its own blank 8 MiB root; no shared learned
  parameters or cross-universe feedback during this phase.
- Do not alter HW4:0 or the production BS0 runner-ring root.

## Geometry

256 addressable nodes; 255 computational heads and 1 reserved network port.
85 non-overlapping triads of 3 computational heads. 32,640 unordered mirror
pairs, each with two 128-byte directional lanes. The 256 diagonal/fixed-point
head cells complete 32,896 canonical cells. Mirror lanes are *orientations*;
they are not independently addressable extra pairs. A temporal slice is
a (nanclock, sequence, resolution, channel, pair/head coordinate, value)
observation, not a second 8 MiB root.

## Specialization matrix

| Universe | Heads | Triads | Hot triads | Global | Wave/spectral | Output |
|---|---|---|---|---|---|---|
| V | directional projections | local inner products | high-energy alignment | centroid/dispersion | rotating vector modes | vector field, Gram |
| T | local rank-1 factors | cross-head outer products | sparse interactions | contraction | tensor mode spectra | rank-3/4 sparse factors |
| D | finite differences | local lag correlations | changepoints | drift | autocorrelation | transition/lag matrix |
| S | phase/amplitude | triad oscillators | resonant groups | coherent mode | FFT/standing-wave energy | complex mode-coupling matrix |
| G | local orientation | triangle holonomy | high-curvature groups | reference frame | Laplacian modes | metric/connection/curvature |
| P | incidence/boundary | simplicial triples | persistent cycles | connected components | Hodge modes | sparse incidence/boundary |
| I | distribution moments | cross-covariance | anomalous correlations | entropy | cross-spectrum | covariance/information |

These are **proposed measurement operators**. In particular, Born-like
normalization is only a probability analogy and must not be labeled a
physical quantum measurement.

## Output contract

A common export envelope carries corpus digest, universe type, schema version,
training step, nanclock tick, sub-tick index, channel, tensor rank, dimensions,
sparse indices, dtype, quantization, value(s), checksum, and provenance.
Actual values are tensor-only: no raw text, glosses, lemmas, or synset labels.
A deterministic decoder must round-trip the exported binary format.

Store a full baseline at a coarse interval, with sparse deltas at high
frequency; also sample high-frequency hotspot transitions. Do not write all
32,640 pairs every byte unless explicitly benchmarking the upper bound.

## Measurement gates before any combination

1. Correctness: pair-index bijection, directional mirror lanes, timestamps,
   deterministic replay, numerical stability, and recoverable binary export.
2. Throughput: frames/sec, emitted elements/frame, bytes/frame, compression
   ratio, wall-clock training cost, and root/scratch/peak RSS.
3. Structure: active-pair occupancy, effective rank, singular spectrum,
   temporal autocorrelation, redundancy and mutual information estimates.
4. Quality: frozen held-out prediction/retrieval probes versus random,
   untrained, shuffled-corpus, and matched-parameter conventional baselines.
5. Independence: cross-universe canonical correlation and complementary
   information, computed *only after* individual runs are frozen.
6. HW4 reference: independently validate snapshot SHA-256, graph fact count,
   mapping version, and generation; compare aligned held-out relational
   structures without leaking WordNet labels into sensor training.

A reference is not 'persistent' merely because an experiment once used
an mlocked 8 MiB root or a snapshot restored locally. Require durable
snapshot plus hyperedge arena, provenance, checksum, restart recovery,
versioned pointer, and live HW4:0 service proof.

## Scope boundary

No overmind, decoder, cross-universe fusion, or upper-universe code until
the independent exported tensor streams have been measured and audited.
The existing temporal C256 sampler is a **synthetic codec/sensor smoke test**,
not a WordNet-trained specialized universe.
