# Mathematical specialist universes — same WordNet corpus, different tensor readings

**Experiment branch only. No upper universe, overmind, decoded language, or model
merger is built here.** No NANNET production state is changed.

## Shared input and addressing

The source is a pinned `data.noun` file (its complete SHA-256 is included in
every output). The WordNet parser produces a fixed, ordered stream:

1. A synset vertex event, with numeric gloss-length stimulus.
2. Its outgoing noun-pointer relation events, retaining relation polarity and
   typed pointer identity numerically.
3. A monotonically increasing **logical nanclock tick for every event**, with
   the same ticks and events presented to all specialists.

This is a WordNet **noun graph experiment**, not a complete WordNet 3.1 corpus,
and does not directly sample unseen gloss token sequences or other POS tables.
Its corpus duration is events × epochs. No network transport is inferred from
the path supplied to the program. Native HW4:0 has a reserved route but **its
live resident WordNet matrix has not been verified**.

The source-to-coordinate projection agrees with NaNROUTER's
`wordnet_mirror_adapter.py`: SHA-256 of
`b"nanrouter.wordnet.concept/1\\0" + b"synset:" + offset + b"-n"`,
first four digest bytes interpreted as big-endian u32, `head = id & 255`.
This maps distinct synsets to the same head and is **not a semantic embedding**.
Reference comparison must retain exact synset identities at HW4 and aggregate
them through precisely this coordinate projection; do not compare to another
repository's FNV-based leaf projection as though the hashes were equivalent.

The 256th coordinate, head 255, is ingress-only: no mathematical specialist
writes state into its 128-byte head record. Source relations touching that
coordinate may still be sampled in the common C(256,2) pair space. The
`ingress_coordinate_events` and `hash_collision_events` counters expose
collisions/ingress projections.

One independent *fresh* 8,388,608-byte mapped universe per specialist:

- 256 × 128-byte palindromic head registers: 32,768 bytes.
- 32,640 × 256-byte mirror-pair records: 8,355,840 bytes.
- No text, glosses, lemma strings, or synset keys are exported.
- Every pair stores its current **canonical low→high** scalar in forward lane
  offset 0 and the **previous value of that same scalar** in the mirror
  lane offset 128. It is a deliberately **experimental temporal-lag use of
  a mirror register**, not the independent inverse-relation semantics of the
  canonical WordNet matrix. The two physically oriented address views remain
  tied to one unordered pair. The mirror lane's lag can be tested exactly.

Each specialist works independently. Sharing a clock, hash rule and input
stream does *not* share weights or learned representations.

## The five mathematical measurement profiles

| Specialist | State updated on each graph event | Distinct measurements / hypothesis |
|---|---|---|
| **Topology** | Signed orientation-weighted pair evidence and head incidence | Local degree/flow, change in edge magnitude, triad/field activity, occupied-pair count |
| **Geometry** | Wrapped U(1)-like head phase and typed edge connection | Angular displacement `1-cos(theta)`, triad holonomy `wrap(phi_ab + phi_bc - phi_ac)` |
| **Wave** | Causal second-order 255-head oscillator, triad coupling and directed pair excitation | Per-tick wave amplitude, four graph-index Fourier harmonics, oscillator energy and mirror-lagged pair energy |
| **Information** | Observed head frequencies and pair occurrence mass | Smoothed log co-occurrence ratio, Shannon entropy over head counts, triad probability mass |
| **Triadic** | Shared head/edge activations with multiplicative local interactions | `head_a*head_b*pair_ab`, `head_3g*head_3g+1*head_3g+2`, local hotspots and global energy |

All five report point, triad, activated triad (top-k by absolute magnitude),
and global-field readouts through one consistent tensor ABI.
Specialized channels differ; zero and negative values are preserved.

The geometric **holonomy** is a U(1) analogy; Fourier modes are mathematical
oscillation measurements; information probability is a normalized frequency
estimate. They are **not quantum mechanics measurements**, and `entropy`
is Shannon entropy, not evidence of quantum decoherence. The goal is to test
whether specialist readouts carry distinct useful statistical structure.

## Tensor export contract

Each specialist writes **one .c256tensor file**. Header:
`b"MXS2561\\0"`, little-endian u32 header byte length, UTF-8 JSON
(metadata, profile, corpus hash, channel vocabulary, and mapping version).
The remaining bytes are one streaming zlib payload of little-endian records:

```
<QIHbBf
uint64  tick         corpus event clock, preserving high temporal resolution
uint32  cell         0..255=head; 256..32895=canonical pair
uint16  channel      named specialist measurement
int8    orientation  0 non-directed; +1 current; -1 mirror lag
uint8   reserved     zero
float32 value        finite signed measurement
```

**20 bytes per raw record**, compressed without source text. Every event produces
measurements; `--snapshot-every 64` additionally exports all 255 active heads
and 85 triad values as diagnostic anchors each 64 events. This is a sparse,
time-indexed C(256,2) measurement **mapping**, not a 32,896-cell dense tensor
materialized for every tick and not lossless history of every head between
anchors. The sparse records remain in the same 32,896-cell coordinate basis.

Each profile has its own new blank 8 MiB file during training. Roots are
discarded at completion by default, leaving just the five compressed tensor
maps and `summary.json`. Use `--retain-roots` only for local inspection,
never upload retained 8 MiB roots as tensor artifacts.

The summary reports event count, distinct matrix coordinates, mirror-pair
updates, nonzero entries, min/max and actual compressed bytes. These values
are measurements of the **sensor representation**, not compression quality
or language-model accuracy. A later overmind decoder must beat matched
baselines on source-disjoint gloss and relation tasks before it can claim to
recover concepts.

## Reproduce

```sh
python3 tools/matrix_probe_bank.py \\
  --data-noun tests/fixtures/data.noun \\
  --out /tmp/math-probes \\
  --epochs 2 --snapshot-every 32 --hotspots 4
python3 -m unittest discover -s tests -p 'test_matrix_probe_bank.py' -v
```

For a larger test supply a **verified and licensed** WordNet `data.noun` and
record its SHA-256. The same stream and identical sample order reach all five
specialists. The official full WordNet matrix state at HW4:0 should become a
**frozen, read-only, generation- and digest-pinned reference** once an actual
HW4 snapshot and exact synset index are found or installed. Do not assume
that reservation of `HW4:0` constitutes a persisted WordNet matrix.

## Evaluation, before combining any universes

Measure, per specialist and tick resolution:

1. Output count/byte rate, actual sparsity and matrix occupancy.
2. Autocorrelation across lag 1, 2, 4, 8, 16, 32; temporal information
   beyond the current graph event and beyond the mirror-lag baseline.
3. Pair/triad/global readout redundancy, variance and diversity.
4. Source-disjoint prediction of typed relations or synset gloss classes
   using a **separately evaluated lightweight probe** and matched hashed-
   identity, count-feature, and random-projection controls.
5. Before any comparison to HW4, reconcile reference WordNet version,
   exact synset IDs, hash projections, occupied pairs, corpus digest, and
   snapshot generation. Hash collision identities must not be assumed equal.

**Open gates:** a live authenticated HW4 WordNet stream; confirmed persisted
HW4 state; real full-corpus experiments; blind tests of concept decodability.
The fixture CI only proves shape, arithmetic execution, deterministic sharing
of input, and mirror-timing properties.
