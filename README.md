# NaNLM HW0:0 — one native C(256,2) universe

This branch implements one persistent memory-mapped hardware universe.
No second universe is instantiated, and no general language ability is claimed.

## Physical memory and addressing

The mmap Root is exactly 8,388,608 bytes (8 MiB): 256 palindrome heads
at 128 bytes each, plus 32,640 physical mirror pairs at 256 bytes each.
Every pair contains two independent 128-byte directed lanes, 16 support
and 16 refutation float channels in each lane.

An additional 65,536-element 12-byte leaf table (786,432 B) maps every
raw address to its inverse, physical offset, orientation and endpoints.
A 32,640-double persisted phase array takes another 261,120 bytes.
Root + mapping + phase = 9,436,160 B nominal, excluding dynamic event
journals, hyperedges, malloc overhead and process runtime. Residency
is guaranteed only by the *process lifetime* of the mmap; OS page
eviction remains possible because mlock is not claimed.

The exact raw address encoding is (a<<8) | reverse8(b).
Full 16-bit reversal corresponds to the logical swap (a,b)->(b,a).
There are 256 palindromic points, 32,640 mirror pairs, 32,896
canonical physical cells, and 65,536 logical leaves.

The 256 heads are encoded as 4 sign/quiet-NaN class variants,
4 nibble groups and 16 selector indices: 4 x 4 x 16 = 256.
Each stores a 16-bit Boolean selector mask, learning state,
3D source/target vectors and gauge potential. NaN addresses are opaque
integers, never executed as actual floating-point NaNs.

Head 255 is reserved as HW0:0 nannet gateway. Routing to HW4:0 is
a packet-intent stub; there is no transport driver yet.
Raw byte FF is reserved from the current learner (valid UTF-8 excludes it).

## Exact mathematical insight: Hodge cycle decomposition

Consider the complete undirected graph K256. Its oriented edge field
has E = C(256,2) = 32,640 components. There are V=256 head potentials,
only V-1=255 independent gauge differences.

Under vertex gauge g, an oriented connection theta_ab transforms as
theta_ab -> theta_ab + g_a - g_b.

An oriented triangle's Abelian holonomy is
Omega_abc = theta_ab + theta_bc + theta_ca.

The independent cycle-space dimension is E - V + 1 = 32,385.
Therefore, although C(256,3) = 2,763,520 triples exist, their
scalar triangle circulations have only 32,385 independent degrees of
freedom, already representable through the 32,640 pair phases.

The 32,385 triangles (0,a,b), 1<=a<b<=255, form a cycle basis.
Every other triangle is algebraically reconstructed by
Omega_abc = Omega_0ab + Omega_0bc + Omega_0ca.
No dense triple allocation is required for Abelian triangle reasoning.

Important distinction: scalar triangle geometry is determined by
pair connections; *genuinely independent* three-way facts require
a sparse hyperedge overlay or noncommuting matrix transport.
A future SO(3) quaternion edge transport can measure non-Abelian
holonomy Q_ab Q_bc Q_ca, sensitive to the order of rotations.
A physical directed knowledge claim is not automatically the
mathematical inverse of a quaternion transport.

## Current teaching circuit

The native teacher performs a 255-way conditional prediction using
trainable 3D source/target vectors stored inside the existing heads,
trigonometric basis functions, directional pair evidence, and
gauge-covariant pair phases. Proposed gradient updates are computed
in separate scratch arrays. A change is committed only when it
reduces the actual sample softmax cross-entropy. Accepted updates
write the modified heads, the directed PRECEDES/FOLLOWS inverse
relation, and a small covariant phase correction into the one fabric.

This proves online model modification and physical field teaching,
not generalization to unseen prose. Test-set geometric advantage
is still open. Training data are not a sufficient scientific
control; require held-out baselines.

The teacher currently uses one owner thread. A real multi-writer
version needs locking/transactions. The event journal currently
grows per accepted sample; bounded compaction is a prerequisite
for large-scale training. Root and overlay commits are not yet
crash-atomic as a pair, even though overlay file rename is atomic.

## WordNet

Exact 32-bit synset IDs, lemma names, glosses, pointer types and
provenance remain in the exact concept overlay. Physical 16-bit
leaf anchors may collide and must never be treated as unique
synset identity. The pair field holds a lossy projection of WordNet
edges, while the exact concept index remains authoritative.
The first export is an inspectable TSV. Full reload of that exact
index, collision audits and bulk deduplication remain open.

## Reproducibility

Run make clean test. The GitHub workflow builds with g++ and
clang++ and tests all 65,536 addresses, gauge triangles, temporal
contradiction state, persistence, WordNet fixture and the online
head teacher.

Build with make nanlm, then invoke:
- ./nanlm /tmp/hw0 init
- ./nanlm /tmp/hw0 resident

The resident command keeps a single Universe mapped during a
command loop. Supported commands: status, train 23 40,
predict 23, sync, exit.

For sequential text:
- ./nanlm /tmp/hw0 teach-stream corpus.txt
- ./nanlm /tmp/hw0 eval-stream unseen.txt

For WordNet 3.1 noun data:
- ./nanlm /tmp/hw0 import-wordnet /path/to/dict/data.noun

Each GitHub Actions runner is ephemeral and does not maintain its
process between workflow jobs. Persistent learning across jobs
requires restoring the root and overlay as an artifact or other
external persistent storage.

## Research gates before any second universe

1. Reconcile the Round 043 archive and verify Round 044 provenance.
2. Add bounded journals, power-loss recovery and long-term resident tests.
3. Prove held-out WordNet/gloss gains over matched count, low-rank and
   conventional-memory baselines; audit collisions and contradictions.
4. Compare scalar U(1) vs noncommuting SO(3) geometry with controlled
   three-body tasks, temporal generalization and uncertainty calibration.
5. Only then integrate Nannet transport to HW4:0.

## Causal byte-language milestone (experimental branch)

The byte model uses **the same HW0:0 Universe** plus a bounded, checksum-verified
sparse `.byte-lm` context overlay. It is not a second 256-head universe.
The legacy 8,388,608-byte mmap layout and Nannet head 255 remain unchanged.

Build: `make nanlm test_byte_model CXX=g++` (or `CXX=clang++`).

- `./nanlm /tmp/hw0 train-text train.txt 3` learns three passes of causal
  byte statistics, preserving a bounded context up to five prior bytes.
- `./nanlm /tmp/hw0 eval-text independent.txt` computes prequential
  next-byte held-out cross-entropy in nats, bits/byte, accuracy and unseen
  context count, without modifying the model.
- `./nanlm /tmp/hw0 generate "the tiny " 200 11 0.85 0.92` generates
  200 continuation bytes with deterministic seed 11, temperature 0.85 and
  top-p 0.92. The output contains the prompt followed by new bytes.
- Optional `train-text corpus.txt 1 32` enables a slow existing
  geometric teacher update every 32 bytes. The default deliberately
  excludes teacher training to isolate the contextual baseline.

Implementation: each observed symbol increments the root unigram and up
to five suffix-context nodes. Predict interpolates longer suffixes with a
strictly positive shorter-context distribution, with more shrinkage for
rare contexts. Memory is capped at 500,000 sparse nodes; once full, new
contexts fall back to their existing shorter suffixes. This means
**bounded representation**, not bounded compute or bounded complete
storage footprint: the full physical universe plus indexes, journals,
maps and overlay may consume additional memory. A 128 MiB read cap is
enforced for the overlay.

The format has an eight-byte versioned magic, order, token count, sorted
records and an FNV-1a checksum with atomic rename on save. Writes across
the root, temporal journal and this overlay are not jointly transactional.
Do not share one checkpoint between uncoordinated writer processes.

**Research honesty:** the model is a count-based autoregressive reference,
not yet a fluent generative neural model. Simple, repetitive held-out
sentences are not genuinely out-of-domain. Run paragraph- and
document-disjoint tests, a matched order-5 n-gram baseline, a conventional
small recurrent/attention model, and byte-level free-run diversity,
UTF-8 validity and repetition checks before making any claims of language
generalization or of the geometry's utility. The current sampler is
byte-level; UTF-8 validity of generated text is not guaranteed.
