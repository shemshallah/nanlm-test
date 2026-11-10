# HW4 FastTRAIL: native hybrid experiment

Status: new native C++ training executable, tools/hw4_fasttrail_native.cpp,
sharing the original NaNLM 8,388,608-byte student and exact FabricTeacher
rank-three head vectors. BPTT remains disabled. This is a draft experiment.

Build: make nanlm hw4-onepass hw4-trail hw4-fasttrail CXX=clang++

Run: ./hw4-fasttrail STUDENT_ROOT CORPUS EXPECTED_BYTES [WINDOW_STRIDE=64] [BLEND=0.25]

## Combined algorithm

1. On a frozen student, precompute both 255-way forward and inverse
   conditional probability tables, using the original two-rail TRAIL method.
2. Scan every byte of the complete immutable corpus and accumulate the
   255 by 255 adjacent-byte transition histogram.
3. On every WINDOW_STRIDE-th contiguous 2048-byte window, advance the two
   eligibility rails with gamma=0.93 and lambda=0.82. Reset eligibility at
   window boundaries, but accumulate credit across selected windows.
4. Calculate the original fast native aggregate CE surrogate gradient
   in rank-three head-vector coordinates. This gradient differentiates the
   4*dot(source,target) score term only, not all nonlinear teacher terms.
5. Convert the signed forward and inverse credit to a temporal direction,
   normalize it separately, and remove any component opposing the fast
   direction in the linearized CE surrogate.
6. Backtracking evaluates both fast-only and blended directions against
   the SAME full-corpus byte-pair CE, from the SAME starting 8 MiB root.
   Select the lower-loss accepted candidate; otherwise preserve the root.
   The entry rail_selected=1 only if blended credit actually wins.

This is one logical model. The 2 copies in the A/B experiment are only
control/treatment replicas from an identical SHA-256 checkpoint. The
executable itself operates on a single persistent root, with head 255
reserved for ingress and no modifications of the HW4 reference.

## Performance and interpretation

Default stride 64 samples approximately 1/64 of available 2048-byte
intra-window transitions for expensive 255-head temporal computation.
All corpus transitions still enter the fast histogram and final CE gate.

The credit estimate is a DETERMINISTIC, TRUNCATED SAMPLE and is not
mathematically equivalent to full uninterrupted TRAIL. Window resets
alter temporal eligibility. The initial test does not train the wave,
recurrent memory, triad kernel or semantic-role decoder. It also does
not optimize an independent temporal holdout objective. Therefore a
lower byte-pair CE is not proof of better fluent generation, semantic
reasoning or language understanding.

The existing 575x speed comparison is BETWEEN DIFFERENT older
algorithms and is not a measured hybrid speedup.

## Exact reproducibility

Workflow: .github/workflows/hw4-fasttrail-hybrid.yml

- Compile with strict C++20 warnings and smoke test 65,536 synthetic bytes.
- Restore the 103-epoch student from GitHub run 42741208752, root SHA-256:
  e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1
- Verify the immutable HW4 archive and full 58,341,529-byte corpus hashes.
- Clone that checkpoint into a fast-only control and sampled FastTRAIL
  treatment (same initialization, different update rule).
- Compare actual train CE, prose-fixture CE and top-1, timings, roots,
  sample fraction, cosine alignment and selected update.
- Preserve both final comparison roots, logs, receipts and checksums;
  exclude the copyrighted WordNet corpus.

The ablation suite now includes a TRUE fast-only same-optimizer control
(blend=0; no TRAIL transition updates are computed) and independent
0.05 / 0.10 / 0.25 / 0.50 temporal-blend treatments, all starting from
the same SHA-256 root. This separates line-search improvements from actual
temporal-credit gains. Only rail_selected=1 establishes that the mixed
candidate influenced a persisted update.

The small 974-byte prose fixture is a weak generalization probe.
Next tests should hold out WordNet families and unrelated prose,
measure multi-epoch learning, generation, relation recall, and
stride/blend ablations against both baselines.

**Do not merge as a measured improvement until CI and held-out
evaluation demonstrate one.**
