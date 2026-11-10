# One complete HW4:0 WordNet epoch (experimental fast geometric pass)

This lives entirely in `nanlm-test`. HW4:0 is the pinned immutable
NaNROUTER/MC0:0 reference at
`nan://GM0:SC0:MC0:BS0:0:HW4:0`. We DO NOT write its matrix, SQLite
ledger, pointer, ring epoch, or BS0 main ring. One independent 8 MiB
student root stores the fitted head geometry.

## Exact corpus coverage

The source is the release
`NaNROUTER/hw4-wordnet31-full-g1`, SHA-256
`87dbcdd2d209db95e8d15be68109ef3642e78ef540ea12ad71f3b04982876695`.
We verify CURRENT.json, release manifest and both root and exact SQLite
SHA-256. All 117,791 synsets, 207,235 senses, 53,869 synonym records
and 378,203 directed pointer rows form the **entire 58,341,529-byte
readout**. Previously split `train/dev/test` files are merely controls;
this experiment trains all sources, so none can be called held out.

## Exactly what one pass trains

`hw4-onepass` makes **one sequential read of every source byte** to
accumulate the 255×255 causal byte-transition contingency matrix. Every
sample affects the statistics exactly once. Then it updates the existing
NaNLM `FabricTeacher` 255 source/target rank-three head vectors using
a full-data, normalized geometric cross-entropy gradient and 16-step
backtracking acceptance gate. A candidate is accepted only if the **actual
native teacher's aggregate CE on all counted transitions** decreases.

This is an exact one-epoch **aggregated geometric co-occurrence
optimization**. It is NOT 58 MB of sequential recurrent attention/oscillator
updates, NOT a full per-byte TRAIL eligibility epoch, NOT semantic relation
training, and NOT a new BPTT implementation. It does **not** write
statistical language co-occurrence counts as formal source-backed graph
claims. The update changes only pre-existing head-local 3D parameters
inside one 8 MiB student root. The source's 32,640-pair matrix remains
read-only, and we do not mislabel copying it as learning.

All corpus lines include lexical definitions and relation names as
training text, but the reduced-rank batch optimization tests what a
low-cost geometric representation can learn from one complete corpus pass.
Actual general-language quality must be evaluated separately. We report
rank-three training CE and a separate fixed non-WordNet prose fixture;
no source-held-out WordNet evaluation exists when all WordNet is trained.

## Reproduce

```sh
python3 tools/hw4_reference_corpus.py --download-release --output /tmp/hw4
make nanlm hw4-onepass CXX=clang++
./nanlm /tmp/learner-root init
./nanlm /tmp/learner-root eval tests/fixtures/unified_quality_heldout.txt
./hw4-onepass /tmp/learner-root /tmp/hw4/all.txt
./nanlm /tmp/learner-root eval tests/fixtures/unified_quality_heldout.txt
```

CI `.github/workflows/hw4-onepass-full.yml` archives the learner root,
source digest/coverage, head-gradient acceptance result and before/after
quality fixture, but not the copyrighted WordNet archive or corpus.

## Further research

Existing local TRAIL eligibility credits and the 85 triads remain
available for later online refinement. Their computationally expensive
per-byte updates are not claimed to have been trained 58 million times.
A subsequent experiment can check semantic relation ranking, paraphrase
generalization and full-pass recurrent gradients against this fast
geometric baseline. This full-pass result is a **baseline**, not a proof of
semantic understanding.
