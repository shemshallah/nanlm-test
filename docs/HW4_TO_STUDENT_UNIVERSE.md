# HW4 WordNet 3.1 as teacher; another NaNLM universe as learner

## Architecture

HW4:0 remains an immutable WordNet reference on the NaNROUTER/MC0:0 ring.
The student is the existing independent 8 MiB root in nanlm-test with
geometric temporal credits, 255 heads, 85 triads and no BPTT. No training
data is written to HW4, its exact SQLite ledger or the BS0 main ring.

Read from a digest-verified snapshot, not from the lossy 32,640-pair
aggregate matrix alone. The current implementation acquires the read-only
HW4 generation through its NaNROUTER version-pinned GitHub release or
from an installed read-only store. It is NOT an authenticated live ring
stream; that will require a generation-fenced paging/lease protocol.

The build checks the SHA-pinned release, CURRENT pointer, manifest and both
data hashes. It produces natural-language gloss, sense, synonym and typed
relation statements, plus separate exact typed-edge TSV files. It preserves
117,791 synsets, 207,235 sense-index records and 378,203 relations.

## Parallel tracks of learning

1. Predictive language: native one-root byte prediction and geometric
   eligibility credits on training-only corpus text; do not enable BPTT.
   Long-context memory is an optional standing-wave readout, not a forced
   representation of exact quotes or names.

2. Relation geometry: future sparse updates on exact source/predicate/target
   events. Train the student on type-conditioned relation candidate rankings,
   including hard negatives sampled from candidate alternatives. Open-world
   missing links are unknown, NOT refutations. Preserve positive and
   refuting provenance separately. Do not directly copy HW4 pair counts
   into the student weights: that would be a memorization oracle, not
   tested generalization.

3. Retrieval: exact long-distance facts and polysemy remain indexed and
   citation-verifiable against HW4 generation. The learned field supplies
   routing and salience, not a substitute for this ledger.

4. Curriculum: definitions and morphology, polysemous contextual senses,
   directed pointers, two-hop composition, and then prose plus grounded
   retrieval. Compare each stage to matched lexical/ngram controls and
   temporal-credit ablations.

## Dataset and evaluation

SHA-256 per exact source synset selects 80/10/10 train/dev/test. This is
source-held-out, NOT fully entity-disjoint: outgoing relation targets can
have names appearing elsewhere. The test set must never be passed to
train-chunk. Score CE, next-byte accuracy, relation ranking Hits@K/MRR,
unknown abstention, four-valued contradictions, bytes/s, root+overlay
size, and heldout source identity.

Verified from actual HW4 release:

| Split | Exact synsets | Text bytes |
|---|---:|---:|
| train | 94,515 | 46,796,369 |
| dev | 11,554 | 5,766,417 |
| test | 11,722 | 5,778,743 |
| total | 117,791 | 58,341,529 |

There are 53,869 synonym statements in addition to 207,235 sense statements,
117,791 definitions and 378,203 relation statements.

The 50-pass request across the training split is 2,339,818,450 bytes.
The short pilot reports 340 B/s inclusive of overhead, which extrapolates
to about 1,912 hours if sustained, but this is NOT a measured end-to-end
speed. Optimize larger batches, native event training, and spectrum rebuild
frequency before launching an expensive whole-corpus campaign. Never claim
that 50 epochs completed if the counter remains zero.

The measured 2,048-byte training probe used a disjoint-source test prefix
(2,047 evaluated bytes). Heldout CE improved 5.69018495 to 3.40694795,
and next-byte accuracy 0.83 percent to 29.12 percent. No BPTT attempted.
2,031 temporal credits accepted. Turning off long-context credit increased
CE from 3.406947948 to 3.411115316 (only 0.004167368). This is an
encouraging SHORT FORMAT-BIASED smoke test, not yet evidence of robust
semantic generalization or a causal long-context mechanism.

## Reproduce on nanlm-test

    make nanlm CXX=clang++
    python3 -m unittest tests.test_hw4_reference_corpus -v
    python3 tools/hw4_reference_corpus.py \
      --download-release --output /tmp/hw4-corpus
    python3 tools/hw4_credit_benchmark.py \
      --engine ./nanlm \
      --corpus /tmp/hw4-corpus/train.txt \
      --manifest /tmp/hw4-corpus/train.manifest.json \
      --heldout /tmp/hw4-corpus/test.txt \
      --train-bytes 2048 --eval-bytes 2048

For the bounded 50-pass experiment, run the existing geometric-credit trainer
against the NEW student root, not the HW4 image:

    python3 tools/wordnet31_50epochs.py \
      --corpus /tmp/hw4-corpus/train.txt \
      --manifest /tmp/hw4-corpus/train.manifest.json \
      --heldout /tmp/hw4-corpus/test.txt \
      --state-dir /tmp/hw4-student \
      --epochs 50 --budget-seconds 900

Beware: a full ~5.8 MB heldout evaluation at every checkpoint may be slower
than one budgeted training run. The frequent short check and the final full
test score should be labeled separately; a failure/interruption between
model sync and progress journal is not yet cross-file atomic.

Reference snapshot and provenance:
https://github.com/shemshallah/NaNROUTER/releases/tag/hw4-wordnet31-full-g1

Real source ingest, training and paired heldout ablation:
https://github.com/shemshallah/nanlm-test/actions/runs/42734153551
