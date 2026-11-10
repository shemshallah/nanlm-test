# Full Princeton WordNet 3.1 — 50 epoch geometric-credit experiment

Branch: `experiment/wordnet31-50epochs` (separate from the production PR
and entirely within `nanlm-test`).

## Exact source

The user requested `wn3.1.dict.zip`. No such attached archive was
available; the public **official WordNet 3.1 dictionary** is
`https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz` (16,358,468 bytes),
SHA-256 `3f7d8be8ef6ecc7167d39b10d66954ec734280b5bdcd57f7d9eafe429d11c22a`.
Both the Princeton endpoint and FreeBSD distfile cache are accepted only if
their original bytes match that digest. An explicitly supplied ZIP can be
parsed with `--archive`, but the manifest always identifies actual bytes.

## Complete coverage, not gloss-only

`tools/wordnet31_full_corpus.py` concatenates the *verbatim bytes* of
all four `data.noun/verb/adj/adv` files, four `index.*` files,
`index.sense`, the four `*.exc` morphology files, `lexnames` and
available additional database resources (`cntlist`, verb examples, etc.).
Each file has deterministic begin/end sentinels, no omitted synset lines, and
a per-member SHA-256 and line count. Full dataset and archive SHA-256,
synset count and total training bytes are recorded. No source corpus is
committed to GitHub; retain the WordNet copyright and license on all source
copies.

## Exactly one model, 50 target passes

The default `UnifiedWaveLanguage::train` remains the *existing* geometric
eligibility teacher. Its optional BPTT path is never activated.
`train-chunk CORPUS OFFSET LENGTH ABSOLUTE_STEP` trains contiguous source
bytes on the one 8 MiB root. The 64 preceding bytes warm its causal state
**without duplicated training**; the absolute learning step preserves
event-time and periodic local update schedules across processes.

The Python runner targets **50 actual full-corpus epochs**, not a 50-epoch
label over a sample. It uses bounded native training chunks so a single
CI run can report honest progress and checkpoint the same model and
overlay for a later resume. The per-run budget is deliberately limited
to 900 seconds due to computational cost and journal growth; its final
`status`, `epochs_completed`, `offset_into_next_epoch`, bytes seen,
online CE, temporal credit acceptance and source-disjoint held-out CE
identify what really finished. **Never report 50 epochs as trained
unless `epochs_completed == 50`.**

A root plus progress journal can be resumed via the Actions
`workflow_dispatch` parameter `resume_run` (prior Actions run ID).
Rerunning a cold branch starts another independent experiment.
The source archive, dictionary text and unpublished samples are not
included in model-checkpoint artifacts.

## Repro

```sh
make nanlm CXX=clang++
python3 tests/test_wordnet31_full_corpus.py
python3 tools/wordnet31_full_corpus.py --cache /tmp/wn3.1.dict.tar.gz --output /tmp/wn31-full.txt
python3 tools/wordnet31_50epochs.py \
  --corpus /tmp/wn31-full.txt \
  --manifest /tmp/wn31-full.manifest.json \
  --state-dir /tmp/wn31-state --epochs 50 --budget-seconds 900
```

Training the whole dictionary at the current character-level throughput
may require more time than a single six-hour GitHub runner supports. The
preflight quantifies that before pretending the 50-epoch target is feasible.
If the physical journal would exceed its configured 256 MiB cap, the
script reports `OVERLAY_CAPACITY` rather than silently deleting evidence
or spawning separate models.

**Caveats:** training every byte includes synset offsets and index syntax;
corpus memorization is not semantic understanding. Interrupted process
between a synced model update and journal commit can cause a chunk to be
replayed on resume; this is a research checkpoint, not a crash-atomic
distributed transaction. Synthetic parser tests never stand in for the
real full WordNet download and checks.

## Real archive preflight repair

The first real download passed the official 16,358,468-byte SHA-256 check,
then correctly reported the Princeton archive does **not** contain a
separate `lexnames` file. That metadata is now optional; all four
`data.*`, four `index.*`, `index.sense` and four `*.exc` files
remain **strictly required**. The inventory lists every available
additional WordNet data file and flags any missing required member.
This distinction avoids pretending metadata from WordNet 3.0 was shipped
in the 3.1 dictionary-only release.

The inherited signed-geometric-credit fixture also now rebuilds its
derived spectral cache after inserting new test evidence. Without this
step it compared a stale in-memory spectrum to a freshly rebuilt
spectrum on restart, producing a false persistence failure.
