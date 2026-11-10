# Experimental temporal aggregation and journal-grounded conceptual translation

This is a **read-only** consumer of the verified `MATS` v1 time-slice sidecar
and (for concept words only) the original `NANLM_SENSOR_V2` source journal.
No extra universe, separate language model, NaN16 root writes, or additional
sidecar capacity. The numeric history stays fixed at 8 MiB.

## Commands

```sh
python3 tools/sensor_temporal_aggregate.py snapshot.sensor.tsv.slices aggregate --format jsonl
python3 tools/sensor_temporal_aggregate.py snapshot.sensor.tsv.slices aggregate --format csv
python3 tools/sensor_temporal_aggregate.py snapshot.sensor.tsv.slices concepts --journal snapshot.sensor.tsv --format jsonl
python3 -m unittest discover -s tests -p 'test_sensor_temporal_*.py' -v
```

## Four independent aggregation tracks

1. **Coordinate history**: per `(level, ID, head-membership)` aggregation,
   separately for heads, triheads, hot triheads, hot field and global field;
   mean 3-vector, full mean 3x3 tensor, activity, signed evidence charges,
   number of distinct observed ticks and sample count.
2. **Temporal persistence**: hot-triad *sampled occupancy* is the fraction of
   retained ticks on which that coordinate was present. Contiguous runs require
   adjacent sequence and adjacent clock ticks. Same-tick measurements are not
   double-counted as persistence; missing sparse rows are not zero.
3. **Mirror coupling**: pair keys are canonical C(256,2) physical offsets.
   Opposite directed reads are first converted into a common flux orientation
   before computing mean flux, absolute flux and phase lock.
4. **Oscillation screening**: a deliberately conservative, untrained vector
   autocorrelation heuristic on at least 12 consecutive ticks and at least 3
   cycles. This emits **candidate**, **not detected**, or **insufficient**.
   It cannot establish a physical standing wave or a predictive effect.

## Turning numerical patterns into words: two different meanings

**Descriptor vocabulary:** Numeric telemetry may legitimately be called
"persistent sampled hot-triad activity", "coexisting positive and negative
charges", "recurrent vector pattern candidate", or "mirror-pair phase
alignment". These describe observed sensor geometry only, not a statement
about external reality.

**Source-grounded claim vocabulary:** The separate V2 journal contains source,
polarity, subject, relation, object and original event clock. The optional
`concepts` command obtains *all* lexical material from those literal journal
fields, groups evidence by exact typed statement, and emits `supports`,
`refutes`, `both` or (if no evidence) `unknown`. Per-event NaNclock can be
aligned to a verified numerical frame, but temporal coincidence **does not**
prove that a specific hotspot encodes that claim. Support and refutation never
cancel. A missing or inconsistent journal blocks the conceptual output.

Example: `penguin ISA animal` supported by `src_b` and refuted by `src_a`
becomes the attributed proposition `penguin is an animal` with `evidence_state:
"both"`; this is **not** a fluent learned translation. Numerical component
values are not decoded as text. Journal/frame source and fact counters are
cross-checked at every retained frame; counts alone are **not cryptographic
binding**, and frame files should be read from stable snapshots.

## Scientific tests still required

- **Permutation and ablation**: shuffle source labels or time order; measure
  whether predictive utility disappears while static summaries are preserved.
- **Matched baselines**: one-step prediction with static vector/tensor only,
  versus static + temporal recurrence + mirror + hot-persistence features.
- **Source-disjoint validation**: split sources and semantic facts into held-out
  groups, not adjacent duplicated measurements from the same retained trace.
- **Learned alignment**: train a bounded discriminative prototype/index mapping
  of source-attached **concept IDs** to temporal windows; report top-k and
  calibration alongside lexical and most-frequent-class baselines. No free-form
  text generation until the concept decoder outperforms baselines.
- **Longer observation**: six retained frames are nowhere near enough to claim
  a stable oscillatory mode, a memory benefit, or improved cross-source learning.

The numerical stream is information-losing and its lexical routing is hashed;
there is no mathematically justified unique inverse from a 3-vector to words.
For real language, maintain a separate evidence-backed vocabulary/provenance
index and label inferred relationships as inferences, not observations.

## Source-disjoint numerical-to-lexical retrieval probe

`tools/sensor_temporal_concept_probe.py` is a separate *experiment* that tries
retrieval (not generative decoding) from verified numeric changes alone:

```sh
python3 tools/sensor_temporal_concept_probe.py snapshot.sensor.tsv.slices \
    --journal snapshot.sensor.tsv
```

For each source-journal event with a verified adjacent before/after frame, it
computes which sparse head channels gained support/refute charge **beyond**
the default 0.97 per-tick decay. Newly observed heads are marked present, not
interpreted as a missing zero measurement. It forms a sparse head-ID signature,
indexes signatures with their *training* journal concept labels and searches
for the nearest label by Jaccard similarity. Test-event source labels are
withheld from indexing and reintroduced only to score correctness. Predictions
with weak similarity or ambiguous ties **abstain**. The output reports eligible
known-label events, coverage, correctness, refutation-polarity recovery and a
most-frequent-training-concept baseline. Never report only accuracy among
non-abstentions or infer unknown concept names.

The source-disjoint test tests **repeated known facts under new sources**;
it cannot demonstrate comprehension of previously unseen words or concepts.
The experiment assumes a leak coefficient of 0.97, no dropped intermediary
frames, and one assertion per NaNclock ingest. The stored three-vectors and
head IDs remain a lossy hash-routed representation; their relationship to
words is acquired by supervision, not mathematical inversion.

The next meaningful advance would add an unsupervised concept cluster index,
then a supervised calibrated retriever tested on held-out source/claim splits,
compare these representations to frozen static snapshots, and only then
incorporate them as optional geometric credit signals into the model.