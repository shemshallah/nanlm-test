# Experimental role-probe trained lexical decoder (NLRP1)

## Purpose

The earlier mixed-head signature has excellent bounded hash-inversion accuracy
but **zero** compositional accuracy in the small hash-blind additive-credit
fixture. A separate native experiment tests an architectural alternative:
retain distinct subject, relation, and object **numerical observation
channels at ingestion**, then learn label readouts from those channels.

This is **not** a proof that the original 3-vector/3×3 tensor telemetry
contains decodable semantic roles. The role instrument reads the original
assertion words in the V2 provenance journal to compute three per-role
256-bit hashed hypervectors; it reports only three 6-of-255 head-address
signatures. Therefore it measures a **new, explicit, role-preserving sensor
interface** rather than inverting existing mixed MATS geometry.

## Native observation (no new universe or persistent binary sidecar)

`src/sensor_role_probe.cpp` loads and validates the native V2 source journal
through the actual C++ sensor library. It produces the read-only ASCII
numerical stream `NLRP1 TAB event_tick TAB six_subject_head_ids TAB
six_relation_head_ids TAB six_object_head_ids`. IDs are sorted and bounded
to 0..254. No word, source ID or claim text is present in this numeric stream.
The original journal remains the source of truth for labels and provenance.

Example:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude \
    src/sensor_role_probe.cpp -o sensor-role-probe
./sensor-role-probe /path/to/snapshot.sensor.tsv > /tmp/role-probe.tsv
python3 tools/sensor_temporal_role_decoder.py \
    /path/to/snapshot.sensor.tsv.slices \
    --journal /path/to/snapshot.sensor.tsv \
    --role-trace /tmp/role-probe.tsv --training-events 6
```

This experimental output has *no new persistent allocated universe*. The
existing `MATS` sidecar is not modified. The independently decoded MATS
frames are used to cross-check event ticks, unique-source and fact counters
against the V2 journal. The link between V2 and numeric NLRP1 is temporal
and syntactic, **not a cryptographic or atomic cross-file binding**.

## Supervised learner

Three independent sparse multiclass softmax models are trained with SGD.
The input consists exclusively of six observed head IDs for the respective
role. The labels are the corresponding **training** journal words. All
255-channel classifier weights start at zero. On prediction, the decoder
never calls the native hash, candidate generator, or journal word fields.
A bounded seen-anchor overlap and softmax margin gate cause abstention on
unsupported or ambiguous numeric observations.

A deliberately small trial trains on six statements spanning 3 subject
words, 2 relation words and 3 object words, with independent sources. Six
held-out statements use the same components in **previously unseen whole
triples** from new sources. Report exact triple accuracy, coverage,
source-disjointness, one-missing-head ablation, time-shuffled and role-swapped
controls, and abstentions. Source-disjointness alone is not concept-disjoint.

The learned output is a supervised lexical lookup, **not** semantic meaning:
the native role signatures are deterministic hash-derived identities and do
not encode similarity between the concepts. In particular, this cannot
answer whether a penguin actually is a bird or generate new words outside
the observed training lexicon. The V2 source journal and exact logic graph
still decide support, refutation and source attribution.

## Scientific limitations / next gates

- Small synthetic and native fixtures with repeated component words; evaluate
  on 100s/1000s of source- and whole-statement-disjoint assertions and
  explicit unknown-role rejection before interpreting results broadly.
- Compare against *matched* non-role baseline at similar information and
  parameter budget. A role-aware input has strictly more identifiable
  information than the original mixed-head sample; its improved decoding
  accuracy must **not** be credited to temporal learning.
- Test semantic transfer with WordNet link prediction under held-out synsets,
  lexical ambiguity, synonym and antonym distinctions, and source provenance.
  A relation-label spelling or hash match does not establish semantic
  inference. The unavailable WordNet corpus should not be fabricated.
- Require precision/recall/calibration versus abstention; evaluate confusion
  and noise. Do not claim a lower language-model cross-entropy without an
  actual held-out language-model experiment.

All experimental code remains confined to `nanlm-test` PR #29. No production
NaN16 root, NaNROUTER ring, HW4, Android APK or broader network instance is
modified.