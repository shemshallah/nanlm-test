# Experimental lexical composition from temporal head changes

**Status: read-only exploratory test.** This builds on PR #29's verified
`NLSLICE1` / `NSF1` / `MATS` sidecar decoding. No new model instance, native
universe writes, BPTT, or expansion of the bounded 8 MiB history.

## What this does

A claim `(subject, relation, object)` is encoded by the current *deterministic
and untrained* hash-routing implementation in `include/nanlm/sensor_memory.hpp`:

1. FNV-1a64 words are mixed with the known 64-bit mixer.
2. Subject, predicate and object each use their own role key and four 64-bit
   basis words; those vectors are XOR-bound to one 256-bit claim vector.
3. A claim routes to the six closest of 255 fixed head bases by Hamming
   distance, ties by head ID.
4. Observing adjacent valid `MATS` frames, residual support/refute charge
   after subtracting the known sensor leak (`0.97`) identifies the six changed
   head IDs, with polarity kept separate. This is an **event** descriptor,
   not the cumulative state of the model.
5. A candidate lexicon is formed **only from training events** (all subjects,
   relation types and objects). Every bounded Cartesian-product triple is
   forwarded through the exact native routing formula to compute an *expected*
   six-head signature. A unique, exact signature match can return words for an
   entirely unseen *whole triple*. Ties, unknown tokens, absent features and
   non-exact signatures abstain.

No test-event words enter the candidate vocabulary. Test truth is read from
source journal only **after** building the index and only for scoring.

## Native test (compositional closed vocabulary)

Chronological training statements (six):

```text
penguin ISA bird
eagle ISA animal
whale ISA fish
penguin LIKES fish
eagle LIKES bird
whale LIKES animal
```

Held-out statements (six):

```text
penguin ISA animal
eagle ISA fish
whale ISA bird
penguin LIKES animal
eagle LIKES fish
whale LIKES bird
```

There are 3 subjects × 2 relations × 3 objects = **18** candidate triples.
Every test triple is new, but every component word and relation appeared in
training. The runner builds the actual C++ `sensor-resident`, ingests twelve
source-separated assertions, independently decodes the real 8 MiB sidecar,
and evaluates the Python readout against the V2 journal. Both GCC and Clang
must pass before reporting measured results.

```sh
python3 tools/sensor_temporal_compositional_probe.py recording.slices \
  --journal recording --training-events 6 --max-candidates 4096
```

## Matched controls and interpretation

- **Temporal event readout:** exact six-head change after known decay.
- **Static-only readout:** all nonzero support/refute head addresses after an
  event, with no previous frame. It is deliberately a *simple* occupancy
  baseline, not the strongest possible frozen-vector/tensor comparator.
- **Temporal-shuffle control:** rotate the six held-out event signatures
  among test labels; preserves feature counts, destroys label alignment.
- **Claim-disjoint known-label prototype:** the previous nearest-neighbor
  probe necessarily abstains on statements not in its training label set;
  this new experiment uses known *components* to form unseen combinations.
- **Unknown token:** exact lookup cannot return a word missing from training
  vocabulary. `no_exact_candidate` is an abstention, not a wrong guess.
- **Collision:** if multiple candidate triples map to one observed six-head
  signature, output is ambiguous and must abstain.

**This is not genuine learned compositional semantics.** The candidate decoder
possesses the same deterministic hash-routing rule as the native encoder.
It solves a bounded reverse-search problem in a closed vocabulary, not an
open-world meaning or language-generation problem. A high score proves only
that enough *address information* survived numerical projection to permit
this inverse lookup at this small candidate size. It cannot establish
WordNet semantic transfer, infer an unobserved factual relation, or demonstrate
cross-entropy / accuracy improvements to the actual language model.

## Next scientific gates

1. Compute candidate collisions and false-match/abstention rates as the closed
   vocabulary grows (256, 1K, 4K triples), **without** expanding disk storage.
2. Compare a candidate scorer from full 3-vectors and 3×3 tensors against
   the strict six-head signature; evaluate on sources and exact triples held
   out together, and test whether time history adds signal beyond occupancy.
3. Add a **different** mode using learned geometric-credit concept-prototype
   mappings trained from source-anchored examples. The native hash-based
   re-encoder must be excluded from that model's predictions.
4. Validate concept relationships against a source-disjoint WordNet-derived
   holdout with explicit unknowns and contradictory claims; report coverage,
   correct/eligible and positive/negative controls.
5. Keep the 8 MiB ring, the exact source journal and provenance distinct;
   never silently fill overwritten frames or label unsupported geometry.