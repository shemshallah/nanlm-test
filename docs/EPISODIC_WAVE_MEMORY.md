# NaNLM HW0: Episodic exact memory + optional wave readout (R1)

Status: **implemented in native C++ on experimental branch**
`feat/hw0-episodic-wave-memory`, stacked on TriadWaveFormer PR #24.
Code: `include/nanlm/episodic_memory.hpp`, `include/nanlm/unified_wave.hpp`,
`tests/test_episodic_memory.cpp`, and `src/main.cpp`.

## Motivation and separation of responsibilities

The 64-dimensional, four-timescale global field and 255 coupled standing-wave
oscillators can carry approximate temporal signatures. They cannot guarantee
byte-exact reproduction of a quotation or identifier after a 1,200-byte gap:
every decaying, finite-precision state eventually loses information.

**Therefore, separate exact evidence from statistical prediction:**

1. **Episodic exact evidence**: source-attributed, explicitly imported
   `cue<TAB>value` records, checkpointed at `ROOT.episodes`.
2. **Recurrent field and standing waves**: trainable / calibratable approximate
   temporal memory for *prediction*, not the sole storage of literal facts.
3. **One production language model**: the existing shared `UnifiedWaveLanguage`
   computes one next-byte distribution. Exact cue matches take precedence over
   speculation; there is no second model and no parallel LM decoder.

## Interface

```sh
make nanlm test_episodic_memory
./test_episodic_memory
printf 'Beacon code: \tAURORA-8273\n' > /tmp/nanlm-facts.tsv
./nanlm /tmp/nanlm-root memorize /tmp/nanlm-facts.tsv trusted-fixture-v1
./nanlm /tmp/nanlm-root recall 'Beacon code: '
./nanlm /tmp/nanlm-root generate 'Beacon code: ' 11 19
# Prints: Beacon code: AURORA-8273
# Explicitly index the opening phrases and remaining text of each long line:
./nanlm /tmp/nanlm-root memorize-lines /tmp/quotes.txt public-domain-book
# Indexed sources carry line tags such as public-domain-book#L18
```

Cues are strict UTF-8 strings of 6–64 bytes ending with a space.
Values are strict UTF-8, nonempty, up to 4,096 bytes, on one logical
line; sources are nonempty UTF-8 labels up to 128 bytes.
This version deliberately uses exact anchor matches, **not semantic
approximation**. Wrong or unrecognized cues fall back to the native predictor;
a mismatching continuation deactivates the copy.

`memorize` is the *only* write path. Ordinary training, evaluation and
generation load the checkpoint read-only. No automatic ingestion from
held-out evaluation data, no query-time online learning, and no claim that
the model has discovered facts on its own. A repeated (cue, source) is an explicit
revision of that source's value. Different sources remain separate; if they
assert different values for one cue, it is **contested**: direct recall
reports `UNIFIED_EPISODIC_CONTESTED` and automatic exact copying is disabled
for that cue. No arbitrary winner is chosen.

To reproduce exact content longer than the 64-byte context window,
the stream state stores a lightweight record index and next-value offset.
Once the cue matches, the cursor advances on every observed byte; the original
cue may leave the window entirely. Sampling is bypassed during a valid exact
copy and standard UTF-8 constraints are still enforced.

## Storage and provenance

The NaN16 root remains exactly **8,388,608 bytes**.
The separate bounded `ROOT.episodes` file has a fixed format marker,
little-endian record sizes, byte-exact UTF-8 strings, full source labels
and FNV-1a record digests. It has a maximum of 4,096 records and
24 MiB on-disk file limit. It is a **versioned model-attached memory sidecar**,
not a second root, neural expert, or replication of the 32,640 mirror pairs.
Replacing the file uses write-new-then-rename semantics, so a failed import
does not replace the previous checkpoint. FNV-1a is an error-detection
checksum, **not** an authentication hash or digital signature.
Power-loss durability and simultaneous multiwriter coordination require a
future fsync/journal/transaction layer.

The root's 255th head (network ingress) is not written by this feature.
An attached remote memory provider must eventually supply signed provenance,
authorization, version checks and merge semantics before any multi-device use.

## Standing waves remain optional, not obligatory

The previously unused head 0 byte 122 carries a backward-compatible
bounded direct-wave-readout scale. The untouched byte denotes 1.0; explicit
scales include zero (disable direct wave and T/T* auxiliary readout),
and values through 2.0. The exact direct-wave adjoint and local output-gain
updates incorporate this scale. The `calibrate DEV_TEXT` command tests
wave scales on **development** text only, reports the choice, and retains
zero if it improves dev loss. `audit` includes `no_wave` and
`no_episodic` on the *same* root; disabling `no_wave` now also disables
the otherwise unconditional BPTT direct-wave readout.

**Scope of zero wave scale:** direct standing-wave logits vanish. Internal
wave recurrence, attention, hyperedge readout, mirror effects, triad field
inputs and geometry still execute; claiming the entire oscillator is disabled
would be incorrect. A future runtime compute gate can be added only after
dependency tracing proves equivalent behavior or calibrated tolerances.

Importantly, exact episodic recall works with wave direct-readout scale
0. Thus standing waves are a useful candidate *approximate temporal
representation* but do not gate literal memory retrieval.

## Validated tests and remaining experiments

Native `test_episodic_memory` covers: 1,200-byte irrelevant gap,
full continuation longer than 200 bytes, 64-byte history wrap,
source attribution, strict UTF-8, TSV batch import, persistence on restart,
checkpoint corruption rejection, unchanged 8 MiB root and ingress head,
non-mutating generation and wave-scale=0 independent exact copy.
Workflows run under both GCC and Clang and verify actual production CLI.

Next:
- add a **source-aware free-text extractor** that requires explicit user
  approval for facts and quotations; index canonical source spans and
  offsets rather than guessing entity-value relations
- add optional wave-derived candidate ranking and learned retrieval gating,
  but only commit an answer after *byte-exact verification* against the
  persistent source span
- evaluate 1k, 10k, 100k and 1M byte intervening gaps; compare memory
  on/off, wave on/off, exact-hit/miss, wrong cue, corrupted checkpoint,
  retrieval latency, disk size, peak RSS, and held-out language CE
- design multiwriter transactional merge, source signatures, provenance,
  conflict handling and revocation; keep refutation distinct from support
- do not score exact indexed quotations as unseen-language generalization:
  retrieval success and language modeling generalization are separate metrics

This is the **first grounded exact-retrieval mechanism**, not a solution
for general semantic understanding, paraphrase matching or AGI.

## Optional phase-resonant cue ranking (experimental)

The exact episodic store now offers an optional phase-matching search mode.
For each cue, four pairs of damped sine/cosine oscillators produce an
8-component normalized signature. Candidate cues are ranked by cosine
similarity of this phase code to the current history suffix, with a bounded
secondary priority term from the **live 255-head standing-wave state**.
The longest matching exact suffix always takes precedence; every candidate
must pass a byte-for-byte equality test before a copy starts.
Conflicting source values abstain even if their phase scores are high.

This mode is **disabled by default**; a caller opts in with
`set_resonance_retrieval_enabled(true)`. This is a candidate-ordering
experiment, not evidence of faster retrieval, superior language CE, or
semantic paraphrase retrieval. Because the candidate list is fully sorted,
it may use *more* CPU than direct scanning for small memory banks.
The native test compares resonance-on and resonance-off against the same
exact cue, then challenges both with a near-miss and contested cue.

### Multi-candidate routing benchmark

A separate native fixture inserts 180 equal-length decoy cues before the
true cue, then compares baseline exact scanning with phase-ranked searching.
Both must resolve the **same exact record** and reject a similar-looking
nonmatch. A fixed 50-repeat microbenchmark prints equality-comparison counts
and wall-clock microseconds for each probe. These are diagnostic CPU timings,
not whole-model generation rates; use them to decide whether the optional
resonance route earns its runtime cost. No blanket speedup is presumed.

### Native exact-hash addressing (new default)

The basic exact-cue path now indexes **reversed 64-bit byte hashes by cue
length**. At a cue boundary, a 64-byte reverse hash walk creates candidate
buckets for every possible suffix length; longest verified suffix wins.
Hash matches are always byte-verified, and all distinct source claims
are still checked for contradictions. The hash index is reconstructed from
records on checkpoint open/import; it is not a second persisted root.

Phase ranking is still opt-in. For operator-assisted approximate lookups,
`suggest CUE [TOP_K]` returns only candidate cue strings, sources, scores
and an explicit `verified` flag, **never source-backed fact values**.
`probe-memory CUE [RESONANCE_0_OR_1]` exposes the exact search result,
number of candidates, comparison count and the phase score. These controls
are diagnostic and do not automatically authorize copying from a near miss.
`generate` accepts a final optional 0/1 resonance-routing flag; it defaults
0 for predictable low overhead.

The adversarial benchmark now compares **hash-indexed** exact retrieval with
phase ranking, not a linear scan. Any faster hash result is independent of
whether wave-based associative search is useful.
