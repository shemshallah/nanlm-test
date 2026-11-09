# Binary-phase fabric: implementation and limits

Built on nanlm-test `00c2eba` (including the latest matched-prior control updates), with NaN0S
`24b358403226fa5bb5b5d108960f03cd9e32536b` as the carrier/VM reference.

## One object, several mathematically distinct views

K256 has C(256,2)=32,640 off-diagonal edges; adding 256 real heads gives
32,896 canonical cells, representing 65,536 oriented logical addresses.
Head 255 remains the local HW0:0 gateway. `address(a,b)=(a<<8)|rev8(b)`
is unchanged. Full bit reversal swaps endpoints. Directed evidence, exact
WordNet identities, and reversible geometric connections are distinct views
of this object, not interchangeable encodings.

`nan16.h` provides a 514-byte immutable Q15 quarter-wave LUT, integer
sine/cosine, 16-step integer CORDIC, all Boolean functions, and explicit
NaN0S memory-byte pack/unpack. The quarter-wave endpoint never reads S[257].
The phase group is Z/65536Z; additions and conjugation wrap exactly, but
trig and learned real parameters remain approximate. Full-turn resolution
is 2*pi/65536 = 9.58738e-5 radians. Two lanes A:C give a 32-bit accumulator
with 1.46292e-9-radian steps. The current trig lookup consumes only A;
fractional C reduces accumulation error, not this lookup's output error.

Measured exhaustive sine error is 3.13894e-5 with scale 32767. Interpolation
error alone understates total error. Q15 outputs cannot carry 32-bit phase
information. There are exactly 16 bits of discrete phase capacity per edge,
not an exponential increase in semantic information. Byte class (4), nibble
group (4), and selector (16) enumerate the 256 heads. Separately, two NaN
headers have four legal classes each, forming 16 Boolean truth tables.
A native NaN16 carrier is **64 bits**, with a 32-bit payload and two headers.
Its tag sign is never used as phase conjugation.

`HermitianView` uses 130,560 off-diagonal bytes (phase + unsigned amplitude)
and 512 real diagonal bytes. It is optional scratch, not another resident
universe. Reversal conjugates phase, with one canonical amplitude. A generic
Hermitian matrix can have negative eigenvalues: it is not automatically a
quantum density matrix. `WaveState` instead constructs psi from predicted
probabilities and covariant connection phases. rho=psi psi* is PSD, trace one,
and rank one; Born probabilities equal the learner's probabilities. Pairwise
interference is available explicitly. This is classical simulation, with no
quantum speedup or new evidence implied by coherent superposition.

The FFT works in complex scratch, using integer twiddle phases and Q15 trig.
Existing rev8 input may skip permutation. It cannot run correctly on only a
phase lane, and transforming rows independently does not preserve Hermiticity.
For an operator basis change use F H F*, not F H. LUT twiddles are approximate,
so spectral mass has small numerical error. The `spectrum` command is a
coherent diagnostic, not another trained language model.

The existing gauge/cycle algebra remains: E-V+1=32,385 independent cycles,
with triangles anchored at vertex zero forming a basis. No dense three-body
allocation is needed for Abelian holonomies. Independent higher-order facts
still require the existing sparse hyperedge overlay. The existing quaternion
layer remains available for noncommuting transport; this change does not claim
SO(3) is better than scalar phase.

## Learning and self-modification

`Harmony` uses 19 normalized experts: conditional counts with eight unigram
pseudocounts, unigram with 0.5 pseudocount per byte, a shared learned circular
translation, and all 16 bitwise Boolean relations. Counts use existing
PRECEDES/FOLLOWS evidence. Global circular moments learn translation from
training data; no evaluation label enters updates. The trig expert has bounded
concentration four. Boolean scores use popcount(BOOL_n(source,candidate)).
These are explicit structural priors on byte IDs, not lexical semantics.

Two sets of Bayesian expert weights handle first-observation contexts and
previously observed contexts. The switch uses only existing row counts.
Each observed target updates its regime's weights by normalized likelihood.
This prevents successful memorization from suppressing transfer rules in
contexts where there are no counts. For each regime, cumulative prequential
mixture loss is bounded by each expert's loss plus log(19) from a uniform
prior (log(2) for the count/backoff ablation), up to floating-point error.
This is not a held-out or distribution-shift guarantee. The fixed experts
can themselves learn online; their predictions must precede each observation.

The best Boolean class is written as a legal `BOOL r1,r2,r3` NaN16 carrier.
`nan-program` exports it. This is bounded learned program selection among
16 functions, not unrestricted code synthesis. All 16 translations execute
in the upstream NaN0S VM. `guest/phase.nas` runs the integer trig operation
natively, exhaustively checking a weighted checksum over all 65,536 inputs.
The full mixture trainer is host C++: a native NaN16 port of training,
probability normalization, sampling, and persistence is still future work.

Generation samples normalized probabilities with a fixed PRNG mapping and
seed. Byte FF never participates. Output is raw bytes and may be invalid
UTF-8, control characters, or nonsense. Generated output is not automatically
accepted as training evidence. There is no claim of open-domain fluency.

## Storage ownership and restart

The 8 MiB root layout is unchanged. New fields occupy previously unused bytes:
heads 0..18 and 32..50 at offset 80 hold double expert log weights;
heads 0..254 at offset 88 hold uint64 incoming counts. Gateway 255 holds
circular moment doubles at 32 and 40, the selected 8-byte carrier at 56,
format marker at 68, and observation count double at 80. The gateway's
identity at 120 and routing remain unchanged. These fields must be reserved
in any future root schema. 38 weights cost 304 bytes within the root.

Binary phase helpers and the view are compact, but the authoritative legacy
phase overlay still stores doubles to preserve file compatibility. No overall
memory reduction is claimed. Training writes quantized phases to that overlay.
Directed evidence remains float32; journals remain unbounded, and root plus
overlay commits remain non-atomic. Use separate roots for independent
experiments; alternating legacy and Harmony trainers mixes their evidence.
The new unigram counters only count Harmony observations, not old journals.

## WordNet and NaNNet

Exact synset IDs, lemmas, glosses, pointer types, and provenance remain in the
existing `.wordnet.tsv` overlay. A 16-bit anchor is never a unique synset ID.
`wordnet-link LEMMA` resolves the exact overlay and creates the existing route
intent through local point 255, labeled `HW4:0::0:HW0`, preserving the exact
synset ID in packet payload. The existing transport is a **local intent stub**;
this implementation does not transmit to an HW4 service. The user's address
notation is retained as a link label, not asserted as a validated wire format.

## Measured evaluation

Same source split, eight epochs, deterministic shuffles, seeds 11/29/47 as
`test_learning_validation`: sources 16..175; every fourth source held out;
target=(source+19)%255. Evaluation is read-only. For all three seeds:

| Model | Trained CE | Unseen CE |
|---|---:|---:|
| Original add-one count baseline | 3.375 | 5.541 |
| Matched Harmony count/backoff control | 0.685361 | 7.68478 |
| Harmony with structured experts | 0.685361 | 3.97032 |

The gain on trained contexts also changes smoothing, so it must not be
attributed to geometry. Structured transfer beats both count controls on this
synthetic translation task. Natural prose, WordNet gloss splits, shuffled
relations, and broader distribution shifts remain necessary before claiming
language generalization. Existing seven-arm geometry benchmarks are retained. The latest pure shared
rotor and its matched offset control score 3.390766 unseen CE, better than
Harmony's softer distribution; no superiority to that control is claimed.
See [sequence learning and aggregation](DISTRIBUTED.md) for the broader design.
