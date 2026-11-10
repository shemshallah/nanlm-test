# Native joint UnifiedWave semantic-credit experiment

Status: experimental, not merged. Parent: PRs #50 and #52. This proof-of-mechanism evaluates a **new shared differentiable 24D readout of actual UnifiedWaveLanguage::State**, with byte, role and sense losses. It does **not** yet reuse the parameter matrices in ContextualRoleLearner/ContextualSenseLearner: it reuses their supervision schema and exact WordNet identifiers. Full integration of the preceding contextual encoder remains outstanding.

## One root, three supervised losses

The existing UnifiedWaveLanguage advances its NaN16 recurrent state over raw bytes. For each nonempty prefix, a fixed read of the actual wave and inverse-wave arrays produces 255 bounded inputs:

    x[i] = tanh(0.25 * wave[i] + 0.25 * inverse_wave[i])
    z[d] = tanh(bias[d] + sum_i P[i,d] * x[i] + order[d] * role_orientation)

All three objectives update the **same** 255 x 24 projection P and bias.

- Byte: softmax(log(UnifiedWave.predict(state).p) + B * z) over 255 bytes, CE on next byte.
- Role: relation-keyed sigmoid with supervised, ordered subject/object and binary BCE; orientation is derived from exact entity token positions. Readout remains a learnable semantic classifier, not an asserted fact.
- Sense: masked synset softmax over 2..64 exact WordNet candidate IDs. A candidate absent from training causes abstention. Supervised cross-entropy updates the same P and bias.

The first, detached-readout experiment differentiates through the shared projection while keeping the wave input x fixed. The *subsequent* opt-in eight-step semantic recurrence path differentiates through the oscillator and can update physical recurrent input-gain registers following guarded replay. Neither path differentiates triad routing, QKV kernels or FabricTeacher; neither asserts a WordNet claim. This is truncated recurrent BPTT, not full end-to-end differentiation of every language component or unsupervised semantics.

All joint readout parameters persist as an FNV-integrity-checked, atomic-renamed single experimental sidecar (ROOT.joint-wave), with bounded parser and corruption rejection. Saving a sidecar is not a transactional checkpoint of all root components. A separate byte-only and semantic-shuffled model runs on the **same frozen underlying wave**, not a second universe.

## Native reproduction

    g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude tests/test_joint_wave_semantics.cpp -o test_joint_wave_semantics
    ./test_joint_wave_semantics

Run both GCC and Clang with warnings-as-errors in the native-joint-wave-semantics.yml workflow; or use make test-joint-wave-semantics. The fixed synthetic fixture has role-reversal pairs with names absent from training, two exact bank WordNet sense IDs, held-out sentence recombinations and a separate held-out byte string. Training data are explicitly synthetic and small: no Gutenberg semantics claim follows.

Required controls:

1. Matched byte-only training without semantic gradients.
2. Joint role + sense training after matched byte warm-up.
3. Seeded, fixed-permutation semantic-label control (balanced counts preserved).
4. Role-only and sense-only ablations with identical byte warm-up.
5. Verify semantic gradients shift the byte probability and root symbolic events remain unchanged.
6. Restart with the sidecar, reject corrupted sidecar, and verify production UnifiedWave prediction remains exactly unchanged.

Compare held-out **cross-entropy**, not only accuracy. If joint CE is worse than byte-only, report interference. If it is better, verify that the advantage survives more seeds, larger splits, negative controls and naturalistic prose before claiming semantic transfer.

## First cross-compiler receipt, before task-specific ablation revision

GitHub Actions run 42752883590, native GCC and Clang jobs succeeded on the first implementation. The first GCC receipt reported:

| Outcome | Value |
|---|---:|
| Joint held-out byte CE | 4.22553 |
| Byte-only held-out byte CE | 4.22605 |
| Shuffled-control byte CE (original alternating-label control) | 4.22606 |
| Production UnifiedWave CE | 4.22618 |
| Held-out role accuracy | 62.5% |
| Held-out sense accuracy | 75% |
| Shuffled-control sense accuracy | 75% |
| Semantic-induced shift in prob(space) | +0.0000682806 |
| Byte / role / sense updates | 452 / 280 / 210 |
| NaN16 journal events after joint training | 113, unchanged |

Caveats: the very small byte CE delta is insufficient to establish a statistically reliable benefit, and the sense accuracy equals its negative control on this small test. The first shuffled-label implementation changed the target every epoch; the subsequent revision replaces it with a fixed seeded permutation and adds separate role-only and sense-only ablations. **Do not use old shuffled metrics to describe the revised protocol.**

## Revised protocol receipt (fixed seeded labels; separate semantic objectives)

Run 42753102857 completed successfully for both GCC and Clang. The GCC job reported:

| Held-out measure | Joint role+sense | Byte only | Role only | Sense only | Shuffled labels |
|---|---:|---:|---:|---:|---:|
| Byte CE, synthetic heldout | 4.22553 | 4.22605 | 4.22605 | 4.22558 | 4.22661 |

The unchanged production UnifiedWave byte CE was 4.22618. In this one small seed, the positive byte shift primarily tracks the **sense** objective; role-only was indistinguishable at the printed precision. Heldout role accuracy was 62.5% versus 50% randomized, and heldout sense accuracy was 75% versus 25% randomized. The joint shared-readout prediction for the space byte moved +0.0000682806 under semantic training.

This is a valid initial mechanistic/control result, but these data do not resolve significance, robust improvement across seeds, generalization to unseen relations, or natural semantics. The next benchmark must report confidence intervals and absolute/proportional deltas across independently generated heldout examples.

## Recurrent semantic adjoint (experimental verification)

The next commit adds a **read-only, eight-transition reverse-mode derivative through the actual UnifiedWave oscillator**. This is a separate capability from training the 24D readout. The semantic role/synset softmax supplies a terminal gradient with respect to 255 forward and 255 inverse wave activations; UnifiedWave propagates these gradients through both second-order recurrences, directed scattering, the input-gain register and frequency shift. It does not differentiate unrelated teacher, attention or triad parameters.

The native test uses a complete eight-byte causal tape and independently perturbs the *existing* input-gain register by plus/minus 0.005, replaying the exact physical wave from the tape's fixed starting state. Raw original gain-register bytes, including the magic marker, are restored after each test. The analytic role and sense derivatives must agree with central differences within a numerical tolerance. An opt-in safeguarded update then uses the analytic semantic gradient to propose changes to the *existing* mapped recurrent input-gain registers, replays the entire sentence to verify semantic CE, and compares against a separate byte-development sample before committing. Every rejected proposal restores all original gain bytes and magic markers. A valid single-step optimizer still does not prove robust long-run semantic acquisition.

## Verified physically committed recurrent step

Run 42754322831 passed on GCC and Clang. The analytic and finite-difference gradients matched:

| Objective | Analytic | Finite difference |
|---|---:|---:|
| Exact synset loss | 0.00884468 | 0.00884466 |
| Ordered role loss | 0.00691626 | 0.00691634 |

The first guarded proposal was accepted and persisted directly in the mapped NaNLM root:

| Measured objective | Before | After |
|---|---:|---:|
| Synset supervision CE | 0.393033 | 0.392068 |
| Independent byte development CE | 4.12775 | 4.12773 |

Run 42754322831 initially allowed up to +0.01 byte CE but in fact reduced the measured byte CE. **Run 42754454073** subsequently enforced the stricter **zero allowed byte regression** acceptance gate and passed on GCC and Clang with the same accepted update and the same before/after metrics. These are one-sentence, one-development-sample micro-tests and do not imply a general improvement.

## Next required experiment

1. Connect the preceding shared contextual lexical embeddings to the wave readout, preserving end-to-end derivatives to both roles and senses instead of creating parallel encoders.
2. Repeat the safeguarded recurrent updates across independent corpora and seeds, audit the frequency-gain gradient and compare to detached readout on unseen-book byte loss.
3. Compare multiple fixed seeds and label permutations; evaluate unseen templates, unseen predicates and real Gutenberg heldout books, with corpus disjointness and leakage controls.
4. Gate acceptance on non-interference in language CE and exact WordNet reasoning. Keep exact claims separate from probabilistic outputs.


## Pinned cross-author Gutenberg transfer and interference audit

The next stage implements a source-verified natural-language test in
`tools/joint_gutenberg_audit.cpp`, prepared by
`tools/prepare_joint_gutenberg.py` and summarized by
`tools/report_joint_gutenberg.py`. It reuses the existing
`tools/public_domain_benchmark.py` pinning of GITenberg Alice and Dickens
repositories by immutable commit and SHA-256. The sources are downloaded only
during Actions; only content hashes, offsets, exact split bytes and metric
receipts go into artifacts. No copyrighted source text is checked into GitHub.

| Corpus | Role | Body character slice |
|---|---|---:|
| Alice's Adventures in Wonderland | Wave and byte readout training | 1,000–3,600 |
| Alice's Adventures in Wonderland | Independent byte-development gate | 50,000–50,640 |
| A Christmas Carol | Cross-author early heldout test | 1,200–2,170 |
| A Christmas Carol | Cross-author later heldout test | 28,000–28,970 |

The recurrent NaNLM root is initialized using only the first 1,300 training
bytes. For each of three seeded causal-chunk training orders, three matched
readouts share the same frozen native UnifiedWave foundation:

1. Byte-only supervised next-byte CE on Alice train.
2. The same byte updates plus small supervised synthetic role/synset gradients.
3. The same byte updates plus fixed-permutation randomized role/synset labels.

The semantic training examples are synthetic and **are not acquired from the
Gutenberg sources**. This evaluates whether a semantic signal learned elsewhere
transfers to natural prose while separating shuffled-label effects.

The opt-in exact-root recurrent update uses the pretrained joint readout and
one supervised synthetic sense. Only noncontiguous Alice development CE may
authorize the write, with **zero permitted byte-development regression**.
The test book is never an optimizer input or a selection criterion. Cross-author
byte CE is measured both before and after any accepted recurrent update;
independent semantic role and sense CE are also recorded to detect interference.

The output includes per-seed, per-book-part CE, paired byte-only/joint/shuffled
deltas, recurrent accepted/rejected status, exact 8 MiB root size, unchanged
symbolic event count, and an aggregated JSON report. Three orders from one
deterministic initialization are **not three independent model initializations**,
and cannot support a population confidence interval, p-value or a general
language-improvement claim. Negative results are acceptable, reported outcomes.

Reproduce in GitHub Actions via
`joint-gutenberg-audit.yml`. To reproduce manually, first run
`python3 tools/prepare_joint_gutenberg.py <TEMP_DIR>` on a connected
machine, compile the C++ audit with GCC/Clang C++20 -O2 -Werror, run it with
the four split paths and the integer seed count 3, then run
`tools/report_joint_gutenberg.py` with the receipt and manifest.



## First successful cross-author receipt: negative transfer

[GitHub Actions run 42757461252](https://github.com/shemshallah/nanlm-test/actions/runs/42757461252) completed successfully with GCC/Clang compilation under -Werror, pinned Gutenberg source digests, the 8 MiB native root, all three training-order seeds and preserved numerical artifacts. **This is a negative result; do not treat a passing CI job as a language-quality victory.**

| Training-order seed | Byte-only cross-author CE | Naive joint cross-author CE | Joint minus byte (nats/byte) |
|---|---:|---:|---:|
| 20261109 | 2.959716540 | 2.972994472 | +0.013277931 |
| 20261212 | 2.980040214 | 2.993224216 | +0.013184003 |
| 20261315 | 2.970036977 | 2.985696344 | +0.015659366 |

Average penalty: **+0.014040433 nats/byte**, sample standard deviation across the three training-order seeds **0.001402823**. Each seed evaluates the same 1,938 Dickens bytes (969 early and 969 later) after 2,636 Alice training bytes. The improvement previously measured on tiny synthetic contexts **does not carry over** to this book-disjoint evaluation. There is no inference that this sample represents all Gutenberg literature.

The single optional semantic recurrent step was accepted for every seed based on disjoint Alice development CE, but slightly worsened Dickens CE each time: post-minus-pre joint CE **+0.000002826**, **+0.000002553**, **+0.000005275**. This reveals the limitation of a narrow, same-author development gate. All exact symbolic events remained unchanged.

Synthetic heldout role accuracies were 0.50 / 0.25 / 0.50 across the seeds; sense accuracy was 0.75 in all three, but in one seed its shuffled control also reached 0.75. The current semantic task is weakly learned; no WordNet-scale natural sense induction is established.

**Next ablation (under test): Pareto-gated semantic readout.** Train a matched fourth readout on the identical physical wave; offer batches of three synthetic semantic epochs; commit the entire readout atomically in memory only if independently heldout Alice byte CE does not increase. Keep the disjoint Dickens text outside the acceptance gate. Report accepted/rejected semantic batches and the resulting book-disjoint CE, even if zero batches are accepted. A zero-interference result achieved by rejecting all semantic updates must be called a failed joint-learning attempt, not successful transfer.


## Causal audit revision: shared-projection line search

The initial Pareto candidate offered only a full semantic batch and could
reject every batch, leaving a byte-only readout with no semantic learning
at all. The revised implementation draws a precise distinction between
**positive shared credit** and **head-only specialization**.

After three synthetic supervision epochs, the candidate is evaluated on
the Alice-only byte-development gate using fixed projection interpolation
weights `1, 1/4, 1/16, 1/64, 1/256`, in that order. The shared 255×24
projection and bias are interpolated, while supervised relation/synset
heads are retained. Only an interpolation weight greater than zero that
does not raise development byte CE can count as accepted shared credit.

If every positive weight is rejected, a strictly zero-weight fallback
adopts only the semantic heads and role orientation. The byte projection,
bias and byte output weights must then give **bitwise-identical byte
probabilities** on a native regression probe. That fallback is recorded
as head-only, **not evidence of semantic-to-language learning**.

The aggregate report checks that all six batches per training-order seed
are accounted for, the alpha sum is consistent, byte-only development CE
remains exact when every batch fell back, and cross-author Dickens
performance is still reported regardless of the gate outcome. The native
unit test independently checks alpha=0, alpha=1, invalid alpha, and
unrelated-readout rejection. This only adds experimentally controlled
credit attribution; it does not make the natural-language semantic
training corpus any larger or establish a reliable transfer advantage.

**Status:** implementation committed to the draft experimental branch;
the new GitHub Actions run must pass before reporting new metrics.


## Split semantic heads from shared-byte credit — verified fourth arm

Cross-author [GitHub Actions run 42758760796](https://github.com/shemshallah/nanlm-test/actions/runs/42758760796) passed C++20 strict GCC/Clang compilation, source verification and all three training-order audits. It adds an opt-in `JointWaveSemanticLearner::blend_semantics_from` operation:

* `alpha=1` proposes the whole semantic projection/bias delta, alongside newly learned role/sense heads and role-specific orientation.
* The fixed line search tries `alpha={1, 0.25, 0.0625, 0.015625, 0.00390625}` against **Alice-only** heldout development CE. Nonzero shared credit is accepted only with zero permitted byte CE increase.
* On rejection, `alpha=0` still copies the trained task-specific semantic heads and role orientation while leaving all 255x24 shared projection coefficients, bias and byte-head weights exactly unchanged. This is semantic specialization on a frozen language encoder, **not joint credit**.
* The method refuses unrelated learners, byte-update mismatches and nonfinite/out-of-range interpolation coefficients.

| Seed | Shared semantic batches accepted | Head-only batches retained | Role steps retained | Sense steps retained | Dickens CE delta against byte-only |
|---|---:|---:|---:|---:|---:|
| 20261109 | 0 | 6 | 144 | 108 | 0.000000000 |
| 20261212 | 0 | 6 | 144 | 108 | 0.000000000 |
| 20261315 | 0 | 6 | 144 | 108 | 0.000000000 |

Semantic heldout test role CE for the constrained arm was 0.69627 / 0.69586 / 0.69456, and sense CE was 0.69048 / 0.68946 / 0.69354. These values are weak: the unconstrained joint model achieved lower sense CE on all three seeds, although at a consistent Dickens byte cost. The constrained head-only arm did not establish cross-author language improvement. Zero byte regression follows from freezing the language readout and projection, **not from discovering a beneficial joint gradient**.

The native synthetic test additionally verifies `alpha=0` byte-distribution identity for all 255 predictions, finite semantic task metrics, `alpha=1` equivalence to the trained proposal, strict scale validation, checkpoint/reload invariance, and unchanged exact NaN16 journal. The 8 MiB root remains the only physical Universe: the readout variants are scoped experimental heads.



## Direct frozen-encoder arm (fifth variant)

Two optional C++20 parameters now allow semantic head optimization without
creating a candidate projection drift:

`teach_role(state, example, lr, /*update_shared=*/false)` and
`teach_sense(state, example, wordnet, lr, /*update_shared=*/false)`.

In this mode, the pretrained byte projection and bias are held *exactly fixed*
while role and synset classification heads receive supervised gradients.
Role-specific orientation can still learn because it is not read by the byte
predictor. Both the full-gradient default and the direct frozen mode share the
same 255-component actual UnifiedWave state. There is still exactly **one**
8 MiB NaN16 Universe; experimental readouts are not additional cognitive
instances or disconnected copied roots.

The pinned Alice/Dickens audit now includes a fifth model: **direct-frozen**.
It receives the same byte warm-up, the same semantic examples and epoch order
as the unrestricted joint model. The test compares it with the previous
head-only *copy-back* model, which trains the candidate on a moving projection
then discards that projection. These are not equivalent optimizers.

The direct-frozen arm asserts exact equality of the byte predictor's
development and cross-author CE and of the full shared-projection norm,
while reporting independent heldout role/sense CE and accuracy. This is a
test of whether purely semantic heads can learn on the existing language
features **without** perturbing those features; it does not by itself
demonstrate beneficial semantic-to-language transfer. No Dickens byte data
are used in the optimizer or any gate.



### Direct frozen-encoder measurement

[Run 42759291547](https://github.com/shemshallah/nanlm-test/actions/runs/42759291547)
successfully ran the expanded five-arm benchmark against the same immutable,
author-disjoint source splits. The native GCC and Clang C++20 compilation
completed with `-Werror`; the test receipt and seed CSV-equivalent rows
are preserved in the Actions artifact.

| Seed | Copy-back role CE | Direct frozen role CE | Copy-back sense CE | Direct frozen sense CE |
|---|---:|---:|---:|---:|
| 20261109 | 0.69627018 | 0.69613290 | 0.69047737 | 0.69039797 |
| 20261212 | 0.69586395 | 0.69578957 | 0.68946462 | 0.68937233 |
| 20261315 | 0.69455841 | 0.69448584 | 0.69354477 | 0.69345664 |

The direct-frozen method lowers semantic heldout CE slightly on all three
training-order seeds versus copying heads from an incompatible changed
projection, but **does not improve semantic heldout accuracy** (roles:
50% for every seed; senses: 75%, 75%, 50%). These datasets are too
small to assert a significant generalizable advantage.

All direct-frozen Gutenberg and Alice development byte CE matched byte-only
**exactly** to reported precision; both arms used 144 role and 108 sense
updates per seed. The naive joint model still worsens Dickens CE by
+0.014040433 nats/byte on average. No beneficial gradient into the shared
language projection has been verified. This establishes a conservative
fallback path with functioning task-specific learning and zero byte
interference, but not improved general-purpose semantics or text generation.



## Projected shared semantic updates: calibration/gate separation

Two experiments separate **failure of a straight semantic step** from
**failure of all semantic information**. Both use the same exact 8 MiB physical
Universe and synthetic role/synset supervision; neither trains on Dickens.

Define the proposed shared vector as
\(\Delta\theta_s=\theta_{\rm semantic}-\theta_{\rm byte}\), with
\(\theta=(P,b)\) for the 255-by-24 projection and its bias. Compute the
actual byte cross-entropy gradient \(g_b\) while leaving byte output weights
and all physical recurrent registers frozen:

\[
\Delta\theta_{\perp}=
\begin{cases}
\Delta\theta_s-\frac{g_b^\top\Delta\theta_s}{g_b^\top g_b}g_b,
&g_b^\top\Delta\theta_s>0,\\
\Delta\theta_s,&\text{otherwise.}
\end{cases}
\]

A nonlinear, independent byte-CE gate tests five alpha values
\(1,1/4,1/16,1/64,1/256\) and commits only a nonzero shared update
with no measured Alice DEV regression. Otherwise it retains semantic
heads at alpha=0 and records a shared-update rejection. This experiment
does **not** add a separate byte-gradient descent step.

### Initial DEV-calibrated diagnostic: negative, non-independent gate

[Run 42829538991](https://github.com/shemshallah/nanlm-test/actions/runs/42829538991):
all 18 semantic proposals had positive first-order byte-CE conflict and
all 18 projected line searches were rejected. Dickens CE remained
exactly equal to the byte-only baseline. This diagnostic used Alice DEV
both for gradient projection and as the acceptance gate: that double
use does **not** constitute a strictly independent development gate.

### Corrected TRAIN-calibrated, DEV-gated experiment

[Run 42830808712](https://github.com/shemshallah/nanlm-test/actions/runs/42830808712):
the gradient projection is now computed from Alice TRAIN only, and
nonoverlapping Alice DEV selects the acceptance alpha. The independent
cross-author Dickens test remains entirely excluded from gradients
and selection. This corrected protocol passed native GCC/Clang and
all three pinned training-order seeds.

| Seed | Shared batches accepted | DEV CE byte-only | DEV CE projected | Dickens CE delta |
|---|---:|---:|---:|---:|
| 20261109 | 6/6 | 3.413456847 | 3.412298695 | -0.000198173 |
| 20261212 | 6/6 | 3.446424859 | 3.445514599 | -0.000080712 |
| 20261315 | 6/6 | 3.422222556 | 3.420564737 | -0.000087182 |

The mean cross-author improvement is **-0.000122023 nats/byte** over
the matched byte-only readout. These are nonzero updates to parameters
read by the byte predictor. The native regression now also verifies
the shared byte gradient against an independent three-point
finite-difference derivative
([GCC/Clang run 42830808692](https://github.com/shemshallah/nanlm-test/actions/runs/42830808692)).

**Interpretation boundaries:** The positive result is a repeatable,
small proof of mechanism on one model initialization, one fixed pair of
books and three correlated training-order seeds. Labels remain
synthetic rather than Gutenberg-acquired. Changes could arise from
optimization/regularization rather than meaningful role/synset content.
Role generalization is weak. It is not statistical significance,
general semantic understanding or a population-scale result.

### Required matched label control

The seventh arm added in commit
[e461150d](https://github.com/shemshallah/nanlm-test/commit/e461150d2769257620931646dd010aa17e07f6f9)
runs the **same train-gradient projection, alpha schedule, byte warm-up,
Alice DEV gate and training order with permanently permuted semantic
labels**. Compare true-vs-shuffled projected shared updates on heldout
Dickens byte CE and synthetic semantic test CE. Until the matched
negative control is verified, do **not** attribute the tiny positive
cross-author byte improvement specifically to correct semantic labels.


### Matched shuffled-label projected-gradient result: semantic attribution fails

The seventh-arm source-pinned Gutenberg audit
[Actions #42831634140](https://github.com/shemshallah/nanlm-test/actions/runs/42831634140)
passed compilation, its three-seed native comparison, strict accounting,
corpus checks and report generation. The correct-label projected arm and
fixed-permutation shuffled-label projected arm had **identical** byte warmup,
synthetic example order, projection algorithm (gradients computed exclusively
on Alice TRAIN), fixed line search and Alice DEV gate. Dickens was excluded
from all gradient and acceptance decisions.

| Seed | Correct-label projected batches accepted | Shuffled-label projected batches accepted | Dickens CE delta: correct vs byte-only | Dickens CE delta: shuffled vs byte-only | Correct minus shuffled |
|---|---:|---:|---:|---:|---:|
| 20261109 | 6 | 0 | -0.000198173 | 0.000000000 | -0.000198173 |
| 20261212 | 6 | 6 | -0.000080712 | -0.002625949 | +0.002545237 |
| 20261315 | 6 | 1 | -0.000087182 | -0.000119058 | +0.000031876 |
| **Mean** | **6** | **2.33** | **-0.000122023** | **-0.000915003** | **+0.000792980** |

The shuffled semantic supervision achieved **greater mean heldout Dickens byte
CE improvement** even though its synthetic semantic predictions were often
worse. Therefore the preceding proof of nonzero shared gradient and
cross-author improvement is **not evidence that correct semantic labels
improve language transfer**. A plausible alternative is that projection
surgery plus an adaptive byte-development gate can act as regularization,
independently of task label correctness. The fixed heldout evaluation also
remains too small and correlated to establish significance.

Positive, bounded observations that survive this falsification are:
1. Correct-label semantic training produced an actual nonzero change to
   parameters read by the language predictor in 18/18 gated batches.
2. Strict disjoint Alice DEV nonregression held, and Dickens byte CE improved
   slightly versus byte-only for each of the three train-order seeds.
3. A **strong, fully matched shuffled-label control** matched or outperformed
   the correct-label variant on Dickens CE in two of three seeds and by
   0.000792980 nats/byte in the three-seed average.

The open scientific target is now more specific: *identify real context-
conditioned semantic features and an optimizer whose transfer gains beat
matched shuffled labels on separately selected model initializations and
corpus/test splits*. Keep PR #53 experimental and draft. Do not replace a
negative attribution result with a generic "joint semantics improved language"
claim.


## Independent V2 readout initialization: negative attribution replication

Native regression [#42855669538](https://github.com/shemshallah/nanlm-test/actions/runs/42855669538)
passes GCC and Clang with a new independently seeded 24D projection, orientation
and semantic-head initializer. The checksum-protected sidecar records the seed
under the V2 marker; default-seed V1 persistence remains compatible. An
incorrect seed is rejected on reload. Each benchmark seed initializes all
matched readout arms identically while preserving the **same physical root**
and its existing Alice-trained recurrent wave. Thus the experimental variation
is in readout initialization, **not** three independently trained whole models.

[Three-seed pinned audit #42855592846](https://github.com/shemshallah/nanlm-test/actions/runs/42855592846)
passed source hashing, both native compiler builds, strict Alice DEV gates,
heldout evaluation, all sample-count assertions, 8 MiB root and unchanged
symbolic journal.

| Training-order seed | Readout seed | True-label shared batches accepted | Shuffled shared batches accepted | True-label Dickens CE delta | Shuffled Dickens CE delta |
|---|---:|---:|---:|---:|---:|
| 20261109 | 271828 | 6/6 | 6/6 | -0.000003973 | -0.000054249 |
| 20261212 | 376557 | 0/6 | 5/6 | +0.000000000 | +0.000086613 |
| 20261315 | 481286 | 0/6 | 6/6 | +0.000000000 | -0.000162943 |
| **Mean** | — | **2/6** | **5.67/6** | **-0.000001324** | **-0.000043526** |

The shuffled-label model again outperformed the correct-label model on
**mean** cross-author byte CE. The six-of-six true-label acceptance observed
under the original unseeded initializer is **not robust** to new readout
initializations. With these small fixed corpora we cannot attribute language
transfer to meaningful semantic labels or claim generality. The naive
unconstrained joint model also retained a **+0.014481449 nats/byte** mean
Dickens CE penalty in the reseeded audit.

The previously failing 48-step root-reload test was independently diagnosed:
the original probe had been advanced before the graph was mutated and its
spectral/scattering operator refreshed; post-restart replay used the **new**
graph. Correct causal replay produces exactly equal forward/inverse wave,
triad routing and byte prediction on GCC/Clang. That isolated test fix was
merged as [PR #54](https://github.com/shemshallah/nanlm-test/pull/54)
into the parent experimental development branch, without merging semantic
learning into production.

**Next experiment:** [draft PR #55](https://github.com/shemshallah/nanlm-test/pull/55)
derives contextual word-order positive and corrupted examples from actual
pinned Gutenberg training text and matches them against shuffled labels.
This does not equate textual order discrimination with WordNet sense
understanding, but directly tests whether an independently sourced,
language-aligned auxiliary signal is better than arbitrary projected-gradient
regularization.
