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

