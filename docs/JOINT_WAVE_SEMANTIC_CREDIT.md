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

Gradients are analytic through the shared projection and heads; they stop at the **fixed** wave input x. No semantic gradient goes through the oscillator recurrence, triad routing, FabricTeacher, or physical NaN16 pair cells. This is not full BPTT or unsupervised semantics.

All joint readout parameters persist as an FNV-integrity-checked, atomic-renamed single experimental sidecar (ROOT.joint-wave), with bounded parser and corruption rejection. Saving a sidecar is not a transactional checkpoint of all root components. A separate byte-only and semantic-shuffled model runs on the **same frozen underlying wave**, not a second universe.

## Native reproduction

    g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude tests/test_joint_wave_semantics.cpp -o test_joint_wave_semantics
    ./test_joint_wave_semantics

Run both GCC and Clang in the native-joint-wave-semantics.yml workflow. The fixed synthetic fixture has role-reversal pairs with names absent from training, two exact bank WordNet sense IDs, held-out sentence recombinations and a separate held-out byte string. Training data are explicitly synthetic and small: no Gutenberg semantics claim follows.

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

The native test uses a complete eight-byte causal tape and independently perturbs the *existing* input-gain register by plus/minus 0.005, replaying the exact physical wave from the tape's fixed starting state. Raw original gain-register bytes, including the magic marker, are restored after each test. The analytic role and sense derivatives must agree with the central difference within a numerical tolerance. The test does **not** yet accept any persistent recurrent update. A gradient audit, even when it succeeds, does not prove a safe optimizer or a general language benefit.

## Next required experiment

1. Connect the preceding shared contextual lexical embeddings to the wave readout, preserving end-to-end derivatives to both roles and senses instead of creating parallel encoders.
2. After the read-only semantic recurrence gradient is verified, implement guarded, replay-validated root updates and compare with detached readout on heldout byte loss.
3. Compare multiple fixed seeds and label permutations; evaluate unseen templates, unseen predicates and real Gutenberg heldout books, with corpus disjointness and leakage controls.
4. Gate acceptance on non-interference in language CE and exact WordNet reasoning. Keep exact claims separate from probabilistic outputs.
