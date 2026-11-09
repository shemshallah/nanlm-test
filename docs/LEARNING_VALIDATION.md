# HW0:0 learning-design validation — controlled experiment

## Scope

One mmap root and one teacher instance exist per sequential diagnostic run. The control replicas are independent temporary runs, never additional production universes. The benchmark does not establish open-domain fluency, a WordNet generalization advantage, or non-Abelian superiority. Printed loss values are measurements, not predefined winning thresholds.

Commands:

    make clean test CXX=g++
    make validate CXX=g++
    make validate CXX=clang++

Three deterministic shuffle seeds: 11, 29, 47. Target mapping: y = (x+19) modulo 192. The source heads 16–175 are partitioned by (x-16) modulo 4: 120 conditioning contexts are trained for eight epochs; 40 source contexts are never used as training predictors. Training is within one universe per run, with identical example order and training budget across arms. Target heads may occur as sources on the held-out side, because the output embeddings are shared; source-disjoint does not mean wholly untouched vectors. No head 255 is trained.

## Ablations

| Arm | Dot gain | U(1) gain | SO(3) gain | Interpretation |
| --- | ---: | ---: | ---: | --- |
| flat | 4.45 | 0 | 0 | Shared 3D low-rank dot baseline |
| u1 | 4.45 | 0.08 | 0 | Scalar phase |
| so3 | 4.00 | 0 | 0.45 | Non-Abelian transport |
| hybrid | 4.00 | 0.08 | 0.45 | Both transports |
| count | — | — | — | Add-one smoothed bigram |
| shuffled_labels | 4.00 | 0.08 | 0.45 | Negative-control permuted targets, first seed |

Identity quaternion transport gives the same initial effective dot coefficient in the flat and so3 configurations: 4.00 + 0.45 = 4.45. Scalar phase changes initial logits slightly, so it is not strictly initially equivalent. The count baseline is analysis-only and not another production memory space. Disabled-geometry configurations retain the same head record allocations; this is a feature ablation, not a perfectly capacity-matched conventional LLM.

## Output and falsifiability

CSV records include seen and unseen-source before/after next-byte CE, top-1 accuracy, accepted/attempted updates, the number of post-commit regressions among accepted steps, and the largest gap between a proposed scratch loss and the real committed loss. Each trained model is synchronized, unloaded, reloaded, and checked for identical held-out prediction losses. Missing or nonfinite probabilities and persistence drift fail the test.

Evaluate all three seeds. A win on seen sources alone is memorization, not evidence of rule transfer. A gain on a simple arithmetic rule is not a language result. The acceptance optimizer presently judges proposed head updates before evidence/phase/observation and some quaternion writes, so a trial scratch improvement is not proof of a final improvement. This audit makes such discrepancies measurable.

The PASS line means the protocol ran and persistence checks succeeded. It is not an assertion that geometry won. Publishable geometric superiority additionally requires longer and genuinely unseen text, matched conventional memory, independent evaluation, meaningful effect size, and free-running generation metrics.
