# NaNLM native geometric-credit learning: 1,000-epoch Gutenberg transfer audit

## Status

**Publishable partial experimental result (narrow benchmark only).** Three consecutive GitHub Actions runs are completed and successful, with pinned source/corpus, an unchanged eight-megabyte student lineage, native checks, recorded model SHA-256, and a title-disjoint book evaluation. This supports a *low-rank conditional-byte prediction* claim. It does not establish semantic or recurrent language understanding or superiority to the count-based baseline.

## Scope and method

- Original source root: read-only NaNROUTER HW5:0 Gutenberg C(256,2) reference, archive SHA-256 `906a9030a11a67cc2635a6700344826d4c924e49596e57949a969cf78e1a7a18`.
- The complete source is **20,597,424 bytes** from **21 books**, split by book title: **17,571,596 train**, **1,138,500 dev**, and **1,887,328 test** bytes. Dev/test excluded from Gutenberg optimizer input.
- Student: one continuous **8,388,608-byte** NaNLM root initially restored from verified **WordNet epoch-103** (SHA-256 `e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1`).
- Every Gutenberg step scans **the complete train split** and performs **one rank-three aggregated conditional-byte geometric-credit proposal** on the two FabricTeacher head vectors, checked by a native improvement gate. No BPTT, no per-byte recurrent optimizer, and no claim of an online temporal training step at each source byte.
- Validation: read-only dev evaluations and dev-only best-checkpoint selection; final test evaluated after selection in each continuation. **The same test set was already evaluated in earlier experiments and therefore is not a virgin holdout for future experiment design.**
- Separate count control: add-0.5 smoothed full-train byte bigram table, with the same document-disjoint dev/test split. Control has many more free parameters than rank-three head factorization; it is a predictive baseline, not a parameter-matched architecture ablation.

## Reproducible checkpoint lineage

| Gutenberg epoch | Verified GitHub Actions | Student root SHA-256 |
| ---: | --- | --- |
| 0 | [WordNet-103 source run](https://github.com/shemshallah/nanlm-test/actions/runs/42741208752) | `e10325f7f9ba71cd9e43419722f98b417e86e59ac74b15885478f831beca88f1` |
| 20 | [HW5 transfer](https://github.com/shemshallah/nanlm-test/actions/runs/42744089393) | `83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c` |
| 200 | [HW5 longitudinal](https://github.com/shemshallah/nanlm-test/actions/runs/42745053257) | `a975fc792ec6b505d9afb41424c2ea7e52649e61eed43070d217e77ccc4d9df1` |
| 1,000 | [HW5 convergence](https://github.com/shemshallah/nanlm-test/actions/runs/42745423118) | `44b9615ce25139ff9dd538e5590ec1855967084cbc8c2172bada2df163e12eb5` |

## Actual native predictive performance

*CE in natural-log nats per adjacent-byte transition; accuracy is top-1 next-byte prediction. Lower CE is better.*

| Checkpoint | Train CE | Unseen-book dev CE | Unseen-book test CE | Unseen-book test top-1 |
| --- | ---: | ---: | ---: | ---: |
| WordNet-103 before Gutenberg transfer | 3.197934584 | 3.115421204 | 3.188367801 | 20.8118% |
| Gutenberg epoch 20 | 2.928596716 | 2.849494061 | 2.939172723 | 22.9201% |
| Gutenberg epoch 200 | 2.730788180 | 2.699913994 | 2.731607700 | 26.5539% |
| Gutenberg epoch 1,000 | 2.691954681 | 2.676346498 | 2.702507209 | 26.7051% |
| Smoothed byte-count control | — | 2.474040507 | **2.496766480** | **27.6431%** |

Cumulatively, Gutenberg training processed **17,571,596,000 training bytes** (1,000 complete train-split passes), counted as byte scans, not as 17.6 billion independently updated online tokens. Across the epoch-200 to epoch-1000 segment, the native trainer spent **76.4383 seconds** on 800 scans and accepted all 800 proposals. This excludes compilation, source retrieval, integrity checks, and read-only evaluations.

### Findings

1. **Cross-domain conditional-byte transfer is observed**: Gutenberg test CE improves from 3.188367801 (the original WordNet-trained student) to 2.702507209 (after Gutenberg epoch 1000), and test top-1 rises from 20.8118% to 26.7051%.
2. **The learning curve is flattening**: test CE improvement across 20→200 epochs is 0.207565023 nats, versus just 0.029100491 nats across 200→1000 epochs (despite 800 more scans). Dev CE still improved at the latest checkpoint, but the marginal returns are now small.
3. **The simpler count control is not beaten**: at epoch 1000, rank-three native geometry has test CE 2.702507209 vs control 2.496766480, and test top-1 26.7051% vs control 27.6431%.
4. **Scope is narrow**: byte pairs cannot establish semantic role decoding, true cross-sentence memory, robust generation, or concept reasoning. The slow HW4 TRAIL and other recurrent paths are distinct experiments, with different computational budgets and objectives.
5. **Metric selection caveat**: dev sets were used for checkpoint selection; test books were excluded from optimizer input, but have been reported multiple times in successive runs. A fully new book/author split and explicit seed controls are needed before strong generalization claims.

## Next diagnostic tests before changing architecture

- Rank-three versus rank-eight/rank-sixteen *parameter-matched* factorized heads, with equivalent optimizer and corpus budgets; preserve each baseline's free parameter count and throughput.
- A new Gutenberg author/title split not viewed in prior experiments, along with stratified confidence intervals across books (not across correlated individual bytes).
- A shuffled-transition and unigram-only negative control, plus provenance and root-hash audits after all read-only evaluations.
- Separate genuinely sequential recurrent/temporal memory and semantic retrieval tasks; do not re-label the byte-pair success as whole-model cognitive learning.

These are experimental benchmark results suitable for a short reproducibility note, **not** proof of a general language model or a scientific breakthrough in semantics.
