# HW5:0 Gutenberg — Native Learning and Generalization Ledger

**Address:** \`nan://GM0:SC0:MC0:BS0:0:HW5:0\`. **HW5:0 is hardware slot 5, not an experiment counter.** All records below belong to the same Gutenberg-corpus hardware address. Do not allocate an HW6 slot.

## Controlled evidence, in chronological order

| Readout / experiment | Held-out documents | Cross-entropy (nats/byte) | Top-1 | Assessment |
|---|---:|---:|---:|---|
| Native Gutenberg add-0.5 bigram | Original HW5 test: 2 books | 2.4967693 | 27.6431% | Diagnostic count control |
| Train-only native exact triples + mirror memory | Original HW5 test: 2 books | **2.1287557** | **37.0607%** | **Validated memory improvement** |
| Frozen existing geometry + memory (dev-selected positive-geometry config) | Original HW5 test: 2 books | 2.1276304 | 37.0609% | Small exploratory geometric gain, not statistically robust |
| Native train-only CP rank-8 geometry + memory (dev-selected epoch 8, rho=0.01, scale=2) | Original HW5 test: 2 books | 2.1268614 | 37.0606% | Positive on familiar split, not independent generalization |
| Novelty-gated CP8 vs global CP8 | Same original HW5 dev split, 296 choices | Global mix selected (\`gate=0\`) | — | Novelty-only gate not selected |
| Native exact triple memory baseline | **Fresh Gutenberg texts: 4 books** | **2.0039019** | Report artifact | Better than CP8 on fresh source-disjoint prose |
| Frozen native CP8 geometric mixture | **Same fresh 4 books** | 2.0046448 | Report artifact | **Regressed** on fresh prose |

**Published reproducible GitHub Actions run IDs:**
- Original train-only mirror and exact ordered triple success: [42746674921](https://github.com/shemshallah/nanlm-test/actions/runs/42746674921).
- Per-book native geometric attribution, positive/negative support strata and cluster tests: [42748656401](https://github.com/shemshallah/nanlm-test/actions/runs/42748656401).
- Train-only CP8 native SGD/Adam geometric credit: [42749845164](https://github.com/shemshallah/nanlm-test/actions/runs/42749845164).
- Global-vs-novelty gate, 296 dev candidates: [42750439700](https://github.com/shemshallah/nanlm-test/actions/runs/42750439700).
- **Precommitted fresh-book out-of-source audit:** [42751004704](https://github.com/shemshallah/nanlm-test/actions/runs/42751004704).

## Why the first improvement did not generalize reliably

The original two-book held-out attribution, using a **book** rather than a byte as the independent sampling unit, showed:
- Seen exact triples: **1,881,592** transitions, or **99.6963%** of that test. Geometry increased CE from 2.1107823 to 2.1118363 in the original frozen geometric readout.
- Previously seen pair, new continuation: **5,109** transitions. The frozen geometric readout reduced CE from 8.6031074 to 7.8016932.
- Unseen pair contexts: **623** transitions. Geometry reduced CE from 3.3183278 to 3.2980205.
- Whole-book delta: 0.00112535 nats in favor of the frozen geometry on this familiar split; exact book-sign-flip p = **1.0** (two test books).

Training CP8 (rank 8, 3 roles × 8 components = 96 persistent bytes per cognitive head, same native 8 MiB root) improved familiar-split CE from **2.1287557 to 2.1268614** after eight native Adam epochs and dev-only selection among 152 configurations. This establishes a real, reproducible positive familiar-split result; **it does not establish transferable geometry or semantic understanding**.

The novelty-preserving gate conserved known exact-triple probabilities and redistributed only their unknown-mass budget. Its 296-option dev-only search selected **global mixing** instead of novelty gating. This is a real negative selection result.

The fresh-prose audit fixed CP8 parameters and source SHA-256 **before acquisition**. It excluded IDs from all 21 original reference books and chose four valid predeclared candidate texts (Project Gutenberg IDs **608, 5200, 158, 768**), stripped Gutenberg license boilerplate and measured **1,780,743** new byte predictions. The native readouts reported:
- Memory-only mean CE **2.0039018861**
- Frozen CP8 mean CE **2.0046447636**
- \`memory_ce - cp8_ce = -0.0007428775\` nats (negative = memory better).
- Geometric CP8 lost on **3 of 4** books. Exact book-sign-flip p = **0.25**. The four-book bootstrap is too small to establish a reliable population conclusion.

**Result:** Our strongest publishable partial result remains the large improvement from train-only, higher-order native associative memory over bigram controls. Rank-eight geometric learning **has not yet demonstrated a robust out-of-source advantage**. The fresh-book negative result is an essential part of the evidence, not a failure to be hidden.

## Causality, model selection, and invariants

1. Gutenberg training records supply only train-only physical C256 mirror adjacency and lag-two evidence plus sparse ordered triples. Development books choose readout coefficients and CP8 checkpoint epoch. Independent books must never affect fitting or selection.
2. The WordNet→Gutenberg student root and exact memory ledger are SHA-pinned. The native CP8 geometry is fitted in the same logical 8 MiB root's existing 128-byte head registers, leaving all 32,640 physical mirror-pair cells and gateway head 255 unchanged.
3. CP8 is a rank-8 trigram-conditioned probability factorization, not end-to-end semantic grounding, not a full transformer, and not proof of agency or AGI. Its optimizer is in-memory training state and does not instantiate a new persistent root.
4. The original held-out test was inspected repeatedly; new variants evaluated on it are exploratory even when test parameters were frozen before a given run. Independent fresh books are stronger evidence, although four books remain too few for confident generalization claims.
5. All samples, model keys, checkpoints, source hashes, train/dev/fresh split lineage, original native logs, score reports, and unit tests are retained in Actions artifacts or source control.

## Next experiments (precommit evaluation protocol before inspecting new outcomes)

- **Book-disjoint train-source cross-validation**: multiple genuinely excluded books per fold, re-build mirror memory without the fold's books, fit CP8 without seeing held-out fold labels, select hyperparameters on a different development fold. Evaluate transfers by book.
- **Train-only pseudo-unseen next-byte objective**: mask some source triple continuations during credit assignment so geometry is explicitly rewarded for source-disjoint or masked-context generalization rather than just fitting seen exact counts.
- **Source-conditioned calibration and abstention**: compare whether geometric contributions should be suppressed when lexical evidence is strong, and measure confidence calibration, not just top-1.
- **Semantic validation on facts/relations**: separate structured signed ISA and compositional relation tasks from byte-trigram prediction. Do not equate lower CE with semantic understanding.
- **New predeclared Gutenberg books** for subsequent confirmatory tests; **do not tune against IDs 608/5200/158/768** after observing their scores.

No additional hardware address should be allocated for these stages: all remain **HW5:0**.
