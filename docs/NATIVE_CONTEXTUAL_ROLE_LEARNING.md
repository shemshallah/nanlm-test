# EXPERIMENT — Native contextual relation-credit v1

This branch extends PR #44 on the **same NaNLM Universe root**; it does not merge a separate semantic service or change the 8 MiB physical address space. The experimental model persists as an integrity-checked sparse sidecar at ROOT.context-role. Inference and training run in C++20, and both clang++ and g++ are CI targets.

## What the gradient actually updates

Each supervised sample is a five-column TSV:

    SENTENCE<TAB>SUBJECT<TAB>OBJECT<TAB>RELATION<TAB>LABEL_0_OR_1

A sentence is tokenized *in order*. The two exact subject/object mentions must each occur exactly once. Each nonentity lexical token is assigned a position slot based on its placement relative to BOTH entity mentions. A shared 24D hidden state is then computed as:

    h = tanh(b + s o + (1/sqrt(T)) SUM_j g[slot(j)] ⊙ e[token(j)])
    p(y=1 | text, sub, obj, rel) = sigmoid(b_rel + r_rel · h)

Here s is the subject-before-object orientation, o is a learned orientation vector, g are learned slot gates, e are learned lexical vectors and r is a learned typed relation vector. All five trainable parameter families receive native analytic cross-entropy gradients. There is **no exact tuple lookup** in this classifier. Entity identities determine role-mention positions but their names are deliberately omitted from the lexical state, enabling a test of generalization to new entity names.

The model supports **binary classification of explicitly named queried roles**, not arbitrary free-form prose interpretation, unobserved grammatical constructions, cross-sentence coreference, or automatically inferred WordNet senses. It accepts ASCII word tokens in the bounded initial experiment. Unseen relation keys abstain rather than acquire fabricated truth.

Source-attributed signed graph claims in ROOT.semantic.tsv and exact WordNet nodes in ROOT.wordnet.tsv remain separate canonical stores. This supervised contextual model does not insert its probabilistic predictions into either exact knowledge ledger. It does not yet backpropagate into UnifiedWaveLanguage's next-byte parameters; that coupling, with byte-performance regression controls, remains the next experiment. A shared root path does not by itself imply shared model parameters.

## Reproducible evaluation

The fixed native test generates entity-disjoint training (288), validation (144) and test (144) *binary classification examples*. It contains 12 controlled transfer/follow sentence patterns: each has an asserted forward role and a matched query with the subject/object roles reversed. Thus a trivial always-positive predictor is 50%. Test entity strings do not appear in training, but the same lexical patterns occur in all splits, so this is NOT a held-out-author or unseen-grammar evaluation.

Controls:

1. Exact role reversal for every test pair.
2. Labels shuffled on the original training sentences with a fixed seed; require held-out advantage over corrupted supervision.
3. Cross-compiler compilation under C++20 with -Werror.
4. Save/reload equivalence for exact numeric outputs and update count.
5. Checksum rejection of a corrupted saved checkpoint.
6. No writes to the canonical Universe event journal from training the probabilistic overlay.
7. Unknown typed relation abstention.

The held-out test is artificial and designed to isolate the missing role-learning mechanism. It must not be relabeled as Gutenberg transfer, CLUTRR performance, general semantic acquisition, or proof of geometric superiority.

## Run

    make nanlm test_contextual_roles CXX=clang++
    ./test_contextual_roles
    ./nanlm /tmp/one-root init
    ./nanlm /tmp/one-root context-train train.tsv 90 0.065
    ./nanlm /tmp/one-root context-eval heldout.tsv
    ./nanlm /tmp/one-root context-query "alice sent parcel to bob" alice bob transfer

The fixed train/dev/test fixture is generated inside tests/test_contextual_roles.cpp. Evaluate on heldout TSV without training on it.

## Next controlled coupling

Learn a shared sequence state that feeds both byte generation and semantic classification, and differentiate a semantic loss through that SAME state. Then evaluate held-out title/author byte loss, disjoint semantic entities, role reversal, OOV lexical cues, shuffled labels, and root/ledger rollback. Do not silently feed model predictions into the authoritative symbolic store.
