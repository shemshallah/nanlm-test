# Shared context WordNet-sense probe (stacked on native role learning)

This experimental branch adds trainable **exact WordNet sense disambiguation** on the same root and the same lexical/positional embedding arrays as the native role encoder (PR #50). It does **not** create another Universe or modify the invariant 8 MiB physical root. The output heads are keyed by **full uint32 synset identities** drawn from WordNet's exact candidate list, never lossy 8-bit head indices.

For a unique occurrence of an ambiguous target word, context words on either side are read into the **same trainable lexical embeddings and position gates** used for role classification. A 24D tanh context vector is compared to exact-synset output vectors by a candidate-masked softmax. Cross-entropy updates the candidate synset vectors, their biases, the lexical embeddings, the position gates and the shared context bias. All training and evaluation is native C++.

The model stores a checksum-protected ROOT.context-senses sidecar in addition to the **shared** ROOT.context-role checkpoint; a header step count refuses to load a misaligned pair. Each individual file is written by atomic rename, but the two files are **not yet one atomic transaction**; a crash between writes needs a recovery/checkpoint manifest.

## CLI

    make nanlm test_contextual_senses CXX=clang++
    ./test_contextual_senses
    ./nanlm /tmp/one-root sense-train training-sense.tsv 120 0.07
    ./nanlm /tmp/one-root sense-eval heldout-sense.tsv
    ./nanlm /tmp/one-root sense-query "the bank approved a loan" bank

Sense TSV lines are: SENTENCE<TAB>TARGET_WORD<TAB>EXACT_WORDNET_SYNSET_UINT32. Requires an existing ROOT.wordnet.tsv. Rejects unknown gold senses and disambiguation when the lexical target appears zero or multiple times. Untaught candidates cause abstention.

## Falsifiable scope

The fixed fixture has two *synthetic WordNet* synsets sharing "bank", one financial and one river. The test compares held-out sentence word-order combinations with a class-balanced shuffled-label negative control; confirms full ID fidelity, checkpoint persistence, source roots unchanged and corruption rejection. Distinguishing two synthetic senses is **not a full WordNet WSD benchmark**, not induction of senses from unlabeled prose and not end-to-end language understanding. Test templates reuse lexical cue words in training; that is deliberate and must be identified as limited in-domain transfer.

The next scientific audit should use externally labeled WordNet sense contexts with split-by-sentence and out-of-domain contexts, plus the same model's retained role accuracy before/after sense training and a learned semantic gradient through the *production* UnifiedWave sequence state. No such result is claimed here.
