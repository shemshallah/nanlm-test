# Source-grounded contextual-order credit

**Status: experimental; stacked on draft PR #53.** This experiment replaces
invented supervision examples with a task derived from the actual, pinned Alice
training split and evaluated on independent Dickens excerpts.

## Research question

Does a language-compatible, source-derived contextual gradient beat a fully
matched permanently shuffled-label control on unseen-book contextual-order
classification and next-byte cross-entropy?

## Corpus and positive/negative labels

The immutable GITenberg Alice and Dickens source revisions, explicit
character-offset splits and SHA-256 manifest are inherited from the existing
Gutenberg benchmark. Book text is downloaded only during CI; no source
passages are checked into the repository.

The native C++ test divides each book fragment into sentence segments and
extracts twelve consecutive ASCII tokens at fixed stride. Both endpoint
anchors must be distinct and unique. For each source window:

- Positive label: original source token sequence
- Negative label: same sequence with the fourth and ninth interior tokens
  swapped
- Anchor words, their order and their positions stay identical in both
  cases, preventing a trivial anchor-orientation answer

This is **word-order compatibility learned from a public-domain source**,
not a gold WordNet sense annotation, a semantic role ground truth, or a
natural-language-understanding benchmark. Corrupted passages may sometimes
remain plausible; report the limitations.

## Four matched readouts, one physical root

All training uses the existing single 8 MiB NaN16 Universe, its fixed
UnifiedWave recurrent state, and a persistable experimental V2 readout
initializer. Every variant within a seed has identical byte warmup.

1. Byte-only language-control readout
2. Correct source-order classification, with projected shared updates
3. Permanently shuffled labels, same examples, learning rates, projection,
   line search, and acceptance policy
4. Correct-label semantic-head-only training on the frozen byte encoder

The byte-loss gradient used for projected-gradient surgery is computed only
from Alice TRAIN. The independent Alice DEV split is used only to accept or
reject nonzero shared updates, requiring zero byte CE regression.
Dickens is never an optimizer or selection input.

Test metrics: positive/negative Dickens contextual classification CE and
accuracy, and original (uncorrupted) Dickens next-byte CE. All shared-update
acceptances/rejections and exact-journal invariance are reported.

## Reproduction

Run the pinned corpus preparer, then compile and execute the benchmark:

    g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude tools/grounded_context_credit.cpp -o grounded_credit
    ./grounded_credit grounded-corpus/train_alice.txt grounded-corpus/dev_alice.txt grounded-corpus/test_carol_early.txt grounded-corpus/test_carol_late.txt 3 > grounded-credit-receipt.log
    python3 tools/report_grounded_context.py grounded-credit-receipt.log grounded-corpus/manifest.json grounded-credit-report.json

CI additionally compiles Clang under warnings-as-errors and independently
executes two seeds. Reports retain all negative outcomes.

## Scientific interpretation

A language CE improvement that also appears under shuffled labels does not
establish learned semantics. Correct labels must reliably beat matched
randomized labels on contextual heldout performance and preferably on
cross-author byte CE across more independent initializations, corpora, and
corruption families before broader credit is claimed. Three correlated
training-order seeds on one fixed pair of books cannot support population
significance or general semantic competence.
