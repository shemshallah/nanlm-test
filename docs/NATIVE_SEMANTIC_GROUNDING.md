# NaNLM native semantic grounding: bounded prototype and route to learned meaning

## Root-cause diagnosis

The successful Gutenberg rank-three trainer collapses each complete training split into a 255-by-255 conditional-byte transition histogram and then updates source/target head vectors. Every conditional distribution depends only on the immediately prior byte. This training path cannot preserve the entity, grammatical roles, long-range context, discourse state, or causal relations necessary for propositional meaning. Increasing rank may improve conditional byte modeling without addressing information loss. The recurrent/wave architecture elsewhere in the repository is distinct from this aggregate trainer and should not be conflated with it.

## Implemented native bridge

New C++ header: `include/nanlm/semantic.hpp`, using the existing `Universe` and `WordNetIndex`. The same `nanlm ROOT` executable adds:

```sh
make nanlm test_semantic_bridge CXX=clang++
./test_semantic_bridge
./nanlm /tmp/one-root import-wordnet tests/fixtures/data.noun
printf 'Entity can exist.\nObject cannot exist.\nArtifact can exist.\n' > /tmp/claims.txt
./nanlm /tmp/one-root semantic-train /tmp/claims.txt fixture-1
./nanlm /tmp/one-root semantic-ask isa artifact entity
./nanlm /tmp/one-root semantic-ask ability artifact exist
```

- Reads exact WordNet noun synset IDs, synonyms and typed ISA edges from the existing `ROOT.wordnet.tsv` index.
- Ingests **explicitly supervised controlled-English** `X can VERB`, `X cannot VERB`, `X is a Y`, `X is not a Y`.
- Rejects unrecognized syntax, unknown lemmas and ambiguous senses; explicit `n:ID` works for known synsets.
- Persists claims, polarity and provenance in the exact `ROOT.semantic.tsv` sidecar and also projects source-attributed typed evidence into the existing same-root NaN16 mirror lanes.
- Answers transitive ISA proofs and feature inheritance across 2+ unseen graph edges; uses nearest-specificity exception selection, while preserving equally specific support+refutation as **contested**.
- Returns `0 unknown, 1 supports, 2 refutes, 3 both`, plus path depth and source counts.
- Does not instantiate independent production models, conflate hash/head collisions with synsets, or claim that controlling a parser constitutes broad learned language comprehension.

## Explicit falsifiable tests

The native regression checks 3-hop ISA entailment absent as direct edge; reverse-direction negative control; inheritance; closer contradictory exceptions; equally specific contested facts; unknown predicate abstention; WordNet synonym identity; provenance idempotence; unsupported English refusal; source persistence and no duplicate NaN16 projections after restart. Two compilers plus CLI run in GitHub Actions.

This is a **symbolic semantic inference milestone**, not an end-to-end learned semantics result. No held-out free-form natural language benchmark has been passed, and no evidence yet shows a learned world model.

## Parallel conceptual tracks

| Track | Role in unified mind | Required independent validation |
| --- | --- | --- |
| Context encoder | Replace order-one byte aggregation with word/phrase/sentence state and attention | Agent/patient swaps and long-distance coreference |
| Lexical grounding | Align multiple forms to exact WordNet senses without 8-bit collisions | Held-out polysemy/context sense labeling |
| Relation learning | Learn subject-relation-object representations and typed forward/inverse operators from graph supervision | Held-out relation edges, unseen entity compositions, hard corruptions |
| Explicit memory | Keep source-tagged, time-aware factual/negative assertions in one logical universe | Conflict preservation, counterfactual replay and provenance |
| Symbolic/latent reasoning | Compose typed graph paths with continuous geometric similarities | 2–6-hop inference, noisy evidence, negative and disconnected controls |
| World modeling | Predict intervened entity state transitions, not just next byte | Out-of-distribution causal and temporal tasks |
| Retrieval-grounded decoding | Fetch proofs and memories before saying facts | Faithfulness, calibrated abstention and unsupported-claim detection |
| Distributed transactions | Atomically merge source-tagged contributions to one shared logical space | Replay determinism, rollback, no lost time steps |
| Joint optimizer | Preserve byte fluency while optimizing roles, entailment, denoising and contrastive objectives | Dev-heldout semantic gains without regression on independent prose |

## What must be fixed next

1. Replace the *aggregate bigram-only training objective* in the semantic path with sequence-aware native encoders that read token boundaries and subject/object roles; do not reuse a 255-by-255 histogram as a surrogate for semantic supervision.
2. Implement signed typed relation contrastive learning on genuine WordNet train edges, with **train/development/test splits by connected subgraphs/entities** (and a separate new author set), not a trivial random negative table.
3. Couple proof retrieval to output generation via a gated, auditable interface rather than silently inserting asserted facts as true.
4. Add negative controls that scramble labels, swap roles and erase 1 hop. Compare exact graph search, low-rank relation embeddings, and sequence predictors under controlled computation budgets.
5. Introduce atomic root + overlay checkpoint manifests: the current sparse semantic ledger uses rename-on-save, but the root, event journal and ledger are not transactionally committed as a unit.

The prototype intentionally separates facts taught in a controlled grammar, logical inference from those facts, and meaning learned from unconstrained prose. Treating any of these as interchangeable would invalidate the experiment.

## Full HW4 WordNet semantic-scale evaluation

The SHA-verified immutable HW4 SQLite source is converted by `tools/hw4_semantic_export.py` into an exact noun-synset index, with source-derived canonical IDs and typed ISA edges. The independent Python converter generates 300 shortest 3–5-hop positive query pairs not directly stated as edges and 300 unreachable negative controls. **Native C++ performs the evaluated inference**, not Python.

The `nanlm ROOT semantic-project-wordnet` command explicitly projects all exact WordNet edges into the existing 8 MiB C256 NaN16 mirror fabric, including reverse typed lanes. It refuses to bootstrap into an already populated event journal. The exact sparse concept/edge index stays authoritative because physical head projections may collide.

The `semantic-batch` command checks every label against an independent gold TSV, returns nonzero on error, and reports total accuracy, positives, negatives, mismatches, synsets and edges. The CI validates that read-only batch evaluation leaves the physical 8 MiB learner root unchanged. The full-scale test exists in `.github/workflows/native-semantic-grounding.yml` and in the separate `hw4-semantic-full-audit.yml` experiment.

**Scientific scope:** composed query pairs were absent as direct edges, but constituent hypernym edges were loaded. Passing shows graph composition and error abstention, *not* automatically acquired meaning from Gutenberg prose. Do not claim full-WordNet CI success before its results are verified.

## Distinct learning objectives

1. Aggregate rank-three byte-pair geometry estimates order-one byte probabilities. It cannot model multiword roles merely by adding epochs.
2. Train-only exact trigram memory improves held-out byte prediction by preserving one additional byte of causal context; it is still count-based memory, not a trained semantic encoder.
3. The native WordNet bridge provides exact supervised symbolic meaning and multi-hop reasoning.
4. The **remaining open objective** is to train contextual role/sense embeddings and typed relation operators *from raw linguistic context*, coupled to native recurrent/attention states and evaluated on genuinely new concepts, roles, sources and authors.

Do not silently equate any one of these with the others.
