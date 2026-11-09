# NaNLM: three-resolution mathematical sensor memory (experiment)

**Non-production test branch.** Existing HW0 8 MiB universe, 255-head predictor and NaN16 root layout remain unchanged. This experiment exercises a host-side mathematical sensor and source-aware hypergraph overlay. It is not an additional LLM.

## Three resolutions, one logical information space

1. **255 head branches:** a head is a mathematical detector/address, not one fixed byte, word or concept. Each branch contains bounded keyed leaves. A leaf has a 256-bit role-bound vector, a 3x3 tensor, explicit positive/refuting charge, activation, last tick and source identities.
2. **85 triads:** sum tensor/charge/activity over the head mapping, preserve phase/triangle holonomy; mapping may be imported from native adaptive-triad registers. No separate expert models.
3. **One global field:** a single 3x3 aggregate tensor and signed evidence channels, plus exact counts of unique facts, independent source names and the NaNclock tick.

A sensor layer binds the subject/predicate/object with distinct role salts, then routes the resultant hypervector to the six nearest detector heads. Rotated three-component feature vectors form second-order tensor products. Six is a probe setting, not a theoretical optimum. Random hash routing **does not yet learn synonyms or semantics**. Triangle curvature initially uses deterministic antisymmetric phases; \`import_native_geometry\` can read live \`Universe::triangle\`.

## Evidence vs activation

The authoritative ordered typed hyperedge graph preserves *positive and negative evidence independently*: state 0 = unknown, 1 = support, 2 = refutation, 3 = both. A zero activation is **not** a refutation. A source contributing the same claim twice is deduplicated. Individual head leaves may evict older tensors under their budget, but exact claims and proof provenance persist in the graph and journal.

Idle \`tick()\` decays transient sensor/tensor activities and recomputes the global field **without deleting facts**. This is a callable clock mechanism, not an always-running hosted service. Persistence is a versioned TSV replay snapshot with temp-file rename. No fsync, checksum, cross-file transaction or 8 MiB native overlay integration is claimed.

## Reasoning test

Source A states \`penguin ISA bird\`; B states \`bird ISA animal\`; C refutes \`penguin ISA animal\`. The graph composes A and B to infer the conclusion, records both supporting sources, and keeps C's direct refutation. Status is **3 (both)**, without entailing unrelated claims. Only the \`ISA\` relation is declared transitive; source names are provenance, not guaranteed evidence independence. Proof search is capped at 5 hops and 4096 expansions. This is **restricted typed symbolic inference**, not general-language comprehension.

The fixture also has competing claims about flying. A repeated source cannot vote twice. Tests verify exact replay, 255/85/1 activity, bounded leaves, ingress head 255 exclusion, native membership mapping, source-preserving inference, contradiction non-explosion and clock-only decay.

## Mathematical containers to test next

| Container | Mathematical use | Risk |
|---|---|---|
| Binary VSA and FHRR | role/filler binding, superpositions, phase | approximate cleanup; hash collision |
| Exterior/geometric algebra | oriented bivectors, triangle area, chirality | high-grade expansion |
| Hodge / simplicial chains | vertex gradient, edge curl, harmonic cycles | costly incidence and gauge control |
| Cellular sheaves | local sensor views with global consistency | untrained restriction operators |
| Low-rank CP/Tucker/tensor-train | third- and fourth-order relation tensor | rank-collapse / missed conjunctions |
| Factor graphs and proof semirings | typed multi-source inference with provenance | rule soundness and search scaling |
| Hyperbolic product spaces | inheritance and taxonomies | geometry does not certify facts |
| Graph spectral wavelets | multiscale mirror topology signals | eigensolver costs |
| Persistent homology | persistent loops / robust motifs | best computed on slow ticks |
| Reservoirs and Koopman operators | event dynamics and timescales | cannot preserve exact distant facts |
| Sparse distributed memory | content-addressed activation | threshold and capacity tuning |
| Quantized residual codes | separate abstraction and lossless reconstruction | training, bit budget |

## Corpus tokenization experiments

Keep parallel channels rather than one global segmentation: UTF-8 bytes (lossless); graphemes/Unicode with fallback; BPE or unigram subwords; words and morphological features; variable-length n-grams; typed subject–relation–object events; source-tagged temporal assertions; graph motifs; learned vector-quantization codes with residual; and entropy/change-point segmentation.

**Bootstrap output boundary:** corpus bytes + source-tagged claims -> mathematical leaf/tensor/curvature detectors -> 255 heads -> 85 triads -> global field + proof-bearing hypergraph -> symbolic answer packet -> HW0 conditional language decoder -> UTF-8. The native HW0 generator exists, but **learning a language decoder from this proof packet is not implemented here**. Invertible byte transport and semantic reasoning must remain separate until trained, evaluated and audited.

## Test commands

\`\`\`sh
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/test_sensor_memory.cpp -o test_sensor_memory
./test_sensor_memory
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/sensor_memory_probe.cpp -o sensor-memory-probe
./sensor-memory-probe tests/fixtures/sensor_sources.tsv penguin ISA animal
\`\`\`

GCC and Clang are covered by the branch workflow. Before claiming language improvement, integrate the native hypergraph register journal, implement source-aware fact extraction, test entity resolution, learn a proof-to-language decoder, run source-disjoint evaluation and measure CE, retrieval F1, latency, memory, and multi-source reasoning accuracy.


## Native ABI bridge and always-resident protocol

The new tests/test_sensor_native_bridge.cpp constructs a real mapped Universe
and UnifiedWaveLanguage, imports live head_triad() membership and triangle()
curvature, checks NaN16 hypermatrix leg/inverse address equality, and ensures
experimental intake does not write a single production root byte or native
hyperedge. The root remains exactly 8,388,608 bytes. This demonstrates
read-only adapter compatibility, NOT native persistence.

The src/sensor_resident.cpp component maintains 255/85/1 state while running.
It accepts line-oriented tab-delimited commands:

- INGEST [source] [+/-] [subject] [relation] [object]
- QUERY [subject] [relation] [object]
- TICK, STATUS, SYNC, QUIT

Each INGEST snapshots newly accepted observations immediately. TICK changes
transient activity without erasing exact graph evidence. CI launches a second
process from the snapshot and checks the cross-source proof survives restart.

This is live only while a host runs the resident process. It is a separate
sidecar, not the native atomic .overlay format. Rename alone is not fsync-safe.
Native NaNclock scheduling, network intake, CRC and transactional journaling
are required before calling this an always-on permanent native overlay.
