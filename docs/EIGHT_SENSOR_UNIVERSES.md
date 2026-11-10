# NaNLM: Eight mathematical sensor universes (V1–V8)

Status: experimental numeric projections. No trained universes and no language-model improvement is claimed. The source is include/nanlm/sensor_family_maps.hpp with tests/test_sensor_family_maps.cpp.

## Shared observation contract

All eight sensors consume the identical corpus, event ordering, source identities, NaNclock and dynamic head-to-triad membership. Every frame has 255 point heads (experimental v1 IDs 0..254), 85 triheads, dynamic hot triheads (K<=12, original IDs not hot slots), a global field over ALL 85 triheads, and a separately measured hot field. Signed support and refutation must remain distinct. MATS source frame retains C(256,2)=32640 canonical mirror-link location and oriented flux; the new family codec exports its sparse link COUNT, not the full link matrix. A consumer must reference the original MATS slice to recover full links.

Important version distinction: head 255 is reserved for ingress in nanlm-test v1, whereas head 255 is the field counter in NaNLMind v2. The v2 router also reserves control role views around head 0. An explicit cross-repo role/address adapter is required; the family prototype does NOT rewrite native heads or introduce a second language model.

## Six measured features per row

| Family | Six metrics | Limits |
|---|---|---|
| V1 Vector | magnitude, unit x/y/z, xy angle, activity | Direction undefined for zero vector; zero is exported by convention |
| V2 Tensor | trace, Frobenius norm, determinant, deviatoric norm, Qxx, Qxy | Eigenmodes, rank and contractions beyond trace deferred |
| V3 Trigonometry | xy phase, wrapped angular speed per tick, cosine, sine, magnitude, has-previous flag | Phase is a projected diagnostic, not intrinsic global synchronization |
| V4 Spectral | mean magnitude, three DFT powers, dominant nonzero bin, contiguous-window length | Max eight contiguous tick samples; gaps reset spectrum; not a graph eigensolver |
| V5 Temporal | magnitude, first/second finite differences, elapsed ticks, continuity weight, previous hot activation | Does not assert Markov structure or learned dynamics |
| V6 Topology | hot-link degree, stored triad curvature, total links, magnitude, activity, reserved | Actual cycle/Betti/persistence invariants not implemented yet |
| V7 Information | binary support/refute entropy, conflict fraction, total charge, log(1+magnitude), activity, source count | Mutual information not estimated yet; shared source does not imply independent evidence |
| V8 Complex amplitude | normalized real/imaginary, normalized squared magnitude, phase, total scale energy, valid flag | Complex-number and interference precursor only, not quantum physics |

In V8, z_h = v_x + i v_y and p_h = abs(z_h)^2 / sum_j abs(z_j)^2 across rows within the same scale. An active scale sums to one. A global field row normalizes separately from the hot field. Do not mistake this for a physical wavefunction.

The V4 DFT is over per-row magnitude on contiguous NaNclock ticks, where Fourier index k has frequency k/N cycles/tick. Missing time steps reset the sample window. Report spectral eigensolver residuals when a future native graph spectral kernel is added.

A family projector has history keys (family, scale, original point/triad ID), preventing hot-slot reorder from changing recorded identity. It is deterministic, not trained to WordNet or another semantic map.

## Numeric output format

Binary payload FAM1 is explicit little-endian. It contains family id 1..8, version 1, NaNclock u64, source/fact counts u32, active mirror link count u32, sparse row counts u16, and rows. Each row consists of original id u8, three member head ids u8, six f32 metrics, separate support f32 and refutation f32. There is one global and one hot-field row with marker id 255. Decoder checks family id, dimensions, duplicates, finite values and trailing bytes.

Eight independent sensor-family journal charts are now implemented as fixed 8,388,608-byte experimental sidecars with a 16,384-byte header slot and 511 circular data slots of 16,384 bytes each. Each data slot stores an FREC/1 header (sequence, NaNclock, family ID, payload length and FNV-1a32 damage checksum) plus one FAM1 payload. Records are bounded and wrap, recover the latest valid sequence after reopen, reject wrong-family charts and skip torn records. They are NOT NaN16-native universes, have no fsync/atomic cross-file commit, and are not a durable authoritative semantic ledger. The existing MATS chart remains the source of the lower-layer observations, and the production HW0 root remains unmodified.

## WordNet comparison is a reference, not imitation

The proposed BS0:0::0:HW4:0::HW0 route requires a verified persistent HW4 chart, corpus digest, state digest, and NaNclock mapping. A cached local WordNet file or a successful prior run is not proof of an instantiated persistent HW4 mapping. Keep HW4 read-only.

Match events across families by source/event ID and time, then compare rank statistics, CKA/HSIC structural similarity, phase/spectral features and held-out relation retrieval. Use source-disjoint and shuffled/null controls. Never fit all eight outputs directly to one WordNet axis set, and never claim improved language loss without actual held-out tests.

## Verification and next milestones

The family test runs the same five-source contradictory fixture through all eight measurements and checks original row IDs, triad/hot ID preservation, binary codec roundtrip, vector normalization, tensor invariants, Shannon entropy, complex amplitude norm, phase velocity, DFT on a known periodic signal, irregular-clock gap behavior and no mutation of the MATS source.

Next: corpus-scale write bandwidth/retention benchmark across eight 8 MiB charts, cross-journal transactional recovery, V6 cycle rank and triangle count on synthetic graph controls, V7 bias-corrected mutual-information estimator, V4 eigenresidual-based graph spectral operator, V8 pairwise complex interference, independently trainable native NaN16 detector kernels and source-disjoint WordNet comparison. MTH kernels must run inside native NaN16 for a v2 production port; host float arithmetic remains a v1 research reference only.

New persistence regression: tests/test_sensor_family_chart.cpp verifies eight separate 8 MiB allocated files, correct family-specific records, explicit numerical roundtrips, rolling restart across 511 slots and rejection of a deliberately corrupted newest frame. Sample payload bytes for each family are printed by the tests. Empty-capacity guarantee does not imply corpus-level retention or throughput.
