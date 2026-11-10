# NaNLM mathematical sensor — time-sliced vector/tensor matrices

Status: experimental branch. This is measurement telemetry of one shared sensor;
it creates no other language model and never changes the production 8 MiB NaN16 root.

## Four simultaneously measured resolutions

| Resolution | Vector matrix | Tensor matrix | Other features |
| --- | --- | --- | --- |
| Point/head | 255 x 3 | 255 x 3 x 3 | activity; supporting/refuting charge |
| Trihead | 85 x 3 | 85 x 3 x 3 | dynamic three-head membership; curvature |
| Hot trihead | K x 3 (K <= 12 default) | K x 3 x 3 | original trihead IDs, prior activation, rank |
| Global and hot-field | 2 x 3 | 2 x 3 x 3 | independent full/hot field views |
| Sparse mirror links | antisymmetric directed flux | tensor cosine, phase lock | shared sources, physical pair addresses |

At each accepted input event e, the role-bound hypervector encoder produces
a rotated three-component feature x_(e,h) at each routed head h.
At NaNclock t with leak parameter lambda, the sensor measures

    v_h(t) = SUM_e lambda^(t-t_e) polarity(e) confidence(e) x_(e,h)
    Q_h(t) = SUM_e lambda^(t-t_e) confidence(e) x_(e,h) x_(e,h)^T

where polarity is +1 for support and -1 for refutation. Tensor Q is
symmetric; support and refutation have separate counters so cancellation in
the signed vector does not erase contradictory evidence. These are
deterministic mathematical measurements, not trained embeddings.

Trihead vectors/tensors sum their actual member head values. The global
field takes the mean of all 85 triheads, including cold ones, while the hot
field weights only dynamically selected triheads. Native adaptive memberships
are supported. No new independent model heads are introduced.

## NaN16 mirror-hypermatrix spatial coordinates

The 256-address topology has C(256,2)=32,640 unordered canonical mirror
pairs. For the sparse hot-trihead links, we store the two anchor head IDs,
NaN16 address(a,b), its reflected address, and the shared physical offset.
A directed link and its bitwise mirror refer to ONE physical canonical cell.
The oriented vector flux

    F(a,b) = v_a,x v_b,y - v_a,y v_b,x

changes sign when the address is reflected; tensor cosine and phase-lock
are mirror-invariant. MatrixTimeSlice::mirror_at returns either directed
reading with the same physical byte offset. Inactive pairs are not written
and must not be interpreted as negative evidence. Head 255 is still reserved
for ingress; it is not repurposed as a cognitive point.

This is a sparse indexed matrix overlay over the existing NaN16 geometry,
NOT a second dense 65,536-slot matrix or another 8 MiB universe.

## Temporal binary storage

The numeric recording lives in an isolated fixed 8,388,608-byte .slices file,
separate from the unchanged 8,388,608-byte production root.

- File header: 64 bytes, version, endianness/schema, capacity, alignment.
- Every record starts on a 64-byte boundary. Record header: 32 bytes with
  sequence, NaNclock, payload length and a damage-detection checksum.
- Payload rows contain explicit IDs, vectors (3 binary32), tensors (9
  binary32), positive/refuting activity and, for mirror links, the canonical
  pair offset, orientation and relation measurements.
- Only active head/trihead rows, hot triheads and sparse links are written.
  An idle tick can be much smaller than a fully active sensor state.
- The byte encoding is explicit little-endian with finite IEEE-754 binary32.
  It does not write raw NaN payloads or C++ struct padding to disk.
- The bounded log wraps, overwriting older numerical frames. On restart it
  scans and checks frames; invalid/torn newest records are rejected.
- FNV-1a32 checks accidental damage only. There is no authenticated commit,
  fsync-based power-loss guarantee, or atomic transaction shared with the
  provenance graph snapshot.

Actual payload bytes and retained frame count depend on sensor density and
must be measured with representative corpora. The format has an individual
frame limit of 64 KiB, not an unlimited matrix dump.

## Clock and provenance

INGEST records one frame after its logical clock advance; TICK records one
frame after idle decay; MEASURE captures one extra frame on demand without
advancing the clock. Each record also has a monotonic separate frame sequence
number, so multiple samples at one clock tick remain distinguishable.

The exact graph journal uses NANLM_SENSOR_V2, which stores event ticks and
the current clock, preserving idle periods on restart. The old V1 reader is
retained. An analytical fast-forward is used for long gaps, with potential
small floating-point rounding differences versus individual tick replay.
Source facts, contradictions and proof paths are never encoded solely in
the bounded tensor trace.

Resident protocol commands (tab delimiters for parameters):

    INGEST <TAB> sourceA <TAB> + <TAB> penguin <TAB> ISA <TAB> bird
    MATRIX
    SENSOR_VECTOR <TAB> GLOBAL
    SENSOR_TENSOR <TAB> GLOBAL
    SENSOR_VECTOR <TAB> HEAD <TAB> 42
    SENSOR_TENSOR <TAB> TRIAD <TAB> 10
    SENSOR_VECTOR <TAB> HOT <TAB> 7
    MEASURE
    TICK
    TIMELINE <TAB> 8
    HOTMAP
    QUIT

HOT addresses use original trihead IDs, not ephemeral rank/slot numbers.

## Reproducible gates

tests/test_sensor_timeslices.cpp checks: four-resolution nondegenerate vector
and tensor measurements; tensor symmetry; native Universe leaf/pair physical
mapping, directed reverse16 flux, ingress exclusion; exact little-endian
roundtrips; 64-KiB ring wrap and monotonic restart; deliberately corrupted
frame recovery; and clock-faithful journal replay across source gaps.
The GCC/Clang GitHub CI workflow compiles and runs this test plus the resident
readout. Its log reports actual row counts and sample payload bytes.

Remaining experiments: learned semantic encoder gradients, source-disjoint
evaluation against the existing HW0 model, benchmarking corpus-scale
recording bandwidth, transaction-safe native temporal overlays and a trained
HW0 mathematical-state-to-language decoder.
