# Experimental temporal sensor export and change analysis

This is a **read-only** consumer of the native `MATS` v1 measurements
written in the fixed `.slices` sidecar by `sensor-resident` on the
`experiment/math-sensor-hypervectors-hw0-20261109` branch. It does not
open, rewrite, or instantiate the 8 MiB native NaN16 universe. It does not
reconstruct source claims or replace the source-provenance journal.

## Run

```sh
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv summary
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv.slices frames --format jsonl
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv.slices samples --format csv > samples.csv
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv.slices mirrors --format csv > mirrors.csv
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv.slices deltas --format csv > deltas.csv
python3 tools/sensor_temporal_export.py snapshot.sensor.tsv.slices deltas --limit 32
python3 -m unittest discover -s tests -p test_sensor_temporal_export.py -v
```

The CLI accepts the actual `.slices` path or the snapshot filename (it
appends `.slices`). The default output is one compact JSON object per
line. CSV rows describe measured numerical data; none contain source text.

## Native compatibility

- 64-byte `NLSLICE1` header containing version, capacity and alignment.
- 32-byte `NSF1` record headers with monotonic sequence, NaNclock,
  bounded byte length and FNV-1a32 digest over payload + sequence + clock.
- 32-byte `MATS` v1 payload header with event-source/fact counters and
  independent head/triad/hot/mirror row counts.
- 76-byte spatial rows: integer coordinate, three member heads, 3-vector,
  independent 3×3 tensor, and six scalar measurements.
- 26-byte canonical mirror rows: endpoint IDs, directed addresses, pair
  offset, tensor cosine, phase lock, flux and shared-source counter.
- Two field rows (hot and global) follow the mirror rows.
- All scalar coordinates use finite IEEE-754 binary32 in **little-endian**;
  no native C++ padding or floating-point NaN payload serialization.
- Every frame is length-checked, checksum-checked, dimension-checked and
  ID-checked before export. Invalid candidates are skipped and counted.
  Duplicate valid sequence numbers or a regressing nanclock during temporal
  analysis are treated as errors, not silently resolved.

This is damaged-frame detection, **not cryptographic authenticity or a
transactional source/native commit**. Files may be read during concurrent
native writes; for a consistent export, run against a stable sidecar copy.

## Temporal meaning

For adjacent **retained, verified** frames, a stable row coordinate is
matched by `(level, id)`. The trihead membership triple must also match;
a remapped row is emitted with `membership_changed` and without any numeric
delta. A missing sparse row or vanished hotspot is **not** a zero
measurement. Hot rows use original triad IDs, not rank positions.

When the same stable row exists at both frame boundaries:

```text
delta_vector = v(t2) - v(t1)
vector_delta_norm = sqrt(sum_i delta_vector_i^2)
delta_tensor = Q(t2) - Q(t1)
vector_derivative = delta_vector/(t2-t1) when delta_ticks > 0
tensor_derivative = delta_tensor/(t2-t1) when delta_ticks > 0
tensor_delta_fro = sqrt(sum_ij (Q_ij(t2)-Q_ij(t1))^2)
angular_change = acos(clamp(dot(v1,v2)/(||v1|| ||v2||), -1, 1))
vector_speed_per_tick = vector_delta_norm/(t2-t1)
angular_rate_rad_per_tick = angular_change/(t2-t1)
tensor_speed_per_tick = tensor_delta_fro/(t2-t1)
tensor_trace_delta = sum_i (Q_ii(t2)-Q_ii(t1))
```

Angles of zero vectors are **undefined**, not zero. Two frames at the
same NaNclock have a valid displacement but **no temporal rate**. A tick
gap larger than one is measured using the *actual* elapsed NaNclock, not
the number of stored samples. Sequence gaps are reported explicitly;
the bounded ring cannot infer anything within an overwritten interval.

The raw `mirrors` export includes both forward and reflected address, shared
physical offset, directed flux, ascending-anchor canonical flux, tensor cosine,
phase lock, and shared-source count.

Mirror links are keyed by the same physical `C(256,2)` pair offset;
direction is normalized against ascending anchor IDs before computing
flux differences. Opposite directed addresses read the same physical cell,
with reversed oriented flux. This is not evidence that a link was written
to the native root.

## Research interpretation

The metrics can reveal persistent or changing patterns, but they do not
demonstrate improved reasoning, trained semantic representations, or lower
language-model loss. Directly compare measured temporal feature stability
and predictive utility against matched held-out baselines before making
those claims. Source-level support and refutation remain in the journal;
the signed vector alone may cancel contradictions.

The GitHub Actions workflow builds `sensor-resident` with GCC and Clang,
records an actual multi-source evidence trace, independently decodes it
with the Python reader, checks the 8 MiB storage boundary, and runs a
synthetic format/temporal-invariant suite including intentional corruption,
ring wrap, same-tick samples, remapped triads and reversed mirror orientation.
