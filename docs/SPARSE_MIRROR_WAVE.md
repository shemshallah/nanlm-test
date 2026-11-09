# Sparse mirror-arc execution for the single-root standing wave

The production universe still stores 32,640 physical NaN16 mirror pairs
and 255 cognitive heads; head 255 remains network ingress. This change
is NOT a sparse separately trained model. It is a rebuildable execution
index over the exact same directed pair evidence as before.

Every spectral refresh constructs two per-head adjacency lists:
incoming edges T and outgoing/inverse edges T*. The original dense
observe loop multiplied all 255×255 directed entries every byte,
including zeros. The new loop visits only nonzero cached arcs in the
same j order, and reuses the exact degree normalizers.
No additional persistent weights are allocated.

`tests/test_sparse_wave.cpp` builds forward and refuting directed
relations with nontrivial phase. It checks every coordinate of both
next-wave arrays against an independent, full 255×255 dense oracle using
the same damping, modal frequency and spectral terms. The allowed
numerical disagreement is less than 1e-11 per coordinate. GCC and Clang
both compile and run the test.

The test also prints elapsed microseconds for a sequence of 512
observations as a diagnostic, without a host-dependent performance
assertion. The principal motivation is reducing the per-byte temporal
cost before training BPTT over larger external prose.

The index is invalidated and rebuilt by the existing
`refresh_spectrum()` and periodic spectral rebuild mechanisms, so
new pair evidence is incorporated at the same moment as the spectrum.
