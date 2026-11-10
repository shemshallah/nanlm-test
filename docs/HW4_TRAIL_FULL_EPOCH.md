# HW4:0 complete two-rail TRAIL sweep — full WordNet once

This experimental native kernel consumes every byte of the pinned
HW4:0 exact-ledger-derived 58,341,529-byte WordNet source corpus in
one sequential traversal, after restoring the actual trained 8 MiB
student from PR #31. The NaNROUTER/MC0:0 HW4 root is never writable.

The default NaNLM long_context_signal is a *48-byte feature-gradient*
with no full two-rail time-difference recurrence. This new experimental
sweep uses forward and inverse TD eligibility e_plus/e_minus, both
with gamma=0.93 and lambda=0.82, and signed credit
C = sum(delta_plus*e_plus - delta_minus*e_minus).
Every transition updates all 255 3D head eligibility coordinates.
Forward and backward conditional values are frozen from the same
pretrained FabricTeacher probability matrix. The reverse conditional
is obtained by normalizing its transposed score matrix. This is a
frozen-value full-corpus batch TRAIL credit experiment, not complete
online updating of all native teacher, triad, standing-wave and graph
parameters. There is no BPTT.

Only the existing learner's 255 source and target 3D head vector slots
are candidates for an aggregate credit update, after a bounded scratch
acceptance test on the exact all-corpus byte-pair objective. If the
combined two-rail credit fails the objective, the model remains unchanged
and that result is reported.

The GitHub Actions job independently confirms all synsets, senses,
synonyms, source pointers and SHA256s, downloads PR31's trained
checkpoint, compares the very same 974-byte prose fixture before and
after, runs a one-pass native fixed-gamma/lambda sweep, and reports
precompute seconds, complete corpus scan seconds, commit seconds,
total native seconds, bytes per second, accepted/rejected credit and
all losses. It saves the single 8 MiB updated model root, source
hashes and logs, not the WordNet dictionary.

Reproduction from a verified HW4 all.txt and PR31 checkpoint:

    make nanlm hw4-trail CXX=clang++
    ./hw4-trail student-root hw4/all.txt 58341529

A full pass is not proof of semantic knowledge, and this evaluation
fixture is short. Crucially, unlike the earlier purely rank-three
aggregated count update, TRAIL carries separate persistent-in-time
forward and inverse eligibility across every observed byte before
its full-data credit is committed.
