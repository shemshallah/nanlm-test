# NaNLM adaptive triad topology

There are still exactly 255 active head registers, 85 three-head voting
groups, and one full-field output distribution. Adaptive assignment
does not create any separate expert language model.

## Physically mapped membership

Group coefficients remain in the existing 85 group-holder palindrome
registers (heads 2,5,…,254, bytes 12–15). The group membership ID for
every head 1–254 occupies otherwise-unused bytes 64 (ID) and 65
(marker), while head 0 stays pinned in group 0 because its bytes
64–71 are shared calibration values. Head 255 is untouched.

The first construction initializes the mapping to consecutive groups of
three. On every restart, all 85 group sizes must be exactly three;
corrupt mappings are rejected rather than silently repaired.

## Hierarchical self-modification

An update proposes a **swap** of two nonzero heads from different
groups, preserving group sizes exactly. The complete probability of
the supervised next byte is measured against the unchanged model
state. A swap commits only if the normalized next-byte likelihood
improves by more than a fixed threshold. Otherwise both mapped
membership bytes are restored. Current online training proposes one
deterministic context-based swap every 128 training tokens.

These acceptance decisions can overfit single training positions and
must be validated separately on unseen documents. The grouping
proposal rule is not yet an independently trained routing network,
nor a globally optimized graph partition.

`tests/test_adaptive_triads.cpp` directly tests a favorable swap,
an unfavorable rollback, membership conservation, persistence, and
the reserved ingress head. Its GCC/Clang workflow is
`.github/workflows/adaptive-triads.yml`.
