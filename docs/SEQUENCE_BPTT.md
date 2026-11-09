# Multi-position sequence credit assignment in one NaNLM

The existing eight-transition reverse-mode BPTT originally optimized
only the *last* next-byte target in a truncated causal tape. This
experiment changes the objective to the average teacher-forced
next-byte loss at every available **intermediate causal state**
and the terminal label. No new model, vocabulary, register table or
persistent universe has been created.

Let the detached state be `s_0`, transitions `x_0…x_{k-1}`
produce `s_1…s_k`, and the terminal next byte be `y`.
The objective is:

`L = (CE(s_k,y) + Σ_{j:s_j has history} CE(s_j,x_j)) / M`

where `M` is the number of valid targets. The backward pass starts
with the terminal output adjoint. At each earlier state it adds the
gradient of that state's own target prediction **before** transporting
the combined wave-state gradient backward through the second-order
oscillation and both transpose-directed mirror scattering operators.

This is a real sum of temporal losses with shared parameters, not
a bag of independent head-level cross-entropy updates. The same
bounded recurrent input gain per palindromic head is trained.
The first detached state has no incoming gradient; the 8-step truncation
and fixed graph basis remain explicit constraints.

`tests/test_sequence_bptt.cpp` checks multi-target analytic
gradients against independently replayed central finite-difference
gradients for three mapped recurrence parameters, and checks that
an accepted transaction decreases the *full sequence objective*.
The previously merged single-terminal numerical gradient test remains
unchanged and must continue to pass.

The method remains an experiment until source-disjoint and
cross-author CE and generated samples are measured. Training-position
loss improvements alone are not language-quality claims.
