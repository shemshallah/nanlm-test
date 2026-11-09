# Training ablation for actual backpropagation

`nanlm ROOT train TRAIN_TEXT 2 1` (default) runs native truncated BPTT.
`nanlm ROOT train TRAIN_TEXT 2 0` performs exactly the same training
operations except that it disables the eight-step recurrent BPTT
parameter update. Both retain the same teacher, mirror wave, head
coefficients, triad voting, and predictive distribution.

The comparison is diagnostic only: its two temporary roots are
counterfactual **copies** created by the benchmark, not separately
deployed or separately designed production language models.
Each starts with fresh deterministic native state, runs two training
passes on the same training split, calibrates using only the development
split, and measures three never-trained held-out topics.

The audit reports all per-domain before/after CE values, actual
generation samples and the number of accepted BPTT gradient steps.
It does **not** require BPTT to win; deterioration is a concrete finding.

This isolates learning by recurrent gradients from unrelated changes
such as the addition of an output wave readout. The parameter updates
are self-contained and persisted in the original 8 MiB HW0 root,
and the native ingress head 255 is not trained.
