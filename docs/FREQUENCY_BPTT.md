# Native trainable standing-wave frequency

The first merged BPTT learns one recurrent input amplitude per cognitive
head. This experimental round adds **one trainable oscillator frequency
shift per head** to the same second-order forward/inverse standing-wave
recurrence. Head 255 is never trained.

The original model frequency `ω_i = 0.53 + 0.1 tanh(λ_i)` becomes

`ω_i = 0.53 + 0.1 tanh(λ_i) + Δω_i`

where the eigenvalue-derived modal contribution `λ_i` still comes
from the physically shared mirror-pair graph. `Δω_i` is a bounded
float32 value stored at bytes 116–119 of each existing 128-byte
cognitive head; marker byte 121 indicates the new register field.
The initial offset is exactly zero, preserving the prior recurrence.

For each causal transition, reverse-mode differentiation includes

`∂x_{t+1,i}/∂Δω_i = -2d sin(ω_i) x_{t,i}`

and the analogous inverse-wave expression, both gated by the output
clamp derivative. Gradients are summed across the truncated time tape
and global-norm-clipped together with input-amplitude gradients.
The optimizer proposes both parameter banks, replays the same causal
history, and commits only when the exact normalized loss decreases.
A failed proposal restores all bytes and both register markers.

This is **not** a separate recurrent model or an independent eigenbasis,
and it is not gradient descent through eigenvector power iteration.
It trains a small phase modulation of the *existing* shared oscillator.

`tests/test_frequency_bptt.cpp` checks seven-transition exact
frequency gradients against central finite differences, nonzero
remote gradients, accepted true-loss improvement, root restart and
byte-exact preservation of the 256th network ingress head.

The feature remains experimental until the authored three-domain
and pinned cross-author public-domain evaluations have been inspected.
