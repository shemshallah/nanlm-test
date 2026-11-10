# NaNLM 48-step causal eligibility learning

The production predictor remains one native `UnifiedWaveLanguage`, one mapped
8 MiB NaN16 root, and 255 coupled cognitive heads. No recurrent model has
been replicated.

## Memory and causal contract

The causal input history grows from 16 to **48** bytes; ordinary attention
continues to attend to the most recent 16. Earlier symbols contribute to a
decayed, signed PRECEDES/FOLLOWS evidence trace for each candidate output
head. This uses the SAME directed mirror pairs as the teacher, standing
wave, and factorized hypermatrix.

The trace's trainable readout coefficient is a `float32` in bytes 44–47
of each of the 255 existing head records. This slot sits between the
teacher source vector (32–43) and target vector (48–59). Zero is a
backward-compatible initial value, and the reserved ingress head 255 is
never updated.

A teacher-forced supervised label generates softmax errors for all 255
candidates. Exact scratch normalization checks trial gradient steps,
accepting a new set of coefficients only when the immediate training
loss strictly decreases. This is **truncated feature-gradient temporal
credit assignment**, not full backpropagation through recurrent time.

## Falsifiable test

`no_long_context` is a readout ablation using the identical stored
parameters. It should have exactly zero effect for inputs of length 16
or less, and may have a measurable effect after the tail has historical
evidence and training has changed the register coefficient.

`tests/test_temporal_credit.cpp` checks both cases, accepted gains,
byte-255 preservation, and restart determinism. This track deliberately
does not assert that 48 bytes are sufficient for prose-level dependency
modeling or that local gradient acceptance implies held-out improvement.

For the 974-byte authored prose fixture, calibrated CE in the branch
was 2.647527 versus 2.656093 with long-context readout disabled. This
is a regression sentinel, not independent language-quality evidence.

## Default training mode

Default `UnifiedWaveLanguage::train` uses the existing native local and
long-context eligibility updates **without storing a BPTT tape**.
The prior reverse-mode path remains explicitly opt-in for historical
comparison. The temporal-credit test checks that mapped long-context gains
change while BPTT attempted/accepted counts remain zero.

This does not create another TRAIL implementation; the forward/inverse rail
semantics of NaNLM's already-existing local credit are a separate parity
question, and no new geometric advantage is claimed.

## Measured existing geometric readout (no new learner)

The existing temporal eligibility learner now exposes a per-process
`temporal_credit_accepted` count of scratch-validated long-context head
gain updates. Default training must show **zero BPTT attempts**, yet can
accept existing temporal-credit updates in the same 8 MiB root. The native
regression checks positive credit acceptance and a finite paired readout
ablation after a long-distance training sample.

`nanlm ROOT eval-credit HELDOUT.txt [WAVE_EXEC_0_OR_1]` measures the
SAME model root on the SAME held-out bytes with the existing long-context
readout enabled and disabled. It emits `UNIFIED_CREDIT_EVAL` with CE,
accuracy, token parity and the signed held-out loss difference. It does not
train or create a new TRAIL implementation. A positive signed delta means
the stored geometric readout helped; a negative delta means it hurt.
