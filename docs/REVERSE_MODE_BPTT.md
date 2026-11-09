# Reverse-mode truncated BPTT in the one shared NaNLM

This is the first implementation in this line that differentiates through
multiple **causal recurrent state transitions** rather than updating only
a single-token output/readout feature.

## Parameter and topology contract

The same 255 cognitive heads, 85 mutable three-head voting groups,
32,640 NaN16 mirror-pair cells and 8,388,608-byte root remain active.
Head 255 stays reserved for ingress. A scalar recurrent input gain
in each cognitive head's unused register bytes 112–115 has initial
value 1.0, bounded to [0.6,1.4] with a marker at byte 120.
No extra language model, dense four-tensor, or trainable second graph
is constructed.

## Causal gradient

The optimizer retains a detached oldest state, then exactly eight
subsequent byte inputs and forward/inverse wave states. It computes
the full normalized next-byte cross-entropy of the current output
distribution and differentiates that loss with respect to the current
and previous forward/inverse wave coordinates.

The output adjoint includes:
- the wave readout (including the new small always-visible wave probe)
- attention's weighted sum of source wave activations
- the factorized hypermatrix's temporal tanh product

Reverse-mode then propagates through the actual second-order standing-wave
recurrence, the transposed directed scattering operator, the
forward operator, and saturation/clamp masks. The oldest state is
detached, and graph/eigen/teacher weights stay fixed inside each tape.

Every 16 observed bytes (with at least four transitions available),
a global-norm-clipped gradient proposes mapped-head gain updates.
The eight-state sequence is **replayed** under the proposal. Only
strictly improving *full normalized current-label* loss is committed;
all 255 weight bytes and their register markers are restored when
a proposal is rejected. Accepted replay states replace the tape so
the live stream evolves under its updated weights. Graph refreshes
every 128 tokens invalidate and reset the tape.

**Scope:** this is *single-target, eight-step truncated BPTT* on the
existing recurrent input gains. It is not yet a full-length
sequence-to-sequence gradient through the teacher, attention keys,
quaternions, graph eigenvectors, or every sparse relation write.

## Validation

`tests/test_wave_bptt.cpp` builds a directed NaN16 graph, traces
seven transitions, and compares the analytic gradient for three
physical input gains with central finite differences of the actual
normalized prediction objective. The early register must receive
a nonzero gradient through the intervening recurrences.

The same test requires an accepted true-loss improvement, nonzero
recurrent gain change, root persistence after a new Universe
is constructed, and byte-exact preservation of all of ingress
head 255. Both GCC and Clang have passed this regression.
`no_tbptt_readout` is a specific readout ablation, **not** a complete
roll-back of the learned recurrent gains.

## Empirical limitation

The authored 974-byte prose fixture after training produced about
2.65573 nats/byte with the new readout active versus 2.65554
without its direct readout contribution. This small difference is
not evidence of held-out quality improvement. The separate
source-disjoint and external public-domain benchmarks must be
consulted before claiming fluent generation.

The online CLI explicitly reports `bptt_attempted` and
`bptt_accepted`; acceptance is a training-position outcome,
never a guarantee of generalization.
