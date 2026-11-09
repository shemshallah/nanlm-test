# Rank-four temporal mirror interaction

This is a **fourth-order factorized feature** of the same native language
model, not a second LM, a dense 255^4 array, or a lookup table of
independently trained n-grams.

Given the last three observed bytes `a,b,d` and a proposed next
byte `c`, the readout multiplies three consecutive directed pair
evidence terms from the canonical NaN16 mirror-pair cells, adds a
closing pair interaction, and applies a U(1) triangle factor and a
rank-four CP product from the already-trained head vectors. Exact
four-address features are computed on demand and not separately stored.

Each head stores one bounded trainable `float32` fourth-order readout
coefficient. This uses formerly unused bytes 66–69 in heads 1–254
(beside the adaptive-triad map at 64–65), and bytes 105–108 in head
0 (beside the two tensor-calibration scales); head 255 is not modified.

Supervised next-byte softmax gradients propose a new full set of
head coefficients. The exact normalized loss is checked in scratch,
and the proposed updates are accepted only when the immediate
prediction loss decreases. This is *feature-gradient learning*, not
end-to-end differentiation through the entire standing-wave dynamics.

`no_fourth_order` ablates only this higher-order readout and leaves
the wave, hypermatrix, and adaptive triad routes active. The native
regression confirms that it is zero before three history symbols,
affects predictions after training on a chain of actual native
mirror relations, and persists across root reloads. The 8 MiB mapped
storage size remains unchanged.

**Measurement rule:** a visible ablation difference is a wiring proof,
not evidence of transferable language modeling. Report new development,
held-out and generation samples before claiming gains.
