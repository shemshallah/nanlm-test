# Unified mirror-wave HW0:0 — implementation and evaluation contract

## One model, one root

The production `nanlm` command constructs exactly one `UnifiedWaveLanguage`
on one 8,388,608-byte mapped `Universe`. Heads 0–254 are simultaneous
coupled language channels; head 255 is reserved for ingress and is not
a token. The shared 32,640 unordered physical mirror pairs carry two
ordered lanes of 16 typed relation weights each. The full 65,536-address
space maps to 32,896 canonical cells. No 255-model ensemble is created.

Persistent language parameters include the teacher's vectors, phases,
typed evidence, quaternion transports, four gating gains per
palindromic head, and one additional hyperedge gain per head. Group
consensus is held in 85 head-register slots and the field vote in a
single head-0 slot. All four gains live inside that head's existing
128-byte record; the causal wavefront and spectral decomposition are
derived activations. Root and overlay writes are not crash-atomic.

## Temporal inverse and spectrum

The forward and inverse waves use the directed pair operator T and
its adjoint T* respectively, with damped second-order recurrence and
shared input. This is a forward-time causal adjoint, not access to
future labels. Four approximate symmetric eigenmodes modulate the
oscillator and output. U(1) cycles, ordered SO(3) quaternion products,
16-step context attention and a factorized third-order hypermatrix
contribute additional features. The eigensolver is currently an approximation,
not a differentiable neural attention layer.

## Falsifiable ablation gates

After training on the training split, run:

```sh
./nanlm /tmp/hw0 audit heldout.txt
```

Eight read-only modes process identical held-out bytes, with the same
stored weights, seed-independent inference and initial causal state:

- `full`: full production prediction.
- `no_wave`: removes the wave readout, not the wave dynamics.
- `no_attention`: removes the 16-position attention readout.
- `no_spectral`: removes the spectral readout, not wave frequency modulation.
- `no_cycle`: removes the gauge-cycle readout.
- `no_inverse_evidence`: removes the extra inverse-consensus logit.
- `no_hyperedge`: removes both the factorized third-order interaction
  and its explicit signed sparse residual, leaving the single teacher
  and all other shared readouts unchanged.
- `teacher_only`: ablates all six additive logit terms; this is one
  existing native teacher's projection, not another production model.

These are diagnostic **readout** ablations, not independent architectural
experiments: the teacher still uses native geometry, relations, gates
and trainable vectors. Ablating the spectral readout does not disable
its influence on recurrent dynamics. Do not describe these measurements
as causal feature isolation or attribute improvements without controls.

## Quality criteria

Preserve disjoint train/development/test files with no repeated documents.
Freeze parameters and thresholds before test evaluation, report bytes
scored, mean NLL in nats and top-1 accuracy. Repeat over meaningful
source-disjoint corpora and seeds; report a matched baseline including
unigram and order-matched count models as separately named *diagnostics*
only, not production ensemble members. Record generated continuations
and repetition, validity, novelty, latency and memory. Native
compilation, valid Unicode and persistence do not establish fluency.

## Longer prose sentinel (not a language-quality claim)

The GitHub workflow `.github/workflows/unified-quality.yml` scores an
untrained initialized root against a held-out prose file, trains that **same
root** for two passes on a separate prose file, and then re-scores all eight
readout modes. It records the signed after-minus-before CE difference, text
samples, and SHA-256 corpus hashes without declaring improvement by fiat.
The authored fixtures are tiny and do not establish transfer to new topics,
styles or languages; the sentinel is an instrumentation gate only. No new
production model, root, or expert ensemble is constructed.

## Development-calibrated shared readout strengths

The `calibrate DEV_TEXT` CLI command performs deterministic coordinate
search over four gain scales in {0, 0.5, 1, 1.5, 2}. It measures each
candidate on an independent development text, never on the final held-out
test. Four unsigned Q8.8 values occupy previously unused bytes 64–71 of
palindrome head 0; byte 61 marks the schema. Its gauge potential remains
at 72–79 and existing head-local expert gains remain at 80–95. A scale
is one shared readout multiplier applied across 255 heads of the **same**
model. Subsequent online learning includes that multiplier in its gate
gradients. Calibration minimizes development CE by construction and is
not guaranteed to improve held-out CE; both outcomes must be reported.

## Register-coupled hypermatrix and three-level self-modification

A hypermatrix is a **logical view** on the resident NaN16 memory, not an
extra 255³ array or a second production model. For distinct heads
`a,b,c`, the predicted third-order term uses the ordered PRECEDES /
FOLLOWS mirror-lane evidence on `a→b`, `b→c`, and `a→c` and a
gauge-invariant U(1) triangle holonomy. The product of pair operators
is non-additive: evidence for one leg conditions the others. These
terms are derived directly from the same 8 MiB memory-mapped `Root`.

Each logical triple also has a concrete NaN16 location mapping:
`Universe::hypermatrix_address(a,b,c)` identifies three oriented legs
`a→b`, `b→c`, `c→a`, their exact mirror addresses, physical offsets
into the canonical pair bank, and the three fixed-point head offsets.
A directed leg and its inverse have **the same physical offset**;
neither address creates a new independent tensor cell.

The factorized tensor additionally reads the existing trainable 3-vectors
in heads `a,b,c` as a CP-rank-three multilinear product, which
multiplicatively modulates the pair evidence. Both forward and inverse
standing-wave amplitudes provide a small temporal modulation. This
couples head register learning, mirror topology and the causal wave to
the same higher-order readout, rather than adding an unconnected
hyperedge lookup.

There is an additional **signed sparse correction** for explicitly
observed `(a,b,c,relation)` tuples. The persistent overlay maintains
the previous `Hyperedge` binary representation, with a rebuilt hash
index for constant-time ordered-triple reads. Support and refutation
remain independent. Unlike the old exact-triple-only readout,
`no_hyperedge` can now change predictions for a triple that has **no
exact sparse entry** if its constituent pair legs have evidence.

All trainable hyperedge readout gains occupy previously unused bytes
of the **existing** 128-byte palindromic register records:

| Layer | Physical location | Parameterization |
|---|---|---|
| Individual | Each of heads 0–254, bytes 28–31 | 255 local float32 gains |
| Triad | Head 2, 5, …, 254, bytes 12–15 | 85 group float32 gains, three adjacent heads per group |
| Field | Head 0, bytes 12–15 | 1 shared float32 gain |
| Markers | Byte 62 (local), byte 63 (triad/field) | Layout and restart validation |

The selected head, its three-head group, and the field contribute to
**one** hyperedge logit. Individual gradients, three-head aggregated
gradients, and the field-wide aggregate propose a coupled change.
The candidate is evaluated against the exact normalized softmax
cross-entropy, and only a strictly improving step is committed. Sparse
observed-triple writes additionally require at least two positive
voters among head/group/field and cap positive residual strength;
they no longer depend on every-eighth-step scheduling or an unrelated
teacher update. These acceptance rules are **single-token local
optimization**, not proof of corpus-scale generalization.

**Important limitations:** the 85 groups currently use static
consecutive head addresses, not learned semantic group assignments.
The 255³ hypermatrix remains *factorized*, not a dense arbitrary
third-order tensor. Sparse records remain in a separate on-disk overlay
and can grow beyond the mapped 8 MiB root; the full resident working
set is therefore larger than 8 MiB. The learner does not yet provide
multi-step backpropagation through time. These distinctions should
remain explicit in performance reports.

### Regressions and ablation interpretation

`make test_hypermatrix && ./test_hypermatrix` checks a never-stored
ordered triple receiving a third-order signal via native pair legs,
signed residual independence, restart determinism, all three
learnable voter levels, physical address/mirror equality, and sensitivity
to a live head-vector perturbation. `make test_unified_wave && ./test_unified_wave`
checks the full standing-wave system. Both are required under GCC and
Clang in `.github/workflows/unified-wave.yml`.

Report *both* `full` and `no_hyperedge` held-out CE with identical
model state. A nonzero difference proves the feature is active on that
sample; only source-disjoint evaluation can show generalization.
