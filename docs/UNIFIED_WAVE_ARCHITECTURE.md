# Unified mirror-wave HW0:0 — implementation and evaluation contract

## One model, one root

The production `nanlm` command constructs exactly one `UnifiedWaveLanguage`
on one 8,388,608-byte mapped `Universe`. Heads 0–254 are simultaneous
coupled language channels; head 255 is reserved for ingress and is not
a token. The shared 32,640 unordered physical mirror pairs carry two
ordered lanes of 16 typed relation weights each. The full 65,536-address
space maps to 32,896 canonical cells. No 255-model ensemble is created.

Persistent language parameters include the teacher's vectors, phases,
typed evidence, quaternion transports and four gating gains per
palindromic head. All four gains live inside that head's existing
128-byte record; the causal wavefront and spectral decomposition are
derived activations. Root and overlay writes are not crash-atomic.

## Temporal inverse and spectrum

The forward and inverse waves use the directed pair operator T and
its adjoint T* respectively, with damped second-order recurrence and
shared input. This is a forward-time causal adjoint, not access to
future labels. Four approximate symmetric eigenmodes modulate the
oscillator and output. U(1) cycles, ordered SO(3) quaternion products,
16-step context attention and native sparse hyperedges contribute
additional features. The eigensolver is currently an approximation,
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
- `no_hyperedge`: removes the sparse hyperedge logit.
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
