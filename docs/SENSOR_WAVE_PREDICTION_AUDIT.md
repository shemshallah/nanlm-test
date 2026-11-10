# Experimental recurrent signatures and long-window holdout prediction

Status: experimental, **numeric-only**. The system reads the independent,
verified 8 MiB `NLSLICE1`/`NSF1`/`MATS` sidecar without opening the native
NaN16 universe, rewriting journal claims, allocating a new model instance,
or using BPTT. `tools/sensor_temporal_wave_forecast.py` computes a bounded
AR(2) readout in standard Python without changing any stored state.

## Why a held-out prediction is necessary

A high vector autocorrelation can arise from a trivial decay coefficient, a
constant field, a repeated measurement, or event-routed hash geometry. None
of these establishes a standing wave or memory function. The evaluation fits
one global two-lag recurrence on **earlier** valid contiguous ticks and scores
one-step forecasts on strictly later ticks, using actual previous observations
at each test tick. The temporal alternatives are:

- **Persistence:** predict next vector = current vector.
- **Known leak:** predict next vector = `0.97 * current` (the actual sensor's
  idle transition, not a learned result).
- **Fitted recurrence:** predict `x_next = alpha*x_now + beta*x_previous`.

The fitted coefficients use regularized closed-form least squares; no
backpropagation through time is involved. A candidate oscillatory mode is
reported only if the fitted recurrence has conjugate-complex characteristic
roots, supports at least three cycles in a sufficiently long contiguous
window, and beats **both** the persistence and known-leak baselines on held-out
observations by 20%. Even then it is an **oscillatory predictive candidate**,
not physical proof of a standing wave or learning-accuracy gain.

## Boundaries and correctness

- Select `HEAD`, `TRIAD`, `HOT`, `HOTFIELD`, or `GLOBAL` plus stable numerical ID.
- Verified frames are ordered by monotone sequence. Sequence gaps, tick gaps,
  vanished sparse rows and remapped triad memberships end an observation run.
- Multiple measurements at the same tick use the last sample as the
  representative of **one** tick; duplicates never count as cycles.
- Only the longest complete contiguous segment is evaluated; short or
  interrupted histories explicitly return `insufficient_contiguous_history`.
- Fit on first `N-holdout` ticks; report RMSE on the subsequent holdout only.
- Forecast is *teacher-forced one step*: the actual immediately preceding
  sensor values are available to each prediction. This is not autonomous
  roll-out nor evidence that the sensor internally forecast anything.

## Commands

```sh
python3 tools/sensor_temporal_wave_forecast.py recording.slices \
  --level GLOBAL --id 255 --min-ticks 32 --holdout-ticks 10
python3 -m unittest discover -s tests -p 'test_sensor_temporal_wave_forecast.py' -v
```

## Reproducible positive and negative controls

1. A 90-tick synthetic 12-tick sinusoidal trajectory is fitted and its period
   independently recovered, with lower holdout error than fixed decay.
2. An 80-tick pure exponential decay must **not** be classified as an
   oscillatory mode, even when AR(2) predicts it accurately.
3. Clock gaps, missing hot-triad coordinates, changed memberships and
   same-tick duplicate samples are explicitly tested.
4. An actual native C++ sensor fixture ingests three graph assertions then
   advances through 55 idle NaNclock ticks, writing its fixed 8 MiB sidecar.
   The trace is measured with the same decoder; expected result is **not** an
   oscillatory candidate. The baseline that knows native decay should be
   strong on this trace. This is a negative-control experiment, not a claim
   of persistent waves.

The scientific next gate is to test controlled recurrence **inside the sensor
architecture** (e.g., localized injected perturbations with defined,
provenance-preserving propagation), then evaluate whether its predictive
advantage survives source-disjoint corpora and time-shuffling. Only a
measured downstream loss/accuracy gain permits a language-learning claim.