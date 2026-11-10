#!/usr/bin/env python3
"""Read-only, strictly chronological AR(2) forecast of verified MATS vectors.

The output labels oscillations as *candidates*, never as learned concepts or
standing-wave memory proofs. Numeric-only, stdlib and no BPTT. No NaN16 writes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import sensor_temporal_export as exp

LEVELS = ('HEAD', 'TRIAD', 'HOT', 'HOTFIELD', 'GLOBAL')


def contiguous_series(frames: list[dict], level='GLOBAL', identifier=255):
    """Stable row and membership, contiguous seq/ticks; latest same-tick wins."""
    segments = []
    current = []
    last_frame = None
    duplicate_samples = 0
    for frame in frames:
        tick = frame['nanclock']
        sample = next((r for r in frame['rows']
                       if r['level'] == level and r['id'] == identifier), None)
        valid = sample is not None and all(math.isfinite(x) for x in sample['vector'])
        if last_frame is not None and frame['seq'] != last_frame['seq'] + 1:
            if current:
                segments.append(current)
            current = []
        if not valid:
            if current:
                segments.append(current)
            current = []
        elif current:
            old_tick, _, old_heads, _ = current[-1]
            if tick < old_tick:
                raise ValueError('regressing clock')
            if sample['heads'] != old_heads or tick > old_tick + 1:
                segments.append(current)
                current = []
            elif tick == old_tick:
                # Same-tick sequence is a separate event, not a new period.
                duplicate_samples += 1
                current[-1] = (tick, frame['seq'], sample['heads'],
                               tuple(float(x) for x in sample['vector']))
                last_frame = frame
                continue
        if valid:
            current.append((tick, frame['seq'], sample['heads'],
                            tuple(float(x) for x in sample['vector'])))
        last_frame = frame
    if current:
        segments.append(current)
    return segments, duplicate_samples


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fit_ar2(training, ridge=1e-8):
    if len(training) < 5 or ridge <= 0:
        raise ValueError('need >=5 training ticks and positive ridge')
    aa = bb = ab = ay = by = 0.
    for k in range(2, len(training)):
        prev = training[k-1][3]
        prev2 = training[k-2][3]
        target = training[k][3]
        aa += dot(prev, prev)
        bb += dot(prev2, prev2)
        ab += dot(prev, prev2)
        ay += dot(prev, target)
        by += dot(prev2, target)
    aa += ridge
    bb += ridge
    det = aa * bb - ab * ab
    if det <= 0:
        raise ValueError('non-positive regularized normal matrix')
    alpha = (ay * bb - by * ab) / det
    beta = (by * aa - ay * ab) / det
    if not all(map(math.isfinite, (alpha, beta))):
        raise ValueError('nonfinite AR coefficients')
    return alpha, beta


def forecast(frames, level='GLOBAL', identifier=255, min_ticks=24,
             holdout_ticks=10, leak=0.97, ridge=1e-8):
    if level not in LEVELS or not 0 <= identifier <= 255:
        raise ValueError('unknown sensor level/address')
    if min_ticks < 8 or holdout_ticks < 3 or not 0 <= leak < 1:
        raise ValueError('invalid forecast parameters')
    segments, duplicates = contiguous_series(frames, level, identifier)
    segment = max(segments, key=lambda x: (len(x), -x[0][0]), default=[])
    result = {'level': level, 'id': identifier, 'segments': len(segments),
              'retained_distinct_ticks': len(segment),
              'collapsed_same_tick_samples': duplicates,
              'status': 'insufficient_contiguous_history',
              'oscillatory_predictive_candidate': False}
    if len(segment) < max(min_ticks, holdout_ticks + 8):
        return result
    train = segment[:-holdout_ticks]
    alpha, beta = fit_ar2(train, ridge=ridge)
    sums = {'ar2': 0., 'persistence': 0., 'known_leak': 0.}
    for k in range(len(train), len(segment)):
        actual = segment[k][3]
        p, q = segment[k-1][3], segment[k-2][3]
        guesses = {
            'ar2': tuple(alpha*p[i] + beta*q[i] for i in range(3)),
            'persistence': p,
            'known_leak': tuple(leak*p[i] for i in range(3)),
        }
        for key, guess in guesses.items():
            sums[key] += sum((actual[i] - guess[i])**2 for i in range(3))
    rmse = {key: math.sqrt(val / (3 * holdout_ticks)) for key, val in sums.items()}
    discriminant = alpha*alpha + 4*beta
    radius = math.sqrt(-beta) if beta < 0 else None
    period = None
    if discriminant < -1e-9 and radius and radius > 1e-12:
        angular = math.acos(max(-1., min(1., alpha/(2*radius))))
        if angular > 1e-9:
            period = 2*math.pi/angular
    # A physical interpretation requires independent evidence. This gate is
    # merely a heuristic on one vector trajectory, not a wave proof.
    improvement = rmse['ar2'] < 0.8 * min(rmse['known_leak'], rmse['persistence'])
    candidate = (period is not None and period >= 2 and
                 len(segment) >= 3*period and improvement and
                 max(sum(x*x for x in row[3]) for row in segment) > 1e-10)
    result.update({
        'status': 'evaluated', 'start_tick': segment[0][0], 'end_tick': segment[-1][0],
        'train_ticks': len(train), 'holdout_ticks': holdout_ticks,
        'ar2_alpha': alpha, 'ar2_beta': beta,
        'characteristic_discriminant': discriminant,
        'candidate_period_ticks': period,
        'holdout_rmse': rmse,
        'beats_known_leak': rmse['ar2'] < rmse['known_leak'],
        'beats_persistence': rmse['ar2'] < rmse['persistence'],
        'oscillatory_predictive_candidate': bool(candidate),
        'forecast_mode': 'one-step teacher-forced chronological holdout',
        'warning': ('Oscillation candidate is NOT a confirmed standing wave, '
                    'memory mechanism, causal information, or language improvement.'),
    })
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecar')
    p.add_argument('--level', choices=LEVELS, default='GLOBAL')
    p.add_argument('--id', type=int, default=255)
    p.add_argument('--min-ticks', type=int, default=24)
    p.add_argument('--holdout-ticks', type=int, default=10)
    args = p.parse_args(argv)
    try:
        frames, _ = exp.load(args.sidecar)
        print(json.dumps(forecast(frames, args.level, args.id,
                                  args.min_ticks, args.holdout_ticks),
                         sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print('wave forecast rejected: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())