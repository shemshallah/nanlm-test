#!/usr/bin/env python3
"""Experimental supervised *role-probe* decoder; NOT latent MATS semantic inversion.

Input NLRP1 is a numerical, three-role channel emitted by an independent
native C++ instrument from each accepted claim's words at ingestion time.
The training journal supplies role labels only for the training interval.
Model predictions see numeric role head IDs ONLY, never held-out words.
Cannot recover open vocabulary or semantics from the existing mixed MATS rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import sensor_temporal_aggregate as ag
import sensor_temporal_export as ex

HEADS = 255
ROLES = ('subject', 'relation', 'object')


def load_role_trace(path: str | Path, max_rows=100000) -> list[dict]:
    path = Path(path)
    if path.stat().st_size > 16_000_000:
        raise ValueError('role trace exceeds bounded size')
    out = []
    last_tick = 0
    with path.open('r', encoding='ascii', newline='') as stream:
        for lineno, text in enumerate(stream, 1):
            if lineno > max_rows:
                raise ValueError('too many role-probe records')
            if len(text) > 160 or not text.endswith('\n'):
                raise ValueError('overlength or unterminated probe record')
            parts = text.rstrip('\n').split('\t')
            if len(parts) != 5 or parts[0] != 'NLRP1':
                raise ValueError('bad NLRP1 format')
            raw_tick = parts[1]
            if not raw_tick.isascii() or not raw_tick.isdecimal():
                raise ValueError('invalid probe clock')
            tick = int(raw_tick)
            if tick <= last_tick:
                raise ValueError('duplicate or regressing probe tick')
            last_tick = tick
            channels = []
            for text_ids in parts[2:]:
                nums = text_ids.split(',')
                if len(nums) != 6 or any(not s.isascii() or not s.isdecimal() for s in nums):
                    raise ValueError('bad role head list')
                ids = tuple(int(s) for s in nums)
                if ids != tuple(sorted(set(ids))) or ids[0] < 0 or ids[-1] >= HEADS:
                    raise ValueError('invalid role head topology')
                channels.append(ids)
            out.append({'tick': tick, 'channels': tuple(channels)})
    return out


def align_and_split(frames: list[dict], final_tick: int, events: list[dict],
                    trace: list[dict], train_events: int):
    """Provenance and tick cross-check; do not treat missing frames as zeros."""
    if not 1 <= train_events < len(events):
        raise ValueError('invalid train/test split')
    ag.concepts(frames, final_tick, events)
    if len(events) != len(trace):
        raise ValueError('event/probe record count mismatch')
    verified_ticks = {f['nanclock'] for f in frames}
    data = []
    for e, t in zip(events, trace):
        if e['tick'] != t['tick']:
            raise ValueError('unaligned event/probe ticks')
        if e['tick'] not in verified_ticks:
            raise ValueError('event has no verified native MATS frame')
        data.append({'tick': e['tick'], 'channels': t['channels'],
                     'truth': (e['subject'], e['relation'], e['object']),
                     'source': e['source'], 'polarity': e['polarity']})
    train, test = data[:train_events], data[train_events:]
    seen = {r['truth'] for r in train}
    if any(row['truth'] in seen for row in test):
        raise ValueError('test repeats a trained whole claim')
    return train, test


def fit(train: list[dict], epochs=200, learning_rate=0.35) -> dict:
    """Three 255D sparse softmax classifiers; no hash encoder or test labels."""
    if not train or not 1 <= epochs <= 4000 or not 0 < learning_rate <= 1:
        raise ValueError('invalid supervised fit settings')
    models = {}
    for role, name in enumerate(ROLES):
        lexicon = sorted({x['truth'][role] for x in train})
        mapping = {word: i for i, word in enumerate(lexicon)}
        w = [[0.0] * HEADS for _ in lexicon]
        bias = [0.0] * len(lexicon)
        anchors = defaultdict(set)
        for row in train:
            anchors[row['truth'][role]].add(frozenset(row['channels'][role]))
        # Sequential SGD, not BPTT. The sensor supplies the independent
        # role-specific measurement, and no original words enter prediction.
        for _ in range(epochs):
            for item in train:
                ids = item['channels'][role]
                y = mapping[item['truth'][role]]
                scores = [bias[c] + sum(w[c][h] for h in ids) / 6.0
                          for c in range(len(lexicon))]
                top = max(scores)
                exps = [math.exp(s - top) for s in scores]
                total = sum(exps)
                for c in range(len(lexicon)):
                    p = exps[c] / total
                    grad = learning_rate * (float(c == y) - p)
                    bias[c] += grad * 0.05
                    for h in ids:
                        w[c][h] += grad / 6.0
        models[name] = {'labels': lexicon, 'weights': w, 'bias': bias,
                        'anchors': dict(anchors)}
    return models


def decode_role(model: dict, observed: tuple[int, ...],
                confidence_floor=0.5, overlap_floor=4, margin_floor=0.10):
    """Open-set guard requires near-neighbor observed anchor + unique result."""
    if len(observed) not in (5, 6) or len(set(observed)) != len(observed):
        return None, 'invalid_probe', 0.0
    active = frozenset(observed)
    scores = [bias + sum(weights[h] for h in observed) / 6.0
              for bias, weights in zip(model['bias'], model['weights'])]
    mx = max(scores)
    exp = [math.exp(v - mx) for v in scores]
    denom = sum(exp)
    probs = [e / denom for e in exp]
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], model['labels'][i]))
    best_idx = order[0]
    best_prob = probs[best_idx]
    nxt = probs[order[1]] if len(order) > 1 else 0.0
    overlaps = {}
    for label, anchors in model['anchors'].items():
        overlaps[label] = max((len(active & a) for a in anchors), default=0)
    nearest = max(overlaps.values(), default=0)
    nearest_labels = [label for label, n in overlaps.items() if n == nearest]
    if nearest < overlap_floor:
        return None, 'unseen_numeric_pattern', best_prob
    if len(nearest_labels) != 1:
        return None, 'ambiguous_numeric_anchors', best_prob
    label = model['labels'][best_idx]
    if label != nearest_labels[0]:
        return None, 'numeric_model_disagreement', best_prob
    if best_prob < confidence_floor or best_prob - nxt < margin_floor:
        return None, 'low_classifier_margin', best_prob
    return label, 'learned_role', best_prob


def decode_triple(models: dict, channels, **kwargs):
    roles = [decode_role(models[name], channels[i], **kwargs)
             for i, name in enumerate(ROLES)]
    if any(r[0] is None for r in roles):
        return None, [r[1] for r in roles], [r[2] for r in roles]
    return tuple(r[0] for r in roles), [r[1] for r in roles], [r[2] for r in roles]


def evaluate(frames, final_tick, events, trace, train_events, epochs=200):
    train, test = align_and_split(frames, final_tick, events, trace, train_events)
    models = fit(train, epochs=epochs)
    vocab = [set(models[name]['labels']) for name in ROLES]
    out = []
    for i, sample in enumerate(test):
        channels = sample['channels']
        guess, states, confidence = decode_triple(models, channels)
        # Deliberately strip source->event timing and re-use unchanged
        # sensor measurements under a different held-out truth.
        shifted = test[(i + 1) % len(test)]['channels'] if len(test) > 1 else channels
        shuffle_guess, _, _ = decode_triple(models, shifted)
        role_swap = (channels[2], channels[1], channels[0])
        swap_guess, _, _ = decode_triple(models, role_swap)
        missing = tuple(tuple(v[1:]) for v in channels)
        drop_guess, _, _ = decode_triple(models, missing)
        out.append({'tick': sample['tick'], 'truth': sample['truth'],
                    'guess': guess, 'correct': guess == sample['truth'],
                    'shuffled_correct': shuffle_guess == sample['truth'],
                    'swapped_role_correct': swap_guess == sample['truth'],
                    'one_missing_head_correct': drop_guess == sample['truth'],
                    'statuses': states, 'role_confidences': confidence,
                    'recoverable_vocabulary': all(sample['truth'][r] in vocab[r]
                                                  for r in range(3))})
    n = len(out)
    return {
        'experiment': 'native_role_probe_supervised_decoder_v1',
        'evaluation': 'chronological_whole_claim_disjoint',
        'training_events': len(train), 'test_events': n,
        'train_sources': len({r['source'] for r in train}),
        'test_sources': len({r['source'] for r in test}),
        'source_disjoint': not ({r['source'] for r in train} &
                                {r['source'] for r in test}),
        'role_vocab_sizes': {name: len(models[name]['labels']) for name in ROLES},
        'seen_complete_test_claims': sum(r['truth'] in {x['truth'] for x in train}
                                         for r in test),
        'in_vocab_test_events': sum(r['recoverable_vocabulary'] for r in out),
        'correct': sum(r['correct'] for r in out),
        'coverage': sum(r['guess'] is not None for r in out) / n,
        'accuracy_all': sum(r['correct'] for r in out) / n,
        'one_missing_head_correct': sum(r['one_missing_head_correct'] for r in out),
        'time_shuffled_correct': sum(r['shuffled_correct'] for r in out),
        'role_swapped_correct': sum(r['swapped_role_correct'] for r in out),
        'samples': out,
        'limitations': ('Role probe encodes subject/relation/object separately FROM '
                        'THE INPUT TEXT; it is auxiliary read-only instrumentation, '
                        'not semantic discovery from mixed MATS frames. Supervised '
                        'softmax learns each known lexical role, but cannot derive '
                        'unseen words or WordNet meanings; weak source-disjoint '
                        'small fixture, no causal held-out loss improvement.'),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecar')
    p.add_argument('--journal', required=True)
    p.add_argument('--role-trace', required=True)
    p.add_argument('--training-events', type=int, required=True)
    p.add_argument('--epochs', type=int, default=200)
    args = p.parse_args(argv)
    try:
        frames, _ = ex.load(args.sidecar)
        final_tick, events = ag.read_journal(args.journal)
        trace = load_role_trace(args.role_trace)
        output = evaluate(frames, final_tick, events, trace,
                          args.training_events, epochs=args.epochs)
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print('role decoder rejected: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())