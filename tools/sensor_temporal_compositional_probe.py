#!/usr/bin/env python3
"""Experimental closed-vocabulary compositional readout of MATS event signatures.

This is a forward-coded inverse lookup of the sensor's PUBLIC deterministic
hash routing, not learned semantic understanding or free-form generation.
Test labels are never used to build the candidate vocabulary. The vocabulary
is extracted exclusively from the training events. A unique exact six-head
match is required for a lexical reconstruction; otherwise abstain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

import sensor_temporal_aggregate as ag
import sensor_temporal_concept_probe as probe
import sensor_temporal_export as exp

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
BASIS_FACTOR = 0xD6E8FEB86659FD93
BASIS_SEED = 0x4E614E4C4D485730
SUBJECT_ROLE = 0x5355424A454354
PREDICATE_ROLE = 0x50524544494354
OBJECT_ROLE = 0x4F424A454354
HEAD_COUNT = 255
DEFAULT_MAX_CANDIDATES = 4096


def mix(x: int) -> int:
    x = (x + GAMMA) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK


@lru_cache(maxsize=32768)
def fingerprint(text: str) -> int:
    h = 14695981039346656037
    for b in text.encode('utf-8'):
        h = ((h ^ b) * 1099511628211) & MASK
    return mix(h)


@lru_cache(maxsize=32768)
def basis(label: int) -> tuple[int, int, int, int]:
    seed = BASIS_SEED ^ ((label * BASIS_FACTOR) & MASK)
    return tuple(mix((seed + i * GAMMA) & MASK) for i in range(4))


@lru_cache(maxsize=32768)
def role_vector(text: str, role: int) -> tuple[int, int, int, int]:
    return basis(mix(fingerprint(text) ^ role))


def claim_vector(subject: str, relation: str, obj: str) -> tuple[int, int, int, int]:
    s, r, o = (role_vector(subject, SUBJECT_ROLE),
               role_vector(relation, PREDICATE_ROLE),
               role_vector(obj, OBJECT_ROLE))
    return tuple(a ^ b ^ c for a, b, c in zip(s, r, o))


@lru_cache(maxsize=255)
def head_basis(head: int) -> tuple[int, int, int, int]:
    return basis(0x100000000000 + head)


@lru_cache(maxsize=131072)
def expected_heads(subject: str, relation: str, obj: str) -> frozenset[int]:
    v = claim_vector(subject, relation, obj)
    ranked = []
    for head in range(HEAD_COUNT):
        d = sum((a ^ b).bit_count() for a, b in zip(v, head_basis(head)))
        ranked.append((d, head))
    return frozenset(head for _, head in sorted(ranked)[:6])


def training_vocabulary(training: list[dict]) -> tuple[tuple[str, ...], ...]:
    return (tuple(sorted({r['truth'][0] for r in training})),
            tuple(sorted({r['truth'][1] for r in training})),
            tuple(sorted({r['truth'][2] for r in training})))


def candidate_index(training: list[dict], max_candidates=DEFAULT_MAX_CANDIDATES):
    if max_candidates <= 0:
        raise ValueError('max_candidates must be positive')
    vocabs = training_vocabulary(training)
    count = len(vocabs[0]) * len(vocabs[1]) * len(vocabs[2])
    if count > max_candidates:
        raise ValueError(f'candidate product {count} exceeds bound {max_candidates}')
    index: dict[frozenset[int], list[tuple[str, str, str]]] = {}
    for label in product(*vocabs):
        signature = expected_heads(*label)
        index.setdefault(signature, []).append(label)
    return index, vocabs, count


def decode_exact(index, observed: frozenset[int]):
    """Never invent unseen words: exact head agreement + unique candidate only."""
    if len(observed) != 6:
        return None, 'invalid_head_count'
    matched = index.get(observed, ())
    if len(matched) == 1:
        return matched[0], 'unique_exact'
    if matched:
        return None, 'ambiguous_collision'
    return None, 'no_exact_candidate'


def fit_role_credit_prototypes(training, epochs=120, rate=0.15):
    """Hash-blind local residual-credit comparison, never BPTT.

    Each observed 255-channel head-change pattern is a target; each lexical
    role token has a 255D real-valued prototype, updated using only that
    event's residual. This learner may fail because the hash routing was not
    designed to preserve additive lexical meaning.
    """
    if epochs < 1 or not 0 < rate <= 1:
        raise ValueError('invalid local-credit optimizer settings')
    proto = {(role, token): [0.] * HEAD_COUNT
             for event in training for role, token in enumerate(event['truth'])}
    for _ in range(epochs):
        for event in training:
            keys = [(role, token) for role, token in enumerate(event['truth'])]
            parts = [proto[k] for k in keys]
            active = event['heads']
            for i in range(HEAD_COUNT):
                error = float(i in active) - sum(p[i] for p in parts)
                step = rate * error / 3.
                for p in parts:
                    p[i] += step
    return proto


def predict_role_credit(proto, observed, vocabs):
    """Rank candidate triples without invoking claim_vector or expected_heads."""
    best = None
    best_score = -float('inf')
    norm_x = len(observed) ** 0.5
    if norm_x == 0:
        return None, None
    for triple in product(*vocabs):
        parts = [proto[(role, token)] for role, token in enumerate(triple)]
        dot = sum(sum(part[i] for part in parts) for i in observed)
        norm_p2 = sum(sum(part[i] for part in parts)**2 for i in range(HEAD_COUNT))
        score = dot / (norm_x * norm_p2**0.5) if norm_p2 > 1e-16 else -1.
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12
                                         and (best is None or triple < best)):
            best, best_score = triple, score
    return best, best_score


def static_heads(frame: dict) -> frozenset[int]:
    """Intentionally weak static-only control, includes accumulated history."""
    return frozenset(r['id'] for r in frame['rows']
                     if r['level'] == 'HEAD' and (r['support'] > 1e-4 or
                                                 r['refute'] > 1e-4))


def evaluate(frames: list[dict], journal_tick: int, events: list[dict],
             training_events: int, max_candidates=DEFAULT_MAX_CANDIDATES) -> dict:
    if training_events < 1:
        raise ValueError('training_events must be positive')
    ag.concepts(frames, journal_tick, events)  # Bound journal counts and chronology.
    rows = probe.signatures(frames, events)
    events_ordered = sorted(events, key=lambda e: e['tick'])
    if len(events_ordered) <= training_events:
        raise ValueError('no held-out events')
    cutoff_tick = events_ordered[training_events - 1]['tick']
    training = [r for r in rows if r['tick'] <= cutoff_tick]
    test = [r for r in rows if r['tick'] > cutoff_tick]
    expected_test_ticks = {r['tick'] for r in events_ordered[training_events:]}
    if not training or set(r['tick'] for r in test) != expected_test_ticks:
        raise ValueError('not every test event has an adjacent complete sensor record')
    idx, vocabs, count = candidate_index(training, max_candidates=max_candidates)
    seen = {r['truth'] for r in training}
    local_role_credit = fit_role_credit_prototypes(training)
    last_frame_at_tick = {f['nanclock']: f for f in frames}
    answer = []
    for i, sample in enumerate(test):
        truth = sample['truth']
        # Train only on earlier timeline. Test events are checked for
        # correctness AFTER the candidate index is frozen.
        if truth in seen:
            raise ValueError('test includes a previously indexed whole statement')
        guess, reason = decode_exact(idx, sample['heads'])
        frame = last_frame_at_tick[sample['tick']]
        static_guess, static_reason = decode_exact(idx, static_heads(frame))
        credit_guess, credit_score = predict_role_credit(
            local_role_credit, sample['heads'], vocabs)
        # Temporal-shuffle control: deterministic cyclic permutation of TEST
        # signatures, preserving label frequencies, head frequency, and
        # sequence lengths but severing event-to-signature alignment.
        shuffled = test[(i + 1) % len(test)]['heads'] if len(test) > 1 else frozenset()
        shuffled_guess, _ = decode_exact(idx, shuffled)
        answer.append({
            'test_tick': sample['tick'], 'truth': list(truth),
            'recoverable_lexicon': all(truth[k] in vocabs[k] for k in range(3)),
            'temporal_guess': list(guess) if guess else None,
            'temporal_status': reason,
            'temporal_correct': guess == truth,
            'static_guess': list(static_guess) if static_guess else None,
            'static_status': static_reason,
            'static_correct': static_guess == truth,
            'shuffled_correct': shuffled_guess == truth,
            'local_role_credit_guess': list(credit_guess) if credit_guess else None,
            'local_role_credit_cosine': credit_score,
            'local_role_credit_correct': credit_guess == truth,
            'observed_head_count': len(sample['heads']),
        })
    n = len(answer)
    eligible = sum(r['recoverable_lexicon'] for r in answer)
    return {
        'experiment': 'closed_vocab_forward_hash_composition_v1',
        'evaluation': 'chronological whole-claim-disjoint',
        'training_events': len(training), 'test_events': n,
        'previously_seen_test_statements': 0,
        'candidate_product': count,
        'lexicon_sizes': {'subjects': len(vocabs[0]), 'relations': len(vocabs[1]),
                          'objects': len(vocabs[2])},
        'recoverable_test_events': eligible,
        'temporal_exact': sum(x['temporal_correct'] for x in answer),
        'static_exact': sum(x['static_correct'] for x in answer),
        'shuffled_exact': sum(x['shuffled_correct'] for x in answer),
        'local_role_credit_exact': sum(x['local_role_credit_correct'] for x in answer),
        'abstentions': sum(x['temporal_guess'] is None for x in answer),
        'temporal_accuracy_all': sum(x['temporal_correct'] for x in answer) / n,
        'static_accuracy_all': sum(x['static_correct'] for x in answer) / n,
        'shuffled_accuracy_all': sum(x['shuffled_correct'] for x in answer) / n,
        'local_role_credit_accuracy_all': sum(x['local_role_credit_correct'] for x in answer) / n,
        'random_closed_vocab_accuracy_reference': 1. / count if count else None,
        'results': answer,
        'limitations': ('Known component vocabulary only; deterministic native hash '
                        'routing replicated in the candidate generator, not a learned '
                        'concept decoder. The hash-blind role-credit comparator is a '
                        'small local update surrogate, not production NaNLM geometric '
                        'credits. Unseen words cannot be recovered; exact '
                        'head signature can collide; static-only control is a basic '
                        'accumulated-head occupancy feature; not a language CE gain.'),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecar')
    p.add_argument('--journal', required=True)
    p.add_argument('--training-events', type=int, required=True)
    p.add_argument('--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES)
    args = p.parse_args(argv)
    try:
        frames, stats = exp.load(args.sidecar)
        tick, events = ag.read_journal(args.journal)
        report = evaluate(frames, tick, events, args.training_events, args.max_candidates)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        print('compositional probe rejected: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())