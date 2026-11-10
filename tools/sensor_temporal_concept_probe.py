#!/usr/bin/env python3
"""Read-only source-disjoint lexical retrieval from temporal head changes.

Not a generative decoder or a trainable production model. Learn a sparse
prototype INDEX from journal-labeled event signatures; test unseen sources.
Only labels seen in training can be returned. Requires adjacent complete
MATS frames and the default experimental per-tick sensor leak (0.97).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import sys

import sensor_temporal_aggregate as ag
import sensor_temporal_export as exp


def signatures(frames, events, leak=0.97, tolerance=1e-4):
    if not 0 <= leak < 1 or tolerance <= 0:
        raise ValueError('invalid leak or residual threshold')
    # Choose first verified post-ingest sample at its tick, and latest sample
    # at previous tick. Sequence adjacency prevents false event attribution.
    by_tick = defaultdict(list)
    for frame in frames:
        by_tick[frame['nanclock']].append(frame)
    result = []
    for event in events:
        tick = event['tick']
        old = by_tick.get(tick - 1, [])
        new = by_tick.get(tick, [])
        if not old or not new:
            continue
        before, after = old[-1], new[0]
        if after['seq'] != before['seq'] + 1:
            continue
        prev = {r['id']: r for r in before['rows'] if r['level'] == 'HEAD'}
        curr = {r['id']: r for r in after['rows'] if r['level'] == 'HEAD'}
        changed = set()
        pos = neg = 0.0
        for head, row in curr.items():
            old_row = prev.get(head)
            if old_row is None:
                # Newly observed sparse address, not an assumed numerical zero.
                if row['support'] > tolerance or row['refute'] > tolerance:
                    changed.add(head)
                pos += row['support']
                neg += row['refute']
                continue
            support_gain = row['support'] - leak * old_row['support']
            refute_gain = row['refute'] - leak * old_row['refute']
            if max(abs(support_gain), abs(refute_gain)) > tolerance:
                changed.add(head)
                pos += max(0.0, support_gain)
                neg += max(0.0, refute_gain)
        if not changed:
            continue
        polarity = '-' if neg > pos else '+' if pos > neg else 'undetermined'
        result.append({
            'tick': tick, 'seq': after['seq'], 'source': event['source'],
            'truth': (event['subject'], event['relation'], event['object']),
            'truth_polarity': event['polarity'], 'recovered_polarity': polarity,
            'heads': frozenset(changed), 'head_count': len(changed),
        })
    return result


def similarity(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def predict(train, feature, threshold=0.5, margin=0.1):
    """Prototype nearest neighbor over numeric head IDs, not lexical hashes."""
    prototypes = defaultdict(list)
    for row in train:
        prototypes[row['truth']].append(row['heads'])
    ranked = sorted(((max(similarity(feature, sample) for sample in samples), label)
                     for label, samples in prototypes.items()),
                    key=lambda pair: (-pair[0], pair[1]))
    if not ranked:
        return None, 0.0
    best, label = ranked[0]
    next_best = ranked[1][0] if len(ranked) > 1 else 0.0
    if best < threshold or best - next_best < margin:
        return None, best
    return label, best


def evaluate(frames, journal_tick, events):
    # Check source alignment before turning any numerical sample into a label.
    ag.concepts(frames, journal_tick, events)
    signatures_all = signatures(frames, [e for e in events
                                         if frames and e['tick'] <= frames[-1]['nanclock']])
    sources = sorted({x['source'] for x in signatures_all})
    eligible = correct = predicted = polarity_correct = 0
    majority_correct = 0
    new_labels = 0
    by_source = []
    for withheld in sources:
        train = [r for r in signatures_all if r['source'] != withheld]
        test = [r for r in signatures_all if r['source'] == withheld]
        labels = Counter(r['truth'] for r in train)
        majority = sorted(labels, key=lambda k: (-labels[k], k))[0] if labels else None
        n = good = returned = unseen = 0
        for sample in test:
            if sample['truth'] not in labels:
                unseen += 1
                new_labels += 1
                continue  # Cannot recognize labels absent from dictionary.
            n += 1
            eligible += 1
            baseline = majority == sample['truth']
            majority_correct += int(baseline)
            guess, _score = predict(train, sample['heads'])
            if guess is not None:
                returned += 1
                predicted += 1
                if guess == sample['truth']:
                    good += 1
                    correct += 1
            if sample['recovered_polarity'] == sample['truth_polarity']:
                polarity_correct += 1
        by_source.append({'heldout_source': withheld, 'eligible': n, 'correct': good,
                          'predicted': returned, 'unseen_labels': unseen})
    return {
        'experiment': 'event_head_fingerprint_source_disjoint_probe_v1',
        'source_folds': len(sources), 'event_signatures': len(signatures_all),
        'eligible_known_concept_events': eligible,
        'unseen_concept_events_excluded': new_labels,
        'predicted': predicted, 'correct': correct,
        'coverage': predicted / eligible if eligible else None,
        'accuracy_at_prediction': correct / predicted if predicted else None,
        'accuracy_all_eligible': correct / eligible if eligible else None,
        'most_frequent_training_label_baseline': majority_correct / eligible if eligible else None,
        'recovered_polarity_accuracy': polarity_correct / eligible if eligible else None,
        'folds': by_source,
        'limitations': ('Known-claim retrieval only; no unseen vocabulary or free text; '
                        'source labels withheld during matching; no downstream CE gain measured; '
                        'fixed 0.97 experimental leak; complete adjacent frames required'),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecar')
    p.add_argument('--journal', required=True)
    args = p.parse_args(argv)
    try:
        frames, _ = exp.load(args.sidecar)
        journal_tick, events = ag.read_journal(args.journal)
        result = evaluate(frames, journal_tick, events)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, UnicodeError, ValueError, OverflowError) as exc:
        print('concept probe rejected: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())