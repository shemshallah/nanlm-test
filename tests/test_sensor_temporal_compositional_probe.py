#!/usr/bin/env python3
"""Bounded closed-vocabulary decode and negative-control unit tests."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import sensor_temporal_compositional_probe as cp


class CompositionTests(unittest.TestCase):
    def test_hash_route_is_bounded_deterministic_and_role_sensitive(self):
        x = cp.expected_heads('penguin', 'ISA', 'bird')
        self.assertEqual(len(x), 6)
        self.assertEqual(x, cp.expected_heads('penguin', 'ISA', 'bird'))
        self.assertNotEqual(x, cp.expected_heads('bird', 'ISA', 'penguin'))
        self.assertNotEqual(x, cp.expected_heads('penguin', 'LIKES', 'bird'))
        self.assertTrue(all(0 <= h < 255 for h in x))

    def test_unseen_whole_claim_from_seen_words(self):
        triples = [('penguin', 'ISA', 'bird'), ('eagle', 'ISA', 'animal'),
                   ('penguin', 'LIKES', 'animal'), ('eagle', 'LIKES', 'bird')]
        train = [{'truth': t, 'heads': cp.expected_heads(*t)} for t in triples]
        index, lex, n = cp.candidate_index(train)
        self.assertEqual(n, 8)
        unseen = ('penguin', 'ISA', 'animal')
        self.assertNotIn(unseen, triples)
        guess, status = cp.decode_exact(index, cp.expected_heads(*unseen))
        self.assertEqual((guess, status), (unseen, 'unique_exact'))
        self.assertEqual(len(lex), 3)

    def test_unknown_word_abstains_and_collision_is_never_guessed(self):
        training = [{'truth': ('penguin', 'ISA', 'bird')}]
        index, _, _ = cp.candidate_index(training)
        guess, reason = cp.decode_exact(index, cp.expected_heads('platypus', 'ISA', 'bird'))
        self.assertIsNone(guess)
        self.assertEqual(reason, 'no_exact_candidate')
        key = cp.expected_heads('penguin', 'ISA', 'bird')
        guess, reason = cp.decode_exact({key: [('a', 'r', 'b'), ('c', 'r', 'd')]}, key)
        self.assertEqual((guess, reason), (None, 'ambiguous_collision'))
        self.assertEqual(cp.decode_exact(index, frozenset())[1], 'invalid_head_count')

    def test_candidate_explosion_rejected(self):
        training = [{'truth': (f's{i}', 'ISA', f'o{i}')} for i in range(10)]
        with self.assertRaisesRegex(ValueError, 'exceeds bound'):
            cp.candidate_index(training, max_candidates=50)


    def test_chronological_whole_claim_disjoint_ablation_synthetic(self):
        # Self-consistency simulation; native C++ interoperability is checked
        # independently by the GCC/Clang workflow.
        train = [('penguin', 'ISA', 'bird'), ('eagle', 'ISA', 'animal'),
                 ('whale', 'ISA', 'fish'), ('penguin', 'LIKES', 'fish'),
                 ('eagle', 'LIKES', 'bird'), ('whale', 'LIKES', 'animal')]
        heldout = [('penguin', 'ISA', 'animal'), ('eagle', 'ISA', 'fish'),
                   ('whale', 'ISA', 'bird'), ('penguin', 'LIKES', 'animal'),
                   ('eagle', 'LIKES', 'fish'), ('whale', 'LIKES', 'bird')]
        events, frames, state = [], [], {}
        def snapshot(tick):
            rows = [{'level': 'HEAD', 'id': h, 'support': vals[0],
                     'refute': vals[1]} for h, vals in sorted(state.items())]
            frames.append({'seq': tick + 1, 'nanclock': tick, 'rows': rows,
                           'sources': len(events), 'facts': len(events)})
        snapshot(0)
        for tick, triple in enumerate(train + heldout, 1):
            for val in state.values():
                val[0] *= .97
                val[1] *= .97
            for head in cp.expected_heads(*triple):
                state.setdefault(head, [0., 0.])[0] += 1.
            events.append({'tick': tick, 'source': f'source_{tick}',
                           'subject': triple[0], 'relation': triple[1],
                           'object': triple[2], 'polarity': '+'})
            snapshot(tick)
        r = cp.evaluate(frames, 12, events, training_events=len(train))
        self.assertEqual(r['test_events'], 6)
        self.assertEqual(r['previously_seen_test_statements'], 0)
        self.assertEqual(r['candidate_product'], 18)
        self.assertEqual(r['temporal_exact'], 6)
        self.assertEqual(r['shuffled_exact'], 0)
        self.assertEqual(r['static_exact'], 0)
        self.assertEqual(r['abstentions'], 0)

    def test_static_field_is_not_an_event_change(self):
        old = {'rows': [{'id': 1, 'level': 'HEAD', 'support': 1., 'refute': 0.},
                        {'id': 2, 'level': 'HEAD', 'support': .5, 'refute': .5}]}
        self.assertEqual(cp.static_heads(old), frozenset({1, 2}))


if __name__ == '__main__':
    unittest.main()