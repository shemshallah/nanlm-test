#!/usr/bin/env python3
"""Scientific-gate tests for sparse temporal concept retrieval."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import sensor_temporal_export as exp
import sensor_temporal_aggregate as agg
import sensor_temporal_concept_probe as probe
from test_sensor_temporal_export import encoded_frame, make_sidecar


def active_head(i, support):
    return struct.pack('<4B18f', i, i, 255, 255, 0., 0., 0., *(0.,)*9,
                       1., support, 0., 0., 0., 0.)


def native_format_sample(tick, sources, facts, heads):
    payload = bytearray(encoded_frame(tick, heads=heads))
    struct.pack_into('<II', payload, 16, sources, facts)
    return payload


class ConceptProbeTests(unittest.TestCase):
    def test_source_disjoint_known_claim_retrieval_beats_majority(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 's.slices'
            journal = Path(td) / 's'
            journal.write_text('NANLM_SENSOR_V2\t4\n'
                'a\t+\tx\tISA\ty\t0x1p+0\t1\n'
                'a\t+\tz\tISA\tq\t0x1p+0\t2\n'
                'b\t+\tx\tISA\ty\t0x1p+0\t3\n'
                'b\t+\tz\tISA\tq\t0x1p+0\t4\n', encoding='utf-8')
            frames = [
                native_format_sample(0, 0, 0, []),
                native_format_sample(1, 1, 1, [active_head(i, 1.) for i in (1, 2)]),
                native_format_sample(2, 1, 2, [active_head(i, .97) for i in (1, 2)] +
                                               [active_head(i, 1.) for i in (3, 4)]),
                native_format_sample(3, 2, 2, [active_head(i, 1.9409) for i in (1, 2)] +
                                               [active_head(i, .97) for i in (3, 4)]),
                native_format_sample(4, 2, 2, [active_head(i, 1.882673) for i in (1, 2)] +
                                               [active_head(i, 1.9409) for i in (3, 4)]),
            ]
            make_sidecar(path, frames)
            verified, _ = exp.load(path)
            end, events = agg.read_journal(journal)
            result = probe.evaluate(verified, end, events)
            self.assertEqual(result['event_signatures'], 4)
            self.assertEqual(result['eligible_known_concept_events'], 4)
            self.assertEqual(result['accuracy_all_eligible'], 1.0)
            self.assertEqual(result['most_frequent_training_label_baseline'], .5)

    def test_unknown_and_collision_abstain(self):
        train = [{'truth': ('x', 'ISA', 'y'), 'heads': frozenset((1, 2, 3))}]
        self.assertIsNone(probe.predict(train, frozenset((7, 8, 9)))[0])
        tied = train + [{'truth': ('a', 'ISA', 'b'), 'heads': frozenset((1, 2, 3))}]
        self.assertIsNone(probe.predict(tied, frozenset((1, 2, 3)))[0])

    def test_event_requires_adjacent_frames(self):
        a = {'seq': 1, 'nanclock': 0, 'rows': []}
        b = {'seq': 4, 'nanclock': 1, 'rows': [{'level': 'HEAD', 'id': 1,
             'support': 1., 'refute': 0.}]}
        events = [{'tick': 1, 'source': 'a', 'subject': 'x', 'relation': 'ISA',
                   'object': 'y', 'polarity': '+'}]
        self.assertEqual(probe.signatures([a, b], events), [])


if __name__ == '__main__':
    unittest.main()