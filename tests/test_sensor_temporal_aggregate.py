#!/usr/bin/env python3
"""Source-grounded semantic bridge and temporal aggregation regression tests."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import sensor_temporal_aggregate as ag
from test_sensor_temporal_export import row, edge, make_sidecar, encoded_frame
import sensor_temporal_export as exp

SCRIPT = Path(ag.__file__)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sidecar = self.root / 'claims.slices'
        self.journal = self.root / 'claims'

    def test_hot_persistence_and_missing_not_zero(self):
        hot = row(3, (1, 2, 3), v=(1., 0., 0.), support=1, refute=1)
        make_sidecar(self.sidecar, [
            encoded_frame(1, hot=(hot,)),
            encoded_frame(2, hot=(hot,)),
            encoded_frame(2, hot=()),
            encoded_frame(4, hot=(hot,)),
        ])
        frames, _ = exp.load(self.sidecar)
        a = next(r for r in ag.aggregate(frames) if r['level'] == 'HOT')
        self.assertEqual(a['samples'], 3)
        self.assertEqual(a['distinct_ticks'], 3)
        self.assertEqual(a['coactive_samples'], 3)
        self.assertEqual(a['max_contiguous_ticks'], 2)
        self.assertEqual(a['tick_occupancy'], 1.)
        self.assertEqual(a['periodicity'], 'insufficient_contiguous_ticks')

    def test_opposite_mirror_address_aggregates_one_canonical_pair(self):
        left = row(4, (0, 1, 2))
        right = row(5, (3, 4, 5))
        make_sidecar(self.sidecar, [
            encoded_frame(1, hot=(left, right), links=(edge(1, 3, 2.),)),
            encoded_frame(2, hot=(left, right), links=(edge(3, 1, -4.),)),
        ])
        frames, _ = exp.load(self.sidecar)
        pairs = [r for r in ag.aggregate(frames) if r['kind'] == 'mirror_pair']
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['samples'], 2)
        self.assertAlmostEqual(pairs[0]['mean_canonical_flux'], 3.)
        self.assertEqual(pairs[0]['positive_flux_samples'], 2)

    def test_oscillation_gate_versus_short_and_monotone_series(self):
        def s(values):
            return [(i + 1, i + 1, (float(x), 0., 0.)) for i, x in enumerate(values)]
        self.assertEqual(ag._periodicity(s([1, -1] * 4))['status'],
                         'insufficient_contiguous_ticks')
        periodic = ag._periodicity(s([1, -1] * 8))
        self.assertEqual(periodic['status'], 'oscillation_candidate_unvalidated')
        self.assertEqual(periodic['period_ticks'], 2)
        self.assertEqual(ag._periodicity(s([float(i) for i in range(16)]))['status'],
                         'not_detected')
        gapped = [(j + 1, j + 1 if j < 8 else j + 4,
                   (float(1 if j % 2 else -1), 0., 0.)) for j in range(16)]
        self.assertEqual(ag._periodicity(gapped)['status'],
                         'insufficient_contiguous_ticks')

    def test_contradiction_preserved_and_words_from_journal_only(self):
        # Two separate sources, three distinct claims; one direct refutation.
        self.journal.write_text('NANLM_SENSOR_V2\t4\n'
            'src_a\t+\tpenguin\tISA\tbird\t0x1p+0\t1\n'
            'src_b\t+\tpenguin\tISA\tanimal\t0x1p+0\t2\n'
            'src_a\t-\tpenguin\tISA\tanimal\t0x1p+0\t3\n', encoding='utf-8')
        make_sidecar(self.sidecar, [encoded_frame(3, (999., 999., 999.)),
                                         encoded_frame(4, (888., 888., 888.))])
        frames, _ = exp.load(self.sidecar)
        final_tick, events = ag.read_journal(self.journal)
        c = ag.concepts(frames, final_tick, events)
        self.assertEqual(len(c), 2)
        both = next(x for x in c if x['object'] == 'animal')
        self.assertEqual(both['evidence_state'], 'both')
        self.assertEqual(both['support_sources'], ['src_b'])
        self.assertEqual(both['refute_sources'], ['src_a'])
        self.assertEqual(both['phrase'], 'penguin is an animal')
        # Does not hallucinate arbitrary '999'/'888' lexical meaning.
        self.assertFalse(any('999' in x['phrase'] for x in c))
        self.assertEqual(both['aligned_sequences'], [1])
        self.assertIn('not attribution', both['temporal_alignment'])

    def test_journal_crosscheck_rejects_mismatched_sources(self):
        self.journal.write_text('NANLM_SENSOR_V2\t1\n'
            'src_a\t+\tx\tISA\ty\t0x1p+0\t1\n', encoding='utf-8')
        make_sidecar(self.sidecar, [encoded_frame(1)])  # fixture advertises 2 sources / 3 facts
        frames, _ = exp.load(self.sidecar)
        t, e = ag.read_journal(self.journal)
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            ag.concepts(frames, t, e)

    def test_journal_rejects_bad_timestamp_and_schema(self):
        self.journal.write_text('NANLM_SENSOR_V1\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'V2'):
            ag.read_journal(self.journal)
        self.journal.write_text('NANLM_SENSOR_V2\t3\n'
            's\t+\tx\tISA\ty\t0x1p+0\t2\n'
            's2\t-\tx\tISA\ty\t0x1p+0\t2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nonmonotonic'):
            ag.read_journal(self.journal)

    def test_cli_rejects_word_guessing_without_provenance(self):
        make_sidecar(self.sidecar, [encoded_frame(1)])
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.sidecar), 'concepts'],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('requires --journal', proc.stderr)
        proc = subprocess.run([sys.executable, str(SCRIPT), str(self.sidecar),
                               'aggregate', '--format', 'csv'],
                              capture_output=True, text=True, check=True)
        self.assertIn('mean_vector_0', proc.stdout)
        self.assertIn('GLOBAL', proc.stdout)


if __name__ == '__main__':
    unittest.main()