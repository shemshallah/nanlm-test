#!/usr/bin/env python3
"""End-to-end numerical signature, evidence state, and history separation."""
import sys
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sensor_temporal_full_readout as full
from test_sensor_temporal_role_decoder import fixture

class FullReadoutTests(unittest.TestCase):
    def test_decoded_words_have_separate_evidence_and_numerical_history(self):
        frames,events,trace=fixture()
        with mock.patch.object(full.agg,'aggregate',return_value=[]):
            report=full.readout(frames,12,events,trace,6)
        self.assertEqual(report['decoded_items'],6)
        self.assertTrue(report['source_disjoint_training'])
        self.assertEqual(report['numerical_aggregation']['verified_frames'],13)
        self.assertEqual(report['numerical_aggregation']['last_tick'],12)
        self.assertIsNone(report['numerical_aggregation']['global_mean_vector'])
        a=report['items'][0]
        self.assertEqual(a['numeric_role_guess'],('penguin','ISA','animal'))
        self.assertEqual(a['phrase'],'penguin is an animal')
        self.assertEqual(a['evidence_state_before_tick'],'unknown')
        self.assertEqual(a['evidence_state_at_tick'],'supports')
        self.assertEqual(a['journal_sources']['support'],['src_7'])
        self.assertEqual(a['provenance_separation']['lexical_identity'],
                         'supervised numerical role probe')
    def test_refuses_overlarge_query_and_tampered_role_trace(self):
        frames,events,trace=fixture()
        with self.assertRaisesRegex(ValueError,'max_items'):
            full.readout(frames,12,events,trace,6,max_items=999)
        tampered=list(trace)
        tampered[-1]=dict(tampered[-1],tick=999)
        with self.assertRaisesRegex(ValueError,'unaligned'):
            full.readout(frames,12,events,tampered,6)

    def test_missing_prior_tick_is_unavailable_not_prior_evidence(self):
        frames, events, trace = fixture()
        # Insert a genuine idle-clock hole before the first held-out event.
        # The earlier snapshot is verified but cannot stand for tick 7.
        frames = [dict(f, nanclock=f['nanclock'] + (f['nanclock'] >= 7))
                  for f in frames]
        events = [dict(e, tick=e['tick'] + (e['tick'] >= 7)) for e in events]
        trace = [dict(r, tick=r['tick'] + (r['tick'] >= 7)) for r in trace]
        with mock.patch.object(full.agg, 'aggregate', return_value=[]):
            report = full.readout(frames, 13, events, trace, 6)
        first = report['items'][0]
        self.assertEqual(first['tick'], 8)
        self.assertEqual(first['prior_verified_tick'], 6)
        self.assertEqual(first['evidence_before_tick_status'], 'unverified_gap')
        self.assertIsNone(first['evidence_state_before_tick'])
        self.assertIsNone(first['inferred_from_previous_evidence'])
        self.assertEqual(first['evidence_state_at_tick'], 'supports')

if __name__=='__main__':unittest.main()