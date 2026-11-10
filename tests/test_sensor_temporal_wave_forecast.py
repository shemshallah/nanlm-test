#!/usr/bin/env python3
"""Finite, chronological wave candidates and null-case guards."""
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import sensor_temporal_wave_forecast as wave


def frame(tick, vector, seq=None, level='GLOBAL', heads=(255, 255, 255)):
    return {'seq': seq if seq is not None else tick + 1, 'nanclock': tick,
            'rows': [{'id': 255, 'level': level, 'heads': heads, 'vector': vector}]}


class WaveForecastTests(unittest.TestCase):
    def test_sinusoid_predictive_ar2_candidate(self):
        w = 2*math.pi/12
        frames = [frame(t, (math.sin(t*w), math.cos(t*w), 0.3*math.sin(t*w)))
                  for t in range(90)]
        r = wave.forecast(frames, min_ticks=60, holdout_ticks=20)
        self.assertEqual(r['status'], 'evaluated')
        self.assertTrue(r['oscillatory_predictive_candidate'])
        self.assertAlmostEqual(r['candidate_period_ticks'], 12, places=2)
        self.assertLess(r['holdout_rmse']['ar2'], 1e-5)
        self.assertLess(r['holdout_rmse']['ar2'], r['holdout_rmse']['known_leak'])

    def test_decay_is_not_mistaken_for_wave(self):
        frames = [frame(t, (0.97**t, 2*0.97**t, -0.97**t)) for t in range(80)]
        r = wave.forecast(frames, min_ticks=30, holdout_ticks=20)
        self.assertEqual(r['status'], 'evaluated')
        self.assertFalse(r['oscillatory_predictive_candidate'])
        self.assertLess(r['holdout_rmse']['known_leak'], 1e-14)

    def test_gaps_and_sparse_disappearance_abstain(self):
        frames = [frame(t, (1., 2., 3.)) for t in range(15)] + [
            frame(t, (1., 2., 3.)) for t in range(18, 33)]
        r = wave.forecast(frames, min_ticks=24, holdout_ticks=8)
        self.assertEqual(r['status'], 'insufficient_contiguous_history')
        frames = [frame(t, (1., 2., 3.), seq=t+1) for t in range(30)]
        frames[12]['rows'] = []
        self.assertEqual(wave.forecast(frames, min_ticks=28)['status'],
                         'insufficient_contiguous_history')

    def test_same_tick_is_one_sample_not_a_cycle(self):
        frames = [frame(0, (1., 0., 0.), seq=1),
                  frame(0, (2., 0., 0.), seq=2)]
        frames.extend(frame(t, (1., 0., 0.), seq=t+2) for t in range(1, 25))
        r = wave.forecast(frames, min_ticks=24)
        self.assertEqual(r['collapsed_same_tick_samples'], 1)
        self.assertEqual(r['retained_distinct_ticks'], 25)
        self.assertFalse(r['oscillatory_predictive_candidate'])

    def test_membership_change_blocks_temporal_comparison(self):
        frames = [frame(t, (1., 0., 0.), level='HOT',
                        heads=(1, 2, 3) if t < 15 else (1, 2, 4))
                  for t in range(30)]
        r = wave.forecast(frames, level='HOT', min_ticks=24)
        self.assertEqual(r['status'], 'insufficient_contiguous_history')


if __name__ == '__main__':
    unittest.main()