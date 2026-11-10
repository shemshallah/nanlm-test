#!/usr/bin/env python3
"""Scientific checks: role-probe learner is distinct from mixed-head inverse."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'tools'))
import sensor_temporal_role_decoder as rd
import sensor_temporal_compositional_probe as cp

TRAIN = [
 ('penguin','ISA','bird'),('eagle','ISA','animal'),
 ('whale','ISA','fish'),('penguin','LIKES','fish'),
 ('eagle','LIKES','bird'),('whale','LIKES','animal')]
TEST = [
 ('penguin','ISA','animal'),('eagle','ISA','fish'),
 ('whale','ISA','bird'),('penguin','LIKES','animal'),
 ('eagle','LIKES','fish'),('whale','LIKES','bird')]
ROLES=(cp.SUBJECT_ROLE,cp.PREDICATE_ROLE,cp.OBJECT_ROLE)


def heads_for_role(word,role):
    v=cp.role_vector(word,role)
    ranked=sorted((sum((a^b).bit_count() for a,b in zip(v,cp.head_basis(h))),h)
                  for h in range(255))
    return tuple(sorted(x[1] for x in ranked[:6]))


def fixture():
    events=[];frames=[];role_rows=[];state={}
    def snap(tick):
        frames.append({'seq':tick+1,'nanclock':tick,'sources':len({x['source'] for x in events}),
                       'facts':len({(x['subject'],x['relation'],x['object']) for x in events}),
                       'rows':[{'level':'HEAD','id':h,'support':val,'refute':0.}
                               for h,val in sorted(state.items())], 'mirrors':[]})
    snap(0)
    for tick, triple in enumerate(TRAIN+TEST,1):
        state={h:val*.97 for h,val in state.items()}
        for h in cp.expected_heads(*triple):state[h]=state.get(h,0.)+1.
        role_rows.append({'tick':tick, 'channels':tuple(heads_for_role(word,role)
                                              for word,role in zip(triple,ROLES))})
        events.append({'tick':tick,'source':f'src_{tick}', 'polarity':'+',
                       'subject':triple[0],'relation':triple[1],'object':triple[2]})
        snap(tick)
    return frames,events,role_rows

class RoleDecoderTests(unittest.TestCase):
    def test_native_numeric_format_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'role.trace'
            p.write_text('NLRP1\t2\t0,1,2,3,4,5\t6,7,8,9,10,11\t12,13,14,15,16,17\n',encoding='ascii')
            self.assertEqual(rd.load_role_trace(p)[0]['tick'],2)
            p.write_text(p.read_text()+'NLRP1\t2\t0,1,2,3,4,5\t6,7,8,9,10,11\t12,13,14,15,16,17\n')
            with self.assertRaisesRegex(ValueError,'regressing'):
                rd.load_role_trace(p)
            p.write_text('NLRP1\t2\t0,1,1,3,4,5\t6,7,8,9,10,11\t12,13,14,15,16,17\n')
            with self.assertRaisesRegex(ValueError,'topology'):
                rd.load_role_trace(p)
            p.write_text('NLRP1\t2\t0,1,2,3,4,255\t6,7,8,9,10,11\t12,13,14,15,16,17\n')
            with self.assertRaisesRegex(ValueError,'topology'):
                rd.load_role_trace(p)

    def test_source_and_whole_claim_disjoint_supervised_role_learning(self):
        frames,events,roles=fixture()
        out=rd.evaluate(frames,12,events,roles,6)
        self.assertEqual(out['training_events'],6)
        self.assertEqual(out['test_events'],6)
        self.assertEqual(out['seen_complete_test_claims'],0)
        self.assertTrue(out['source_disjoint'])
        self.assertEqual(out['in_vocab_test_events'],6)
        self.assertEqual(out['correct'],6)
        self.assertLess(out['time_shuffled_correct'],6)
        self.assertLess(out['role_swapped_correct'],6)
        self.assertEqual(out['one_missing_head_correct'],6)

    def test_unseen_role_abstains_and_source_alignment_enforced(self):
        frames,events,roles=fixture()
        train,test=rd.align_and_split(frames,12,events,roles,6)
        models=rd.fit(train)
        unknown=heads_for_role('new-unknown-word',ROLES[0])
        word,status,_=rd.decode_role(models['subject'],unknown)
        self.assertIsNone(word)
        self.assertEqual(status,'unseen_numeric_pattern')
        tampered=[dict(r) for r in roles]
        tampered[-1]['tick']=100
        with self.assertRaisesRegex(ValueError,'unaligned'):
            rd.align_and_split(frames,12,events,tampered,6)
        repeated=events.copy()
        repeated[-1]=dict(repeated[-1],subject='penguin',relation='ISA',object='bird')
        with self.assertRaises(ValueError):
            rd.align_and_split(frames,12,repeated,roles,6)

    def test_missing_frame_and_untrained_token_guard(self):
        frames,events,roles=fixture()
        with self.assertRaisesRegex(ValueError,'no verified'):
            rd.align_and_split([f for f in frames if f['nanclock']!=7],12,events,roles,6)
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            rd.align_and_split(frames,12,events,roles[:-1],6)

if __name__=='__main__':unittest.main()