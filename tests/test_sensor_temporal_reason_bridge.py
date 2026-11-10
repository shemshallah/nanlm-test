#!/usr/bin/env python3
"""Native-logic contract: attributed source claims versus measured telemetry."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import sensor_temporal_reason_bridge as rb

class ReasonBridgeTests(unittest.TestCase):
    def setUp(self):
        self.events=[
            {'tick':1,'source':'src_a','polarity':'+','subject':'penguin','relation':'ISA','object':'bird'},
            {'tick':2,'source':'src_b','polarity':'+','subject':'bird','relation':'ISA','object':'animal'},
            {'tick':3,'source':'src_c','polarity':'-','subject':'penguin','relation':'ISA','object':'animal'},
            {'tick':4,'source':'src_d','polarity':'+','subject':'penguin','relation':'LIKES','object':'fish'}]
        self.frames=[{'seq':e['tick']+1,'nanclock':e['tick'],
                      'sources':e['tick'],'facts':e['tick']} for e in self.events]
        self.frames.insert(0,{'seq':1,'nanclock':0,'sources':0,'facts':0})
    def query(self,*triple,**kwargs):
        return rb.interpret(self.frames,4,self.events,*triple,**kwargs)
    def test_refutation_does_not_erase_positive_isa_proof(self):
        r=self.query('penguin','ISA','animal')
        self.assertEqual(r['evidence_state'],'both')
        self.assertEqual(r['proof_nodes'],['penguin','bird','animal'])
        self.assertEqual(r['proof_sources'],['src_a','src_b'])
        self.assertEqual(r['refute_sources'],['src_c'])
        self.assertEqual(r['proof_event_ticks'],[1,2])
        self.assertIn('Conflicting evidence:',r['sentence'])
    def test_no_arbitrary_inference_for_negative_or_non_isa(self):
        self.assertEqual(self.query('penguin','ISA','moon')['evidence_state'],'unknown')
        self.assertFalse(self.query('penguin','LIKES','animal')['inferred'])
        self.assertEqual(self.query('penguin','LIKES','fish')['support_sources'],['src_d'])
    def test_chain_depth_gate_and_sparse_history_alignment(self):
        r=self.query('penguin','ISA','animal',max_hops=1)
        self.assertEqual(r['evidence_state'],'refutes')
        self.assertFalse(r['inferred'])
        with self.assertRaisesRegex(ValueError,'counts disagree'):
            rb.interpret(self.frames[:-1]+[dict(self.frames[-1],sources=99)],4,self.events,
                         'penguin','ISA','animal')
    def test_subsequent_evidence_not_inferred_into_earlier_frame(self):
        early=self.frames[:3]
        q=rb.interpret(early,4,self.events,'penguin','ISA','animal')
        self.assertTrue(q['inferred'])
        self.assertEqual(q['evidence_state'],'supports')
        self.assertEqual(q['journal_event_count'],2)

if __name__=='__main__':unittest.main()