#!/usr/bin/env python3
"""Auditable numerical->lexical->evidence readout of one NaNLM sensor trace.

Numerical word prediction uses the experimental NLRP1 role-probe classifier.
Meaning/truth is NOT inferred from hashed numbers; it is separately checked
against the source-journal graph at the event and preceding clocks.
"""
from __future__ import annotations

import argparse
import json
import sys

import sensor_temporal_aggregate as agg
import sensor_temporal_export as exp
import sensor_temporal_role_decoder as rd
import sensor_temporal_reason_bridge as reason


def readout(frames, final_tick, events, numeric_roles, train_events, max_items=64):
    if not 1 <= max_items <= 256:
        raise ValueError('max_items outside 1..256')
    results=rd.evaluate(frames,final_tick,events,numeric_roles,train_events)
    data=[]
    for sample in results['samples'][:max_items]:
        tick=sample['tick']
        guess=sample['guess']
        base={'tick':tick,'numeric_role_guess':guess,
              'role_probabilities_uncalibrated':sample['role_confidences'],
              'role_statuses':sample['statuses'],
              'readout_kind':'trained_role_probe' if guess else 'abstain',
              'numeric_predicted_event_is_ground_truth':sample['correct']}
        if guess is None:
            base.update({'phrase':None,'evidence_state_at_tick':None,
                         'evidence_state_before_tick':None,'journal_sources':[],
                         'proof':[]})
        else:
            # After prediction: lookup source evidence *for the predicted
            # triple*, and distinguish evidence before and after this event.
            past=[f for f in frames if f['nanclock']<tick]
            present=[f for f in frames if f['nanclock']<=tick]
            pre=reason.interpret(past,final_tick,events,*guess)
            now=reason.interpret(present,final_tick,events,*guess)
            base.update({'phrase':now['phrase'],
                         'sentence':now['sentence'],
                         'evidence_state_before_tick':pre['evidence_state'],
                         'evidence_state_at_tick':now['evidence_state'],
                         'journal_sources':{'support':now['support_sources'],
                                            'refute':now['refute_sources']},
                         'proof':now['proof_nodes'],
                         'inferred_from_previous_evidence':pre['inferred'],
                         'provenance_separation':{
                             'lexical_identity':'supervised numerical role probe',
                             'evidence':'authoritative V2 journal',
                             'temporal':'verified MATS v1 sidecar',
                             'warning':'no independent semantic generalization or causality'}})
        data.append(base)
    summaries=agg.aggregate(frames)
    global_history=next((s for s in summaries if s.get('level')=='GLOBAL'
                         and s.get('kind')=='coordinate'),None)
    hot=[s for s in summaries if s.get('level')=='HOT'
         and s.get('kind')=='coordinate']
    mirrors=[s for s in summaries if s.get('kind')=='mirror_pair']
    return {'experiment':'temporal_full_readout_v1',
            'decoded_items':len(data),
            'source_disjoint_training':results['source_disjoint'],
            'model_accuracy_in_fixture':results['accuracy_all'],
            'numerical_aggregation':{
                'verified_frames':len(frames),
                'last_tick':frames[-1]['nanclock'] if frames else None,
                'global_mean_vector':global_history['mean_vector'] if global_history else None,
                'hot_triad_histories':len(hot),
                'max_hot_tick_occupancy':max((r['tick_occupancy'] for r in hot),default=None),
                'canonical_mirror_pair_histories':len(mirrors),
                'attribution':'global/hot aggregates are NOT per-word semantics'},
            'items':data,
            'limitations':'Word labels from trained role signatures; source facts from the V2 journal. No unseen word generation, no physical standing-wave conclusion, no language CE gain.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sidecar')
    parser.add_argument('--journal',required=True)
    parser.add_argument('--role-trace',required=True)
    parser.add_argument('--training-events',type=int,required=True)
    parser.add_argument('--max-items',type=int,default=64)
    args=parser.parse_args(argv)
    try:
        frames,_=exp.load(args.sidecar)
        final_tick,events=agg.read_journal(args.journal)
        roles=rd.load_role_trace(args.role_trace)
        report=readout(frames,final_tick,events,roles,args.training_events,args.max_items)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,UnicodeError) as exc:
        print('full readout rejected: '+str(exc),file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())