#!/usr/bin/env python3
"""Read-only, journal-grounded conceptual language and exact ISA proof bridge.

Numeric MATS frames corroborate provenance counts and timeline; they DO NOT
encode/produce graph words. Only positive ISA paths are transitively inferred,
refutations remain direct, and contradictions are not collapsed into 'true'.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import sys

import sensor_temporal_export as exp
import sensor_temporal_aggregate as ag


def interpret(frames: list[dict], final_tick: int, events: list[dict],
              subject: str, relation: str, obj: str, max_hops=5) -> dict:
    if not isinstance(max_hops,int) or not 1 <= max_hops <= 16:
        raise ValueError('max_hops outside bounded path depth')
    if any(not x or len(x)>192 or '\t' in x or '\n' in x or '\r' in x
           for x in (subject,relation,obj)):
        raise ValueError('invalid query text')
    if not frames:
        raise ValueError('cannot ground query without verified frames')
    ag.concepts(frames,final_tick,events)
    tick=frames[-1]['nanclock']
    observed=[e for e in events if e['tick']<=tick]
    direct=[e for e in observed if (e['subject'],e['relation'],e['object'])==
            (subject,relation,obj)]
    support=sorted({e['source'] for e in direct if e['polarity']=='+'})
    refute=sorted({e['source'] for e in direct if e['polarity']=='-'})
    proof_nodes=[];proof_sources=[];proof_ticks=[];proof_edge_sources=[]
    # Native sensor's only supported logical extension is POSITIVE ISA.
    # Deliberately avoid negation-as-failure or transitivity for other types.
    if relation=='ISA' and subject!=obj and max_hops>=2:
        adj=defaultdict(dict)
        for e in observed:
            if e['relation']!='ISA' or e['polarity']!='+':
                continue
            key=(e['subject'],e['object'])
            old=adj[e['subject']].get(e['object'])
            if old is None or (e['source'],e['tick']) < (old['source'],old['tick']):
                adj[e['subject']][e['object']]=e
        todo=deque([(subject,(subject,),(),())])
        best_depth={subject:0}
        steps=0
        while todo and steps<4096:
            current,path,sources,times=todo.popleft();steps+=1
            depth=len(path)-1
            if depth>=max_hops:continue
            for dest,e in sorted(adj[current].items()):
                if dest in path:continue
                np=path+(dest,)
                ns=sources+(e['source'],)
                nt=times+(e['tick'],)
                if dest==obj and len(np)>=3:
                    proof_nodes=list(np)
                    proof_sources=sorted(set(ns))
                    proof_edge_sources=list(ns)  # One source per ordered proof edge.
                    proof_ticks=list(nt)
                    break
                if best_depth.get(dest,999)>depth+1:
                    best_depth[dest]=depth+1
                    todo.append((dest,np,ns,nt))
            if proof_nodes:break
    all_support=sorted(set(support)|set(proof_sources))
    state='both' if all_support and refute else ('supports' if all_support else
            'refutes' if refute else 'unknown')
    article='an' if obj[:1].lower() in 'aeiou' else 'a'
    phrase=(f'{subject} is {article} {obj}' if relation=='ISA' else
            f'{subject} [{relation}] {obj}')
    if state=='both':
        sentence=f'Conflicting evidence: {phrase}; supporting and refuting sources coexist.'
    elif state=='supports':
        sentence=f'Evidence supports: {phrase}.'
    elif state=='refutes':
        sentence=f'Evidence refutes: {phrase}.'
    else:
        sentence=f'No verified claim or supported inference: {phrase}.'
    return {'subject':subject,'relation':relation,'object':obj,
            'phrase':phrase,'sentence':sentence,'evidence_state':state,
            'direct_support_sources':support,'direct_refute_sources':refute,
            'support_sources':all_support, 'refute_sources':refute,
            'inferred':bool(proof_nodes),'proof_nodes':proof_nodes,
            'proof_sources':proof_sources,'proof_event_ticks':proof_ticks,
            'proof_edge_sources':proof_edge_sources,
            'proof_edges':[{'from':left,'to':right,'source':source,'tick':event_tick}
                           for left,right,source,event_tick in zip(
                               proof_nodes,proof_nodes[1:],proof_edge_sources,proof_ticks)],
            'journal_event_count':len(observed),'last_verified_tick':tick,
            'provenance':'journal lexical facts; independent MATS frame count alignment',
            'non_explosion':True, 'semantics':'bounded positive ISA closure only; not model language learning'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sidecar')
    p.add_argument('--journal',required=True)
    p.add_argument('--subject',required=True)
    p.add_argument('--relation',required=True)
    p.add_argument('--object',required=True)
    p.add_argument('--max-hops',type=int,default=5)
    args=p.parse_args(argv)
    try:
        frames,_=exp.load(args.sidecar)
        final_tick,events=ag.read_journal(args.journal)
        r=interpret(frames,final_tick,events,args.subject,args.relation,
                    args.object,args.max_hops)
        print(json.dumps(r,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,UnicodeError) as exc:
        print('reason bridge rejected: '+str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())