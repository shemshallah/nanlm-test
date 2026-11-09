#!/usr/bin/env python3
"""Reproducible diagnostic probes of the two PyTorch NaNLM reference tracks.

Print known gaps as explicit AUDIT_* measurements; do not mistake a passing
smoke test for held-out fluency or transactional optimizer correctness.
"""
from __future__ import annotations
from math import comb
import numpy as np
import torch
from experiments.independent_heads.model import IndependentHead, HeadConfig
from experiments.independent_heads.quant_bank import QuantizedBank, BankConfig, memory_projection
from experiments.independent_heads.field import TriadField
from experiments.sparse_shared.model import SparseConfig, SparseSharedModel, parameter_budget

def check(ok, message):
    if not ok: raise AssertionError(message)

def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    print("AUDIT_VERSION torch="+torch.__version__+" numpy="+np.__version__,flush=True)
    head_cfg=HeadConfig(d_model=16,n_heads=2,mlp=32,context=8)
    model=IndependentHead(head_cfg).eval()
    ids=torch.tensor([[11,22,33,44],[33,11,22,44]],dtype=torch.long)
    with torch.no_grad():
        logits=model(ids)
    delta=float((logits[0,-1]-logits[1,-1]).abs().max())
    print(f"AUDIT_POSITION head_last_logit_permutation_delta={delta:.9g}",flush=True)
    check(delta<1e-5,"expected permutation symmetry of position-free 1-layer head")

    sparse_cfg=SparseConfig(d_model=32,n_heads=4,layers=1,mlp=64,context=8,active=8)
    shared=SparseSharedModel(sparse_cfg).eval()
    usage_before=float(shared.adapters.usage.sum())
    with torch.no_grad():
        y,_=shared(ids,active=8)
    delta2=float((y[0,-1]-y[1,-1]).abs().max())
    usage_after=float(shared.adapters.usage.sum())
    print(f"AUDIT_POSITION trunk_last_logit_permutation_delta={delta2:.9g}",flush=True)
    print(f"AUDIT_SIDE_EFFECT inference_usage_delta={usage_after-usage_before:.0f}",flush=True)
    check(delta2<1e-5,"expected permutation symmetry of position-free 1-layer shared trunk")
    check(usage_after>usage_before,"expected current inference usage mutation")

    # Rejected optimizer candidate: prove that weight-only restore does not
    # restore int8 Adam moments; the training loop calls bank.store on rollback.
    params=sum(v.numel() for v in model.parameters())
    bank=QuantizedBank(BankConfig(heads=1,params=params))
    try:
        bank.store(0,model)
        initial_model={k:v.detach().clone() for k,v in model.state_dict().items()}
        old_m=bank.moments[0].copy()
        model.train()
        batch=torch.tensor([[1,2,3,4,5,6,7,8]],dtype=torch.long)
        model.zero_grad(set_to_none=True)
        model.loss(batch).backward()
        bank.adam_step(0,model,lr=1e-3,t=1)
        model.load_state_dict(initial_model)
        bank.store(0,model)  # current rollback path
        changed=int(np.count_nonzero(bank.moments[0]!=old_m))
        print(f"AUDIT_TRANSACTION rollback_stale_adam_moment_elements={changed}",flush=True)
        check(changed>0,"expected optimizer moment rollback defect")
    finally:
        bank.close()

    # 255 cognitive heads / 85 triads vs 256 physical addresses. The current
    # full-vote implementation uses heads==256, creates triad 85 from ingress,
    # and thereby makes its exact 85-triad acceptance condition unreachable.
    f=TriadField()
    for h in range(256):
        f.vote_head(h,1.0,0.5,0.0,0.0)
    accepted,n_good,n_total=f.field_vote(require_full=True)
    print(f"AUDIT_INGRESS candidate_heads=256 triads_seen={n_total} "
          f"accepted_triads={n_good} field_accepted={int(accepted)}",flush=True)
    check(n_total==86 and not accepted,"unexpected full-vote gateway invariant")

    # If top-k=8 is chosen at HEAD level, 2-of-3 triad activation is
    # uncommon before learned routing begins. Exact hypergeometric expectation.
    p2=3*comb(252,6)/comb(255,8)
    p3=comb(252,5)/comb(255,8)
    expectation=85*(p2+p3)
    print(f"AUDIT_ROUTING random_top8_expected_active_triads={expectation:.9f}",flush=True)
    check(0<expectation<0.3,"unexpected two-of-three routing math")
    print("PASS: audited independent/sparse references and reproduced known design hazards",flush=True)

if __name__=="__main__":
    main()
