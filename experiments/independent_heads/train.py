"""Train independent 1 MiB-budget heads with local CFT/vote credit.

This is a reference harness: each head owns its parameters and local loss;
no gradient crosses heads or the field.  Heads vote on their triad changes,
then triads vote on the global bundle.  The persistent bank is int16 weights
plus int8 blockwise Adam moments; only the active head is dequantized.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import torch

from .field import TriadField, triad_of
from .model import HeadConfig, IndependentHead, parameter_budget
from .quant_bank import BankConfig, QuantizedBank, memory_projection


def _ids(seq: int, offset: int, seed: int = 17) -> torch.Tensor:
    rng = random.Random(seed + offset)
    values = [rng.randrange(256) for _ in range(seq + 1)]
    return torch.tensor([values], dtype=torch.long)


def run(heads: int = 4, steps: int = 2, seq: int = 16, lr: float = 3e-4,
        seed: int = 17, out: str | None = None) -> dict:
    cfg = HeadConfig()
    params = parameter_budget(cfg)["parameters"]
    bank = QuantizedBank(BankConfig(heads=heads, params=params))
    try:
        models = []
        losses_before = []
        losses_after = []
        for head in range(heads):
            model = IndependentHead(cfg)
            bank.store(head, model)
            models.append(model)
        field = TriadField()
        for step in range(1, steps + 1):
            snapshots = [deepcopy(model.state_dict()) for model in models]
            for head, model in enumerate(models):
                ids = _ids(seq, head, seed)
                before = float(model.loss(ids).detach())
                model.zero_grad(set_to_none=True)
                loss = model.loss(ids)
                loss.backward()
                triad = triad_of(head)
                cft_before = field.cft.residual(triad)
                theta = lr * (1.0 if before > 0 else 0.0)
                bank.adam_step(head, model, lr, step)
                after = float(model.loss(ids).detach())
                cft_after = cft_before + theta * (before - after) / 3.0
                accepted = field.vote_head(head, before, after,
                                           cft_before, cft_after)
                losses_before.append(before)
                losses_after.append(after)
                if accepted:
                    bank.store(head, model)
                else:
                    model.load_state_dict(snapshots[head])
                    bank.store(head, model)
            triad_votes = field.vote_triads()
            field_ok, accepted_triads, total_triads = field.field_vote(
                require_full=heads == 256)
            if field_ok:
                for vote in field.head_votes:
                    if vote.improved:
                        field.cft.propose(triad_of(vote.head),
                                          vote.before_loss - vote.after_loss)
            else:
                for head, model in enumerate(models):
                    model.load_state_dict(snapshots[head])
                    bank.store(head, model)
        final_losses = []
        for head, model in enumerate(models):
            final_losses.append(float(model.loss(_ids(seq, head, seed)).detach()))
        result = {
            "schema": "nanlm.independent-heads/1",
            "heads": heads,
            "steps": steps,
            "parameters_per_head": params,
            "weight_bytes_16bit": params * 2,
            "moment_bytes_8bit": params * 2,
            "memory": memory_projection(heads),
            "loss_before_mean": sum(losses_before) / max(1, len(losses_before)),
            "loss_after_mean": sum(losses_after) / max(1, len(losses_after)),
            "loss_committed_mean": sum(final_losses) / max(1, len(final_losses)),
            "committed_losses": final_losses,
            "triad_votes": {str(k): v for k, v in triad_votes.items()},
            "head_votes": [
                {"head": v.head, "before": v.before_loss, "after": v.after_loss,
                 "improved": v.improved}
                for v in field.head_votes
            ],
            "accepted_triads": accepted_triads,
            "total_triads": total_triads,
            "field_accepted": field_ok,
            "cft_residuals": {
                str(t): field.cft.residual(t)
                for t in sorted({triad_of(h) for h in range(heads)})
            },
            "candidate_only": heads != 256,
            "independence": "local loss/backward per head; no shared parameters",
        }
        if out:
            Path(out).write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8")
        return result
    finally:
        bank.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out")
    parser.add_argument("--audit-memory", action="store_true")
    parser.add_argument("--audit-heads", type=int, default=256)
    args = parser.parse_args(argv)
    if args.audit_memory:
        projection = memory_projection(args.audit_heads)
        print(json.dumps(projection, indent=2))
        return 0
    if args.heads < 1 or args.heads > 256 or args.steps < 1:
        raise SystemExit("heads must be 1..256 and steps >= 1")
    result = run(args.heads, args.steps, args.seq, args.lr, args.seed, args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
