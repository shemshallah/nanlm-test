"""Train/evaluate the shared trunk + sparse 255-adapter reference."""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from .model import SparseConfig, SparseSharedModel, parameter_budget


def _batch(seq: int, batch: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    data = [[rng.randrange(256) for _ in range(seq + 1)] for _ in range(batch)]
    return torch.tensor(data, dtype=torch.long)


def _grad_norms(model: SparseSharedModel) -> torch.Tensor:
    down = model.adapters.down.grad
    up = model.adapters.up.grad
    norms = torch.zeros(model.cfg.heads)
    if down is not None:
        norms += down.reshape(model.cfg.heads, -1).norm(dim=1)
    if up is not None:
        norms += up.reshape(model.cfg.heads, -1).norm(dim=1)
    return norms


def run(*, dense: bool = False, random_topk: bool = False, active: int = 8,
        steps: int = 3, seq: int = 16, batch: int = 2, seed: int = 17,
        cfg: SparseConfig | None = None, same_batch: bool = False) -> dict:
    cfg = cfg or SparseConfig()
    torch.manual_seed(seed)
    model = SparseSharedModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    times = []
    tokens = 0
    for step in range(1, steps + 1):
        ids = _batch(seq, batch, seed if same_batch else seed + step)
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, aux = model.loss(ids, active=active, dense=dense,
                               random=random_topk)
        loss.backward()
        model.adapters.update_hotspots(_grad_norms(model))
        opt.step()
        times.append(time.perf_counter() - start)
        losses.append(float(loss.detach()))
        tokens += ids.shape[0] * (ids.shape[1] - 1)
    elapsed = sum(times)
    result = {
        "schema": "nanlm.sparse-shared/1",
        "mode": ("dense" if dense else
                 "random_topk" if random_topk else "sparse_hotspot"),
        "config": cfg.__dict__,
        "parameters": parameter_budget(cfg),
        "steps": steps,
        "loss": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "tokens": tokens,
        "seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "active_fraction": 1.0 if dense else active / cfg.heads,
        "usage_nonzero": int((model.adapters.usage > 0).sum().item()),
        "hotspot_nonzero": int((model.adapters.hotspot > 0).sum().item()),
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sparse", "dense", "random"], default="sparse")
    parser.add_argument("--active", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mlp", type=int, default=256)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    cfg = SparseConfig(d_model=args.d_model, layers=args.layers, mlp=args.mlp,
                       context=args.context)
    result = run(dense=args.mode == "dense", random_topk=args.mode == "random",
                 active=args.active, steps=args.steps, seq=args.seq,
                 batch=args.batch, seed=args.seed, cfg=cfg)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
