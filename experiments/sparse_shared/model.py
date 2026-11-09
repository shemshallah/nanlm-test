"""Shared deep trunk + 255 low-rank head adapters + sparse hotspot routing.

Reference harness for NaNLM v1.  One shared transformer trunk does the deep
thinking; each head is a small persistent adapter and a routing interface.
Only the top-k heads run their adapters.  The router uses learned scores plus
a triad-gradient/hotspot bias; inactive heads keep the cheap wave state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseConfig:
    d_model: int = 64
    n_heads: int = 4
    layers: int = 4
    mlp: int = 256
    vocab: int = 256
    context: int = 64
    heads: int = 255
    active: int = 8
    rank: int = 4


class Block(nn.Module):
    def __init__(self, cfg: SparseConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mlp1 = nn.Linear(cfg.d_model, cfg.mlp)
        self.mlp2 = nn.Linear(cfg.mlp, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.cfg.n_heads
        dh = d // h
        qkv = self.qkv(x).view(b, t, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, d)
        x = self.ln1(x + self.proj(ctx))
        return self.ln2(x + self.mlp2(F.gelu(self.mlp1(x))))


class SharedTrunk(nn.Module):
    def __init__(self, cfg: SparseConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.ln(x)


class AdapterBank(nn.Module):
    def __init__(self, cfg: SparseConfig):
        super().__init__()
        self.cfg = cfg
        self.down = nn.Parameter(torch.empty(cfg.heads, cfg.d_model, cfg.rank))
        self.up = nn.Parameter(torch.empty(cfg.heads, cfg.rank, cfg.d_model))
        self.router = nn.Linear(cfg.d_model, cfg.heads, bias=True)
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=0.02)
        # Persistent hotspot bias from triad gradients; not a gradient path.
        self.register_buffer("hotspot", torch.zeros(cfg.heads))
        self.register_buffer("usage", torch.zeros(cfg.heads))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.router(x) + self.hotspot[None, None, :]

    def forward(self, x: torch.Tensor, *, active: int | None = None,
                dense: bool = False, random: bool = False):
        b, t, d = x.shape
        k = self.cfg.heads if dense else (active or self.cfg.active)
        k = max(1, min(k, self.cfg.heads))
        logits = self.scores(x)
        if random:
            idx = torch.stack([
                torch.randperm(self.cfg.heads, device=x.device)[:k]
                for _ in range(b * t)
            ]).reshape(b, t, k)
        else:
            idx = torch.topk(logits, k, dim=-1).indices
        if dense:
            weights = torch.softmax(logits, dim=-1)
        else:
            selected = torch.gather(logits, -1, idx)
            weights = torch.softmax(selected, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(k):
            head = idx[..., slot]
            head_out = torch.einsum(
                "btd,btdr->btr", x, self.down[head])
            head_out = torch.einsum(
                "btr,btrd->btd", head_out, self.up[head])
            out = out + head_out * weights[..., slot, None]
        if dense:
            self.usage += idx.reshape(-1).bincount(
                minlength=self.cfg.heads).to(self.usage.dtype)
        else:
            self.usage += torch.bincount(
                idx.reshape(-1), minlength=self.cfg.heads).to(self.usage.dtype)
        return out, {"indices": idx, "weights": weights, "logits": logits}

    def update_hotspots(self, grad_norms: torch.Tensor, *, decay: float = 0.95,
                        threshold: float | None = None) -> None:
        """Promote triad-gradient hotspots; 2/3 triad members share the bias."""
        with torch.no_grad():
            self.hotspot.mul_(decay)
            triads = grad_norms[:255].reshape(85, 3)
            triad_score = triads.mean(dim=1)
            if threshold is None:
                threshold = float(triad_score.mean() + triad_score.std())
            hot = triad_score > threshold
            for t in torch.nonzero(hot, as_tuple=False).flatten().tolist():
                self.hotspot[t * 3:t * 3 + 3].add_(float(triad_score[t]))


class SparseSharedModel(nn.Module):
    def __init__(self, cfg: SparseConfig | None = None):
        super().__init__()
        self.cfg = cfg or SparseConfig()
        self.trunk = SharedTrunk(self.cfg)
        self.adapters = AdapterBank(self.cfg)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab)

    def forward(self, ids: torch.Tensor, *, active: int | None = None,
                dense: bool = False, random: bool = False):
        x = self.trunk(ids)
        if ids.shape[1] > self.cfg.context:
            raise ValueError("sequence exceeds configured context")
        delta, aux = self.adapters(x, active=active, dense=dense, random=random)
        return self.head(x + delta), aux

    def loss(self, ids: torch.Tensor, **kwargs):
        logits, aux = self(ids[:, :-1], **kwargs)
        target = ids[:, 1:].reshape(-1)
        return F.cross_entropy(logits.reshape(-1, self.cfg.vocab), target), aux


def parameter_budget(cfg: SparseConfig | None = None) -> dict[str, int]:
    cfg = cfg or SparseConfig()
    trunk_per_layer = 4 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * cfg.mlp + 4 * cfg.d_model
    trunk = cfg.vocab * cfg.d_model + cfg.layers * trunk_per_layer + 2 * cfg.d_model
    adapters = cfg.heads * (2 * cfg.d_model * cfg.rank) + cfg.heads
    router = cfg.d_model * cfg.heads + cfg.heads
    head = cfg.d_model * cfg.vocab + cfg.vocab
    total = trunk + adapters + router + head
    return {
        "trunk": trunk, "adapters": adapters, "router": router,
        "head": head, "total": total,
        "active_adapter_fraction": cfg.active / cfg.heads,
    }
