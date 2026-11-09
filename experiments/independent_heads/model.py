"""One independent 1 MiB-budget micro-transformer head (PyTorch reference)."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HeadConfig:
    d_model: int = 128
    n_heads: int = 2
    mlp: int = 1504
    vocab: int = 256
    context: int = 8
    layers: int = 1


class HeadBlock(nn.Module):
    def __init__(self, cfg: HeadConfig):
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
        scores = (q @ k.transpose(-1, -2)) / (dh ** 0.5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(b, t, d)
        x = self.ln1(x + self.proj(ctx))
        y = self.mlp2(F.gelu(self.mlp1(x)))
        return self.ln2(x + y)


class IndependentHead(nn.Module):
    """A complete per-head model; no parameters are shared between instances."""

    def __init__(self, cfg: HeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or HeadConfig()
        self.embed = nn.Embedding(self.cfg.vocab, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            HeadBlock(self.cfg) for _ in range(self.cfg.layers))
        self.ln = nn.LayerNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] < 2:
            raise ValueError("sequence must contain a next token")
        logits = self(ids[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab), ids[:, 1:].reshape(-1))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def parameter_budget(cfg: HeadConfig | None = None) -> dict[str, int]:
    model = IndependentHead(cfg)
    params = model.parameter_count()
    del model
    return {
        "parameters": params,
        "weight_bytes_16bit": params * 2,
        "moment_bytes_8bit": params * 2,
        "total_bytes_per_head": params * 4,
        "head_budget_bytes": 1024 * 1024,
        "differences": 1024 * 1024 - params * 2,
    }
