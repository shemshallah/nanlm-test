"""Persistent 16-bit weights + 8-bit blockwise Adam moments.

The 256-head budget is 256 * (1 MiB weights + 1 MiB moments) = 512 MiB.
Only the active head is dequantized to float32 for a local step; the
persistent bank remains int16/int8.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from .model import HeadConfig, IndependentHead, parameter_budget

BLOCK = 512


def _blocks(count: int) -> int:
    return (count + BLOCK - 1) // BLOCK


@dataclass
class BankConfig:
    heads: int = 256
    params: int = 0
    block: int = BLOCK

    def __post_init__(self):
        if self.params <= 0:
            self.params = parameter_budget(HeadConfig())["parameters"]


class QuantizedBank:
    def __init__(self, cfg: BankConfig | None = None, path: str | Path | None = None):
        self.cfg = cfg or BankConfig()
        self.params = self.cfg.params
        self.heads = self.cfg.heads
        self.block = self.cfg.block
        self.blocks = _blocks(self.params)
        self._tmp = None
        if path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="nanlm-independent-", suffix=".bank", delete=False)
            handle.close()
            self._tmp = handle.name
            path = self._tmp
        self.path = Path(path)
        weight_bytes = self.heads * self.params * 2
        moment_bytes = self.heads * 2 * self.params
        scale_bytes = self.heads * 2 * self.blocks * 4
        weight_scale_bytes = self.heads * 4
        total = weight_bytes + moment_bytes + scale_bytes + weight_scale_bytes
        with open(self.path, "wb") as handle:
            handle.truncate(total)
        self.weights = np.memmap(
            self.path, dtype=np.int16, mode="r+",
            shape=(self.heads, self.params), offset=0)
        self.moments = np.memmap(
            self.path, dtype=np.int8, mode="r+",
            shape=(self.heads, 2, self.params), offset=weight_bytes)
        self.scales = np.memmap(
            self.path, dtype=np.float32, mode="r+",
            shape=(self.heads, 2, self.blocks),
            offset=weight_bytes + moment_bytes)
        self.weight_scales = np.memmap(
            self.path, dtype=np.float32, mode="r+",
            shape=(self.heads,), offset=weight_bytes + moment_bytes + scale_bytes)
        self.weights[:] = 0
        self.moments[:] = 0
        self.scales[:] = 1.0
        self.weight_scales[:] = 1.0
        self.flush()

    @property
    def weight_bytes(self) -> int:
        return int(self.weights.nbytes)

    @property
    def moment_bytes(self) -> int:
        return int(self.moments.nbytes)

    @property
    def scale_bytes(self) -> int:
        return int(self.scales.nbytes) + int(self.weight_scales.nbytes)

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.moment_bytes + self.scale_bytes

    def flush(self) -> None:
        for array in (self.weights, self.moments, self.scales, self.weight_scales):
            array.flush()

    def close(self) -> None:
        self.flush()
        for array in (self.weights, self.moments, self.scales, self.weight_scales):
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()
        if self._tmp:
            try:
                Path(self._tmp).unlink(missing_ok=True)
            except OSError:
                pass

    def audit(self, touch: bool = False) -> dict[str, int]:
        if touch:
            # Deterministic page touch: first/last byte of every 1 MiB.
            for head in range(self.heads):
                self.weights[head, 0] = self.weights[head, 0]
                self.weights[head, -1] = self.weights[head, -1]
                self.moments[head, 0, -1] = self.moments[head, 0, -1]
        return {
            "heads": self.heads,
            "params_per_head": self.params,
            "weight_bytes": self.weight_bytes,
            "moment_bytes": self.moment_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.total_bytes,
            "total_mib": round(self.total_bytes / (1024 * 1024), 3),
        }

    @staticmethod
    def _q16(t: torch.Tensor) -> tuple[np.ndarray, float]:
        flat = t.detach().float().reshape(-1).numpy()
        scale = float(np.max(np.abs(flat))) if flat.size else 0.0
        scale = max(scale, 1e-12)
        q = np.clip(np.rint(flat / scale * 32767.0), -32768, 32767).astype(np.int16)
        return q, scale

    @staticmethod
    def _q8_blocks(flat: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
        count = flat.size
        blocks = (count + block - 1) // block
        padded = np.zeros(blocks * block, dtype=np.float32)
        padded[:count] = flat
        view = padded.reshape(blocks, block)
        scales = np.max(np.abs(view), axis=1)
        scales[scales < 1e-12] = 1e-12
        q = np.clip(np.rint(view / scales[:, None] * 127.0), -127, 127).astype(np.int8)
        return q.reshape(-1)[:count], scales.astype(np.float32)

    @staticmethod
    def _dq8_blocks(q: np.ndarray, scales: np.ndarray, block: int) -> np.ndarray:
        blocks = scales.size
        padded = np.zeros(blocks * block, dtype=np.float32)
        padded[:q.size] = q.astype(np.float32)
        out = padded.reshape(blocks, block) / 127.0 * scales[:, None]
        return out.reshape(-1)[:q.size]

    @staticmethod
    def _params(model: IndependentHead) -> list[torch.Tensor]:
        return [p.detach().reshape(-1) for p in model.parameters()]

    def store(self, head: int, model: IndependentHead) -> None:
        q, scale = self._q16(torch.cat(self._params(model)))
        self.weights[head] = q
        self.weight_scales[head] = scale
        self.flush()

    def load(self, head: int, model: IndependentHead) -> None:
        scale = float(self.weight_scales[head]) or 1.0
        flat = self.weights[head].astype(np.float32) / 32767.0 * scale
        offset = 0
        for p in model.parameters():
            n = p.numel()
            with torch.no_grad():
                p.copy_(torch.from_numpy(flat[offset:offset + n]).reshape(p.shape))
            offset += n
        self.flush()

    def adam_step(self, head: int, model: IndependentHead, lr: float,
                  t: int, beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8) -> None:
        params = torch.cat(self._params(model)).float().numpy()
        grads = torch.cat([
            (p.grad.detach().reshape(-1) if p.grad is not None
             else torch.zeros(p.numel())) for p in model.parameters()
        ]).float().numpy()
        m = self._dq8_blocks(self.moments[head, 0], self.scales[head, 0],
                             self.block)
        v = self._dq8_blocks(self.moments[head, 1], self.scales[head, 1],
                             self.block)
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads * grads
        mh = m / (1.0 - beta1 ** t)
        vh = v / (1.0 - beta2 ** t)
        updated = params - lr * mh / (np.sqrt(vh) + eps)
        q_w, scale_w = self._q16(torch.from_numpy(updated))
        self.weights[head] = q_w
        self.weight_scales[head] = scale_w
        q_m, s_m = self._q8_blocks(m, self.block)
        q_v, s_v = self._q8_blocks(v, self.block)
        self.moments[head, 0] = q_m
        self.moments[head, 1] = q_v
        self.scales[head, 0] = s_m
        self.scales[head, 1] = s_v
        # The active model receives dequantized updated weights.
        flat = updated.astype(np.float32)
        offset = 0
        for p in model.parameters():
            n = p.numel()
            with torch.no_grad():
                p.copy_(torch.from_numpy(flat[offset:offset + n]).reshape(p.shape))
            offset += n
        self.flush()


def memory_projection(heads: int = 256) -> dict[str, int]:
    budget = parameter_budget(HeadConfig())
    params = budget["parameters"]
    blocks = _blocks(params)
    weight = heads * params * 2
    moment = heads * params * 2
    scales = heads * 2 * blocks * 4 + heads * 4
    return {
        "heads": heads,
        "parameters_per_head": params,
        "weight_bytes": weight,
        "moment_bytes": moment,
        "scale_bytes": scales,
        "total_bytes": weight + moment + scales,
        "total_mib": round((weight + moment + scales) / (1024 * 1024), 3),
        "under_512_mib": weight + moment + scales <= 512 * 1024 * 1024,
    }
