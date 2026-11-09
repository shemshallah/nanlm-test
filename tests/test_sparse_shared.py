"""Reference tests for the shared deep trunk + sparse 255-adapter field."""
from __future__ import annotations

import unittest

import torch

from experiments.sparse_shared.model import (
    SparseConfig, SparseSharedModel, parameter_budget)
from experiments.sparse_shared.train import run


class SparseSharedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SparseConfig(d_model=32, layers=2, mlp=64, context=8)
        torch.manual_seed(7)
        self.model = SparseSharedModel(self.cfg)

    def test_budget_has_one_shared_trunk_and_255_adapters(self):
        budget = parameter_budget(self.cfg)
        self.assertGreater(budget["trunk"], 0)
        self.assertGreater(budget["adapters"], 0)
        self.assertLess(budget["active_adapter_fraction"], 0.05)

    def test_sparse_topk_is_unique_and_only_active_adapters_get_grad(self):
        ids = torch.randint(0, 256, (1, 8))
        logits, aux = self.model(ids, active=4)
        self.assertEqual(logits.shape, (1, 8, 256))
        selected = aux["indices"]
        for row in selected.reshape(-1, 4):
            self.assertEqual(len(set(row.tolist())), 4)
        self.model.zero_grad(set_to_none=True)
        self.model.loss(ids, active=4)[0].backward()
        down = self.model.adapters.down.grad.reshape(self.cfg.heads, -1)
        active = set(selected.reshape(-1).tolist())
        for head in range(self.cfg.heads):
            if head not in active:
                self.assertEqual(float(down[head].abs().sum()), 0.0)

    def test_hotspot_promotes_a_triad(self):
        grad = torch.zeros(self.cfg.heads)
        grad[6:9] = 10.0
        self.model.adapters.update_hotspots(grad, decay=0.0, threshold=1.0)
        self.assertTrue(bool((self.model.adapters.hotspot[6:9] > 0).all()))
        self.assertEqual(float(self.model.adapters.hotspot[:6].abs().sum()), 0.0)

    def test_sparse_training_reduces_fixed_batch_loss(self):
        result = run(steps=5, seq=8, batch=1, cfg=self.cfg, active=4,
                     same_batch=True)
        self.assertLess(result["last_loss"], result["first_loss"])
        self.assertLess(result["parameters"]["active_adapter_fraction"], 0.05)


if __name__ == "__main__":
    unittest.main()
