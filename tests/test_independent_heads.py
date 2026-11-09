"""Reference tests for independent 1 MiB heads, 8-bit Adam and votes."""
from __future__ import annotations

import tempfile
import unittest

import torch

from experiments.independent_heads.field import CFTField, TriadField, triad_members
from experiments.independent_heads.model import HeadConfig, IndependentHead, parameter_budget
from experiments.independent_heads.quant_bank import BankConfig, QuantizedBank, memory_projection
from experiments.independent_heads.train import run


class BudgetTests(unittest.TestCase):
    def test_one_mib_head_and_512_mib_total(self):
        budget = parameter_budget(HeadConfig())
        self.assertLessEqual(budget["weight_bytes_16bit"], 1024 * 1024)
        self.assertGreater(budget["parameters"], 500_000)
        projection = memory_projection(256)
        self.assertTrue(projection["under_512_mib"])
        self.assertLessEqual(projection["total_mib"], 512.0)
        self.assertGreater(projection["total_mib"], 500.0)

    def test_heads_are_parameter_independent(self):
        a = IndependentHead()
        b = IndependentHead()
        before = [p.detach().clone() for p in b.parameters()]
        loss = a.loss(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        loss.backward()
        with torch.no_grad():
            for p in a.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-4)
        for old, new in zip(before, b.parameters()):
            self.assertTrue(torch.equal(old, new))


class QuantBankTests(unittest.TestCase):
    def test_store_load_error_is_bounded(self):
        cfg = HeadConfig(d_model=16, n_heads=2, mlp=32, context=8)
        params = parameter_budget(cfg)["parameters"]
        model = IndependentHead(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            bank = QuantizedBank(BankConfig(heads=2, params=params),
                                 tmp + "/bank.bin")
            try:
                bank.store(0, model)
                before = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
                with torch.no_grad():
                    for p in model.parameters():
                        p.zero_()
                bank.load(0, model)
                after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
                self.assertLess(float((before - after).abs().max()), 1e-2)
            finally:
                bank.close()


class VoteTests(unittest.TestCase):
    def test_triad_requires_two_of_three(self):
        field = TriadField()
        for head, after in ((0, 0.9), (1, 0.9), (2, 0.99)):
            field.vote_head(head, 1.0, after, 0.0, 0.0)
        self.assertTrue(field.vote_triads()[0])

    def test_field_requires_43_of_85(self):
        for accepted in (42, 43):
            field = TriadField()
            field.head_votes = []
            for triad in range(85):
                for head in triad_members(triad):
                    good = triad < accepted
                    field.vote_head(head, 1.0, 0.5 if good else 2.0, 0.0, 0.0)
            ok, count, total = field.field_vote(require_full=True)
            self.assertEqual(total, 85)
            self.assertEqual(count, accepted)
            self.assertEqual(ok, accepted >= 43)

    def test_cft_loop_update_changes_loop_by_delta(self):
        cft = CFTField()
        before = cft.residual(0)
        after = cft.propose(0, 0.3, rate=1.0)
        self.assertAlmostEqual(after - before, 0.1, places=6)


class TrainSmokeTests(unittest.TestCase):
    def test_three_head_candidate_commits(self):
        result = run(heads=3, steps=1, seq=8, lr=1e-4)
        self.assertEqual(result["heads"], 3)
        self.assertTrue(result["triad_votes"]["0"])
        self.assertTrue(result["field_accepted"])
        self.assertLess(result["loss_committed_mean"], result["loss_before_mean"])


if __name__ == "__main__":
    unittest.main()
