"""HW5:0 paired, book-cluster hypothesis tests; no readout model in Python."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw5_paired_attribution import coverage, paired_statistics, read_triplets


def book(n, memory_ce, geometry_ce):
    return {"transitions": n,
            "best_memory_only": {"ce_nats": memory_ce},
            "best_geometry_positive": {"ce_nats": geometry_ce}}


class PairedAttributionTests(unittest.TestCase):
    def test_all_positive_paired_book_deltas(self):
        rows = [book(100, 2.0, 1.9), book(200, 3.0, 2.9)]
        summary = paired_statistics(rows, draws=1000)
        self.assertAlmostEqual(summary["delta_ce_memory_minus_geometry"], 0.1)
        self.assertEqual(summary["positive_book_count"], 2)
        self.assertEqual(summary["negative_book_count"], 0)
        self.assertAlmostEqual(summary["two_sided_sign_flip_p"], 0.5)
        self.assertGreater(summary["bootstrap_95_percentile_interval"][0], 0.09)
        self.assertLess(summary["bootstrap_95_percentile_interval"][1], 0.11)
        self.assertEqual(summary["unit"], "book")

    def test_disagreement_is_not_confident_improvement(self):
        rows = [book(100, 2.0, 1.8), book(100, 2.0, 2.2)]
        summary = paired_statistics(rows, draws=1000)
        self.assertAlmostEqual(summary["delta_ce_memory_minus_geometry"], 0)
        self.assertGreaterEqual(summary["two_sided_sign_flip_p"], 0.99)
        self.assertLess(summary["bootstrap_95_percentile_interval"][0], 0)
        self.assertGreater(summary["bootstrap_95_percentile_interval"][1], 0)

    def test_trigram_reconstruction_is_order_sensitive(self):
        triple_key = (ord("a") << 16) | (ord("b") << 8) | ord("c")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "triples.bin"
            p.write_bytes(b"HW5TRI01" + struct.pack("<QII", 1, triple_key, 2))
            triples, contexts = read_triplets(p)
            stats = coverage(b"abc" * 40, triples, contexts)
            self.assertEqual(stats["transitions"], 118)
            self.assertEqual(stats["seen_exact_triple_count"], 40)
            self.assertEqual(stats["seen_pair_context_count"], 40)
            self.assertAlmostEqual(stats["exact_triple_fraction"], 40 / 118)
            self.assertNotIn((ord("b") << 16) | (ord("a") << 8) | ord("c"), triples)

    def test_malformed_duplicate_record_rejected(self):
        key = (ord("a") << 16) | (ord("b") << 8) | ord("c")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "triples.bin"
            p.write_bytes(b"HW5TRI01" + struct.pack("<QIIII", 2, key, 1, key, 2))
            with self.assertRaises(ValueError):
                read_triplets(p)

    def test_small_books_rejected(self):
        with self.assertRaises(ValueError):
            coverage(b"abc", set(), set())
        with self.assertRaises(ValueError):
            paired_statistics([book(200, 2.0, 2.0)])


if __name__ == "__main__":
    unittest.main()
