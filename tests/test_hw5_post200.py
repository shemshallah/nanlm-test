"""Source lineage regression for the epoch-200 to epoch-1000 experiment."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hw5_post200 import verify_parent, PARENT_SHA

class ParentTest(unittest.TestCase):
    def test_verify_exact_root_and_receipt(self):
        manifest = {"artifacts": {"all.txt": {"sha256": "sourcehash"}}}
        prior = {
            "schema": "nanlm-test.hw5-longitudinal-bookheldout.v1",
            "status": "epoch_limit",
            "last_hw5_epoch": 200, "best_dev_epoch": 200,
            "best_dev_root_sha256": PARENT_SHA, "last_model_sha256": PARENT_SHA,
            "starting_hw5_root_sha256": "83fbd7046444f50b082a5eec8a14a941626725376b92ac28ed98162a8802c28c",
            "completed_additional_epochs": 180, "source_corpus_sha256": "sourcehash",
            "split_bytes": {"train": 10, "dev": 5, "test": 5},
            "selected": {"train": {"ce_nats": 2.73078818},
                         "dev": {"ce_nats": 2.699913994}},
        }
        class FakeRoot:
            def stat(self):
                class S: st_size = 8_388_608
                return S()
        with patch("hw5_post200.sha", return_value=PARENT_SHA):
            verify_parent(prior, FakeRoot(), manifest, prior["split_bytes"])
            broken = dict(prior, best_dev_epoch=190)
            with self.assertRaises(ValueError):
                verify_parent(broken, FakeRoot(), manifest, prior["split_bytes"])

if __name__ == "__main__":
    unittest.main()
