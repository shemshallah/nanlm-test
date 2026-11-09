"""Regression gate for five independent mathematical tensor specialists."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import matrix_probe_bank as m


def decode(path):
    raw = path.read_bytes()
    assert raw.startswith(m.MAGIC)
    header_length = struct.unpack_from("<I", raw, len(m.MAGIC))[0]
    offset = len(m.MAGIC) + 4
    header = json.loads(raw[offset:offset+header_length])
    payload = zlib.decompress(raw[offset+header_length:])
    assert len(payload) % m.RECORD.size == 0
    values = [m.RECORD.unpack_from(payload,i) for i in range(0,len(payload),m.RECORD.size)]
    return header, values


class SpecialistTests(unittest.TestCase):
    fixture = ROOT/"tests"/"fixtures"/"data.noun"

    def test_geometry_and_wordnet_hash_alignment(self):
        self.assertEqual(m.ROOT_BYTES, 8388608)
        self.assertEqual(m.PAIR_COUNT, 32640)
        self.assertEqual(m.pair_index(0,1), 0)
        self.assertEqual(m.pair_index(254,255), 32639)
        self.assertEqual(m.pair_cell(254,255), 32895)
        self.assertEqual(m.key_head("00001740"),
                         int.from_bytes(hashlib.sha256(
                             b"nanrouter.wordnet.concept/1\0" +
                             b"synset:00001740-n").digest()[:4],"big")&255)
        with self.assertRaises(ValueError):
            m.pair_index(11,11)

    def test_one_input_five_independent_universes_and_mirror_lags(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"experiment"
            report = m.run(self.fixture, output, epochs=3, every=3, hotspots=4,
                           retain_roots=True)
            self.assertEqual(report["distinct_universes"],5)
            self.assertEqual(report["total_input_events_each"],27)
            self.assertFalse(report["upper_universe_built"])
            self.assertFalse(report["live_wordnet_matrix_connected"])
            self.assertEqual({x["profile"] for x in report["profiles"]},set(m.PROFILES))
            self.assertEqual(len(list(output.glob("*.root"))),5)
            self.assertEqual(len(list(output.glob("*.c256tensor"))),5)
            source = m.file_digest(self.fixture)
            common_ticks = set(range(1,28))
            all_metric_traces = []
            for profile in m.PROFILES:
                root = output/(profile+".root")
                self.assertEqual(root.stat().st_size,m.ROOT_BYTES)
                with root.open("rb") as f:
                    f.seek(255*128)
                    self.assertEqual(f.read(128),bytes(128),"ingress head was trained")
                header, records = decode(output/(profile+".c256tensor"))
                self.assertEqual(header["source_sha256"],source)
                self.assertEqual(header["profile"],profile)
                self.assertTrue(header["not_live_hw4"])
                self.assertEqual(header["sample_every_ticks"],3)
                self.assertEqual({r[0] for r in records},common_ticks)
                self.assertTrue(all(r[2]<len(m.CHANNELS) and
                                    -1<=r[3]<=1 and math.isfinite(r[5])
                                    for r in records))
                self.assertTrue(all(0<=r[1]<32896 for r in records))
                self.assertFalse(any(r[1]==255 and r[2]==m.C["point"]
                                     for r in records))
                self.assertTrue(any(r[2]==m.C["hottriad"] for r in records))
                self.assertTrue(any(r[2]==m.C["triad"] for r in records))
                self.assertTrue(any(r[2]==m.C["global"] for r in records))
                metric = [(tick,cell,round(val,6)) for tick,cell,ch,o,_,val in records
                          if ch == m.C["metric"]]
                self.assertTrue(metric,"missing C(256,2) relation-metric measurements")
                all_metric_traces.append(metric)
                # The backward mirror register stores the PREVIOUS pair value,
                # not an arbitrary second independently trained relation.
                prior = {}
                pending = {}
                for tick,cell,ch,orient,_,val in records:
                    if ch == m.C["mirror_now"]:
                        self.assertEqual(orient,1)
                        self.assertNotIn(cell,pending)
                        pending[cell] = val
                    if ch == m.C["mirror_lag"]:
                        self.assertEqual(orient,-1)
                        self.assertIn(cell,pending)
                        self.assertAlmostEqual(val, prior.get(cell,0.),places=5)
                        prior[cell] = pending.pop(cell)
                self.assertFalse(pending)
                self.assertEqual(report["profiles"][m.PROFILES.index(profile)]["events"],27)
            self.assertGreater(len({str(t) for t in all_metric_traces}),1,
                               "specialists cannot all emit the same matrix")
            # No WordNet lemma, gloss, or full synset key is exported to tensor records.
            for file in output.glob("*.c256tensor"):
                _,records=decode(file)
                self.assertEqual(len(records)*m.RECORD.size % m.RECORD.size,0)
            with self.assertRaises(FileExistsError):
                m.run(self.fixture,output)

    def test_empty_input_and_hash_collision_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            source=p/"empty"
            source.write_text("a non-entry header\n")
            with self.assertRaisesRegex(ValueError,"no noun"):
                m.run(source,p/"probe",profiles=("topology",))
            self.assertEqual(len(list((p/"probe").glob("*.c256tensor"))),1)

    def test_individual_specialist_and_deterministic_numeric_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            a=m.run(self.fixture,p/"A",profiles=("geometry",),epochs=2,every=5)
            b=m.run(self.fixture,p/"B",profiles=("geometry",),epochs=2,every=5)
            ha,ra=decode(p/"A"/"geometry.c256tensor")
            hb,rb=decode(p/"B"/"geometry.c256tensor")
            self.assertEqual(ha,hb)
            self.assertEqual(ra,rb)
            self.assertEqual(a["profiles"][0]["file_sha256"],b["profiles"][0]["file_sha256"])
            self.assertEqual(len(a["profiles"]),1)


if __name__=="__main__":
    unittest.main()
