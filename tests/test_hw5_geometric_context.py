"""Synthetic native C256 CP gradient, persistence and document-boundary audit."""
from pathlib import Path
import subprocess
import tempfile
import unittest

def command(*args):
    return subprocess.run(["./hw5-geometric-context",*map(str,args)],
       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def get(result,name):
    return float(next(x.split("=",1)[1] for x in result.stdout.split()
          if x.startswith(name+"=")))
class NativeCpContextTests(unittest.TestCase):
    def test_train_and_context_ablation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/"one-root"
            init=command("init",root)
            self.assertEqual(init.returncode,0,init.stdout)
            self.assertEqual(root.stat().st_size,8388608)
            corpus=p/"train";lengths=p/"lengths"
            docs=[b"abcabcabcabc\n"*300,b"bacbacbacbac\n"*300]
            corpus.write_bytes(b"".join(docs))
            lengths.write_text("".join(str(len(x))+"\n" for x in docs))
            first=command("score",root,corpus,lengths)
            self.assertEqual(first.returncode,0,first.stdout)
            self.assertIn("books=2",first.stdout)
            self.assertIn("inference_trigram_lookup=0",first.stdout)
            for _ in range(7):
                trained=command("train",root,corpus,lengths,"16")
                self.assertEqual(trained.returncode,0,trained.stdout)
                if "accepted=0" in trained.stdout:break
                self.assertLess(get(trained,"after_soft_ce"),get(trained,"before_soft_ce"))
            second=command("score",root,corpus,lengths)
            self.assertEqual(second.returncode,0,second.stdout)
            self.assertLess(get(second,"ce_nats"),get(first,"ce_nats"))
            wrong=command("score",root,corpus,lengths,"wrong")
            self.assertEqual(wrong.returncode,0,wrong.stdout)
            self.assertGreater(get(wrong,"ce_nats"),get(second,"ce_nats"))
            lengths.write_text(str(len(docs[0]))+"\n")
            bad=command("score",root,corpus,lengths)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn("extra or failed document input",bad.stdout)
    def test_ingress_is_reserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/"one-root"
            self.assertEqual(command("init",root).returncode,0)
            (p/"bad").write_bytes(b"a\xffz")
            (p/"lengths").write_text("3\n")
            self.assertIn("0xff reserved",command("train",root,p/"bad",p/"lengths").stdout)
if __name__=="__main__":unittest.main()
