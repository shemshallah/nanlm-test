"""HW5 native train-only pair/triple bank, fixed-point and boundary contracts."""
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=8388608
def pid(a,b):
    assert 0<=a<b<256
    return 1+a*(511-a)//2+(b-a-1)
def offset(a,b):
    return 32768+(pid(min(a,b),max(a,b))-1)*256
def run(*args):
    return subprocess.run(["./hw5-context",*map(str,args)],
             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def scalar(data,off):
    return struct.unpack_from("<I",data,off)[0]

class MirrorContextTests(unittest.TestCase):
    def test_c256_train_only_and_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            docs=[b"abcabc\n",b"defdef\n"]
            src=base/"train.txt";src.write_bytes(b"".join(docs))
            sizes=base/"train.lengths";sizes.write_text("".join(f"{len(d)}\n" for d in docs))
            root=base/"train-c256.root"
            overlay=base/"train-triples.bin"
            result=run("build",src,sizes,root,overlay)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertIn("books=2",result.stdout)
            self.assertIn("gateway_head=255",result.stdout)
            self.assertIn("addressed_pairs=32640",result.stdout)
            self.assertIn("reserved_pairs=255",result.stdout)
            self.assertEqual(root.stat().st_size,ROOT)
            data=root.read_bytes()
            self.assertEqual(data[255*128:256*128],bytes(128))
            addressed=0
            for a in range(256):
                for b in range(a+1,256):
                    at=offset(a,b)
                    self.assertEqual(scalar(data,at+160),a)
                    self.assertEqual(scalar(data,at+164),b)
                    self.assertEqual(data[at+168:at+172],b"H5P1")
                    if b==255:
                        self.assertEqual(data[at:at+88],bytes(88))
                    addressed+=1
            self.assertEqual(addressed,32640)
            self.assertEqual(scalar(data,0),0)
            self.assertEqual(scalar(data,ord("a")*128),ord("a"))
            xy=offset(ord("a"),ord("b"))
            yx=offset(ord("b"),ord("a"))
            self.assertEqual(xy,yx)
            self.assertGreater(scalar(data,xy),0)
            self.assertEqual(scalar(data,xy+16),scalar(data,xy))
            self.assertEqual(scalar(data,xy+4),0)
            ac=offset(ord("a"),ord("c"))
            self.assertGreater(scalar(data,ac+16+4),0) # skip-one a->c
            # 'c' is final content in first book and 'd' first in second:
            # only an accidental concatenation would create a c->d edge.
            cd=offset(ord("c"),ord("d"))
            self.assertEqual(scalar(data,cd),0)
            self.assertEqual(scalar(data,cd+4),0)
            self.assertEqual(scalar(data,cd+160),ord("c"))
            self.assertEqual(scalar(data,cd+164),ord("d"))
            raw=overlay.read_bytes()
            self.assertEqual(raw[:8],b"HW5TRI01")
            count=struct.unpack_from("<Q",raw,8)[0]
            self.assertEqual(len(raw),16+count*8)
            triples={}
            for i in range(count):
                key,n=struct.unpack_from("<II",raw,16+8*i)
                triples[key]=n
            def k(a,b,c):return (ord(a)<<16)|(ord(b)<<8)|ord(c)
            self.assertEqual(triples[k("a","b","c")],2)
            self.assertEqual(triples.get(k("b","c","d"),0),0)
            second=base/"again.root";second_overlay=base/"again.bin"
            again=run("build",src,sizes,second,second_overlay)
            self.assertEqual(again.returncode,0,again.stdout)
            self.assertEqual(hashlib.sha256(second.read_bytes()).digest(),hashlib.sha256(data).digest())
            self.assertEqual(second_overlay.read_bytes(),raw)
            sizes.write_text(f"{len(docs[0])}\n") # missing second book boundary
            invalid=run("build",src,sizes,base/"bad.root",base/"bad.bin")
            self.assertNotEqual(invalid.returncode,0)
            self.assertIn("extra source bytes",invalid.stdout)
    def test_ingress_and_underlength(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            source=base/"bad.txt";source.write_bytes(b"a\xffz")
            sizes=base/"sizes";sizes.write_text("3\n")
            p=run("build",source,sizes,base/"root",base/"overlay")
            self.assertNotEqual(p.returncode,0)
            self.assertIn("0xff ingress",p.stdout)
            source.write_bytes(b"ab")
            sizes.write_text("2\n")
            p=run("build",source,sizes,base/"root",base/"overlay")
            self.assertNotEqual(p.returncode,0)
            self.assertIn("invalid book length",p.stdout)

if __name__=="__main__":unittest.main()
