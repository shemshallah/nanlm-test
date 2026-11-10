#!/usr/bin/env python3
"""Offline full-coverage parser fixture, no download required."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from wordnet31_full_corpus import prepare,REQUIRED,extract

with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp)
    content={}
    for name in REQUIRED:
        content[name]=(
            b"00001740 03 n 01 entity 0 0 | something that exists\n"
            if name.startswith("data.") else
            b"test_word test_word 1\n"
        )
    for kind in ("zip","tar.gz"):
        archive=folder/("wn3.1.dict."+kind)
        if kind=="zip":
            with zipfile.ZipFile(archive,"w") as z:
                for name,data in content.items():
                    z.writestr("wn3.1.dict/dict/"+name,data)
        else:
            with tarfile.open(archive,"w:gz") as t:
                for name,data in content.items():
                    import io
                    info=tarfile.TarInfo("wn3.1.dict/dict/"+name)
                    info.size=len(data)
                    t.addfile(info,io.BytesIO(data))
        assert len(extract(archive))==len(REQUIRED)
        out=folder/("corpus-"+kind+".txt")
        source={"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"source":str(archive)}
        summary=prepare(archive,out,source,strict=False)
        assert summary["synsets"]==4,summary
        assert len(summary["members"])==len(REQUIRED)
        assert summary["corpus_bytes"]==out.stat().st_size
        assert out.read_bytes().count(b"something that exists")==4
        assert out.with_suffix(".manifest.json").is_file()
        for name in REQUIRED:
            assert name in summary["members"],name
print("PASS: full WordNet dictionary parser preserves all four POS, index and morphology from ZIP and TAR")
