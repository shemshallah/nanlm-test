#!/usr/bin/env python3
"""Reproducible bounded-memory ingestion diagnostic, not a language-quality test."""
import json
from pathlib import Path
import resource
import tempfile
import time
from context_lm import open_db,train
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    text=b'the little fox walks through the quiet forest. the bird sings near the river.\n'
    corpus=root/'corpus.txt';corpus.write_bytes((text*(1048576//len(text)+1))[:1048576])
    db,order=open_db(root/'model.sqlite',8)
    start=time.monotonic();result=train(db,order,corpus)
    elapsed=time.monotonic()-start
    rows=db.execute('SELECT count(*) FROM counts').fetchone()[0]
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    print(json.dumps({'input_bytes':result['bytes'],'seconds':elapsed,'bytes_per_second':result['bytes']/elapsed,
                      'context_edges':rows,'database_bytes':(root/'model.sqlite').stat().st_size,
                      'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'warning':'repetitive synthetic corpus; diverse corpora require much more disk'}))
