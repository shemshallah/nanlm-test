#!/usr/bin/env python3
"""Disk-backed variable-context byte LM and conservative additive shard merger.
The exact sequence overlay belongs to one HW0 universe; byte 255 is routing only.
Python standard library only. See docs/DISTRIBUTED.md for the protocol limits.
"""
import argparse
from collections import Counter, deque
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys

VERSION = 'nan-context-v1'


def open_db(path, order=8, readonly=False):
    if readonly:
        db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
        db.execute('PRAGMA query_only=ON')
    else:
        if not 0 <= order <= 32:
            raise ValueError('order must be 0..32')
        db = sqlite3.connect(path, timeout=60)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.executescript('''
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS documents(digest TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS counts(context BLOB NOT NULL, target INTEGER NOT NULL CHECK(target BETWEEN 0 AND 254),
             count INTEGER NOT NULL CHECK(typeof(count)='integer' AND count>0), PRIMARY KEY(context,target)) WITHOUT ROWID;
        ''')
        with db:
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('version', VERSION))
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('order', str(order)))
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('link', 'HW4:0::0:HW0'))
            db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('gateway', '255'))
    metadata = dict(db.execute('SELECT key,value FROM meta'))
    if metadata.get('version') != VERSION or metadata.get('gateway') != '255':
        db.close()
        raise ValueError('unsupported context overlay')
    actual = int(metadata['order'])
    if not 0 <= actual <= 32:
        raise ValueError('invalid stored context order')
    if not readonly and actual != order:
        db.close()
        raise ValueError('order mismatch; use the order chosen at creation')
    return db, actual


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


UPSERT = '''INSERT INTO counts VALUES (?,?,?) ON CONFLICT(context,target)
            DO UPDATE SET count=count+excluded.count'''


def train(db, order, path, batch=65536):
    digest = digest_file(path)
    db.execute('BEGIN IMMEDIATE')
    try:
        if db.execute('SELECT 1 FROM documents WHERE digest=?', (digest,)).fetchone():
            db.rollback()
            return {'duplicate': True, 'digest': digest, 'bytes': 0}
        pending = Counter()
        history = deque(maxlen=order)
        n = 0
        verify = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                verify.update(block)
                n += len(block)
                for byte in block:
                    if byte == 255:
                        history.clear()
                        continue
                    context = bytes(history)
                    for k in range(len(context)+1):
                        pending[(context[len(context)-k:] if k else b'', byte)] += 1
                    history.append(byte)
                    if len(pending) >= batch:
                        db.executemany(UPSERT, ((c,t,v) for (c,t),v in pending.items()))
                        pending.clear()
        if verify.hexdigest() != digest:
            raise ValueError('corpus changed during training')
        db.executemany(UPSERT, ((c,t,v) for (c,t),v in pending.items()))
        db.execute('INSERT INTO documents VALUES (?,?)', (digest, n))
        db.commit()
        return {'duplicate': False, 'digest': digest, 'bytes': n}
    except BaseException:
        db.rollback()
        raise


class Model:
    def __init__(self, db, order):
        self.db, self.order = db, order

    @lru_cache(maxsize=4096)
    def row(self, context):
        values = tuple(self.db.execute('SELECT target,count FROM counts WHERE context=? ORDER BY target', (context,)))
        return values, sum(n for _,n in values)

    def probability(self, history, target):
        if not 0 <= target < 255:
            raise ValueError('gateway is not a language target')
        values, total = self.row(b'')
        p = (dict(values).get(target, 0)+.5)/(total+127.5)
        context = bytes(history)[-self.order:] if self.order else b''
        for k in range(1,len(context)+1):
            values, total = self.row(context[-k:])
            if total:
                escape = len(values)
                p = (dict(values).get(target, 0)+escape*p)/(total+escape)
        return p

    def sample(self, history, rng):
        # Sample interpolation recursively without a 255-way softmax at each level.
        context = bytes(history)[-self.order:] if self.order else b''
        for k in range(len(context),0,-1):
            values, total = self.row(context[-k:])
            if total:
                x = rng.randrange(total+len(values))
                if x < total:
                    for target, count in values:
                        if x < count:
                            return target
                        x -= count
        values, total = self.row(b'')
        x = rng.randrange(2*total+255)
        for target, count in values:
            if x < 2*count:
                return target
            x -= 2*count
        return x  # uniform half-count smoothing over 0..254

    def generate(self, prompt, length, seed):
        rng = random.Random(seed)
        history = deque(maxlen=self.order)
        for b in prompt:
            if b == 255:
                history.clear()
            else:
                history.append(b)
        result = bytearray()
        for _ in range(length):
            b = self.sample(history, rng)
            result.append(b)
            history.append(b)
        return bytes(result)

    def evaluate(self, path):
        history = deque(maxlen=self.order)
        total = 0
        loss = 0.
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                for b in block:
                    if b == 255:
                        history.clear()
                        continue
                    loss -= math.log(self.probability(history, b))
                    history.append(b)
                    total += 1
        return {'bytes': total, 'nats_per_byte': loss/total if total else None,
                'bits_per_byte': loss/total/math.log(2) if total else None}


def merge(db, order, source):
    other, source_order = open_db(source, readonly=True)
    try:
        if order != source_order:
            raise ValueError('shard context orders differ')
        other.execute('BEGIN')  # stable source snapshot, including concurrent writers
        documents = dict(other.execute('SELECT digest,bytes FROM documents'))
        db.execute('BEGIN IMMEDIATE')
        known = dict(db.execute('SELECT digest,bytes FROM documents'))
        overlap = documents.keys() & known.keys()
        if any(documents[k] != known[k] for k in overlap):
            raise ValueError('conflicting document receipt')
        if documents.keys() <= known.keys():
            db.rollback()
            return {'duplicate': True, 'documents': 0}
        if overlap:
            raise ValueError('partially overlapping shards: merge disjoint leaf shards instead')
        # Streaming SQL cursor: counts need not fit in host RAM.
        def rows():
            for c,t,n in other.execute('SELECT context,target,count FROM counts'):
                if not isinstance(c, bytes) or len(c)>order or 255 in c or not isinstance(t,int) or not 0<=t<255 or not isinstance(n,int) or n<=0:
                    raise ValueError('invalid shard count')
                yield c,t,n
        db.executemany(UPSERT, rows())
        db.executemany('INSERT INTO documents VALUES (?,?)', documents.items())
        db.commit()
        return {'duplicate': False, 'documents': len(documents)}
    except BaseException:
        db.rollback()
        raise
    finally:
        other.close()


def recall(index, lemma):
    """Read the existing exact WordNet overlay; never identify synsets by leaf."""
    nodes, aliases = {}, {}
    with open(index, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if len(header)!=3 or header[0]!='NANLM-WN1':
            raise ValueError('invalid WordNet index')
        edges = 0
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if fields[0]=='N' and len(fields)>=5:
                sid, leaf = int(fields[1]), int(fields[2])
                if not 0<=sid<=0xffffffff or not 0<=leaf<=65535 or sid in nodes:
                    raise ValueError('invalid synset identity')
                nodes[sid] = {'synset': 'n:'+str(sid), 'leaf': leaf, 'lemma': fields[3],
                              'gloss': fields[4], 'gateway': 255, 'link': 'HW4:0::0:HW0', 'transport': 'local-intent'}
                aliases.setdefault(fields[3].lower(), set()).add(sid)
            elif fields[0]=='L' and len(fields)==3:
                aliases.setdefault(fields[1].lower(), set()).add(int(fields[2]))
            elif fields[0]=='E' and len(fields)>=4:
                edges += 1
            else:
                raise ValueError('invalid WordNet record')
    if len(nodes)!=int(header[1]) or edges!=int(header[2]):
        raise ValueError('WordNet record counts differ')
    return [nodes[sid] for sid in sorted(aliases.get(lemma.lower(), set()))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('database', help='per-universe exact .context.sqlite overlay')
    p.add_argument('--order', type=int, default=8)
    sub = p.add_subparsers(dest='action', required=True)
    t = sub.add_parser('train'); t.add_argument('corpora', nargs='+')
    e = sub.add_parser('eval'); e.add_argument('corpus')
    g = sub.add_parser('generate'); g.add_argument('--prompt', default=''); g.add_argument('--bytes', type=int, default=512); g.add_argument('--seed', type=int, default=11); g.add_argument('--wordnet'); g.add_argument('--lemma')
    m = sub.add_parser('merge'); m.add_argument('shards', nargs='+')
    r = sub.add_parser('recall'); r.add_argument('index'); r.add_argument('lemma')
    sub.add_parser('inspect')
    a = p.parse_args()
    readonly = a.action in ('eval', 'generate', 'inspect', 'recall')
    db, order = open_db(a.database, a.order, readonly)
    try:
        if readonly:
            db.execute('BEGIN')
        if a.action == 'train':
            for path in a.corpora:
                print(json.dumps(train(db, order, path)))
        elif a.action == 'merge':
            for path in a.shards:
                if Path(path).resolve() == Path(a.database).resolve():
                    raise ValueError('cannot merge a database into itself')
                print(json.dumps(merge(db, order, path)))
        elif a.action == 'eval':
            print(json.dumps(Model(db, order).evaluate(a.corpus)))
        elif a.action == 'generate':
            if not 0<=a.bytes<=1000000:
                raise ValueError('output length must be 0..1000000')
            prompt = a.prompt
            if bool(a.wordnet) != bool(a.lemma):
                raise ValueError('--wordnet and --lemma must be supplied together')
            if a.wordnet:
                facts = recall(a.wordnet, a.lemma)
                print(json.dumps({'retrieved': facts}), file=sys.stderr)
                prompt = '\n'.join(f['lemma']+': '+f['gloss'] for f in facts)+'\n'+prompt
            sys.stdout.buffer.write(Model(db, order).generate(prompt.encode(), a.bytes, a.seed))
        elif a.action == 'recall':
            print(json.dumps(recall(a.index, a.lemma)))
        else:
            print(json.dumps({'format': VERSION, 'order': order, 'gateway': 255, 'link': 'HW4:0::0:HW0',
                              'documents': db.execute('SELECT count(*) FROM documents').fetchone()[0],
                              'context_edges': db.execute('SELECT count(*) FROM counts').fetchone()[0]}))
    finally:
        db.close()


if __name__ == '__main__':
    main()
