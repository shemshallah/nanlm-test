import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('context_lm',Path(__file__).resolve().parents[1]/'tools/context_lm.py')
lm=importlib.util.module_from_spec(spec);spec.loader.exec_module(lm)

class ContextTests(unittest.TestCase):
 def test_learning_restart_merge_and_overlap(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp)
   a=d/'a.txt';b=d/'b.txt';held=d/'held.txt'
   a.write_text(('the little fox walks through the quiet forest. the little bird sings in the quiet forest.\n')*20)
   b.write_text(('the little fox sleeps near the river. the little bird flies over the river.\n')*20)
   held.write_text('the little bird walks through the quiet forest. the little fox flies over the river.\n')
   shards=[]
   for i,corpus in enumerate((a,b)):
    path=d/f'{i}.db';db,order=lm.open_db(path,8)
    lm.train(db,order,corpus,batch=17)
    before=db.execute('SELECT sum(count) FROM counts').fetchone()
    self.assertTrue(lm.train(db,order,corpus)['duplicate'])
    self.assertEqual(before,db.execute('SELECT sum(count) FROM counts').fetchone())
    db.close();shards.append(path)
   aggregate=d/'all.db';db,order=lm.open_db(aggregate,8)
   for shard in shards:self.assertFalse(lm.merge(db,order,shard)['duplicate'])
   self.assertTrue(lm.merge(db,order,shards[0])['duplicate'])
   model=lm.Model(db,order)
   ce=model.evaluate(held)['nats_per_byte']
   unigram=lm.Model(db,0).evaluate(held)['nats_per_byte']
   self.assertLess(ce,unigram)
   self.assertAlmostEqual(sum(model.probability(b'the little ',i) for i in range(255)),1.)
   generated=model.generate(b'the little ',256,11)
   self.assertNotIn(255,generated)
   rows=list(db.execute('SELECT * FROM counts ORDER BY context,target'))
   db.close()
   db,order=lm.open_db(aggregate,readonly=True)
   self.assertEqual(generated,lm.Model(db,order).generate(b'the little ',256,11));db.close()
   # Additive aggregation equals sequential disjoint-document training.
   direct,_=lm.open_db(d/'direct.db',8)
   lm.train(direct,8,a);lm.train(direct,8,b)
   self.assertEqual(rows,list(direct.execute('SELECT * FROM counts ORDER BY context,target')));direct.close()
   reverse,_=lm.open_db(d/'reverse.db',8)
   lm.merge(reverse,8,shards[1]);lm.merge(reverse,8,shards[0])
   self.assertEqual(rows,list(reverse.execute('SELECT * FROM counts ORDER BY context,target')));reverse.close()
   partial,_=lm.open_db(d/'partial.db',8);lm.train(partial,8,a)
   with self.assertRaisesRegex(ValueError,'partially overlapping'):lm.merge(partial,8,aggregate)
   self.assertEqual(1,partial.execute('SELECT count(*) FROM documents').fetchone()[0]);partial.close()
   print(f'context held-out recombination CE={ce:.6f}, unigram={unigram:.6f}; sample={generated[:120]!r}')

 def test_exact_wordnet_collision(self):
  with tempfile.TemporaryDirectory() as tmp:
   index=Path(tmp)/'wordnet.tsv'
   index.write_text('NANLM-WN1\t2\t0\nN\t10\t42\tdog\ta canine\nN\t11\t42\tcat\ta feline\nL\thound\t10\n')
   dog=lm.recall(index,'hound')[0];cat=lm.recall(index,'cat')[0]
   self.assertEqual(dog['leaf'],cat['leaf']);self.assertNotEqual(dog['synset'],cat['synset'])
   self.assertEqual('HW4:0::0:HW0',dog['link']);self.assertEqual(255,dog['gateway'])
   self.assertEqual([],lm.recall(index,'missing'))

 def test_gateway_boundary_and_corrupt_merge(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);corpus=d/'bytes';corpus.write_bytes(b'ab\xffcd')
   db,_=lm.open_db(d/'model',2);lm.train(db,2,corpus)
   self.assertEqual([],list(db.execute('SELECT * FROM counts WHERE context=?',(b'b',))))
   db.close()
   bad=lm.sqlite3.connect(d/'model');bad.execute('UPDATE counts SET context=? WHERE context=?',(b'\xff',b'a'));bad.commit();bad.close()
   dest,_=lm.open_db(d/'dest',2)
   with self.assertRaisesRegex(ValueError,'invalid shard'):lm.merge(dest,2,d/'model')
   self.assertEqual(0,dest.execute('SELECT count(*) FROM counts').fetchone()[0]);dest.close()

if __name__=='__main__':unittest.main()
