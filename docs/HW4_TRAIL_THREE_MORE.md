# HW4 TRAIL — three additional complete sweeps

This experiment extends the **successful** [one-pass native two-rail
TRAIL benchmark](https://github.com/shemshallah/nanlm-test/actions/runs/42739524950)
by three sequential additional complete passes over exactly the same
SHA-pinned 58,341,529-byte HW4-derived WordNet corpus.

**Crucially:** pass 2 starts from the successful pass-1 checkpoint
SHA-256 `a95a89f2856264d190d996363d39dc11bb8bb5ca8f30eae98c3cc915a2433695`.
Pass 3 starts from pass 2; pass 4 starts from pass 3. Each sweep freezes
the current 255×255 forward/reverse value matrix for the duration of that
pass, updates all 255 head eligibility tracks for each byte transition,
then attempts one scratch-accepted rank-three head-vector update.
No BPTT, no separate model instances, no HW4 source writes.

It logs per-pass **exact full-coverage bytes**, full-corpus teacher CE and
top-1 accuracy before/after, independent 974-byte prose CE and accuracy,
forward/inverse credit magnitude, update acceptance/rejection, scan wall
time, whole native runtime, and the resulting 8 MiB learner hash.
A rejected scratch proposal is an honest result, not a hidden extra
restart or discarded epoch.

This is a **frozen-value two-rail TD eligibility batch sweep** per epoch.
It is *not* 3×58 million full autoregressive recurrent, triad and wave
optimizer updates. Semantic WordNet knowledge remains unproven. The source
is immutable and its previous dev/test splits are not unseen because
the entire corpus is trained.

The workflow `.github/workflows/hw4-trail-three-more.yml` saves
only the final root, its original lineage, timing files, evaluation
results and checksummed receipt—not duplicate intermediate checkpoint
files and not a copy of the WordNet archive.
