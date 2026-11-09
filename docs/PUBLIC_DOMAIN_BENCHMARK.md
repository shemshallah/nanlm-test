# Cross-author public-domain benchmark

Two independently authored historic public-domain texts are downloaded
from **immutable GITenberg GitHub commit snapshots** during CI. They are
never stored in this repository or uploaded as corpus artifacts:

- Lewis Carroll, *Alice's Adventures in Wonderland*, ebook 11
- Charles Dickens, *A Christmas Carol*, ebook 46

Project Gutenberg describes both as public domain in the United States.
The GITenberg mirrors are historical copies of these Project Gutenberg
ebooks. Follow license requirements and local restrictions outside the US.
The source revisions and byte hashes are pinned in the benchmark script.

The parser requires legitimate START/END ebook boundaries, records raw
source SHA-256 and exact split hashes, trains on a 10,500-character
early slice of Carroll's book, calibrates on a noncontiguous later
Carroll segment, and tests on two nonadjacent chunks of Dickens. No
Dickens passage is used in training or calibration. This isolates
cross-author generalization but does not solve domain distribution shift
or establish independent document-level fluency.

The suite records pretraining, uncalibrated and calibrated cross-author
byte CE, accuracy, and actual free generation. It deliberately does not
assert that training must improve the held-out cross-author score.

The previous live `www.gutenberg.org/cache/epub` endpoint returned
repeated HTTP 503/504 errors in GitHub Actions. This version uses fixed,
content-verified GITenberg commits instead:

- Alice commit `7f18748e114b994d5d9454d5793581bc0b9cf9d7`
- Carol commit `b173bd3787848ecbded8d036ed9f5326c738da19`

The benchmark additionally verifies fixed SHA-256 values of the exact
raw files. Because these snapshots are **different editions from the
earlier live gateway**, previous CE numbers should not be compared directly
to the new baseline. Within each current CI run, training and held-out
comparisons use exactly the same pinned source bytes and split positions.
A transient GitHub download failure is still not a model regression;
the authored offline suite remains independently available.
