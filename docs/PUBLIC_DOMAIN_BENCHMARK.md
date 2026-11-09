# Cross-author public-domain benchmark

Two independently authored historic texts are downloaded directly from
Project Gutenberg during CI, never stored in the repository or uploaded
as corpus artifacts:

- Lewis Carroll, *Alice's Adventures in Wonderland*, ebook 11
- Charles Dickens, *A Christmas Carol*, ebook 46

Project Gutenberg describes both as public domain in the United States.
Follow its license and local legal restrictions for uses outside the US.

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

Since Project Gutenberg content may be updated, archive the SHA-256
recorded in each Actions report before comparing across runs.
The live download is a network-dependent integration test; a transient
download failure is not a model regression. The authored offline suite
remains independently available.
