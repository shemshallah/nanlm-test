# Source-disjoint transfer sentinel

This diagnostic addresses the principal weakness of the earlier
constructed 974-byte prose fixture without claiming open-domain fluency.

## Split policy

The train file consists of nine distinct authored passages about
workshop, coastal, agricultural, archival, and domestic activities.
Development text uses different observational passages and is the
**only** split permitted to select the existing readout scales.
The held-out files cover three separate domains: ocean research,
spaceflight operations, and civic administration.

Each fixture is SHA-256 identified. `tools/transfer_benchmark.py`
validates UTF-8, prohibits reserved 0xFF bytes, requires substantial
length, and rejects every identical 32-byte span shared by any two
files. This is strict literal-overlap control, not a guarantee of
semantic or topic independence. Because the fixtures were authored
during development, this suite should remain an **engineering sentinel**.

## Exactly one model

The benchmark creates one temporary 8,388,608-byte mapped Universe,
audits each test domain before training, trains twice on the train
split, audits again, calibrates using development text only, then
audits each domain again with all diagnostic readout ablations.

No held-out metric participates in optimization. Generation uses one
fixed seed twice to confirm deterministic valid UTF-8. The run records
full reproducibility hashes, per-domain CE and accuracy, ablation
results, a weighted aggregate, and the actual continuation in
`transfer-report.json`.

It deliberately does **not** require held-out improvement, enabling
negative results. It does require finite probabilities, deterministic
generation, no leakage of long literal substrings, correct root size,
and the same number of tokens across readout ablations.

Run: `make nanlm CXX=clang++ && python3 tools/transfer_benchmark.py`.
CI: `.github/workflows/unified-transfer.yml`.
