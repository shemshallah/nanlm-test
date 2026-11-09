# Free-generation diagnostics on one native NaNLM root

The existing source-disjoint benchmark now generates exactly 120 bytes
after the same UTF-8 prefix and deterministic seed **before** and
**after** training, using the same 8 MiB root.

It reports alpha and whitespace fractions, unique four-byte n-gram
ratio, the fraction of generated eight-byte spans found in the training
corpus, and a literal escaped sample. All metrics describe the actual
sample rather than forcing a numeric improvement.

These metrics are not substitutes for linguistic or human evaluation.
High novel n-gram ratios can accompany nonsense; training overlap
is not itself evidence of memorization without a stronger holdout
design. The sample is recorded to make incoherence and excessive
repetition visible instead of assuming fluency from cross-entropy.

`tools/transfer_benchmark.py` writes both samples and all metrics to
`transfer-report.json`, and prints concise `TRANSFER_GENERATION`
lines in the GitHub Actions log. It creates no additional model and
performs no generation-based calibration.
