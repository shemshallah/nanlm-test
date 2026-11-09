# Independent 1 MiB micro-transformer heads (reference)

Status: **PyTorch/C reference harness in NaNLM v1**, not yet ported to
NaNLMind v2 and not a native NaN16 claim. The goal is to validate the memory
budget, local-credit training and two-layer vote protocol before porting.

## Budget

Each head owns one complete micro-transformer:

| component | parameters |
|---|---:|
| QKV, d=128 | 49,152 |
| output projection, d=128 | 16,384 |
| MLP 128→1504→128 | 386,656 |
| token embedding, vocab=256 | 32,768 |
| output head, vocab=256 | 33,024 |
| layer norms | 1,280 |
| **total** | **519,264** |

- 16-bit weights: **1,038,528 B/head** (fits 1 MiB with 10,048 B spare).
- 8-bit blockwise Adam m/v: **1,038,528 B/head**.
- 256 heads: **509.08 MiB** including scales, under the 512 MiB target.
- A 64-head bank was physically allocated and audited at **127.27 MiB**; the
  256-head layout is the same arithmetic (509.08 MiB).

The bank is an `int16` weight matrix plus `int8` moments and Q16 block scales.
Only the active head is dequantized to float32 for a local step.

## Local independence and votes

- Each head has its own parameters, embedding, readout, loss, gradients,
  Adam moments and persistence. No gradient crosses heads.
- A head's candidate is staged; the head votes using its local loss delta and
  a CFT loop residual.
- The 3-head triad accepts with 2/3.
- The 85 triads vote on the global bundle; the full field requires 43/85.
- A rejected candidate is restored from the pre-step snapshot; committed state
  is never partially written.

Reduced-head smoke runs use candidate-only field thresholds because 43/85
cannot be reached with 3 heads; this is labeled `candidate_only` in the result.

## Smoke result

`python3 -m experiments.independent_heads.train --heads 3 --steps 1 --seq 16 --lr 1e-4`

- candidate loss: 5.7118 → 5.6029
- committed loss: 5.6029
- triad vote: true (2/3)
- field: accepted in candidate-only mode
- memory for 3 heads: 5.97 MiB

This is a tiny synthetic next-byte task, not language quality evidence.

## GPT-2 comparison

| model | params | fp32 weights | fp16 weights | int8 weights | training memory (fp32 Adam + grads) |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 124M | ~496 MB | ~248 MB | ~124 MB | ~1.5 GB+ |
| GPT-2 medium | 355M | ~1.42 GB | ~710 MB | ~355 MB | ~4.3 GB+ |
| GPT-2 large | 774M | ~3.1 GB | ~1.55 GB | ~774 MB | ~9.3 GB+ |
| GPT-2 XL | 1.5B | ~6.0 GB | ~3.0 GB | ~1.5 GB | ~18 GB+ |

The 256 independent heads hold **132.9M parameters**, slightly more than
GPT-2 small, in **509 MiB** of 16-bit weights + 8-bit moments. They are not a
GPT-2 replacement: each head is one shallow block with an 8–16 token causal
window and local training, not a deep 1024-token transformer.

## Files

- `experiments/independent_heads/model.py` — per-head QKV/MLP model.
- `experiments/independent_heads/quant_bank.py` — int16 weights, int8 Adam,
  block scales, memory audit.
- `experiments/independent_heads/field.py` — CFT residual + head/triad/field votes.
- `experiments/independent_heads/train.py` — local independent trainer.
- `tests/test_independent_heads.py` — budget, quantisation, independence,
  vote thresholds and smoke training.

## Limits

- PyTorch/float compute; the persistent bank is quantized.
- No full 256-head training run yet; no multi-author corpus, free generation or
  native NaN16 parity.
- CFT is a small loop residual, not a complete geometric replacement for BPTT.
- The v2 port must implement the same vote semantics in native NaN16 and sync
  against canonical nan0s.
