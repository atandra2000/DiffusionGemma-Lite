# Inference pipeline — the eval harness, the metrics, and the honest gaps

> **Canonical** for the evaluation protocol. `DIFFUSION.md` §4.2 stays
> authoritative for the eval design; this page walks the harness, the
> speedup metric, the quality anchor, and — explicitly — what is *not*
> measured yet.

**Depends on:** [sampler](concepts/sampler.md) §5 ·
[diffusion-core](concepts/diffusion-core.md) §5 ·
**Read next:** [benchmarking](guides/benchmarking.md) · [sampler-tuning](guides/sampler-tuning.md)

---

## Table of Contents

1. [The harness](#1-the-harness)
2. [The headline metric: tokens per forward](#2-the-headline-metric)
3. [Quality: held-out x0-NLL](#3-quality-held-out-x0-nll)
4. [Honest gaps](#4-honest-gaps)
5. [Worked example: reading a row at toy scale](#5-worked-example-reading-a-row-at-toy-scale)
6. [What breaks if you change this](#6-what-breaks-if-you-change-this)
7. [Glossary](#7-glossary)
8. [Interview Q&A](#8-interview-qa)

---

## 1. The harness

`inference/evaluate.py:SpeedupEvaluator` measures the three sampler schedules
plus the analytic AR row in one call:

```python
# inference/evaluate.py:SpeedupEvaluator.evaluate (structure)
rows = {name: self._measure(n, prompt, gen, parse_baseline(name), wall_clock)
        for name in ("fixed_T16", "fixed_T32", "adaptive_T32")}
rows["ar_kv_analytic"] = {..., "tokens_per_forward": 1.0, "seconds": None}
```

`parse_baseline("adaptive_T32")` → `SamplerConfig(n_diffusion_steps=32,
adaptive=True)` (`inference/evaluate.py:parse_baseline`). Counting works by
**instrumenting the backbone**: `SpeedupEvaluator._count_forwards` wraps
`model.backbone`, incrementing `forwards` per call and `token_forwards` by
`ids.size(1)` — no code changes inside the model, restored in a `finally`.

## 2. The headline metric

`tokens_per_forward = n_samples · gen_tokens / forwards` — checkpoint-free
by construction:

- **AR + KV cache**: 1 forward per token → `tokens/forward = 1.0` exactly.
  No AR checkpoint is needed to state this — it is what KV-cache decode
  *is* (`inference/evaluate.py:SpeedupEvaluator.evaluate` writes the
  `ar_kv_analytic` row with `seconds=None`).
- **Block-AR**: `(T_eval + 1)` forwards per canvas →
  `tokens/forward = L / (T_eval + 1)`.

| row | forwards | tokens/forward | measured? |
|---|---|---|---|
| `fixed_T16` | 17/canvas | 15.06× | forwards + wall-clock |
| `fixed_T32` | 33 | 7.76× | forwards + wall-clock |
| `adaptive_T32` | ≤ 33 (entropy bond) | ≥ 7.76× | forwards + wall-clock |
| `ar_kv_analytic` | 1/token | 1.0× | analytic only |

Per-schedule detail at production dims (256-token canvases, 1,024-token
generations = 4 canvases): 68 forwards at T=16 vs 1,024 for AR — 15.1× fewer
forwards; token_forwards/token = 17.0 at T=16, 33.0 at T=32
(`tests/test_inference.py::test_flop_counter_adaptive_le_fixed` pins
adaptive ≤ fixed; `test_evaluator_produces_three_rows` pins the row set).

## 3. Quality: held-out x0-NLL

`inference/evaluate.py:heldout_x0_nll` is the quality anchor — the training
objective on held-out windows:

```
per window:  t ~ U{1..T} per canvas (seeded rng)
             xt = q_sample(x0, t)                # corruption, eval protocol
             hidden = model.final_hidden(xt, t)  # NO sc input
             NLL_w = chunked_x0_ce(hidden, E, x0)     # nats/token
return mean over windows
```

- **The AR comparison is direct**: x0-CE(nats/token) on the same shard is
  the same unit as an AR baseline's CE — that parity *is* the quality anchor
  (DESIGN §4.2).
- **No self-cond input** at eval-time NLL: the metric conditions the model
  exactly as the training corruption step does, minus the sc input — matching
  the loss-parity protocol
  (`inference/evaluate.py:heldout_x0_nll`; sc-off is also why the number is
  directly comparable to a no-sc AR baseline).
- Deterministic: fixed seed for window picks *and* corruption draws
  (`tests/test_inference.py::test_heldout_x0_nll_finite`,
  `test_heldout_x0_nll_short_shard_raises` for the too-short-shard error).

## 4. Honest gaps

The harness discloses what it does *not* measure (DESIGN §4.2):

| gap | why | code |
|---|---|---|
| AR wall-clock tokens/s | needs a trained external AR checkpoint; without it the row is `seconds=None`, analytically 1 forward/token | `inference/evaluate.py:SpeedupEvaluator.evaluate` |
| decode preview / text output | tokenizer accepted but unused — counting is tokenizer-free | `inference/evaluate.py:SpeedupEvaluator.__init__` |
| quality-vs-steps curves | single NLL point per shard, not a sweep | `inference/evaluate.py:heldout_x0_nll` |
| sample-quality metrics (perplexity beyond x0-NLL, MAUVE, human eval) | out of harness scope; not yet built | — |

The speedup claim survives these gaps because it is *counting*, not timing:
forwards and token-forwards are exact integers from the instrumented loop
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`).

## 5. Worked example: reading a row at toy scale

Tiny fixture model (`V=256, d=64, L=32`), `T_eval=4`, generate 2 canvases
with `n_samples=1, gen_tokens=64`:

```
forwards       = 2 canvases × (4 denoise + 1 encode) = 10
token_forwards = 10 forwards × 32 tokens             = 320
tokens generated                                    = 64

tokens_per_forward        = 64 / 10        = 6.4
token_forwards_per_token  = 320 / 64       = 5.0
ar_kv_analytic row        = 1.0 (by construction)
speedup (tokens/forward)  = 6.4 × AR
```

Take-away: `tokens_per_forward` and `token_forwards_per_token` differ —
the first charges the prompt canvas too (64 generated tokens over 10
forwards), the second counts *all* tokens pushed through the backbone
(including re-denoising passes over already-committed positions). The
headline is the first; the second is the honest cost accounting
(`inference/evaluate.py:SpeedupEvaluator._measure` returns both).

## 6. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| count forwards inside the sampler (not via backbone wrap) | double-counts encode+denoise differently; rows incomparable | `tests/test_inference.py::test_evaluator_produces_three_rows` |
| AR row wall-clock without a real checkpoint | fabricated baseline — the exact gap §4 forbids | (disclosure contract, `ar_kv_analytic.seconds=None`) |
| NLL with self-cond input on | eval ≠ training-corruption parity; number not comparable to AR CE | `tests/test_inference.py::test_heldout_x0_nll_finite` |
| `heldout_x0_nll` on the training shard | leakage; quality metric meaningless | (protocol; shard path caller-provided) |
| `parse_baseline` name format change | config rows fail to parse | `tests/test_inference.py::test_parse_baseline` |

## 7. Glossary

| symbol | meaning | code |
|---|---|---|
| `forwards` | backbone call count (denoise + encode + prefill) | `inference/evaluate.py:SpeedupEvaluator._count_forwards` |
| `token_forwards` | sum of sequence lengths over forwards | same |
| `tokens_per_forward` | generated tokens / backbone forwards | `inference/evaluate.py:SpeedupEvaluator.evaluate` |
| `heldout_x0_nll` | chunked x0-CE on held-out windows, no sc | `inference/evaluate.py:heldout_x0_nll` |
| `parse_baseline` | `"fixed_T16"` → `SamplerConfig` | `inference/evaluate.py:parse_baseline` |
| `ar_kv_analytic` | AR row: 1 forward/token by construction | `inference/evaluate.py:SpeedupEvaluator.evaluate` |

## 8. Interview Q&A

**Q: What exactly does the 15.06× number measure?**
A: Generated tokens per backbone forward at T=16: 256 tokens per canvas /
17 forwards (16 denoise + 1 encode) vs AR's exactly-1 — the ratio is 15.06
(`inference/evaluate.py:SpeedupEvaluator.evaluate`). It is checkpoint-free:
forwards are counted by instrumenting `model.backbone`.

**Q: Why is the AR baseline "analytic"?**
A: A KV-cached AR decoder performs exactly 1 forward per generated token by
construction — measuring it would need an external trained checkpoint, so
the harness states the row analytically and leaves wall-clock as the
disclosed gap (`inference/evaluate.py:SpeedupEvaluator.evaluate`).

**Q: Why is x0-NLL the quality anchor, not perplexity of generated text?**
A: It is the training objective evaluated on held-out data at sampled t —
directly comparable to an AR model's CE on the same shard, checkpoint-free,
and deterministic under the fixed seed
(`inference/evaluate.py:heldout_x0_nll`). Sample-quality metrics
(MAUVE, human eval) are the disclosed gap.

**Q: Why no self-conditioning input in the eval NLL?**
A: The quality anchor compares against an AR baseline's CE on the same
tokens; conditioning on the model's own no-grad guess would change the
objective being measured
(`inference/evaluate.py:heldout_x0_nll` — no `sc_input` in the call).

**Q: What would you add first to strengthen this harness?**
A: An AR wall-clock row from a real checkpoint (the one `None` in the
output), then decode previews — both are constructor-ready
(`SpeedupEvaluator.__init__` already accepts the tokenizer).