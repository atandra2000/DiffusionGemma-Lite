# Concept: memory engineering — the byte budget that shaped the design

> **Canonical** for the peak-VRAM accounting, the chunked-CE memory
> argument, and the VRAM-for-MFU trade. `DIFFUSION.md` §4 stays authoritative
> for rulings; this page derives every byte. All numbers below are computed
> from `utils/memory.py:estimate_model_memory_gb`'s accounting at the
> production config (`configs/pretrain_a100_380m.yaml`).

**Depends on:** [foundations](foundations.md) §7, §16 ·
[diffusion-core](diffusion-core.md) §8 ·
**Read next:** [training](../training.md) · [data-pipeline](data-pipeline.md)

---

## Table of Contents

1. [The four budgets](#1-the-four-budgets)
2. [The CE term: why chunked cross-entropy exists](#2-the-ce-term)
3. [The chunk-scaling law](#3-the-chunk-scaling-law)
4. [The VRAM-for-MFU trade (8x4+ckpt vs 16x2)](#4-the-vram-for-mfu-trade)
5. [Worked example: the full estimate, both layouts](#5-worked-example-the-full-estimate-both-layouts)
6. [What breaks if you change this](#6-what-breaks-if-you-change-this)
7. [Glossary](#7-glossary)
8. [Interview Q&A](#8-interview-qa)

---

## 1. The four budgets

Peak training VRAM is the sum of four independent budgets
(`utils/memory.py:estimate_model_memory_gb`), each with its own scaling law:

| budget | formula (production dims) | bytes at N=343,516,160 | scales with |
|---|---|---|---|
| weights (bf16) | `N · 2` | 0.640 GB | params only |
| AdamW state | `N · 12` (m 4 + v 4 + fp32 master 4) | 3.839 GB | params only — *fixed* |
| activations | `L · S · B · d_model · 2` (checkpointed boundaries) | 1.50 GB @ 8×4096 / 3.00 GB @ 16×4096 | layers × seq × micro_bs |
| CE chain | §2 | 4.07 GB chunked / 12.27 GB naive (ms=8) | seq × micro_bs × V |

Fixed cost first: **4.48 GB** of weights + optimizer is unavoidable —
the interesting engineering is all in the last two rows. Add the framework
overhead reserve (`min(13.7, max(2.0, 0.17 · total)) = 13.6 GB` on an 80 GB
A100, `utils/memory.py:estimate_model_memory_gb`) and the floor before a
single activation is ~18 GB.

## 2. The CE term

The LM head produces `(B, S, V)` logits — at production dims `8 × 4096 ×
50,257` fp32 that is **6.13 GB for the tensor alone**, before softmax, before
the loss, before backward's saved tensors. This is the single largest
activation in the model, and it exists *only* because V = 50,257.

`training/losses.py:chunked_x0_ce` restructures the head + CE into vocab
chunks of `vocab_chunk` (runtime `8192·8/micro_bs` — see §3):

- **backward still needs every chunk's logits** → each chunk's bf16 logits
  are retained: total `B·S·V·2` bytes retained across chunks (half of naive's
  fp32) — `training/losses.py:_ChunkTerms`,
- one **transient fp32 chunk** at a time: `B·S·chunk·4` bytes
  (`training/losses.py:chunked_x0_ce`).

Naive vs chunked at micro_bs 8:

| variant | retained | transient | total |
|---|---|---|---|
| naive fp32 CE | `8·4096·50257·4 = 6.13 GB` | (already counted) | **6.13 GB** |
| chunked (chunk 8192) | `8·4096·50257·2 = 3.07 GB` (bf16 logits) | `8·4096·8192·4 = 1.00 GB` | **4.07 GB** |

The swap buys 2.07 GB *and* buys back the head-GEMM checkpoint recompute —
the retained bf16 logits remove the need to recompute the head GEMM in
backward. Equivalence to eager CE (loss **and** gradient direction,
atol 1e-6, partial last chunk) is pinned by
`tests/test_loss.py::test_chunked_equals_eager`,
`test_chunked_matches_eager_grad_direction`, `test_partial_last_chunk`.

At micro_bs 16 the same accounting gives 7.13 GB chunked vs **12.27 GB**
naive fp32 — chunking is what makes the no-checkpoint layout (§4) possible.

## 3. The chunk-scaling law

`vocab_chunk = 8192·8/micro_bs` (`utils/memory.py:estimate_model_memory_gb`
default 8192; trainer passes `8192·8/micro_bs`) keeps the **transient fp32
chunk** constant in bytes across micro-batch sizes:

```
transient = micro_bs · seq · vocab_chunk · 4
          = micro_bs · 4096 · (8192·8/micro_bs) · 4
          = 4096 · 8192 · 32 = 1.07 GB   (independent of micro_bs)
```

while the retained-bf16 term `micro_bs · seq · V · 2` scales *linearly* with
micro_bs (it is irreducible: backward needs every position's full-vocab
logits). So:

| micro_bs | vocab_chunk | transient fp32 (GB) | retained bf16 (GB) |
|---|---|---|---|
| 8 | 8192 | 1.00 | 3.07 |
| 16 | 4096 | 1.00 | 6.13 |

The law: **micro_bs trades against vocab_chunk at fixed transient bytes** —
halve one, double the other, and the fp32 spike stays put. The retained term
is the price of the batch; the chunk only sets the shape of the spike. This
is also why `chunked_p_embed` (the self-cond pre-pass) reuses the identical
mechanism under `no_grad` — it faces the same (B, S, V) hazard with no
gradients at all ([diffusion-core §8](diffusion-core.md),
[self-conditioning §6](self-conditioning.md)).

## 4. The VRAM-for-MFU trade

The config comment is the decision record
(`configs/pretrain_a100_380m.yaml` training block): the §4.0 table was
written for `micro 8 × accum 4 + gradient checkpointing`; the shipped config
is `micro 16 × accum 2` with **no checkpointing**. What that buys:

| | 8×4 + ckpt (§4.0 layout) | 16×2, no ckpt (shipped) |
|---|---|---|
| boundary activations | 1.50 GB | 3.00 GB |
| SwiGLU intermediates | recompute (checkpointed) | **retained: 27.00 GB** |
| CE (chunked) | 4.07 GB | 7.13 GB (chunk 4096) |
| total estimate | 23.6 GB / 80 | 55.2 GB / 80 (margin 24.8) |

The 27 GB SwiGLU retention buys back the every-3-layer recompute
(`grad_checkpoint_every: 3` — inert in the shipped config): ~33% of
checkpointed-layer forwards disappear, which is pure MFU. The estimator
models both layouts (`grad_checkpoint=True/False` branches in
`utils/memory.py:estimate_model_memory_gb`); the guard
`utils/memory.py:assert_fits_in_available_gpu` raises before launch when the
estimate leaves less than a 2 GB safety margin.

## 5. Worked example: the full estimate, both layouts

Toy scale: `N = 101,888` params (the tiny fixture model),
`seq 128, micro 2, d_model 64, layers 2, V 256, chunk 128`, overhead 2.0 GB
(CPU):

```
weights bf16        101,888 · 2   =        0.0002 GB
AdamW m+v+master    101,888 · 12  =        0.0012 GB
boundary acts       2·4096·64·2 B =        0.0000 GB   (layers·seq·B·d·2)
CE retained bf16    8·4096·256·2  =        0.0000 GB
CE transient fp32   8·4096·128·4  =        0.0000 GB
overhead                                    2.0 GB
                    ─────────────
estimate                     ≈ 2.0014 GB
```

Take-away: at toy scale everything except the overhead reserve vanishes —
which is exactly why the estimator's *shape* (param/optim/act/CE terms) is
tested for monotonicity and for dominating the CE term at production scale,
not for absolute values
(`tests/test_utils.py::test_memory_estimator_monotone_in_batch`,
`test_chunked_ce_term_bounds_memory_estimate`).

Production recap (computed, §1–§4): 8×4+ckpt → **23.6 GB**; 16×2 →
**55.2 GB**; both fit 80 GB with the 13.6 GB framework overhead reserve.

## 6. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| full-vocab fp32 CE at micro 16 | +12.27 GB — over the 16×2 budget (55.2 → 60.4, margin gone at larger seq) | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |
| `vocab_chunk` scaled with micro_bs | transient fp32 spike scales with micro_bs; the constant-spike law broken | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |
| chunked logits retained in fp32 | retained term doubles: 4.07 → 8.13 GB (ms=8) | (no pin; VRAM guard at launch) |
| `chunked_p_embed` run under grad mode | every chunk's logits retained *with graph* — silent multi-GB blow-up | `tests/test_loss.py::test_chunked_p_embed_matches_eager` (correctness; memory hazard documented) |
| drop the 13.7 GB overhead cap on a 80 GB part | estimate under-counts allocator fragmentation | `utils/memory.py:estimate_model_memory_gb` overhead rule |
| estimate with `grad_checkpoint=False` on the 8×4 layout | under-counts SwiGLU retention (27 GB surprise) | `utils/memory.py:estimate_model_memory_gb` docstring contract |

## 7. Glossary

| symbol | meaning | code |
|---|---|---|
| `N` | parameter count, 343,516,160 | `training/pretrain.py:count_parameters` |
| `K` | vocab chunk width | runtime `vocab_chunk` (8192·8/micro_bs) |
| `_ChunkTerms` | per-chunk retained bf16 logits + fp32 transient | `training/losses.py:_ChunkTerms` |
| `act_bytes` | checkpointed block-boundary activations | `utils/memory.py:estimate_model_memory_gb` |
| `overhead_gb` | allocator/framework reserve (≤ 13.7 GB) | `utils/memory.py:estimate_model_memory_gb` |
| `assert_fits` | pre-launch VRAM guard, 2 GB margin | `utils/memory.py:assert_fits_in_available_gpu` |

## 8. Interview Q&A

**Q: Walk me through the 80 GB budget.**
A: Weights 0.64 + AdamW 3.84 = 4.48 GB fixed; activations 1.5 GB
(checkpointed) or 3.0 + 27 GB (no-ckpt, SwiGLU retained); CE 4.07 GB chunked
vs 6.13+ naive; framework reserve 13.6 GB — landing at ~23.6 GB (8×4+ckpt)
or ~55.2 GB (16×2 no-ckpt) of 80
(`utils/memory.py:estimate_model_memory_gb`).

**Q: Why is the CE term special?**
A: The LM head materializes `(B, S, V)` logits — 6.13 GB fp32 at micro 8,
12.27 at micro 16. `training/losses.py:chunked_x0_ce` keeps every chunk's
bf16 logits for backward (irreducible) plus one transient fp32 chunk, and
the equivalence is pinned at atol 1e-6
(`tests/test_loss.py::test_chunked_equals_eager`).

**Q: Why keep the bf16 logits instead of recomputing the head in backward?**
A: Recompute costs a full head GEMM per backward; retaining bf16 chunks is
half the bytes of fp32 and trades ~3.3 GB retained for removing that recompute
— the explicit DESIGN §4.0 trade
(`training/losses.py:chunked_x0_ce` docstring).

**Q: What is the chunk-scaling law?**
A: Transient fp32 = micro_bs · seq · vocab_chunk · 4; setting
`vocab_chunk = 8192·8/micro_bs` pins the spike at 1 GB regardless of
micro_bs, while the retained term (micro_bs · seq · V · 2) grows linearly
with the only knob that *must* grow
(`utils/memory.py:estimate_model_memory_gb`).

**Q: Why did the config move from 8×4+checkpointing to 16×2 without?**
A: At 80 GB the no-ckpt layout fits (55.2 of 80 GB) and deletes the
every-3-layer recompute — ~33% of checkpointed-layer forwards — so MFU
improves for free; the §4.0 table describes the older, more conservative
layout (`configs/pretrain_a100_380m.yaml` training block comment).

**Q: How is the estimate validated?**
A: Monotonicity in batch
(`tests/test_utils.py::test_memory_estimator_monotone_in_batch`) and the
chunked-CE term bounding the total at production dims
(`tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate`), plus
the `assert_fits_in_available_gpu` pre-launch guard with a 2 GB margin
(`utils/memory.py:assert_fits_in_available_gpu`).