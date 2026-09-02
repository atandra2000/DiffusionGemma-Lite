# Concept: memory engineering — the byte budget that shaped the design

> **Canonical** for the peak-VRAM accounting, the chunked-CE memory
> argument, and the VRAM-for-MFU trade. DESIGN §4.0 (mirrored as
> `DIFFUSION.md` §5) stays authoritative for rulings; this page derives
> every byte. All numbers below are computed from
> `utils/memory.py:estimate_model_memory_gb`'s accounting at the
> production config (`configs/pretrain_a100_380m.yaml`).

**Depends on:** [foundations](foundations.md) §7, §16 ·
[diffusion-core](diffusion-core.md) §8 ·
**Read next:** [training](../training.md) · [data-pipeline](data-pipeline.md)

**Units.** "GB" in this chapter means GiB (1024³ bytes) — that is the unit
the estimator divides by. Docstrings and `DIFFUSION.md` sometimes quote
decimal GB; 4.07 GB here ≈ 4.4 GB decimal. When reconciling this chapter
against a docstring, convert first, then compare.

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
| weights (fp32 master) | `N · 4` | 1.28 GB | params only |
| AdamW state | `N · 12` (m 4 + v 4 + master 4) | 3.84 GB | params only — *fixed* |
| activations | §1.3 | 1.50 GB @ 8×4096+ckpt / 30.0 GB @ 16×4096 | layers × seq × micro_bs |
| CE chain | §2 | 4.07 GB chunked (ms=8) / 7.14 GB (ms=16) | seq × micro_bs × V |

The decomposition is worth internalizing because each row responds to a
different knob. Weights and optimizer state are **fixed**: they depend only
on N, so no batch/seq/precision decision moves them (see §1.2 for the
optimizer-class question). Activations and the CE chain are **linear in the
token count** `S·B` of a micro-batch: they are the only terms a
training-layout decision can actually trade. The engineering story of this
chapter is: make the fixed part as small as the mixed-precision contract
allows, then spend every remaining byte of VRAM on tokens rather than on
recompute.

### 1.1 The weights: why the master copy is fp32

`utils/memory.py:estimate_model_memory_gb` reads
`p.numel() * p.element_size()` per parameter, and the model is moved with
`.to(self.device)` only (`training/pretrain.py:Pretrainer.__init__`) — no
`.to(torch.bfloat16)` exists anywhere in the training path. Every parameter
is fp32: `param_bytes = N · 4 = 1.28 GB`. bf16 exists only *inside* the
autocast region (`training/pretrain.py:Pretrainer._amp_context`) as
transient per-matmul casts, never retained across a step — which is why the
row says `N · 4`, not the `N · 2` you might expect from "a bf16 model": the
weights are fp32; only the *compute* is bf16.

Why not store weights in bf16 and save 0.64 GB? AdamW updates would round
to bf16 precision: with lr ≈ 3e-4 on weights of scale 0.02, many per-step
updates are smaller than one bf16 ulp and get rounded away — the classic
silent stall of pure-bf16 training. The fp32 master is the standard
mixed-precision contract, 0.64 GB is noise next to the 30 GB activation
term, and it lets AdamW run fused on CUDA
(`training/pretrain.py:Pretrainer.__init__`) — a non-fused optimizer step
is bandwidth-bound dead time, which matters for the same reason §4 cares
about MFU.

### 1.2 The AdamW state: 12 bytes per parameter, and why not less

AdamW keeps two moment buffers plus a master copy — `m` and `v` in fp32
(4 + 4 bytes) plus the 4-byte master weight copy the estimator attributes
to the optimizer — `N · 12 = 3.84 GB`
(`utils/memory.py:estimate_model_memory_gb`). With §1.1 the fixed floor is
`N · 16 = 5.12 GB`, which in decimal GB is 5.50 — exactly the
"params + AdamW fp32 state ~5.5 GB" line in `DIFFUSION.md` §5.

Could this be shrunk? 8-bit optimizers, Lion (no `v`), or optimizer
sharding all exist, and the repo declines them: **PyTorch-only** — 8-bit
optimizers are a third-party package and sharding needs multi-GPU, but the
target is 1× A100; **the wrong target** — 5.12 GB is 6% of an 80 GB part,
and every memory-engineering hour here buys less than one micro-batch of
activation budget (§1.5), because the binding constraint is the linear
term; **reproducibility** — `tests/test_training.py::test_checkpoint_resume_determinism`
pins optimizer round-trips, and a quantized moment buffer changes every
resumed step's numerics and would need its own equivalence pins.

Fixed cost first, then: **5.12 GB** of weights + optimizer is unavoidable —
the interesting engineering is all in the next two rows.

### 1.3 The activations: two layouts, one formula

The activation term has two branches in the estimator
(`utils/memory.py:estimate_model_memory_gb`), matching the two training
layouts:

**Checkpointed** (`grad_checkpoint=True`): each checkpointed layer retains
only its *input hidden state*; everything the block computes inside is
re-derived in backward. Retained bytes per layer are one bf16 hidden:

```
act_bytes = n_layers · seq · micro_bs · d_model · 2
          = 24 · 4096 · 8 · 1024 · 2 = 1.50 GB   (ms=8; 3.00 at ms=16)
```

**No checkpointing** (`grad_checkpoint=False`, the shipped config): the
block inputs are retained *and* the three SwiGLU intermediates survive —
gate, up, and `silu(gate)·up`, each `ffn_dim=3072` wide, each saved by the
down-projection matmul for its backward:

```
act_bytes = L·S·B·d·2 + L·3·S·B·ffn·2
          = 3.00 GB + 24·3·4096·16·3072·2 / 2^30 = 3.00 + 27.00 = 30.0 GB
```

Note the ratio hidden inside these formulas: the SwiGLU term per token is
`3·3072·2 = 18,432` bytes against the hidden's `1024·2 = 2,048` — the
3×-wide FFN makes a block's *interior* 9× the memory of its boundary. That
asymmetry is exactly why gradient checkpointing pays: you trade 9× of
retained interior for one extra forward pass of compute (§4.1). What the
formula does *not* include is deliberate: attention projections, norm
outputs, and fp32 gradients are not separate rows — §1.6 covers where they
went.

### 1.4 The CE chain: one row, one section

The CE chain is listed as a budget because at this vocab size it is a
first-class activation, not a rounding error: 4.07 GB chunked at ms=8
versus 6.13 GB naive fp32 *for the logits alone* — and eager CE roughly
doubles that (§2.2). It gets its own section (§2) and its own scaling law
(§3); here it only needs to appear in the budget table.

### 1.5 The per-token constants: the whole trade in two numbers

Dividing the linear terms by the token count `S·B` gives constants that
summarize everything §4 argues:

| component | bytes per token | derivation |
|---|---|---|
| block-boundary hidden | 49,152 | `24 layers · 1024 d · 2 B` |
| SwiGLU intermediates | 442,368 | `24 · 3 · 3072 · 2` |
| CE retained bf16 logits | 100,514 | `50,257 V · 2` |
| **no-ckpt total** | **592,034** | ≈ 0.57 MB/token |
| checkpointed total | 149,666 | boundary + CE only |

Two facts fall out immediately. First, no-checkpointing costs 3.96× the
checkpointed per-token rate — that is the price of the MFU bought in §4.
Second, the CE retained term is ~17% of the no-ckpt linear budget and 67%
of the checkpointed one: chunked-CE is not a micro-optimization, it is one
of the two large linear terms, and at small vocabularies it would be
neither.

### 1.6 What the estimator deliberately does not count

`utils/memory.py:estimate_model_memory_gb` is a decision record, not a
simulator. It counts the four budget terms and nothing else. The known
under-counts, with magnitudes:

| un-counted term | magnitude at 16×4096 | why it is safe to omit |
|---|---|---|
| fp32 gradients | `N·4 = 1.28 GB` | same shape as §1.1's fixed floor; allocated once |
| attention-path saves (Q, K, V, attn-out, two norm outputs) | ≈ 0.56 GB/layer → 13.5 GB | GQA shrinks K/V to `4·64=256` dims (0.03 GB each vs Q's 0.125); all absorbed below |
| autocast bf16 cast buffers | transient, per-matmul | freed as each op completes |
| flex BlockMask | kilobytes (`models/mask.py:build_block_causal_block_mask` is O((T/256)²) arrays) | the whole point of flex (§4.4) |
| per-chunk `(B,S)` lse parts | ≈ 3 MB for 13 chunks | O(B·S) side outputs of `training/losses.py:chunked_x0_ce` |
| sdpa dense bool mask (non-flex paths) | `4096² = 16 MB` | only on CPU smoke runs / sdpa fallback |

The absorption mechanism is the **framework overhead reserve**:
`overhead_gb = min(13.7, max(2.0, total_gb · 0.17))`
(`utils/memory.py:estimate_model_memory_gb`) — 13.6 GB on an 80 GB A100.
The derivation of the three constants: `0.17` of total VRAM covers CUDA
context, cuDNN workspaces, allocator fragmentation, and the un-counted rows
above at production scale; the `13.7` cap stops the reserve itself from
dominating the estimate on much larger parts; the `2.0` floor keeps CPU and
toy-fixture estimates honest (§5 uses it). On a 40 GB part the same rule
yields 6.8 GB — the reserve scales with the part, which is what makes the
estimate portable across A100 SKUs.

Adding it up: `peak ≈ 5.12 (fixed) + 592,034·S·B bytes (linear) + 1.0 GB
(CE transient spike) + 13.6 (reserve)`. Every layout decision in §4 is this
equation evaluated at candidate `(S, B)`.

---

## 2. The CE term

### 2.1 Why (B, S, V) is the largest tensor in the model

Every backbone activation is `O(B·S·d)` or `O(B·S·ffn)` with `d=1024`,
`ffn=3072`. The LM head alone produces `(B, S, V)` logits with
`V = 50,257` — 49× `d_model`, 16× `ffn_dim`. At production dims
`16 × 4096 × 50,257` that tensor in fp32 is **12.27 GB** (6.13 GB at
micro_bs 8) — before softmax, before the loss, before backward's saved
tensors. It exists *only* because GPT-2 BPE has 50,257 entries; nothing
else in the network comes within 3× of it.

### 2.2 The naive accounting — twice

The first cost is the logits tensor itself: `B·S·V·4` bytes fp32. The
second is easy to miss: eager `F.cross_entropy` is `log_softmax` +
`nll_loss`, and autograd saves the **log-softmax output** for backward — a
second full-vocab fp32 tensor. Naive eager CE therefore retains ~24.5 GB
at micro_bs 16, ~12.3 at micro_bs 8: roughly *double* the logits row
everyone remembers.

This is also why the fix cannot be "just loop over chunks with eager ops":
a plain per-chunk `F.cross_entropy` loop still saves each chunk's fp32
log-softmax, and all chunks are alive simultaneously — the total is the
same `B·S·V·4`. The retained term only shrinks if backward derives the
softmax from something cheaper than an fp32 log-softmax copy. That is the
job of the custom autograd Function.

### 2.3 The chunked contract

`training/losses.py:chunked_x0_ce` restructures head + CE into vocab chunks
of `vocab_chunk` (runtime `8192·8/micro_bs` — see §3), calling
`training/losses.py:_ChunkTerms.apply` once per chunk; the loop over
`c0 in range(0, V, step)` handles the partial last chunk naturally
(`c1 = min(c0 + step, V)`; chunk-count arithmetic below).

**Forward** (`training/losses.py:_ChunkTerms.forward`): the chunk GEMM
`hidden @ weight.to(hidden.dtype).t()` runs in bf16 under autocast; the
chunk's logits are upcast to fp32 once for a `logsumexp` (the local
denominator) and a `gather` of the target logit; **the bf16 logits are
saved for backward**, alongside `hidden`, `weight`, and the clamped local
targets.

**Backward** (`training/losses.py:_ChunkTerms.backward`): the softmax is
re-derived *from the saved bf16 logits*. Writing the loss for one chunk as
`L = lse − tgt`, with logits `z`:

```
∂lse/∂z  = softmax(z) = p
∂tgt/∂z  = onehot(target)
g_logits = p · grad_lse  +  onehot ⊕ grad_tgt      (scatter_add_ of the signed target grad)
g_hidden = g_logits.to(bf16) @ weight              (chain through the GEMM)
g_weight = g_logitsᵀ @ hidden                      (chain through the GEMM)
```

The subtlety that makes the `scatter_add_` safe: `grad_tgt` arrives signed,
and the caller zeroes it for out-of-chunk targets
(`torch.where(in_chunk, tgt, zeros)` in `training/losses.py:chunked_x0_ce`),
so the clamped scatter only ever adds zeros where the target belongs to
another chunk.

**Memory anatomy.** Per chunk, three terms with three lifetimes: **retained**
— every chunk's bf16 logits stay alive for backward, total `B·S·V·2` (half
of naive's fp32 logits, and no log-softmax copy at all); **transient** — one
fp32 chunk copy `B·S·chunk·4` at a time, freed as the iteration advances,
the peak spike bounded by `chunk`, not `V`; **negligible** — `hidden` and
`weight` are saved by *reference* (the same tensor each chunk), and the
`(B,S)` per-chunk `lse`/`tgt` outputs sum to ~3 MB across 13 chunks.

**Chunk arithmetic.** `ceil(50257/8192) = 7` chunks at micro_bs 8;
`ceil(50257/4096) = 13` at micro_bs 16; the last chunk is 1,105 wide in
both cases (`50257 = 6·8192 + 1105 = 12·4096 + 1105`). The partial chunk is
what `tests/test_loss.py::test_partial_last_chunk` pins.

### 2.4 Naive vs chunked, both micro-batch sizes

| variant | retained | transient | total |
|---|---|---|---|
| naive eager fp32 CE, ms=8 | logits 6.13 + log-softmax 6.13 | (already counted) | **12.27 GB** |
| naive eager fp32 CE, ms=16 | 12.27 + 12.27 | — | **24.55 GB** |
| chunked (chunk 8192), ms=8 | `8·4096·50257·2 = 3.07 GB` bf16 logits | `8·4096·8192·4 = 1.00 GB` | **4.07 GB** |
| chunked (chunk 4096), ms=16 | `16·4096·50257·2 = 6.14 GB` | `16·4096·4096·4 = 1.00 GB` | **7.14 GB** |

At micro_bs 8 the swap buys 8.2 GB; at micro_bs 16 it buys 17.4 GB — 17 GB
is the difference between the no-checkpoint layout fitting (§4) and not.
The swap also buys back **the head-GEMM recompute**: the previous scheme
wrapped the head in `torch.utils.checkpoint`, re-running the head GEMM —
`2·B·S·d·V ≈ 6.7 TFLOP` per micro-step at production dims — on every
backward. Retaining the bf16 chunks (3.07 GB at ms=8) deletes that
recompute: one GEMM per step, forward only. This is the explicit DESIGN
§4.0 trade recorded in the `training/losses.py:chunked_x0_ce` docstring;
the eager reference it must match is `models/diffusion.py:x0_ce_loss`.

### 2.5 The equivalence pins

Chunking changes the summation order of a logsumexp, so the result is not
bit-identical to eager: the per-chunk fp32 logsumexps combine through a
final `torch.logsumexp` over the stacked parts
(`training/losses.py:chunked_x0_ce`), a different reduction order than
eager's single pass over V — same math, different rounding. Hence the pin
is **atol 1e-6** for the loss, plus a gradient-direction pin, plus the
partial last chunk (`tests/test_loss.py::test_chunked_equals_eager`,
`tests/test_loss.py::test_chunked_matches_eager_grad_direction`,
`tests/test_loss.py::test_partial_last_chunk`). The pins exist so any
future change to `_ChunkTerms` — a dtype shortcut, a fused kernel — fails
loudly instead of silently drifting from the eager reference.

---

## 3. The chunk-scaling law

### 3.1 The derivation

`vocab_chunk` is set inversely to micro-batch size:
`self.vocab_chunk = max(1024, config.vocab_chunk * 8 // config.micro_batch_size)`
(`training/pretrain.py:Pretrainer.__init__`), with
`training/pretrain.py:TrainingConfig.vocab_chunk = 8192` as the DESIGN §4.0
constant — pipeline-internal, deliberately **not** a yaml key, so the
layout table and the chunk width cannot drift apart. The reason is that the
CE chain's two terms scale differently:

```
transient = micro_bs · seq · vocab_chunk · 4
          = micro_bs · 4096 · (8192·8/micro_bs) · 4
          = 4096 · 8192 · 32 = 1.00 GB    (independent of micro_bs)

retained  = micro_bs · seq · V · 2          (linear in micro_bs — irreducible)
```

The retained term *must* scale with micro_bs: backward needs every
position's full-vocab logits, and there are `micro_bs·seq` positions. The
transient term is a spike whose *width* is a free parameter — the inverse
law pins its height.

### 3.2 The law, tabulated

| micro_bs | vocab_chunk | transient fp32 (GB) | retained bf16 (GB) | chunks |
|---|---|---|---|---|
| 8 | 8192 | 1.00 | 3.07 | 7 |
| 16 | 4096 | 1.00 | 6.14 | 13 |
| 32 | 2048 | 1.00 | 12.27 | 25 |
| 128 | 1024 (floor binds) | 2.00 | 49.07 | 50 |

The law: **micro_bs trades against vocab_chunk at fixed transient bytes** —
halve one, double the other, and the fp32 spike stays put. The retained
term is the price of the batch; the chunk only sets the shape of the spike.

### 3.3 The floor, and when it binds

The `max(1024, …)` floor exists because below ~1024 the chunk GEMM stops
being efficient: per-chunk overhead (kernel launches, the lse stack, the
partial-chunk branch) begins to dominate the arithmetic. The floor binds at
micro_bs 64 (`8192·8//64 = 1024` exactly) and is *violated* by micro_bs
128 — the table's last row shows the transient spike doubling to 2 GB
because `8192·8//128 = 512` floors up to 1024. That row is also a warning:
at micro_bs 128 the retained term alone is 49 GB, so the floor is not the
binding constraint — the layout is. Any micro-batch past 32 has already
left the no-checkpoint regime (§4.3).

### 3.4 The second hazard: the self-cond pre-pass

Self-conditioning needs `p @ E` — softmax over the full vocab, weighted sum
of embedding rows — the same `(B, S, V)` hazard with no gradients at all
([diffusion-core §8](diffusion-core.md),
[self-conditioning §6](self-conditioning.md)). The eager reference is
`models/selfcond.py:SelfConditioning.embed`; the training path is
`training/losses.py:chunked_p_embed`, reusing the identical
chunk-logsumexp mechanism under `torch.no_grad()`
(`training/pretrain.py:Pretrainer.diffusion_loss`, gated at
`self_cond_p = 0.5` with `self_cond_detach = true`).

Memory under `no_grad`: each chunk's fp32 logits (`B·S·chunk·4` — the same
1.00 GB spike as §3.2) is allocated and freed inside the loop; the only
retained output is `p_embed` itself, `(B, S, d_model)` bf16 = 0.125 GB,
detached. The pre-pass costs zero *retained* bytes; its price is FLOPs (a
second backbone forward on ~50% of steps, the "~12% self-cond overhead" in
the config header), not VRAM. Under grad mode the same function is a
hazard: every chunk's logits retained *with graph* — ~6.14 GB of
chunk-logit graph plus a full second-backbone autograd graph. The docstring
of `training/losses.py:chunked_p_embed` forbids the differentiable use, and
`tests/test_loss.py::test_chunked_p_embed_matches_eager` pins the numeric
path that *is* allowed.

---

## 4. The VRAM-for-MFU trade

### 4.1 What gradient checkpointing actually costs

`grad_checkpoint_every: 3` (`training/pretrain.py:TrainingConfig`) means
every third block is checkpointed: layers `i % 3 == 0` — 8 of 24 — store
only their inputs (`models/transformer.py:DiffusionGemma.backbone` applies
`torch.utils.checkpoint.checkpoint` under the
`models/transformer.py:DiffusionGemma.grad_ckpt_every` runtime knob, set
from config by `training/pretrain.py:Pretrainer.__init__`, or `None` when
checkpointing is off). In backward those 8 layers re-run their entire
forward: **8 extra layer-forwards per step, one third of a full forward
pass of pure recompute** — wasted MFU whenever VRAM can hold the
intermediates instead, which is the entire argument of this section.

### 4.2 The two layouts

The config comment is the decision record
(`configs/pretrain_a100_380m.yaml` training block): the DESIGN §4.0 table
was written for `micro 8 × accum 4 + gradient checkpointing`; the shipped
config is `micro 16 × accum 2` with **no checkpointing** — same effective
32×4096 batch, same optimizer-step count (61,000 ≈ 8.0B tokens), different
memory shape. The estimator models both layouts (the
`grad_checkpoint=True/False` branches in
`utils/memory.py:estimate_model_memory_gb`):

| | 8×4 + ckpt (§4.0 layout) | 16×2, no ckpt (shipped) |
|---|---|---|
| params + AdamW | 5.12 GB | 5.12 GB |
| boundary activations | 1.50 GB | 3.00 GB |
| SwiGLU intermediates | recompute (checkpointed) | **retained: 27.00 GB** |
| CE (chunked) | 4.07 GB | 7.14 GB (chunk 4096) |
| CE transient | 1.00 GB | 1.00 GB |
| subtotal | 10.69 GB | 42.26 GB |
| framework reserve | 13.6 GB | 13.6 GB |
| **estimate** | **24.3 GB / 80** | **55.9 GB / 80 (margin 24.1)** |

The 27 GB SwiGLU retention buys back the every-3-layer recompute — ~33% of
checkpointed-layer forwards, §4.1 — which is pure MFU: the FLOPs budget of
the run is unchanged, the wall clock shrinks. The guard
`utils/memory.py:assert_fits_in_available_gpu` raises before launch when
the estimate leaves less than a 2 GB safety margin, so the trade cannot be
made silently on hardware that cannot hold it.

### 4.3 Why not 32×1 — the crossover law

If retaining intermediates is free MFU, why stop at micro_bs 16? Because
the linear budget runs out. Using §1.5's per-token constant:

```
available_linear = 80 − 2 (guard margin) − 13.6 (reserve)
                   − 5.12 (fixed) − 1.0 (CE transient)
                 = 58.3 GB
max_tokens       = 58.3 · 2^30 / 592,034 B ≈ 105,700 tokens per micro-batch
```

Evaluate the candidates at seq 4096:

| micro_bs | tokens | linear bytes | total estimate | verdict |
|---|---|---|---|---|
| 8 | 32,768 | 18.1 GB | ~37.8 GB | fits, pays 33% recompute |
| **16** | **65,536** | **38.1 GB** | **55.9 GB** | **fits — shipped** |
| 25 | 102,400 | 56.4 GB | ~76.2 GB | fits, but only ~3.8 GB above the guard |
| 32 | 131,072 | 72.3 GB | ~92.0 GB | **guard raises at launch** |

So 16×2 is not taste: it is the largest power-of-two micro-batch at
seq 4096 that fits without checkpointing and keeps a two-digit GB margin.
32×1 would have to re-enable checkpointing — which lands at
`5.12 + 6.00 (boundaries) + 12.27 (CE retained) + 1.0 + 13.6 ≈ 38 GB`,
fitting easily but paying the recompute again. The crossover law makes the
checkpointing decision mechanical rather than stylistic: compute
`tokens·592,034` and compare against `available_linear`.

### 4.4 Where the attention memory went (flex)

`attn_impl: "flex"` is the default kernel (`configs/pretrain_a100_380m.yaml`
model block), and it is a memory decision as much as a speed one. Eager
attention materializes `(B, H, T, T)` scores: at production dims
`16·16·4096²·4 = 17.2 GB` **per layer** — plus the softmax copy backward
saves. Fused sdpa/FlexAttention never materialize the score matrix; they
tile the computation so peak score memory is O(block), not O(T²). The
block-sparse mask has the same property in miniature:
`models/mask.py:build_block_causal_block_mask` produces a `BlockMask`
whose arrays are O((T/canvas_len)²) — kilobytes for `T=4096, canvas=256` —
while the sdpa path's dense bool mask
(`models/mask.py:build_block_causal_mask`) is `4096² = 16 MB` per forward.
Both sit inside the §1.6 reserve, but flex also deletes the churn of
allocating and freeing a 16 MB tensor every forward.

Two footnotes. GQA (16 query heads over 4 KV groups, `head_dim=64`) is why
K/V are 256-wide — the difference between the honest 343.5M count and the
early "~380M" that counted full-MHA K/V (DIFFUSION.md §6, SDD Ruling 13) —
and it trims §1.6's attention-path saves by the same factor. And flex has
no CPU backward kernel, so smoke runs fall back to `attn_impl='sdpa'`
(`training/pretrain.py:Pretrainer.__init__`) — the estimator is unchanged,
because attention memory sits inside the overhead reserve on both paths.

### 4.5 When checkpointing comes back

The no-checkpoint layout holds exactly while §4.3's inequality holds. Three
independent triggers flip it: **longer sequences** (seq 8192 at micro 16
doubles every linear term: ~92 GB no-ckpt, guard raises — but ~38 GB with
`grad_checkpoint: true`, because the CE retained term survives
checkpointing untouched, the custom autograd Function's saved bf16 logits
sitting outside checkpoint regions, which is why chunked-CE stays mandatory
even in the checkpointed layout); **bigger micro-batches** (anything past
~25×4096, §4.3's table); **a smaller part** (on a 40 GB A100 the reserve is
6.8 GB and no-ckpt's 55.9 GB cannot fit at all, while 8×4 + ckpt lands at
10.7 + 6.8 ≈ 17.5 GB).

The mechanics are one flag each way: `grad_checkpoint: true` sets
`models/transformer.py:DiffusionGemma.grad_ckpt_every` from
`grad_checkpoint_every` and switches the estimator's branch
(`utils/memory.py:estimate_model_memory_gb`); `--no-checkpoint`
(`training/pretrain.py:main`) forces the no-ckpt branch. Estimator and
runtime knob are wired from the same config keys, so the estimate can never
describe a layout the trainer is not running — that symmetry is what makes
the pre-launch guard trustworthy.

---

## 5. Worked example: the full estimate, both layouts

Toy scale first, so every term is checkable by hand: the tiny fixture model
with `N = 101,888` params, `seq 128, micro 2, d_model 64, layers 2,
V 256, chunk 128`, overhead 2.0 GB (CPU floor):

```
params fp32         101,888 · 4   =   407,552 B   0.0004 GB
AdamW m+v+master    101,888 · 12  = 1,222,656 B   0.0011 GB
boundary acts   2·128·2·64·2       =    65,536 B   0.0001 GB   (L·S·B·d·2)
CE retained bf16    2·128·256·2   =   131,072 B   0.0001 GB
CE transient fp32   2·128·128·4   =   131,072 B   0.0001 GB
overhead (CPU)                                    2.0     GB
                    ─────────────
estimate            1,957,888 B + 2.0            ≈ 2.0018 GB
```

Take-away: at toy scale everything except the overhead reserve vanishes —
which is exactly why the estimator's *shape* (param/optim/act/CE terms) is
tested for monotonicity and for the CE term's dominance at production
scale, not for absolute values
(`tests/test_utils.py::test_memory_estimator_monotone_in_batch`,
`tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate`). The
latter test builds the production model on meta device and asserts naive
fp32 CE (`vocab_chunk=None`) exceeds the 8192-chunked estimate by more than
2 GB — encoding §2.4's table as an invariant.

Production recap — every term is computed in §1–§4 and tabulated in §4.2;
in one line each: fixed 5.12 GB, linear 10.69 GB → **24.3 GB total**
(8×4+ckpt, 55.7 GB of headroom) versus fixed 5.12, linear 42.26 GB →
**55.9 GB total** (16×2 no-ckpt, 24.1 GB of headroom). The shipped config
spends 31.6 GB of the checkpointed layout's headroom on SwiGLU retention
and a doubled CE retained term, and buys back a third of a forward pass per
step (§4.1).

---

## 6. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| full-vocab fp32 eager CE at micro 16 | +17.4 GB vs chunked (24.55 vs 7.14): 55.9 → 73.3, margin gone before any seq growth | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |
| `vocab_chunk` scaled *with* micro_bs | transient fp32 spike scales with micro_bs; the constant-spike law broken | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |
| chunked logits retained in fp32 | retained term doubles: 4.07 → 7.14 GB (ms=8) | (no pin; VRAM guard at launch) |
| `chunked_p_embed` run under grad mode | every chunk's logits retained *with graph* — silent multi-GB blow-up | `tests/test_loss.py::test_chunked_p_embed_matches_eager` (correctness; memory hazard documented) |
| drop the 13.7 GB overhead cap on an 80 GB part | estimate under-counts allocator fragmentation and the §1.6 un-counted rows | `utils/memory.py:estimate_model_memory_gb` overhead rule |
| estimate with `grad_checkpoint=False` on the 8×4 layout | under-counts SwiGLU retention (27 GB surprise) | `utils/memory.py:estimate_model_memory_gb` docstring contract |
| micro_bs 32 at seq 4096, no ckpt | ~92 GB estimate — `utils/memory.py:assert_fits_in_available_gpu` raises at launch | (guard; §4.3 table) |
| seq 8192 at micro 16, no ckpt | same ~92 GB failure; ckpt returns it to ~38 GB | (guard; §4.5) |
| `attn_impl: "eager"` in training | `(B,H,T,T)` scores materialized: 17.2 GB per layer — instant OOM at production dims | (config; flex is the default kernel) |
| `vocab_chunk` below the 1024 floor | per-chunk overhead dominates the chunk GEMM; spike grows once the floor binds (ms=128 → 2 GB) | `training/pretrain.py:Pretrainer.__init__` floor |
| weights cast to bf16 masters | 0.64 GB saved; AdamW updates below bf16 ulp silently round away | (no pin; fp32-master contract, §1.1) |

---

## 7. Glossary

| symbol | meaning | code |
|---|---|---|
| `N` | parameter count, 343,516,160 | `training/pretrain.py:count_parameters` |
| `K` | vocab chunk width | runtime `vocab_chunk` (`8192·8/micro_bs`, floor 1024) |
| `_ChunkTerms` | per-chunk autograd Function: retained bf16 logits + fp32 transient | `training/losses.py:_ChunkTerms` |
| `act_bytes` | block-boundary (+ SwiGLU when uncheckpointed) activations | `utils/memory.py:estimate_model_memory_gb` |
| `overhead_gb` | allocator/framework reserve, `min(13.7, max(2.0, 0.17·total))` | `utils/memory.py:estimate_model_memory_gb` |
| `assert_fits` | pre-launch VRAM guard, 2 GB margin | `utils/memory.py:assert_fits_in_available_gpu` |
| per-token constant | 592,034 B/token no-ckpt; 149,666 checkpointed | §1.5 |
| crossover tokens | ~105,700 tokens/micro-batch — the no-ckpt ceiling at seq 4096 | §4.3 |
| `grad_ckpt_every` | runtime checkpoint stride; `None` when off | `models/transformer.py:DiffusionGemma.grad_ckpt_every` |
| `lse` | per-chunk logsumexp; combines into the global denominator | `training/losses.py:chunked_x0_ce` |
| `p_embed` | `p @ E` softmax-weighted embedding sum, chunked under `no_grad` | `training/losses.py:chunked_p_embed` |
| BlockMask | block-sparse flex mask, kilobytes vs the 16 MB dense bool mask | `models/mask.py:build_block_causal_block_mask` |

---

## 8. Interview Q&A

**Q: Walk me through the 80 GB budget.**
A: Fixed 5.12 GB (fp32 weights 1.28 + AdamW 3.84); linear: boundary 3.0 +
SwiGLU 27.0 + CE retained 6.14 + CE transient 1.0; framework reserve
13.6 GB — ~55.9 GB total (16×2 no-ckpt) or ~24.3 GB (8×4+ckpt) of 80
(`utils/memory.py:estimate_model_memory_gb`).

**Q: Why is the CE term special?**
A: The LM head materializes `(B, S, V)` logits — and eager CE saves a
second full-vocab fp32 log-softmax for backward, so naive is ~12.3 GB at
micro 8 and ~24.5 at micro 16. `training/losses.py:chunked_x0_ce` keeps
every chunk's bf16 logits for backward (irreducible) plus one transient
fp32 chunk, pinned at atol 1e-6
(`tests/test_loss.py::test_chunked_equals_eager`).

**Q: Why keep the bf16 logits instead of recomputing the head in backward?**
A: Recompute costs a full head GEMM — ~6.7 TFLOP per micro-step at
production dims — on every backward; retaining bf16 chunks is half the
bytes of fp32 and no log-softmax copy at all, trading ~3 GB retained for
deleting that GEMM — the explicit DESIGN §4.0 trade
(`training/losses.py:chunked_x0_ce` docstring).

**Q: What is the chunk-scaling law?**
A: Transient fp32 = micro_bs · seq · vocab_chunk · 4; setting
`vocab_chunk = 8192·8/micro_bs` pins the spike at 1 GB regardless of
micro_bs, while the retained term (micro_bs · seq · V · 2) grows linearly
with the only knob that *must* grow
(`training/pretrain.py:Pretrainer.__init__`).

**Q: Why did the config move from 8×4+checkpointing to 16×2 without?**
A: At 80 GB the no-ckpt layout fits (55.9 of 80) and deletes the
every-3-layer recompute — 8 of 24 layers re-forwarding each backward, a
third of a forward pass — so MFU improves for free; the §4.0 table
describes the older, more conservative layout
(`configs/pretrain_a100_380m.yaml` training block comment).

**Q: Why not micro_bs 32, then?**
A: The crossover law: 592,034 B/token linear, ~58.3 GB available after
fixed costs, reserve, spike and guard margin → ~105,700 tokens per
micro-batch. 16×4096 = 65,536 fits; 32×4096 = 131,072 estimates ~92 GB and
the pre-launch guard raises
(`utils/memory.py:assert_fits_in_available_gpu`).

**Q: What does the estimator not count, and why is that safe?**
A: fp32 grads (1.28 GB), attention-path saves (~13.5 GB at 24 layers,
GQA-shrunk K/V included), cast buffers, the flex BlockMask (kilobytes).
The `min(13.7, max(2.0, 0.17·total))` reserve absorbs them, and the 2 GB
guard margin plus the 24 GB gap to 80 GB is the real safety net
(`utils/memory.py:estimate_model_memory_gb`).

**Q: When does gradient checkpointing have to come back?**
A: Any time the linear budget overflows: seq 8192 at micro 16 (~92 GB
no-ckpt vs ~38 GB with), micro_bs past ~25 at seq 4096, or a 40 GB part
where no-ckpt cannot fit at all. One config flag; estimator and
`models/transformer.py:DiffusionGemma.grad_ckpt_every` follow the same key.

**Q: Why is flex attention part of the memory story?**
A: Fused kernels never materialize `(B, H, T, T)` scores — eager would be
17.2 GB per layer at production dims — and the flex BlockMask is kilobytes
where the sdpa dense bool mask is 16 MB per forward
(`models/mask.py:build_block_causal_block_mask`). Flex is the default
kernel in the production config.

**Q: How is the estimate validated?**
A: Monotonicity in batch
(`tests/test_utils.py::test_memory_estimator_monotone_in_batch`), the
chunked-CE term bounding the total at production dims
(`tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate`), the
chunked-CE equivalence suite (`tests/test_loss.py`), and the
`utils/memory.py:assert_fits_in_available_gpu` pre-launch guard with a
2 GB margin.