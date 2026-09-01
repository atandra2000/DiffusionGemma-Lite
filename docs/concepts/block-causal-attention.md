# Concept: block-causal attention — the load-bearing primitive

> **Canonical** for the mask derivation, the three attention paths, GQA +
> RoPE at production dims, and the KV-chaining contract. `DIFFUSION.md` §2
> stays authoritative for rulings; this page derives them.

**Depends on:** [foundations](foundations.md) §4–§5 · **Read next:**
[sampler](sampler.md) · [self-conditioning](self-conditioning.md)

---

## Table of Contents

1. [The mask, from first principles](#1-the-mask-from-first-principles)
2. [GQA at production dims](#2-gqa-at-production-dims)
3. [RoPE on q and untiled k](#3-rope-on-q-and-untiled-k)
4. [The three attention paths](#4-the-three-attention-paths)
5. [FlexAttention: why canvas-sized blocks are free](#5-flexattention-why-canvas-sized-blocks-are-free)
6. [KV chaining (the decode contract)](#6-kv-chaining-the-decode-contract)
7. [Shapes at production dims](#7-shapes-at-production-dims)
8. [What breaks if you change this](#8-what-breaks-if-you-change-this)
9. [Glossary](#9-glossary)
10. [Interview Q&A](#10-interview-qa)

---

## 1. The mask, from first principles

### 1.1 What block-AR attention must enforce

`models/mask.py:build_block_causal_mask(seq_len, canvas_len)` returns a bool
`(1, 1, T, T)` mask, `True` = attend. Two row rules define it; for query
position `i` and key position `k`:

```
allow[i, k] = (k < floor(i/L) · L)  |  (floor(k/L) == floor(i/L))
             strictly earlier canvases   own canvas, all of it
```

- **Across canvases, causal.** Canvas `b` attends to canvases `0..b−1` and
  nothing later — this is what makes the trained model a valid left-to-right
  conditional LM over canvases.
- **Within a canvas, bidirectional.** The 256 positions denoise *together*;
  this is the parallelism the project buys.

The implementation is two broadcast comparisons
(`models/mask.py:build_block_causal_mask`):

```python
idx = torch.arange(seq_len)
q_block = (idx // canvas_len).unsqueeze(1)      # (T, 1)
k_pos   = idx.unsqueeze(0)                      # (1, T)
allow   = (k_pos < q_block * canvas_len) | ((k_pos // canvas_len) == q_block)
```

It asserts `seq_len % canvas_len == 0` (the sampler's prompt prefill relaxes
this with an inline partial-canvas mask of identical row semantics,
`inference/generate.py:_prefix_mask`), and is cached via
`functools.lru_cache` — it was rebuilt on every forward before; treat the
cached return as read-only.

**Worked example (seq 8, canvas 4)** — computed from the rule and verified
against `build_block_causal_mask`:

```
        k: 0 1 2 3 4 5 6 7
q0 (c0) |  1 1 1 1 . . . .
q1 (c0) |  1 1 1 1 . . . .
q2 (c0) |  1 1 1 1 . . . .
q3 (c0) |  1 1 1 1 . . . .
q4 (c1) |  1 1 1 1 1 1 1 1
q5 (c1) |  1 1 1 1 1 1 1 1
q6 (c1) |  1 1 1 1 1 1 1 1
q7 (c1) |  1 1 1 1 1 1 1 1
```

Read a row as "what this position may see": query 2 (canvas 0) sees canvas 0
*including position 3* — within-canvas lookahead is the point, it is what
makes the 256 positions denoise together. Query 5 (canvas 1) sees everything:
canvas 0 is finalized, canvas 1 is its own. Verbatim semantics are pinned by
`tests/test_mask.py::test_block_causal_mask_matches_manual`,
`test_mask_causal_across_canvases`, `test_mask_bidirectional_within_canvas`,
`test_mask_shape_and_dtype`.

### 1.2 Why the factorization is legal

The mask makes one transformer play two roles: a bidirectional **denoiser**
within a canvas and a causal **conditional LM** across canvases. The joint

```
p(c0, ..., c15) = p(c0) · p(c1 | c0) · ... · p(c15 | c0..c14)
```

is exactly as valid as token-AR's factorization; each factor is a 256-token
joint that the bidirectional-within-canvas attention models internally.
Lookahead inside a canvas does not violate conditioning because the *canvas*,
not the token, is the unit of generation ([foundations §4](foundations.md)).

### 1.3 The decode-time mask

While a canvas is in flight it is the **last block** of the window and
everything before it is finalized and frozen.
`models/mask.py:build_canvas_decode_mask(prefix_len, canvas_len)` returns an
**all-ones** `(1, 1, L, prefix_len + L)` view: the finalized prefix is fully
visible, the in-flight canvas sees itself fully. The canvas's *content* is
what's still noisy — its *visibility* is not. This is the all-ones re-encode
contract (SDD Ruling 15), superseding DESIGN §2.5's earlier "zero-mask"
wording (the binding sentence there is "canvas = last block in the mask").

---

## 2. GQA at production dims

Multi-head attention gives every head its own key/value: 16 KV heads →
96 KB/token cached (§1.2 arithmetic in [foundations §1.2](foundations.md)).
GQA shares each KV head across a *group* of query heads: 16 query heads, 4 KV
heads (`models/attention.py:DenoiseAttention`) → 4 query heads share each KV
head; cached tokens shrink 4× to 24 KB/token with the fast kernels consuming
untiled KV (`enable_gqa=True`) — no runtime tiling cost at all. The eager
ground-truth twin *does* expand by hand
(`models/block.py:DenoiseBlock._attention` repeat_interleave) — that
expansion-vs-native difference is exactly what its equivalence tests absorb.

| tensor | shape (B=1) | cached? |
|---|---|---|
| `q` | (16, T, 64) | no |
| `k`, `v` | (4, T, 64) | **yes — the KV cache** |
| out | (16, T, 64) → reshape → out_proj → (1, T, 1024) | no |

Layout pinned by `tests/test_attention.py::test_gqa_kv_heads`.

## 3. RoPE on q and untiled k

> Primer in [foundations §5.4](foundations.md); here: the production-specific
> facts only.

- Canonical GPT-NeoX/LLaMA **rotate-half**; pairs are `(i, i + half)`;
  `rope_theta = 500,000`; `head_dim = 64` → 32 frequency planes
  (`models/attention.py:apply_rope` is the reference implementation).
- Production forwards use `models/attention.py:DenoiseAttention`'s cached fp32
  cos/sin tables sized to `max_seq_len` (`DenoiseAttention._roped_qkv`
  indexes them per forward) — the trig was previously recomputed in every one
  of the 24 layers on every forward.
- RoPE applies to `q` and the **untiled** `k` (4 heads) *before* the cache
  append: cached keys are already roped
  (`models/attention.py:DenoiseAttention._roped_qkv`), which is why the
  sampler's decode chunks never re-rope the prefix.
- Invariants (pinned by
  `tests/test_attention.py::test_rope_preserves_norms_and_relative_position`):
  per-position norms preserved; `q_m · k_n = f(m − n)`; a uniform +c shift of
  all positions leaves attention unchanged (SDD Ruling 12 — the plan's
  original shifted-positions test could only pass against a buggy
  interleaved-frequency RoPE, so the test asserts relative spacing).

## 4. The three attention paths

The same masked attention exists three times, deliberately (`models/mask.py`):

| path | symbol | mechanism | role |
|---|---|---|---|
| production (A100 config) | `models/mask.py:flex_block_causal_attention` | FlexAttention fused block-sparse kernel over a cached canvas-sized `BlockMask` | training + decode on CUDA |
| portable fallback | `models/mask.py:block_causal_sdpa_attention` | `F.scaled_dot_product_attention` with the bool mask, `enable_gqa=True` | CPU tests, non-flex configs |
| ground truth | `models/mask.py:eager_block_causal_attention` | explicit O(T²) scores → masked_fill → softmax → @V | the test oracle |

```python
# models/mask.py:eager_block_causal_attention — the reference twin
scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
scores = scores.masked_fill(~mask, float("-inf"))
return scores.softmax(dim=-1) @ v
```

All three are proven equal by weight-transplant tests
(`tests/test_attention.py::test_attention_matches_eager`,
`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
`tests/test_models.py::test_flex_attn_impl_matches_sdpa`). The eager twin is
deliberate duplication — never consolidate it (AGENTS.md §2). It is also the
only path that expands KV heads by hand (`repeat_interleave`); the two fast
paths consume untiled KV (`enable_gqa`).

## 5. FlexAttention: why canvas-sized blocks are free

`models/mask.py:build_block_causal_block_mask` builds the `BlockMask` via a
`mask_mod` predicate; because 256 = 2 × 128, canvas boundaries align with the
kernel's 128-token blocks — every block pair is **all-or-nothing** (fully
allowed within a canvas or for earlier canvases; fully masked for later
ones). The fused kernel therefore skips masked pairs with zero per-element
mask overhead: full-density speed from a block-sparse mask. The BlockMask is
`functools.lru_cache`d because `create_block_mask` tracing is far too slow to
run per forward. The builder handles a partial first canvas (the sampler's
prefill) natively since the rule is per-element.

Under flex the sampler's decode chunks pass `mask=None` — **no block mask ==
full attention == the all-ones decode mask**, without building the tensor
(`inference/generate.py:BlockDiffusionSampler._decode_mask`). A `past_kv`
forward on non-flex paths must pass an explicit mask
(`models/transformer.py:DiffusionGemma.backbone` asserts this).

## 6. KV chaining (the decode contract)

The sampler's cache grows **once per finalized canvas** (256 tokens), not
once per token:

```
denoise_canvas(kv, prefix_len)   # T_eval forwards on the in-flight canvas
encode_canvas(kv, prefix_len)    # 1 forward; per-layer cat(past_kv, new_kv)
```

The contract (`models/mask.py:build_canvas_decode_mask` + the sampler's
`encode_canvas`): the chained KV must equal what a fresh single-shot
block-causal forward over `prefix + canvas` would write — enforced
**bit-exactly in fp64** by
`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`. Any
attention/KV plumbing change re-runs this test. At equal total length the
cache shape is identical to AR's (every token is eventually encoded); the win
is fewer forwards, not a smaller cache — cache growth cadence is once per
canvas vs once per token ([sampler](sampler.md)).

## 7. Shapes at production dims

Training forward (micro_bs 8, seq 4,096 = 16 canvases × 256):

| tensor | shape | note |
|---|---|---|
| bool mask | (1, 1, 4096, 4096) | cached (`build_block_causal_mask`) |
| BlockMask | 32×32 grid of 128² blocks | canvas-aligned, cached |
| `q` | (8, 16, 4096, 64) | 16 query heads |
| `k`, `v` | (8, 4, 4096, 64) | **the KV cache content** |
| eager scores | (8, 16, 4096, 4096) | ground-truth path only — 4.3 GB fp32 |

Decode chunk (in-flight canvas over a `P`-token prefix):

| tensor | shape |
|---|---|
| canvas `xt` | (B, 256) |
| `q` | (B, 16, 256, 64) |
| `k`, `v` after append | (B, 4, prefix+256, 64) |
| decode mask | all-ones (1, 1, 256, prefix+256), or `None` under flex |

KV bytes/token: `2 · 24 layers · 4 kv_heads · 64 · 2 B = 24,576 ≈ 24 KB`
(§2); 100.7 MB at 4,096 context.

## 8. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| delete the eager attention twin | no implementation-independent oracle | weight-transplant tests lose their reference |
| tile KV heads in the fast paths | 4× cache traffic; contradicts `enable_gqa` | `tests/test_attention.py::test_gqa_kv_heads` |
| zero-mask the in-flight canvas against itself | contradicts the all-ones decode contract (Ruling 15) | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (fp64) |
| allow cross-canvas bidirectionality | left-to-right factorization invalid; sampler conditioning leaks future canvases | `tests/test_mask.py::test_mask_causal_across_canvases` |
| recompute RoPE trig per layer per forward | correctness unchanged; ~24× redundant trig per forward (the reason the fp32 tables exist) | (performance, not a pin) |
| bf16 cos/sin tables | trig drift compounds across 24 layers | `tests/test_attention.py::test_rope_preserves_norms_and_relative_position` |
| non-canvas-aligned flex blocks | block pairs straddle canvases; mask no longer all-or-nothing; sparsity overhead returns | `tests/test_models.py::test_flex_attn_impl_matches_sdpa` |

## 9. Glossary

| symbol | meaning | code |
|---|---|---|
| `L` | canvas length (256) | config `canvas_len` |
| `allow[i,k]` | mask entry: query i may attend key k | `models/mask.py:build_block_causal_mask` |
| BlockMask | flex's block-sparse mask (canvas-aligned) | `models/mask.py:build_block_causal_block_mask` |
| GQA | grouped-query attention (16Q/4KV here) | `models/attention.py:DenoiseAttention` |
| `past_kv` | per-layer (k, v), k already roped | `models/attention.py:DenoiseAttention._roped_qkv` |
| `_prefix_mask` | partial-first-canvas prefill mask | `inference/generate.py:_prefix_mask` |

## 10. Interview Q&A

**Q: Why is the mask called the load-bearing primitive?**
A: One dense transformer plays two roles simultaneously — bidirectional
denoiser within a canvas, causal conditional LM across canvases — and
`models/mask.py:build_block_causal_mask` is the contract that makes both
legal at once. The sampler, KV chaining, and the fp64 equivalence test all
lean on it.

**Q: Why is the decode mask all-ones?**
A: The in-flight canvas is the last block and the prefix is frozen;
`models/mask.py:build_canvas_decode_mask` masks nothing because the canvas's
*content* is noisy — its *visibility* is complete (SDD Ruling 15). Masking
the canvas against itself would contradict the block-AR factorization.

**Q: Why does FlexAttention run at full density here?**
A: Canvas-sized (256 = 2×128) blocks make every (query, key) block pair
all-or-nothing, so the fused kernel skips fully-masked pairs and pays no
per-element mask overhead — `models/mask.py:build_block_causal_block_mask`,
cached because `create_block_mask` tracing is far too slow per forward.

**Q: Why keep the eager attention twin at all?**
A: It is the O(T²) implementation-independent oracle: weight-transplant tests
prove SDPA and Flex agree with it
(`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
`test_flex_attn_impl_matches_sdpa`). Deleting it deletes the ground truth
(AGENTS.md §2).

**Q: What does the KV-chaining test pin, in one sentence?**
A: That the sampler's KV-chained decode equals a fresh single-shot
block-causal forward — bit-exact in fp64
(`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`); any
attention/KV plumbing change re-runs it.

**Q: What does GQA buy, concretely?**
A: 24 KB/token cached vs 96 KB for MHA at the same width — 100.7 MB vs
402.7 MB at 4,096 context — with untiled KV consumption (`enable_gqa=True`)
so there is no runtime tiling cost either
(`models/attention.py:DenoiseAttention`).