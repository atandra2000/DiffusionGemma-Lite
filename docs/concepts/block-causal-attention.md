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

To keep the arithmetic checkable, this chapter uses two dim sets side by
side. **Toy dims** (illustrative only, not a repo config): `canvas_len L = 4`,
`seq_len T = 16` (4 canvases), `d_model = 128`, 4 query / 2 KV heads,
`head_dim = 32`, `n_layers = 2`. **Production dims** (the A100 config):
`L = 256`, `T = 4096 = 16 canvases`, `d_model = 1024`, 16 query / 4 KV heads,
`head_dim = 64`, `n_layers = 24` (`configs/pretrain_a100_380m.yaml`,
`models/transformer.py:DiffusionGemmaConfig`).

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

Read the shapes: `q_block` is `(T, 1)` (each query row carries its canvas
index), `k_pos` is `(1, T)`; broadcasting yields the full `(T, T)` table in
one vectorized op. The first disjunct — "key is strictly before my canvas
starts" — is equivalent to `k_pos // canvas_len < q_block` (since
`q_block * canvas_len` is the first position of the query's canvas); the
second is "key lives in my own canvas". Their union is the
block-triangular-with-full-diagonal-blocks pattern.

Why bool and not an additive float mask: the bool form is what
`F.scaled_dot_product_attention(attn_mask=...)` accepts portably, what
`masked_fill(~mask, -inf)` consumes in the eager twin, and what broadcasts
from `(1, 1, T, T)` over batch and heads without allocation. The additive
form (`0` / `−inf` floats) would buy nothing and doubles mask bytes.

It asserts `seq_len % canvas_len == 0` (the sampler's prompt prefill relaxes
this with an inline partial-canvas mask of identical row semantics,
`inference/generate.py:_prefix_mask`), and is cached via
`functools.lru_cache` — it was rebuilt on every forward before; treat the
cached return as read-only. At production dims the bool mask is
`4096² = 16.8 MB`; rebuilding it per forward would pay an allocation plus two
full-length comparisons on top of the `torch.arange`. The cache is keyed on
`(seq_len, canvas_len, device)`, so a training run pays the build once.

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
canvas 0 is finalized, canvas 1 is its own.

Scale the same rule up one canvas and the causal staircase appears —
**seq 16, canvas 4** (rows = queries grouped by canvas, columns = keys):

```
         k: 0..3   4..7   8..11  12..15
canvas 0 (q0..q3)   | 1111 | .... | .... | .... |
canvas 1 (q4..q7)   | 1111 | 1111 | .... | .... |
canvas 2 (q8..q11)  | 1111 | 1111 | 1111 | .... |
canvas 3 (q12..q15) | 1111 | 1111 | 1111 | 1111 |
```

Every row of canvas `b` is identical: earlier canvases all-visible, own
canvas all-visible, later canvases invisible. That row-constancy within a
canvas is precisely what the FlexAttention section exploits (§5): the mask
has structure at *block* granularity, not just element granularity.

Verbatim semantics are pinned by
`tests/test_mask.py::test_block_causal_mask_matches_manual`,
`test_mask_causal_across_canvases`, `test_mask_bidirectional_within_canvas`,
`test_mask_shape_and_dtype`.

### 1.2 Why the factorization is legal

The chain rule of probability holds for *any* ordering or segmentation of a
joint distribution, not just token-left-to-right. Segment the sequence into
canvases `c_0..c_{n−1}`; then

```
p(c_0, ..., c_{n-1}) = p(c_0) · p(c_1 | c_0) · ... · p(c_{n-1} | c_0..c_{n-2})
```

is an *identity* — no approximation is introduced by conditioning on whole
canvases instead of single tokens. Token-AR is the `L = 1` special case,
full-sequence modeling is `L = T`; every `L` between is an exact
factorization, and what changes is the *granularity of generation*, not the
validity of the probability model.

The mask makes one transformer play two roles simultaneously:

- **Across canvases** it is a causal conditional LM: canvas `b`'s tokens
  attend only to finalized context `c_0..c_{b−1}`, so the network computes
  one factor `p(c_b | c_{<b})` per canvas position.
- **Within a canvas** it is a bidirectional denoiser: the 256 positions all
  condition on each other, which is what lets a *single* forward pass refine
  all of them jointly — the model's estimate of token 10 informs its estimate
  of token 200 and vice versa, every step.

Lookahead inside a canvas does not violate conditioning because the *canvas*,
not the token, is the unit of generation ([foundations §4](foundations.md)):
the factor `p(c_b | c_{<b})` is a 256-token *joint*, and modeling a joint
bidirectionally is exactly what bidirectional attention is for. The trained
quantity is the x0 posterior per position (DIFFUSION.md §1.2); nothing in the
commit-renoise loop ever conditions a finalized canvas on a later one.

### 1.3 Why block-causal beats the alternatives

Three attention topologies were available. The choice is not aesthetic — it
determines the decode economics of the whole project:

| property | token-causal (GPT-style) | fully bidirectional (BERT/MDLM-style) | block-causal (this repo) |
|---|---|---|---|
| generation unit | 1 token | the whole sequence, refined in place | 1 canvas (256 tokens) |
| tokens refined per forward | 1 | all of them, but the whole window | 256 |
| KV cache across steps | yes, per token | **no** — every refinement step changes every token's content, so nothing is frozen | yes, per canvas — finalized canvases are frozen forever |
| conditioning for generation | previous tokens | none (must re-noise the whole sequence every step) | previous canvases |
| forwards to emit `n` tokens | `n` | `T_eval · ceil(n/L)`-ish, no cache reuse | `1 + ceil(n/L)·(T_eval+1)` |
| within-unit parallelism | none | total | full (bidirectional inside the canvas) |

- **vs token-causal.** Token-AR is the `L = 1` corner of the same design:
  causal everywhere, no bidirectional block. It wins on decode *latency per
  token* but pays one forward per token. Block-AR keeps the
  causal-across-units structure — precisely what makes a KV cache legal —
  but amortizes each forward over 256 tokens: at `T_eval = 32` the fixed
  schedule spends 33 forwards per canvas, `256/33 ≈ 7.8` tokens/forward
  (DIFFUSION.md §7), vs AR's 256 forwards for the same tokens. The price is
  honest: a more complex mask, a commit/renoise sampler, and a harder
  training objective (reconstruct `x0` from corrupted `xt`, not predict one
  next token).
- **vs fully bidirectional.** A whole-sequence bidirectional diffusion model
  (the MDLM/D3PM default) denoises *everything at once*: no position is ever
  final, so no KV entry is ever stable enough to cache, and there is no
  left-to-right factorization to condition on. Block-causal takes
  bidirectionality where it pays (inside the canvas being denoised — the
  parallel-refinement bet) and re-imposes causality across canvases so that
  everything left of the in-flight canvas is *frozen*: cacheable, never
  recomputed, a valid conditioning prefix. The decode mask of §1.4 is
  all-ones *because* of this split — content is noisy, context is not.
- **Why not a smaller or larger canvas.** `L` trades parallel width against
  within-canvas staleness: larger `L` refines more tokens per forward but each
  position must commit against more co-denoising peers (and the flex block
  alignment of §5 wants `L` a multiple of 128). `L = 256 = 2 × 128` is the
  chosen operating point; `L = 1` degenerates to AR, `L = T` degenerates to
  the cacheless bidirectional model.

### 1.4 The decode-time mask

While a canvas is in flight it is the **last block** of the window and
everything before it is finalized and frozen.
`models/mask.py:build_canvas_decode_mask(prefix_len, canvas_len)` returns an
**all-ones** `(1, 1, L, prefix_len + L)` view: the finalized prefix is fully
visible, the in-flight canvas sees itself fully. The canvas's *content* is
what's still noisy — its *visibility* is not. This is the all-ones re-encode
contract (SDD Ruling 15), superseding DESIGN §2.5's earlier "zero-mask"
wording (the binding sentence there is "canvas = last block in the mask").

Worked example (`prefix_len = 8`, `canvas_len = 4` — the sampler mid-canvas-2):

```
          k: 0 1 2 3 4 5 6 7 | 8 9 10 11
q8  (canvas 2, pos 0)  | 1 1 1 1 1 1 1 1 | 1  1  1  1
q9                     | 1 1 1 1 1 1 1 1 | 1  1  1  1
q10                    | 1 1 1 1 1 1 1 1 | 1  1  1  1
q11 (canvas 2, pos 3)  | 1 1 1 1 1 1 1 1 | 1  1  1  1
```

Every entry is 1. The mask exists as an object only on non-flex paths, where
SDPA wants *some* `attn_mask` of shape `(1, 1, L, prefix+L)` to broadcast;
under flex the sampler passes `mask=None` and "no block mask" *is* full
attention (§5). Note the rectangular shape — the chunk forwards only the 256
in-flight tokens, while their keys span the whole window via `past_kv` (§6).

This is also the row the in-flight canvas would read as the last canvas of a
full single-shot block-causal sequence — "decode view = last block row of the
training mask" — which is what makes the KV-chaining test of §6 provable, and
why zero-masking the canvas against itself (the pre-Ruling-15 wording) would
have been wrong: it would deny the canvas the bidirectional view its own
training mask grants it.

Pinned by `tests/test_mask.py::test_canvas_decode_mask_all_visible`.

### 1.5 Mask shapes and caching, toy vs production

| tensor | toy dims | production dims | built by |
|---|---|---|---|
| training/prefill bool mask | `(1, 1, 16, 16)` = 256 B | `(1, 1, 4096, 4096)` = 16.8 MB | `models/mask.py:build_block_causal_mask` (lru_cached) |
| decode bool mask | `(1, 1, 4, P+4)` | `(1, 1, 256, P+256)`, all-ones | `models/mask.py:build_canvas_decode_mask` (lru_cached) |
| flex `BlockMask` | 4×4 grid of 4-token blocks (toy flex blocks would be degenerate — see §5) | 32×32 grid of 128² blocks | `models/mask.py:build_block_causal_block_mask` (lru_cached) |
| prefill partial-canvas mask (prompt 100, toy `L=4`) | `(1, 1, 100, 100)` | `(1, 1, P, P)` | `inference/generate.py:_prefix_mask` (built per prefill, not cached) |

The model selects which flavor to build from `attn_impl`
(`models/transformer.py:DiffusionGemma._build_mask`): `BlockMask` for flex,
bool otherwise. All three mask builders share one row rule; the differences
are representation (bool tensor vs block-sparse descriptor vs inline
partial-canvas construction), not semantics.

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

The arithmetic, derived rather than quoted. Per layer, per token, the cache
holds `k` and `v`:

```
kv elements/token/layer = 2 · n_kv_heads · head_dim
production:  2 · 4 · 64  = 512 elements → bf16 (2 B) → 1,024 B/layer
             × 24 layers  = 24,576 B ≈ 24 KB/token
MHA same width: 2 · 16 · 64 = 2,048 elements → 4,096 B/layer → 98,304 B ≈ 96 KB/token
toy:         2 · 2 · 32  = 128 elements → 256 B/layer × 2 layers = 512 B/token
```

The same grouping shrinks the projections: `k_proj`/`v_proj` map
`d_model → n_kv_heads · head_dim` (1024 → 256), not `d_model → 1024`. Per
layer that saves `2 · 1024 · (1024 − 256) = 1,572,864` params vs full MHA,
~37.7M over 24 layers — most of the gap between the early planning docs'
"~380M" (which counted full-MHA K/V at `d_model`) and the honest
343,516,160 parameter count (SDD Ruling 13).

Why *grouped* sharing and not the two extremes: **vs MHA** (16 KV heads), 4×
the cache bytes and decode-time bandwidth for capacity the denoiser does not
need — KV heads carry "what is where", and 4 planes at `head_dim = 64` span
the context geometry, while quality-sensitive capacity lives in the 16 query
heads GQA keeps untouched. **vs MQA** (1 KV head): maximal savings but every
query head shares one K/V subspace, collapsing attention diversity; 4 KV
heads is the standard middle — large savings, no measured quality cliff.

Group semantics, concretely: with `reps = n_heads / n_kv_heads = 4`, query
heads `4h..4h+3` consume KV head `h`. That is exactly the mapping
`repeat_interleave(4, dim=1)` produces on the eager path, and exactly the
interpretation `enable_gqa=True` gives the fast kernels — which is why the
two representations are interchangeable and the equivalence tests can
transplant weights between them.

Shapes (B=1), toy then production:

| tensor | toy dims | production dims | cached? |
|---|---|---|---|
| `q` | (4, T, 32), T=16 | (16, T, 64), T = 4096 | no |
| `k`, `v` | (2, T, 32) | (4, T, 64) | **yes — the KV cache** |
| out | (4, T, 32) → reshape → out_proj → (1, T, 128) | (16, T, 64) → reshape → out_proj → (1, T, 1024) | no |
| per-layer cache bytes/token | 256 B | 1,024 B | — |

Layout pinned by `tests/test_attention.py::test_gqa_kv_heads`.

## 3. RoPE on q and untiled k

> Primer in [foundations §5.4](foundations.md); here: the production-specific
> facts only.

**The mechanism, one derivation down.** RoPE encodes position by rotating
each 2D slice of the head vector by a position-dependent angle. Split the
`head_dim`-vector into `half` pairs; pair `p` (frequency `f_p = θ^(−2p/head_dim)`)
rotates by angle `m · f_p` at position `m`. Because rotations compose,
`R(m) R(n)ᵀ = R((m−n) f_p)` — the dot product `q_m · k_n` depends only on the
*relative* offset `m − n`, never on absolute location. That single property
is why RoPE (unlike a learned position table) survives the KV cache: a key
roped once at write time stays correct for every future query, wherever the
query sits.

- Canonical GPT-NeoX/LLaMA **rotate-half**; pairs are `(i, i + half)`;
  `rope_theta = 500,000`; `head_dim = 64` → 32 frequency planes
  (`models/attention.py:apply_rope` is the reference implementation). In the
  rotate-half layout, `x[..., :half]` pairs with `x[..., half:]`, and the
  cos/sin vectors are built as `cat([f, f])` (`repeat(1, 2)` in the cached
  tables) so one multiplication covers both halves of every pair.
- **Frequency span at production dims.** Plane 0 has `f = 1` (wavelength
  `2π ≈ 6.3` tokens — token-local order); plane 31 has
  `f = 500000^(−62/64) ≈ 3.0e−6` (wavelength ≈ 2.1M tokens — at position
  4095 its angle is ≈ 0.012 rad, the coarsest order only). High planes
  disambiguate adjacent tokens; low planes keep distant positions from
  aliasing. A larger `rope_theta` shifts the spectrum down — the standard
  long-context knob; 500k is comfortably sized for 4,096.
- **Worked example (toy `head_dim = 4`, θ = 100).** `half = 2` planes,
  `inv_freq = [1, 100^(−1/2)] = [1.0, 0.1]`. Key at position 1 carries
  angles `(1.0, 0.1)`; query at position 3 carries `(3.0, 0.3)`. The
  query·key dot factors through `R(3)R(1)ᵀ = R(2)` — the rotation a query at
  position 2 applies — so attention between any pair offset by 2 is
  identical, independent of where the pair sits.
- Production forwards use `models/attention.py:DenoiseAttention`'s cached fp32
  cos/sin tables sized to `max_seq_len` (`DenoiseAttention._roped_qkv`
  indexes them per forward) — the trig was previously recomputed in every one
  of the 24 layers on every forward. The tables are `(max_seq_len, head_dim)`
  fp32 buffers — 1 MB each at production dims, ~48 MB across 24 layers —
  non-persistent so they never enter checkpoints; `apply_rope` recomputes
  identical math and remains the property-test oracle.
- RoPE applies to `q` and the **untiled** `k` (4 heads) *before* the cache
  append: cached keys are already roped
  (`models/attention.py:DenoiseAttention._roped_qkv`), which is why the
  sampler's decode chunks never re-rope the prefix. The ordering is
  load-bearing: rope each key **once, at write time**, and the cache stores
  final keys; caching unroped keys would force every forward to re-rope the
  entire prefix per layer (the exact cost the tables avoid) and ship
  positions through the cache, and tiling first would 4× the rope arithmetic.
  Note `DenoiseAttention.forward` always runs the SDPA kernel; under flex the
  block bypasses it and calls
  `models/attention.py:DenoiseAttention._roped_qkv` directly
  (`models/block.py:DenoiseBlock._attention`) — projections and rope are
  shared, only the kernel swaps.
- Invariants (pinned by
  `tests/test_attention.py::test_rope_preserves_norms_and_relative_position`):
  per-position norms preserved (rotation is orthogonal — `cos² + sin² = 1`
  per pair, so RoPE moves *directions* only); `q_m · k_n = f(m − n)`; a
  uniform +c shift of all positions leaves attention unchanged (SDD Ruling
  12 — the plan's original shifted-positions test could only pass against a
  buggy interleaved-frequency RoPE, so the test asserts relative spacing).

Shapes of the rope path, toy then production:

| tensor | toy | production |
|---|---|---|
| `rope_cos` / `rope_sin` buffers | `(max_seq_len, 32)` — sized by `max_seq_len`, not T | `(4096, 64)` fp32, 1 MB each |
| `q` after rope | (1, 4, T, 32) | (B, 16, T, 64) |
| `k` (untiled, roped) | (1, 2, T, 32) | (B, 4, T, 64) |
| decode-chunk `k` after append | (1, 2, P+L, 32) | (B, 4, P+256, 64) |

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

Read as math, the eager twin *is* the definition:

```
attn(q, k, v)[i] = Σ_k softmax_j( q_i · k_j / √d + m[i, j] ) v_k
m[i, j] = 0 if allow[i, j] else −∞        (the −∞ of masked_fill)
```

Every "faster" implementation is an approximation of this expression; the
eager twin exists so "approximation" is measurable rather than assumed. The
`1/√d` scaling (`d = head_dim`) keeps score variance `O(1)` as heads widen —
without it softmax saturates and gradients vanish. `masked_fill` with `−inf`
(not a large negative number) makes masked weights *exactly* zero after
softmax, so no masked key leaks mass even at fp32.

All three are proven equal by weight-transplant tests
(`tests/test_attention.py::test_attention_matches_eager`,
`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
`tests/test_models.py::test_flex_attn_impl_matches_sdpa`). The transplant
methodology: copy one implementation's parameters into the other, run both on
identical inputs, compare outputs — catching any semantic drift between
kernel flavors (reordered softmax accumulation, a GQA interpretation
mismatch, a mask-application bug) with no hand-written expected values.

Why three and not one: **Flex** is the default (`attn_impl: "flex"` in
`configs/pretrain_a100_380m.yaml`) because it fuses masking into the kernel
and skips fully-masked block pairs (§5), at the cost of needing CUDA-scale
machinery and a `BlockMask`. **SDPA** runs everywhere PyTorch runs (CPU
included) and accepts the plain bool mask — the portability floor, so CI and
CPU-only environments never depend on flex availability. **Eager** is the
only path with no fused cleverness to distrust. The eager twin is deliberate
duplication — never consolidate it (AGENTS.md §2) — and the only path that
expands KV heads by hand (`repeat_interleave`); the two fast paths consume
untiled KV (`enable_gqa`).

The three paths disagree on *mask input*, and the wiring reflects that:

| path | mask argument | shape |
|---|---|---|
| eager / sdpa | bool tensor | `(1, 1, T, T_total)`, broadcast over batch and heads |
| flex (training/prefill) | `BlockMask` | block-grid descriptor (§5) |
| flex (decode chunks) | `None` | no mask == full attention == the all-ones view |

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

The predicate is the row rule of §1.1, restated per element:

```python
def mask_mod(b, h, q_idx, kv_idx):
    q_block = q_idx // canvas_len
    return (kv_idx // canvas_len == q_block) | (kv_idx < q_block * canvas_len)
```

`create_block_mask` traces this symbolically (not per element) and reduces it
to block-level verdicts. At production dims the window is a 32×32 grid of
128-token blocks; canvas `b` spans exactly blocks `2b` and `2b+1`. For query
block `Q` and key block `K`: same canvas → all `128 × 128` pairs satisfy the
first clause → fully allowed; `K`'s canvas strictly earlier → all pairs
satisfy the second → fully allowed; `K`'s canvas strictly later → both
clauses fail everywhere → fully masked.

No block pair ever straddles a decision boundary, so the mask collapses to a
block-boolean grid with **no per-element masking work inside the kernel**.
The density arithmetic makes the win concrete. A canvas-`b` query row sees
`(b+1) · L` keys, so the allowed fraction of the full `T × T` window is

```
Σ_b L · (b+1) L / (nL)²  =  n(n+1)/2 / n²  =  (n+1) / 2n
```

At production (`n = 16` canvases): `17/32 ≈ 53%` — flex skips ~47% of all
block pairs outright; at toy dims (`n = 2`), 75%. The skipped work is real
FLOPs and memory traffic, not a masked-then-ignored computation.

The alignment is a hard constraint, not a nicety: with `L = 300`, a 128-token
block would contain elements of two canvases, the traced predicate would no
longer be block-constant, and flex would apply the mask per element —
recovered correctness, lost speed. `L = 256 = 2 × 128` is chosen so the two
granularities coincide.

Caching: `create_block_mask` traces Python — hundreds of microseconds to
milliseconds per call — and training runs thousands of forwards across 24
layers; a per-forward rebuild would dominate. The `lru_cache` on
`(seq_len, canvas_len, device)` makes the per-forward cost a dictionary
lookup, same contract as the bool mask (read-only return).

Under flex the sampler's decode chunks pass `mask=None` — **no block mask ==
full attention == the all-ones decode mask**, without building the tensor
(`inference/generate.py:BlockDiffusionSampler._decode_mask`). This is the
decode-mask payoff of §1.4: the chunk's visibility really is all-ones, and
flex's default (unmasked) kernel computes exactly that. A `past_kv` forward
on non-flex paths must pass an explicit mask
(`models/transformer.py:DiffusionGemma.backbone` asserts this) — the SDPA
kernel has no "full attention" default that spans a `(L, prefix+L)` window.

## 6. KV chaining (the decode contract)

The sampler's cache grows **once per finalized canvas** (256 tokens), not
once per token:

```
denoise_canvas(kv, prefix_len)   # T_eval forwards on the in-flight canvas
encode_canvas(kv, prefix_len)    # 1 forward; per-layer cat(past_kv, new_kv)
```

(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`,
`inference/generate.py:BlockDiffusionSampler.encode_canvas`.) During the
`T_eval` denoise forwards the cache is **static**: every step re-attends over
the same frozen prefix keys with fresh canvas content — that is what makes
the steps comparable and the commit rule (unchanged-or-stronger posterior)
meaningful. Only when the canvas is final does `encode_canvas` run one
re-encode forward — under the all-ones decode view, at `t = 0` — and append
its per-layer `(k, v)`.

Worked example (production dims, `T_eval = 32`, 256-token prompt = exactly
one canvas):

| stage | forwards this stage | window each forward attends | cache after (per layer) |
|---|---|---|---|
| prefill prompt (256 tokens) | 1 | 256 × 256 block-causal | `(4, 256, 64)` k/v |
| denoise canvas 1 (steps 1..32) | 32 | 256 queries × 256 keys, cache static | unchanged |
| encode canvas 1 | 1 | 256 × 512, all visible | `(4, 512, 64)` |
| denoise canvas 2 | 32 | 256 × 512 | unchanged |
| encode canvas 2 | 1 | 256 × 768 | `(4, 768, 64)` |
| ... after `n` canvases | `1 + 33n` total | — | `(4, 256 + 256n, 64)` |

Two properties fall out. First, the forwards-per-token economics: 34 forwards
produce 256 tokens (`1 + 32 + 1`), ~7.5 tokens/forward at `T_eval = 32` — the
count `1 + n·(T+1)` that `inference/evaluate.py:SpeedupEvaluator` charges.
Second, the cache grows in 256-token slabs, once per canvas, never per denoise
step — the 32 refinement forwards over an in-flight canvas would otherwise
either thrash the cache or (worse) cache *noisy* keys that change meaning
every step.

Why appending is *correct*, not just convenient. A canvas token's `k`/`v`
are per-token projections of its hidden state, and that hidden state is a
function of (token content, position, what the token attended). At encode
time the canvas attends to prefix + itself — exactly the row `allow[i, k]`
grants the last canvas of a full single-shot block-causal forward. Same
visibility, same positions, same content ⇒ same hidden ⇒ same `k`/`v`: the
chained cache equals, layer by layer, what a fresh single-shot forward over
`prefix + canvas` would write.

The contract (`models/mask.py:build_canvas_decode_mask` + the sampler's
`encode_canvas`): the chained KV must equal what a fresh single-shot
block-causal forward over `prefix + canvas` would write — enforced
**bit-exactly in fp64** by
`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`. Any
attention/KV plumbing change re-runs this test. The fp64 demand is the strong
version on purpose: a test passing at `atol=1e-3` would absorb a kernel that
reorders softmax accumulation, double-ropes keys, or misaligns the group
mapping — "small" shifts that compound over 24 layers.

At equal total length the cache shape is identical to AR's (every token is
eventually encoded); the win is fewer forwards, not a smaller cache — cache
growth cadence is once per canvas vs once per token ([sampler](sampler.md)).

## 7. Shapes at production dims

Training forward (micro_bs 8, seq 4,096 = 16 canvases × 256), following one
token row through `models/block.py:DenoiseBlock.forward`:

| tensor | shape | note |
|---|---|---|
| input hidden | (8, 4096, 1024) | embed + canvas-time embed |
| post-RMSNorm (`models/block.py:RMSNorm.forward`) | (8, 4096, 1024) | pre-norm for the attn sublayer |
| bool mask (sdpa/eager flavor) | (1, 1, 4096, 4096) | cached (`build_block_causal_mask`), 16.8 MB, broadcasts over B |
| BlockMask (flex flavor) | 32×32 grid of 128² blocks | canvas-aligned, cached (`build_block_causal_block_mask`) |
| `q` | (8, 16, 4096, 64) | 16 query heads |
| `k`, `v` | (8, 4, 4096, 64) | **the KV cache content** |
| eager scores | (8, 16, 4096, 4096) | ground-truth path only — 8.6 GB fp32 (plus a second transient of the same size for the masked/softmax stage; the reason eager is tests-only) |
| attn out → out_proj | (8, 4096, 1024) | residual add back into hidden |
| SwiGLU `w13` gate/up | (8, 4096, 6144) | `ffn_dim = 3072`, fused 2× projection |

Decode chunk (in-flight canvas over a `P`-token prefix):

| tensor | shape |
|---|---|
| canvas `xt` | (B, 256) |
| `q` | (B, 16, 256, 64) |
| `k`, `v` after append | (B, 4, prefix+256, 64) |
| decode mask | all-ones (1, 1, 256, prefix+256), or `None` under flex |

KV bytes/token: `2 · 24 layers · 4 kv_heads · 64 · 2 B = 24,576 ≈ 24 KB`
(§2); 100.7 MB at 4,096 context. Batched: micro_bs 8 over a full 4,096
window caches `8 × 100.7 MB ≈ 806 MB` — real money on an 80 GB A100 shared
with ~33 GB of activations, and the direct reason §2 rejects MHA's 4× figure.

The eager-scores row is the *definition* of attention priced in bytes:
(8, 16, 4096, 4096) fp32 = `8 · 16 · 16.8M · 4 B = 8.6 GB`, and the
masked_fill/softmax stages transiently double it. Flex and SDPA never
materialize it — the fused kernels tile the score matrix through SRAM —
which is why production never touches eager at these dims and the eager twin
lives only in tests.

## 8. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| delete the eager attention twin | no implementation-independent oracle | weight-transplant tests lose their reference |
| tile KV heads in the fast paths | 4× cache traffic; contradicts `enable_gqa` | `tests/test_attention.py::test_gqa_kv_heads` |
| zero-mask the in-flight canvas against itself | contradicts the all-ones decode contract (Ruling 15) | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (fp64) |
| allow cross-canvas bidirectionality | left-to-right factorization invalid; sampler conditioning leaks future canvases | `tests/test_mask.py::test_mask_causal_across_canvases` |
| make the mask causal *within* canvas too (drop bidirectionality) | the 256 positions can no longer condition on each other; the within-canvas joint the diffusion objective trains against no longer matches attention; parallel denoising degenerates to sequential | `tests/test_mask.py::test_mask_bidirectional_within_canvas` |
| recompute RoPE trig per layer per forward | correctness unchanged; ~24× redundant trig per forward (the reason the fp32 tables exist) | (performance, not a pin) |
| bf16 cos/sin tables | trig drift compounds across 24 layers | `tests/test_attention.py::test_rope_preserves_norms_and_relative_position` |
| cache unroped k and rope at read time | every decode chunk re-ropes the whole prefix; needs positions through the cache; breaks the `past_kv` contract (k is already roped) | `tests/test_attention.py::test_past_kv_prefix_path` |
| non-canvas-aligned flex blocks | block pairs straddle canvases; mask no longer all-or-nothing; sparsity overhead returns | `tests/test_models.py::test_flex_attn_impl_matches_sdpa` |
| rebuild the bool mask / BlockMask every forward | 16.8 MB alloc + tracing per forward; correctness unchanged, throughput craters | (performance, not a pin — the `lru_cache` docstrings name it) |
| drop the `seq_len % canvas_len == 0` assertion | silent mis-blocked rows for partial canvases; the prefill path exists precisely to route around it via `inference/generate.py:_prefix_mask` | `tests/test_mask.py::test_block_causal_mask_matches_manual` (assert is the guard the test relies on) |
| mutate a cached mask in place | the `lru_cache` hands the *same* tensor to every later forward | (contract, not a test — docstrings say read-only) |
| append KV per denoise step instead of per canvas | noisy, step-dependent keys enter the cache; the frozen-prefix premise of §6 dies | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| relax the non-flex `past_kv` mask assert in `models/transformer.py:DiffusionGemma.backbone` | SDPA decode chunk silently attends with a default-shaped mask — wrong window | `tests/test_attention.py::test_past_kv_prefix_path` |

## 9. Glossary

| symbol | meaning | code |
|---|---|---|
| `L` | canvas length (256) | config `canvas_len` |
| `T` | total window length (4096 = 16 canvases) | config `max_seq_len` |
| `allow[i,k]` | mask entry: query i may attend key k | `models/mask.py:build_block_causal_mask` |
| BlockMask | flex's block-sparse mask (canvas-aligned) | `models/mask.py:build_block_causal_block_mask` |
| `mask_mod` | per-element predicate `create_block_mask` traces to build a BlockMask | `models/mask.py:build_block_causal_block_mask` |
| GQA | grouped-query attention (16Q/4KV here) | `models/attention.py:DenoiseAttention` |
| `enable_gqa` | SDPA/flex flag: consume untiled (n_kv_heads) k/v natively | `models/mask.py:block_causal_sdpa_attention` |
| `reps` | query heads per KV head (= 4 here) | `models/block.py:DenoiseBlock._attention` |
| `past_kv` | per-layer (k, v), k already roped | `models/attention.py:DenoiseAttention._roped_qkv` |
| `rope_cos` / `rope_sin` | cached fp32 trig tables, `(max_seq_len, head_dim)`, non-persistent | `models/attention.py:DenoiseAttention._roped_qkv` |
| rotate-half | NeoX/LLaMA RoPE pairing `(i, i+half)` | `models/attention.py:apply_rope` |
| `_prefix_mask` | partial-first-canvas prefill mask | `inference/generate.py:_prefix_mask` |
| density | allowed fraction of the block-causal window: `(n+1)/2n` | §5 |

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

**Q: Block-causal vs full causal vs fully bidirectional — when does each win?**
A: Token-causal wins when you need per-token latency granularity and can pay
one forward per token. Fully bidirectional wins when the whole sequence is
present and refine-in-place is acceptable (encoding, editing) — but it cannot
cache, because no position is ever final. Block-causal wins on *throughput*:
bidirectionality inside the canvas (the parallel-refinement bet) plus
causality across canvases (which makes the prefix KV freezable and the
sampler's conditioning valid). At `T_eval = 32`, L = 256: ~7.8 tokens/forward
vs AR's 1.

**Q: Why must keys be roped before entering the cache?**
A: RoPE is relative-position: a key roped once at its absolute position
composes with any future query via `R(m)R(n)ᵀ = R(m−n)`. Cache roped keys and
no decode chunk ever re-ropes the prefix; cache unroped keys and every chunk
re-ropes the whole prefix (24 layers' worth) and the cache must carry
positions. The cache stores `k` post-rope
(`models/attention.py:DenoiseAttention._roped_qkv`).

**Q: What exactly does `enable_gqa=True` avoid?**
A: The naive GQA path expands `(B, 4, T, 64)` k/v to `(B, 16, T, 64)` with
`repeat_interleave` before the kernel — 4× the bytes touched, zero extra
information. `enable_gqa` has the kernel interpret the 4 KV heads against
the 16 query heads directly (query head `h` reads KV head `h // 4`), so the
tiling happens inside the kernel for free. The eager twin keeps the explicit
expansion so its semantics stay visible and testable
(`models/block.py:DenoiseBlock._attention`).

**Q: Why is `mask=None` safe under flex for decode chunks?**
A: FlexAttention's default with no block mask is full attention over the
given q/k window — and the decode chunk's true visibility *is* "everything":
prefix fully visible, in-flight canvas fully visible, i.e. the all-ones mask
of `models/mask.py:build_canvas_decode_mask`. `None` computes those semantics
without materializing a `(1, 1, 256, prefix+256)` tensor; the same forwards
on non-flex paths must pass the explicit mask
(`models/transformer.py:DiffusionGemma.backbone` asserts it).

**Q: What does the ~53% density number mean operationally?**
A: With 16 canvases the allowed fraction of the 4096² window is
`(16+1)/(2·16) ≈ 53%`; because canvas blocks align with flex's 128-token
blocks, that sparsity is *block-exact* — the kernel never touches the masked
47%, so attention cost scales with the block-sparse structure. It is also why
moving `canvas_len` off a multiple of 128 quietly re-introduces per-element
masking overhead.