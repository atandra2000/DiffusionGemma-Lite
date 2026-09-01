# Concept: block-causal attention

> Full context: [`DIFFUSION.md`](../../DIFFUSION.md) §2. The mask is the
> load-bearing primitive — everything else in the project leans on it.

## The mask

`models/mask.py:build_block_causal_mask(seq_len, canvas_len)` returns a bool
`(1, 1, T, T)` with row semantics

```
allow[i, k] = (k < floor(i/L)·L) | (floor(k/L) == floor(i/L))
```

(`L = canvas_len = 256`):

- **strictly earlier canvases: causal** — canvas `b` attends to canvases
  `0..b−1` only. This preserves a valid left-to-right conditional
  factorization of the sequence.
- **own canvas: bidirectional** — the 256 positions denoise *together*; this
  is the parallelism the project exists for.

One dense transformer therefore plays two roles simultaneously: an
autoregressive LM across canvases and a bidirectional denoiser within one.

The mask asserts `seq_len % canvas_len == 0`. The sampler's prompt prefill
needs a partial first canvas, so it builds the same row semantics inline for
arbitrary prompt lengths (`inference/generate.py:_prefix_mask`).

## Decode-time view

While a canvas is in flight it is the **last block** of the window, and
everything before it is finalized. `models/mask.py:build_canvas_decode_mask`
returns an all-ones `(1, 1, L, prefix_len + L)` view — nothing is masked
*within* the in-flight canvas; the canvas's content is noisy, its visibility
is not. The finalized prefix is fully visible (this is the all-ones re-encode
contract, SDD Ruling 15).

## Attention paths

- Production: `models/mask.py:block_causal_sdpa_attention` →
  `F.scaled_dot_product_attention` with the bool mask (`enable_gqa` consumes
  untiled K/V).
- Fused (A100 config default): `models/mask.py:flex_block_causal_attention`
  → `torch.nn.attention.flex_attention` with a canvas-sized cached `BlockMask`
  (`models/mask.py:build_block_causal_block_mask`); every block pair is
  all-or-nothing, so the block-sparse kernel pays no sparsity overhead. The
  sampler decode chunk passes `mask=None` (full attention == all-ones decode
  mask).
- Ground truth: `models/mask.py:eager_block_causal_attention` — explicit
  scores → mask → softmax → @v. **It is deliberate duplication**, exercised
  by weight-transplant equivalence tests; don't consolidate.

Heads: GQA 16 query / 4 KV, head_dim 64, in
`models/attention.py:DenoiseAttention`; the fast kernels consume untiled KV
heads (`enable_gqa`). Positions: canonical GPT-NeoX/LLaMA rotate-half RoPE
via cached fp32 cos/sin tables sized to `max_seq_len`
(`models/attention.py:apply_rope` is the reference implementation); a
property test pins per-position norm preservation and the relative-position
identity `q_m·k_n = f(m − n)` (consequence: a uniform position shift changes
nothing — SDD Ruling 12).

Blocks wrap attention + SwiGLU with pre-norm
(`models/block.py:DenoiseBlock`, `models/block.py:RMSNorm`), and the top
level wires blocks + time embedding in `models/transformer.py:DiffusionGemma`.

## KV chaining

The sampler's cache grows **once per finalized canvas**
(`inference/generate.py:BlockDiffusionSampler.encode_canvas` appends the re-encoded canvas's KV).
The contract: the chained KV must equal what a fresh single-shot block-causal
forward over `prefix + canvas` writes — enforced bit-exactly in fp64 by
`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`. Cache
shape at equal length is identical to AR (all tokens are eventually encoded);
the win is fewer forwards, not smaller caches.
