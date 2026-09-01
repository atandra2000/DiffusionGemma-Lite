# R3 — Mask & attention API

> Every public symbol in `models/mask.py` and `models/attention.py` with
> exact signatures, shapes, and pinning tests. Semantics derived in
> [block-causal-attention](../concepts/block-causal-attention.md).

## `models/mask.py`

| symbol | signature | returns | notes |
|---|---|---|---|
| `build_block_causal_mask` | `(seq_len: int, canvas_len: int, device=None) -> torch.Tensor` | bool `(1, 1, T, T)`, True = attend | asserts `seq_len % canvas_len == 0`; `functools.lru_cache`d — treat as read-only; pinned by `tests/test_mask.py` (4 tests) |
| `build_canvas_decode_mask` | `(prefix_len: int, canvas_len: int, device=None) -> torch.Tensor` | all-ones `(1, 1, L, prefix_len+L)` | the in-flight canvas sees everything (Ruling 15); `tests/test_mask.py::test_canvas_decode_mask_all_visible` |
| `build_block_causal_block_mask` | `(seq_len, canvas_len, device=None)` | `torch.nn.attention.BlockMask` | flex path; handles a partial first canvas (sampler prefill); cached — `create_block_mask` tracing too slow per forward |
| `block_causal_sdpa_attention` | `(q, k, v, mask, enable_gqa: bool = False)` | attention out | portable path; untiled KV via `enable_gqa` |
| `flex_block_causal_attention` | `(q, k, v, block_mask)` | attention out | production CUDA path; `mask=None` == full attention |
| `eager_block_causal_attention` | `(q, k, v, mask)` | attention out | **the oracle** — O(T²) scores → masked_fill(−inf) → softmax → @V; never consolidate (AGENTS.md §2) |

The three attention paths are proven equivalent by weight-transplant tests
(`tests/test_attention.py::test_attention_matches_eager`,
`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
`test_flex_attn_impl_matches_sdpa`). Derivation and the 8×8 worked mask:
[block-causal-attention §1–5](../concepts/block-causal-attention.md).

## `models/attention.py`

| symbol | signature | contract |
|---|---|---|
| `apply_rope` | `(q, k, positions, theta)` | reference rotate-half RoPE; pairs `(i, i+half)`; norm-preserving; `q_m·k_n = f(m−n)` — `tests/test_attention.py::test_rope_preserves_norms_and_relative_position` (Ruling 12) |
| `DenoiseAttention.__init__` | `(d_model, n_heads, n_kv_heads, head_dim, rope_theta, max_seq_len=4096)` | 16Q/4KV at production dims; precomputes fp32 cos/sin tables for `max_seq_len` |
| `DenoiseAttention.forward` | `(hidden, mask, positions, past_kv=None, return_kv=False)` | q/k/v → RoPE on q + untiled k → cache append → masked attention; cached keys are already roped |

| shape | value |
|---|---|
| `q` | (B, 16, S, 64) |
| `k`, `v` | (B, 4, S, 64) — the KV cache |
| KV bytes/token | 2·24 layers·4·64·2 = 24,576 B ≈ 24 KB |

Pinned by `tests/test_attention.py::test_gqa_kv_heads`,
`test_past_kv_prefix_path`, `test_attention_matches_eager`.

## `models/block.py` and `models/time_embed.py`

| symbol | signature | notes |
|---|---|---|
| `RMSNorm.__init__` | `(dim: int, eps: float = 1e-5)` | `x · rsqrt(mean(x²) + eps) · g` — [foundations §5.1](../concepts/foundations.md) |
| `RMSNorm.forward` | `(x: torch.Tensor) -> torch.Tensor` | shape preserved |
| `DenoiseBlock.__init__` | `(d_model, n_heads, n_kv_heads, head_dim, ffn_dim, rms_norm_eps, attn_impl, rope_theta)` | pre-norm residual pair: attention block + SwiGLU MLP |
| `DenoiseBlock.forward` | `(hidden, mask, positions, past_kv=None, return_kv=False)` | RoPE + GQA + SwiGLU; the eager twin expands KV heads (`repeat_interleave`) |
| `CanvasTimeEmbedding.__init__` | `(d_model, time_dim=256)` | sinusoidal t-embedding added per position |
| `CanvasTimeEmbedding.forward` | `(t: torch.Tensor, T: int) -> torch.Tensor` | `t: (B, n_canvases)` → per-canvas embedding broadcast over the canvas's 256 positions; sin/cos over `time_dim` |

Pinned by `tests/test_models.py` (block wiring, time-embed shapes).