"""Block-causal attention masks (bool for sdpa/eager, BlockMask for flex) and the kernels."""
import functools
import math
import torch
import torch.nn.functional as F

@functools.lru_cache(maxsize=None)
def build_block_causal_mask(seq_len: int, canvas_len: int, device=None) -> torch.Tensor:
    """Bool mask (1,1,T,T), True = attend: causal across canvases, bidirectional within.

    Cached (pure function of seq_len/canvas_len/device) — it was rebuilt on every
    forward before. Treat the return value as read-only."""
    assert seq_len % canvas_len == 0, "seq_len must be a multiple of canvas_len"
    idx = torch.arange(seq_len, device=device)
    q_block = (idx // canvas_len).unsqueeze(1)      # (T, 1)
    k_pos = idx.unsqueeze(0)                        # (1, T)
    allow = (k_pos < q_block * canvas_len) | ((k_pos // canvas_len) == q_block)
    return allow.view(1, 1, seq_len, seq_len)

@functools.lru_cache(maxsize=None)
def build_canvas_decode_mask(prefix_len: int, canvas_len: int, device=None) -> torch.Tensor:
    """Bool mask (1,1,L,prefix_len+L): prefix + in-flight canvas all visible."""
    # Finalized prefix is fully visible; the in-flight canvas is the last block.
    shape = (1, 1, canvas_len, prefix_len + canvas_len)
    return torch.ones(shape, dtype=torch.bool, device=device)

@functools.lru_cache(maxsize=None)
def build_block_causal_block_mask(seq_len: int, canvas_len: int, device=None):
    """FlexAttention BlockMask with the same row semantics as ``build_block_causal_mask``.

    Canvas-sized blocks make every (query, key) block pair all-or-nothing, so the
    fused block-sparse kernel runs at full-density speed. Cached because
    ``create_block_mask`` tracing is far too slow to run per forward. Works for
    partial first canvases (the sampler prefill) since the rule is per-element."""
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // canvas_len
        return (kv_idx // canvas_len == q_block) | (kv_idx < q_block * canvas_len)

    return create_block_mask(mask_mod, None, None, seq_len, seq_len, device=device)

def _sdpa_supports_gqa() -> bool:
    """``enable_gqa`` landed in torch 2.5. On 2.4 the kwarg raises TypeError."""
    import inspect
    try:
        return "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return False


_SDPA_HAS_GQA = _sdpa_supports_gqa()


def block_causal_sdpa_attention(q, k, v, mask, enable_gqa: bool = False):
    """Fast path: F.scaled_dot_product_attention with a bool attn_mask.

    ``enable_gqa=True`` lets the kernel consume untiled (n_kv_heads) k/v directly,
    skipping the 4x repeat_interleave expansion. torch before 2.5 has no such
    kwarg, so fall back to that expansion rather than raising TypeError."""
    if not enable_gqa:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if _SDPA_HAS_GQA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    n_rep = q.size(1) // k.size(1)
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def flex_block_causal_attention(q, k, v, block_mask):
    """FlexAttention path: fused block-sparse kernel with native GQA.

    ``block_mask=None`` means full attention — exactly the all-ones decode mask
    (``build_canvas_decode_mask``) semantics, minus the mask construction."""
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)

def eager_block_causal_attention(q, k, v, mask):
    """O(T^2) ground-truth path; do not replace with a fused kernel."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v