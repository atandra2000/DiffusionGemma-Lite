"""Denoiser block: RMSNorm -> GQA attention (RoPE, block-causal mask) -> SwiGLU."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import DenoiseAttention
from models.mask import (block_causal_sdpa_attention, eager_block_causal_attention,
                         flex_block_causal_attention)


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned (initially one) scale, no bias.

    F.rms_norm is the fused stdlib kernel of the manual rsqrt(mean(x^2)) formula."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.size(0),), self.weight, self.eps)


class DenoiseBlock(nn.Module):
    """One denoiser layer: block-causal attention sublayer + SwiGLU sublayer, both residual."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 ffn_dim: int, rms_norm_eps: float, attn_impl: str, rope_theta: float):
        super().__init__()
        assert attn_impl in ("sdpa", "eager", "flex"), f"unknown attn_impl: {attn_impl!r}"
        self.attn_norm = RMSNorm(d_model, rms_norm_eps)
        self.attn = DenoiseAttention(d_model, n_heads, n_kv_heads, head_dim, rope_theta)
        self.ffn_norm = RMSNorm(d_model, rms_norm_eps)
        self.w13 = nn.Linear(d_model, 2 * ffn_dim)   # fused gate/up SwiGLU projection
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.attn_impl = attn_impl

    def _attention(self, h, mask, positions, past_kv=None):
        """Shared projections/rope/past-append via DenoiseAttention; only the masked
        kernel differs between attn_impl paths (sdpa / flex / eager ground truth).
        sdpa and flex consume untiled k/v (enable_gqa); the eager twin keeps the
        explicit repeat_interleave expansion."""
        a = self.attn
        B, T, _ = h.shape
        q, k, v, kv = a._roped_qkv(h, positions, past_kv)
        if self.attn_impl == "flex":
            out = flex_block_causal_attention(q, k, v, mask)
        elif self.attn_impl == "sdpa":
            out = block_causal_sdpa_attention(q, k, v, mask, enable_gqa=True)
        else:  # eager ground truth
            reps = a.n_heads // a.n_kv_heads
            out = eager_block_causal_attention(q, k.repeat_interleave(reps, dim=1),
                                               v.repeat_interleave(reps, dim=1), mask)
        out = a.out_proj(out.transpose(1, 2).reshape(B, T, a.n_heads * a.head_dim))
        return out, kv

    def forward(self, hidden, mask, positions, past_kv=None, return_kv=False):
        att, kv = self._attention(self.attn_norm(hidden), mask, positions, past_kv)
        h = hidden + att
        gate, up = self.w13(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.w2(F.silu(gate) * up)
        return (h, kv) if return_kv else h