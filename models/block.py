"""Denoiser block: RMSNorm -> GQA attention (RoPE, block-causal bool mask) -> SwiGLU."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import DenoiseAttention, apply_rope
from models.mask import eager_block_causal_attention


class RMSNorm(nn.Module):
    """Root-mean-square norm with a learned (initially one) scale, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DenoiseBlock(nn.Module):
    """One denoiser layer: block-causal attention sublayer + SwiGLU sublayer, both residual."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 ffn_dim: int, rms_norm_eps: float, attn_impl: str, rope_theta: float):
        super().__init__()
        assert attn_impl in ("sdpa", "eager"), f"unknown attn_impl: {attn_impl!r}"
        self.attn_norm = RMSNorm(d_model, rms_norm_eps)
        self.attn = DenoiseAttention(d_model, n_heads, n_kv_heads, head_dim, rope_theta)
        self.ffn_norm = RMSNorm(d_model, rms_norm_eps)
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.w3 = nn.Linear(d_model, ffn_dim)
        self.attn_impl = attn_impl

    def _attention(self, h, mask, positions):
        """sdpa: DenoiseAttention; eager: same projections, eager path, mask stays bool."""
        if self.attn_impl == "sdpa":
            return self.attn(h, mask, positions)
        a = self.attn
        B, T, _ = h.shape
        q = a.q_proj(h).view(B, T, a.n_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(h).view(B, T, a.n_kv_heads, a.head_dim).transpose(1, 2)
        v = a.v_proj(h).view(B, T, a.n_kv_heads, a.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, positions, a.inv_freq)
        reps = a.n_heads // a.n_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        out = eager_block_causal_attention(q, k, v, mask)
        return a.out_proj(out.transpose(1, 2).reshape(B, T, a.n_heads * a.head_dim))

    def forward(self, hidden, mask, positions):
        h = hidden + self._attention(self.attn_norm(hidden), mask, positions)
        f = F.silu(self.w1(self.ffn_norm(h))) * self.w3(self.ffn_norm(h))
        return h + self.w2(f)