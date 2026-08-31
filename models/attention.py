"""GQA denoiser attention with RoPE over a block-causal bool mask."""
import torch
import torch.nn as nn

from models.mask import block_causal_sdpa_attention


def apply_rope(q, k, positions, theta):
    """Rotate-half RoPE at absolute positions; q and k share the incoming token axis."""
    half = q.size(-1) // 2
    inv_freq = theta ** (-2.0 * torch.arange(half, dtype=q.dtype, device=q.device) / q.size(-1))
    angle = positions.to(q.dtype)[:, None] * inv_freq            # (T, half)
    cos = angle.cos()[None, None].repeat_interleave(2, dim=-1)  # (1, 1, T, head_dim)
    sin = angle.sin()[None, None].repeat_interleave(2, dim=-1)

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class DenoiseAttention(nn.Module):
    """GQA attention: rope on q and untiled k, kv tiled to n_heads, bool-mask SDPA.

    past_kv (optional) is a (k, v) tuple with already-roped k: (B, n_kv_heads, P, head_dim);
    output covers only the incoming tokens. No attention sinks; mask stays bool.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, rope_theta: float):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads, self.n_kv_heads, self.head_dim, self.rope_theta = (
            n_heads, n_kv_heads, head_dim, rope_theta)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model)

    def forward(self, hidden, mask, positions, past_kv=None):
        """hidden: (B, T, D) -> (B, T, D); mask: (1, 1, T, T_total) bool."""
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, positions, self.rope_theta)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        reps = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        out = block_causal_sdpa_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)