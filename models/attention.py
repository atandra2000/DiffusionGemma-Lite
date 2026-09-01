"""GQA denoiser attention with RoPE over a block-causal bool mask."""
import torch
import torch.nn as nn

from models.mask import block_causal_sdpa_attention


def apply_rope(q, k, positions, theta):
    """Reference rotate-half RoPE at absolute positions (tests + property oracle).

    Production forwards use DenoiseAttention's cached cos/sin tables instead of
    recomputing trig per layer; the math is identical (fp32 tables, cast to the
    input dtype). theta is the base (float) or a precomputed inv_freq buffer."""
    half = q.size(-1) // 2
    if isinstance(theta, torch.Tensor):
        inv_freq = theta.to(dtype=q.dtype, device=q.device)
    else:
        inv_freq = theta ** (-2.0 * torch.arange(half, dtype=q.dtype, device=q.device) / q.size(-1))
    angle = positions.to(q.dtype)[:, None] * inv_freq           # (T, half)
    cos = angle.cos()[None, None].repeat(1, 1, 1, 2)            # cat([f, f]): pairs (i, i+half)
    sin = angle.sin()[None, None].repeat(1, 1, 1, 2)

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class DenoiseAttention(nn.Module):
    """GQA attention: rope on q and untiled k, bool-mask SDPA with native GQA.

    past_kv (optional) is a (k, v) tuple with already-roped k: (B, n_kv_heads, P, head_dim);
    output covers only the incoming tokens. No attention sinks; mask stays bool.
    RoPE tables are precomputed for ``max_seq_len`` positions (non-persistent
    buffers, fp32, cast per call) — the trig was previously recomputed in every
    one of the 24 layers on every forward. Positions must be < max_seq_len.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_theta: float, max_seq_len: int = 4096):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_heads, self.n_kv_heads, self.head_dim, self.rope_theta = (
            n_heads, n_kv_heads, head_dim, rope_theta)
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(d_model, n_heads * head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model)
        half = head_dim // 2
        inv_freq = rope_theta ** (-2.0 * torch.arange(half).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq)
        angle = torch.arange(max_seq_len, dtype=torch.float32)[:, None] * inv_freq
        self.register_buffer("rope_cos", angle.cos().repeat(1, 2), persistent=False)   # (max_seq_len, head_dim)
        self.register_buffer("rope_sin", angle.sin().repeat(1, 2), persistent=False)

    def _rotate_half(self, x):
        half = self.head_dim // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _roped_qkv(self, hidden, positions, past_kv=None):
        """Projections + cached-table rope at absolute positions; past_kv (roped k, v)
        appended. Returns q (n_heads) and k/v (n_kv_heads, untiled), plus the
        cacheable (k, v) covering past + current tokens."""
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos = self.rope_cos[positions].to(q.dtype)      # (T, head_dim); broadcasts over B, H
        sin = self.rope_sin[positions].to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        return q, k, v, (k, v)

    def forward(self, hidden, mask, positions, past_kv=None, return_kv=False):
        """hidden: (B, T, D) -> (B, T, D); mask: (1, 1, T, T_total) bool.

        Output covers only the incoming tokens; with return_kv=True also the
        new per-layer (k, v) for the cache."""
        B, T, _ = hidden.shape
        q, k, v, kv = self._roped_qkv(hidden, positions, past_kv)
        out = block_causal_sdpa_attention(q, k, v, mask, enable_gqa=True)
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))
        return (out, kv) if return_kv else out