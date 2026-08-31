import math

import torch

from models.attention import DenoiseAttention, apply_rope
from models.mask import build_block_causal_mask


def _rope_ref(x, positions, theta):
    """Self-contained rotate-half reference (not imported from the module)."""
    half = x.size(-1) // 2
    inv = theta ** (-2.0 * torch.arange(half, dtype=x.dtype) / x.size(-1))
    ang = positions.to(x.dtype)[:, None] * inv
    cos = ang.cos()[None, None].repeat(1, 1, 1, 2)
    sin = ang.sin()[None, None].repeat(1, 1, 1, 2)
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def test_attention_matches_eager(device):
    torch.manual_seed(0)
    B, T, D, H, Hkv, HD = 2, 8, 16, 2, 2, 8
    m = DenoiseAttention(D, H, Hkv, HD, rope_theta=10000.0).to(device)
    hidden = torch.randn(B, T, D, device=device)
    mask = build_block_causal_mask(T, 4, device)
    positions = torch.arange(T, device=device)

    out = m(hidden, mask, positions)

    # Reference: project with the module's own weights/biases, rope by hand, eager attention.
    q = (hidden @ m.q_proj.weight.T + m.q_proj.bias).view(B, T, H, HD).transpose(1, 2)
    k = (hidden @ m.k_proj.weight.T + m.k_proj.bias).view(B, T, Hkv, HD).transpose(1, 2)
    v = (hidden @ m.v_proj.weight.T + m.v_proj.bias).view(B, T, Hkv, HD).transpose(1, 2)
    q, k = _rope_ref(q, positions, 10000.0), _rope_ref(k, positions, 10000.0)
    k = k.repeat_interleave(H // Hkv, dim=1)
    v = v.repeat_interleave(H // Hkv, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, H * HD)
    ref = ref @ m.out_proj.weight.T + m.out_proj.bias
    assert out.shape == (B, T, D)
    assert torch.allclose(out, ref, atol=1e-5)


def test_rope_positions_change_output(device):
    torch.manual_seed(1)
    B, T, D, H, Hkv, HD = 1, 8, 16, 2, 2, 8
    m = DenoiseAttention(D, H, Hkv, HD, rope_theta=10000.0).to(device)
    hidden = torch.randn(B, T, D, device=device)
    mask = build_block_causal_mask(T, 4, device)
    out_a = m(hidden, mask, torch.arange(T, device=device))
    # Standard rope cancels a UNIFORM shift (relative positions unchanged), so the
    # comparison must change relative spacing, not just add a constant.
    out_b = m(hidden, mask, torch.arange(0, 2 * T, 2, device=device))
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_gqa_kv_heads(device):
    torch.manual_seed(2)
    B, T, D, H, Hkv, HD = 2, 8, 16, 4, 2, 8
    m = DenoiseAttention(D, H, Hkv, HD, rope_theta=10000.0).to(device)
    assert m.k_proj.weight.shape == (Hkv * HD, D)
    assert m.v_proj.weight.shape == (Hkv * HD, D)
    hidden = torch.randn(B, T, D, device=device)
    mask = build_block_causal_mask(T, 4, device)
    positions = torch.arange(T, device=device)

    out = m(hidden, mask, positions)

    # Reference with per-kv-head slices tiled [k0,k0,k1,k1] — the GQA convention.
    q = (hidden @ m.q_proj.weight.T + m.q_proj.bias).view(B, T, H, HD).transpose(1, 2)
    k = (hidden @ m.k_proj.weight.T + m.k_proj.bias).view(B, T, Hkv, HD).transpose(1, 2)
    v = (hidden @ m.v_proj.weight.T + m.v_proj.bias).view(B, T, Hkv, HD).transpose(1, 2)
    q, k = _rope_ref(q, positions, 10000.0), _rope_ref(k, positions, 10000.0)
    kt = torch.cat([k[:, 0:1], k[:, 0:1], k[:, 1:2], k[:, 1:2]], dim=1)
    vt = torch.cat([v[:, 0:1], v[:, 0:1], v[:, 1:2], v[:, 1:2]], dim=1)
    scores = (q @ kt.transpose(-2, -1)) / math.sqrt(HD)
    scores = scores.masked_fill(~mask, float("-inf"))
    ref = (scores.softmax(-1) @ vt).transpose(1, 2).reshape(B, T, H * HD)
    ref = ref @ m.out_proj.weight.T + m.out_proj.bias
    assert torch.allclose(out, ref, atol=1e-5)


def test_past_kv_prefix_path(device):
    torch.manual_seed(3)
    B, D, H, Hkv, HD = 1, 16, 2, 2, 8
    P, L = 4, 4
    m = DenoiseAttention(D, H, Hkv, HD, rope_theta=10000.0).to(device)
    hidden_all = torch.randn(B, P + L, D, device=device)
    h_prefix, h_new = hidden_all[:, :P], hidden_all[:, P:]

    # Full-sequence reference for the last L tokens.
    full_mask = torch.ones(P + L, P + L, dtype=torch.bool, device=device).view(1, 1, P + L, P + L)
    out_full = m(hidden_all, full_mask, torch.arange(P + L, device=device))[:, P:]

    # Incremental: prefix kv roped at 0..P-1, new tokens at positions P..P+L-1.
    k = (h_prefix @ m.k_proj.weight.T + m.k_proj.bias).view(B, P, Hkv, HD).transpose(1, 2)
    v = (h_prefix @ m.v_proj.weight.T + m.v_proj.bias).view(B, P, Hkv, HD).transpose(1, 2)
    k = _rope_ref(k, torch.arange(P, device=device), 10000.0)
    new_mask = torch.ones(L, P + L, dtype=torch.bool, device=device).view(1, 1, L, P + L)
    out_inc = m(h_new, new_mask, torch.arange(P, P + L, device=device), past_kv=(k, v))

    assert out_inc.shape == (B, L, D)
    assert torch.allclose(out_inc, out_full, atol=1e-5)


def test_rope_preserves_norms_and_relative_position(device):
    torch.manual_seed(4)
    B, H, T, D = 1, 2, 6, 8
    q = torch.randn(B, H, T, D, dtype=torch.float64, device=device)
    k = torch.randn(B, H, T, D, dtype=torch.float64, device=device)
    pos = torch.arange(T, device=device)

    qr, kr = apply_rope(q, k, pos, 10000.0)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), rtol=1e-12, atol=1e-12)
    assert torch.allclose(kr.norm(dim=-1), k.norm(dim=-1), rtol=1e-12, atol=1e-12)

    # <R_m q, R_n k> == <R_0 q, R_{n-m} k>: rope must depend only on n-m.
    theta = 10000.0
    for m, n in [(0, 0), (3, 5), (1, 4), (2, 6)]:
        qm = apply_rope(q, q, torch.full((T,), m, dtype=torch.long, device=device), theta)[0]
        kn = apply_rope(k, k, torch.full((T,), n, dtype=torch.long, device=device), theta)[0]
        krel = apply_rope(k, k, torch.full((T,), n - m, dtype=torch.long, device=device), theta)[0]
        assert torch.allclose((qm * kn).sum(-1), (q * krel).sum(-1), rtol=1e-9, atol=1e-9)
