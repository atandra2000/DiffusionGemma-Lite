import torch, pytest
from models.mask import (build_block_causal_mask, build_canvas_decode_mask,
                         block_causal_sdpa_attention, eager_block_causal_attention)

def test_block_causal_mask_matches_manual(device):
    B, T, H, D = 2, 1024, 4, 32
    x_q = torch.randn(B, H, T, D, device=device)
    mask = build_block_causal_mask(T, 256, device)
    out_sdpa = block_causal_sdpa_attention(x_q, x_q, x_q, mask)
    out_eager = eager_block_causal_attention(x_q, x_q, x_q, mask)
    assert torch.allclose(out_sdpa, out_eager, atol=1e-5)

def test_mask_bidirectional_within_canvas():
    mask = build_block_causal_mask(seq_len=1024, canvas_len=256)
    row = mask[0, 0, 300]          # canvas 1
    assert row[:512].all() and not row[512:].any()

def test_mask_causal_across_canvases():
    mask = build_block_causal_mask(seq_len=1024, canvas_len=256)
    assert not mask[0, 0, 600, 780:].any()

def test_mask_shape_and_dtype():
    mask = build_block_causal_mask(seq_len=512, canvas_len=256)
    assert mask.shape == (1, 1, 512, 512) and mask.dtype == torch.bool

def test_canvas_decode_mask_all_visible():
    m = build_canvas_decode_mask(prefix_len=128, canvas_len=64)
    assert m.shape == (1, 1, 64, 192) and m.all()