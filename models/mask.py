"""Block-causal attention masks and eager/sdpa attention paths."""
import math
import torch
import torch.nn.functional as F

def build_block_causal_mask(seq_len: int, canvas_len: int, device=None) -> torch.Tensor:
    """Bool mask (1,1,T,T), True = attend: causal across canvases, bidirectional within."""
    assert seq_len % canvas_len == 0, "seq_len must be a multiple of canvas_len"
    idx = torch.arange(seq_len, device=device)
    q_block = (idx // canvas_len).unsqueeze(1)      # (T, 1)
    k_pos = idx.unsqueeze(0)                        # (1, T)
    allow = (k_pos < q_block * canvas_len) | ((k_pos // canvas_len) == q_block)
    return allow.view(1, 1, seq_len, seq_len)

def build_canvas_decode_mask(prefix_len: int, canvas_len: int, device=None) -> torch.Tensor:
    """Bool mask (1,1,L,prefix_len+L): prefix + in-flight canvas all visible."""
    # Finalized prefix is fully visible; the in-flight canvas is the last block.
    shape = (1, 1, canvas_len, prefix_len + canvas_len)
    return torch.ones(shape, dtype=torch.bool, device=device)

def block_causal_sdpa_attention(q, k, v, mask):
    """Fast path: F.scaled_dot_product_attention with a bool attn_mask."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def eager_block_causal_attention(q, k, v, mask):
    """O(T^2) ground-truth path; do not replace with a fused kernel."""
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v