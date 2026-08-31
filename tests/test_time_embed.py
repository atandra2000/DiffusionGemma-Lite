import torch
from models.time_embed import CanvasTimeEmbedding

def test_time_embed_shape_bounds_distinctness(device):
    emb = CanvasTimeEmbedding(d_model=64, time_dim=32).to(device)
    t = torch.randint(1, 5, (3, 4), device=device)
    out = emb(t, T=4)
    assert out.shape == (3, 4, 64)
    assert out.abs().max() < 100.0
    # DESIGN §7.4: t=1 and t=T distinct — explicit t on both sides, no random draw
    lo = emb(torch.full((3, 4), 1, device=device), T=4)
    hi = emb(torch.full((3, 4), 4, device=device), T=4)
    assert not torch.allclose(lo, hi)