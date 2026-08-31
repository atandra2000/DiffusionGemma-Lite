import torch

from models.selfcond import SelfConditioning


def test_zero_init_equivalence(device):
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 8, 16
    m = SelfConditioning(D).to(device).double()
    h = torch.randn(B, T, D, dtype=torch.float64, device=device)
    h = h + 0.1  # ensure nonzero
    sc = torch.randn(B, T, D, dtype=torch.float64, device=device)
    E = torch.randn(V, D, dtype=torch.float64, device=device)
    assert torch.allclose(m(h, sc, E), h, rtol=0.0, atol=0)

    # The conditioning path must be live: a non-zero-init proj changes the output.
    with torch.no_grad():
        m.proj.weight.copy_(torch.randn(D, D, dtype=torch.float64))
    assert not torch.allclose(m(h, sc, E), h, atol=1e-6)


def test_embed_is_softmax_weighted_mean(device):
    torch.manual_seed(1)
    B, T, D, V = 2, 3, 8, 5
    E = torch.randn(V, D, dtype=torch.float64, device=device)
    h = torch.randn(B, T, D, dtype=torch.float64, device=device)
    p_ref = (h @ E.T).softmax(-1)
    out = SelfConditioning(D).embed(h, E)
    assert torch.allclose(out, p_ref @ E, atol=1e-8)
    assert torch.allclose(p_ref.sum(-1), torch.ones(B, T, dtype=torch.float64), atol=1e-12)
    assert (p_ref >= 0).all()

    # Saturation: h_norm = k * E[i] with large k makes p nearly one-hot at token i.
    k = 50.0
    h_hot = k * E[2].expand(B, T, D)
    assert torch.allclose(SelfConditioning(D).embed(h_hot, E), E[2].expand(B, T, D), atol=1e-6)


def test_sc_grad_flow(device):
    torch.manual_seed(2)
    B, T, D, V = 2, 4, 8, 16
    m = SelfConditioning(D).to(device).double()
    h = torch.randn(B, T, D, dtype=torch.float64, device=device)
    with torch.no_grad():
        sc = torch.randn(B, T, D, dtype=torch.float64, device=device)
    assert sc.grad_fn is None
    E = torch.randn(V, D, dtype=torch.float64, device=device)

    out = m(h, sc, E)
    out.sum().backward()
    assert m.proj.weight.grad is not None
    assert m.proj.weight.grad.abs().sum() > 0
