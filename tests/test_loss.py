"""Chunked x0-CE: numerical equivalence with the eager reference, differentiability."""
import torch

from models.diffusion import x0_ce_loss
from models.selfcond import SelfConditioning
from training.losses import chunked_p_embed, chunked_x0_ce


def test_chunked_equals_eager(device):
    torch.manual_seed(0)
    hidden = torch.randn(2, 64, 128, device=device)
    E = torch.randn(256, 128, device=device)
    x0 = torch.randint(0, 256, (2, 64), device=device)
    eager = x0_ce_loss(hidden @ E.t(), x0)
    assert torch.allclose(chunked_x0_ce(hidden, E, x0, vocab_chunk=64), eager, atol=1e-6)


def test_partial_last_chunk(device):
    """V=100 with chunk 64 leaves a 36-token final chunk — no off-by-one, no full logits."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 32, 64, device=device)
    E = torch.randn(100, 64, device=device)
    x0 = torch.randint(0, 100, (2, 32), device=device)
    eager = x0_ce_loss(hidden @ E.t(), x0)
    assert torch.allclose(chunked_x0_ce(hidden, E, x0, vocab_chunk=64), eager, atol=1e-6)
    assert torch.allclose(chunked_x0_ce(hidden, E, x0, vocab_chunk=None), eager, atol=1e-6)


def test_chunked_loss_differentiable(device):
    hidden = torch.randn(1, 16, 64, device=device, requires_grad=True)
    E = torch.randn(256, 64, device=device, requires_grad=True)
    x0 = torch.randint(0, 256, (1, 16), device=device)
    loss = chunked_x0_ce(hidden, E, x0, vocab_chunk=100)
    assert loss.requires_grad
    loss.backward()
    assert hidden.grad is not None and E.grad is not None
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(E.grad).all()


def test_chunked_matches_eager_grad_direction(device):
    """Chunked path's gradients agree with the eager reference (same sign pattern)."""
    torch.manual_seed(1)
    hidden = torch.randn(2, 16, 32, device=device)
    E = torch.randn(64, 32, device=device)
    x0 = torch.randint(0, 64, (2, 16), device=device)
    h1 = hidden.clone().requires_grad_(True)
    h2 = hidden.clone().requires_grad_(True)
    E1 = E.clone().requires_grad_(True)
    E2 = E.clone().requires_grad_(True)
    x0_ce_loss(h1 @ E1.t(), x0).backward()
    chunked_x0_ce(h2, E2, x0, vocab_chunk=16).backward()
    assert torch.allclose(h1.grad, h2.grad, atol=1e-5)
    assert torch.allclose(E1.grad, E2.grad, atol=1e-5)


def test_chunked_p_embed_matches_eager(device):
    """Pre-pass p@E ≡ SelfConditioning.embed (softmax-weighted embedding mean)."""
    torch.manual_seed(2)
    h = torch.randn(2, 8, 32, device=device)
    E = torch.randn(256, 32, device=device)
    eager = SelfConditioning(32).embed(h, E)
    chunked = chunked_p_embed(h, E, vocab_chunk=64)
    assert chunked.shape == eager.shape
    assert torch.allclose(chunked, eager, atol=1e-5)