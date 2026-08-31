import torch
from models.diffusion import alpha_bar, corruption_probs, sample_canvas_t, q_sample, x0_ce_loss

def test_alpha_bar_monotone_and_bounded():
    t = torch.arange(1, 17)
    a = alpha_bar(t, T=16)
    assert (a <= 1).all() and (a >= 0).all() and (a.sort(descending=True).values == a).all()

def test_final_step_pure_noise():
    assert alpha_bar(torch.tensor([16]), T=16).item() < 0.01

def test_forward_process_marginals():
    V = 100
    q = corruption_probs(0.8, V, clean_token=7)
    # clean-token mass = ᾱ + (1−ᾱ)/V ; every other token = (1−ᾱ)/V ; sums to 1
    assert abs(q[7] - (0.8 + 0.2 / V)) < 1e-6
    assert abs(q[3] - 0.2 / V) < 1e-6
    assert abs(q.sum().item() - 1.0) < 1e-6

def test_sample_canvas_t_range():
    t = sample_canvas_t(batch=8, n_canvases=16, T=16, device="cpu")
    assert t.shape == (8, 16) and t.min() >= 1 and t.max() <= 16

def test_q_sample_zero_alpha_is_uniform_draw():
    g = torch.Generator().manual_seed(0)
    x0 = torch.arange(5, 10).unsqueeze(0).expand(2, 5).contiguous()
    t = torch.full((2, 1), 16)
    xt, alpha = q_sample(x0, t, canvas_len=5, n_diffusion_steps=16, vocab_size=50257, generator=g)
    assert alpha.max() < 0.01          # t=T ⇒ (near-)zero keep-prob
    assert xt.shape == x0.shape

def test_q_sample_identity_when_alpha_one():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randint(0, 100, (2, 40))
    t = torch.zeros(2, 2, dtype=torch.long)   # invalid in production; only for the identity probe
    # use the internal helper directly instead: corruption with keep-prob 1 returns x0
    from models.diffusion import _corrupt_with_alpha
    xt, _ = _corrupt_with_alpha(x0, torch.ones(2, 40, 1), vocab_size=100, generator=g)
    assert (xt == x0).all()

def test_x0_ce_loss_matches_cross_entropy():
    logits = torch.randn(2, 8, 50)
    x0 = torch.randint(0, 50, (2, 8))
    ref = torch.nn.functional.cross_entropy(logits.view(-1, 50), x0.view(-1))
    assert torch.allclose(x0_ce_loss(logits, x0), ref)