import pytest
import torch

from inference.generate import BlockDiffusionSampler, SamplerConfig


def test_fixed_sampler_runs_full_schedule(tiny_model, device):
    cfg = SamplerConfig(n_diffusion_steps=4, adaptive=False, seed=0)
    ids = BlockDiffusionSampler(tiny_model, cfg).generate(
        torch.randint(0, 256, (1, 16), device=device), max_new_tokens=64)
    assert ids.shape == (1, 80)
    assert ids.dtype == torch.long


def test_commit_rule_monotone(tiny_model, device):
    """Hand-driven steps: once a position commits, its value is frozen forever."""
    V, L = tiny_model.cfg.vocab_size, tiny_model.cfg.canvas_len
    s = BlockDiffusionSampler(tiny_model, SamplerConfig(
        n_diffusion_steps=4, adaptive=False, temp_start=0.8, temp_end=0.4, seed=1))
    kv, prefix_len = s.prefill(torch.randint(0, V, (1, 16), device=device))
    g = torch.Generator(device=device)
    g.manual_seed(2)
    x = torch.randint(0, V, (1, L), device=device, generator=g)
    committed = torch.zeros(1, L, dtype=torch.bool, device=device)

    xs, cs, prev = [], [], None
    for t, tau in [(4, 0.8), (3, 0.6), (2, 0.4)]:
        x, committed, prev, _, _, _ = s._denoise_step(
            kv, prefix_len, x, committed, prev, t=t, tau=tau, sc_input=None, generator=g)
        xs.append(x.clone())
        cs.append(committed.clone())
    assert not cs[0].any()                                # step 1: no history, no commits
    assert cs[1].any()                                    # step 2 commits (sanity)
    assert (cs[2] >= cs[1]).all()                         # commit mask only grows
    assert torch.equal(xs[2][cs[1]], xs[1][cs[1]])        # committed values frozen verbatim


@pytest.mark.numeric
def test_block_ar_chaining_matches_single_shot(tiny_model, device):
    """Chained KV-path logits for canvas 2, step 1 == full-sequence block-causal forward."""
    model = tiny_model.double()
    s = BlockDiffusionSampler(model, SamplerConfig(
        n_diffusion_steps=4, adaptive=False, temp_start=0.0, temp_end=0.0))
    prompt = torch.randint(0, 256, (1, 32), device=device)
    kv, prefix_len = s.prefill(prompt)
    c1, _, _ = s.denoise_canvas(kv, prefix_len)           # tau=0 -> deterministic argmax
    kv, prefix2 = s.encode_canvas(kv, prefix_len, c1)
    assert prefix2 == 64

    g = torch.Generator(device=device)
    g.manual_seed(7)
    x2 = torch.randint(0, 256, (1, 32), device=device, generator=g)   # canvas-2 step-1 input
    logits_chain = s._canvas_step_logits(kv, prefix2, x2, t=4, sc_input=None)

    # Single-shot reference: one block-causal forward over prompt + final canvas1 + noise.
    full = torch.cat([prompt, c1, x2], dim=1)
    tt = torch.tensor([[0, 0, 4]], device=device)
    logits_full = model.head_forward(model.backbone(full, tt, time_steps=4))
    assert torch.allclose(logits_chain, logits_full[:, prefix2:], atol=1e-5)

def test_adaptive_stopping_fires(tiny_model, device):
    """A converged canvas (near-one-hot logits injected) stops in < the full schedule."""
    V, L = tiny_model.cfg.vocab_size, tiny_model.cfg.canvas_len

    def converged_head(h_norm, sc_input=None):
        logits = torch.full((h_norm.size(0), L, V), -50.0, device=device, dtype=h_norm.dtype)
        pos = torch.arange(L, device=device)
        logits[..., pos, pos % V] = 50.0          # paired indexing: per-position one-hot
        return logits

    tiny_model.head_forward = converged_head
    s = BlockDiffusionSampler(tiny_model, SamplerConfig(
        n_diffusion_steps=8, adaptive=True, entropy_threshold=1.0, stability_steps=2, seed=3))
    kv, prefix_len = s.prefill(torch.randint(0, V, (1, 16), device=device))
    canvas, steps_used, entropies = s.denoise_canvas(kv, prefix_len)
    assert steps_used < 8
    assert steps_used == 2                   # one step under threshold + stability step
    assert len(entropies) == steps_used
    assert canvas.shape == (1, L)


def test_adaptive_threshold_calibratable(tiny_model, device):
    """Untrained model runs the full schedule at a low threshold; a huge one stops early."""
    V = tiny_model.cfg.vocab_size
    kv, prefix_len = None, None
    for threshold, expected in [(0.1, 8), (1_000_000.0, 2)]:
        s = BlockDiffusionSampler(tiny_model, SamplerConfig(
            n_diffusion_steps=8, adaptive=True, entropy_threshold=threshold,
            stability_steps=2, seed=4))
        kv, prefix_len = s.prefill(torch.randint(0, V, (1, 16), device=device))
        _, steps_used, _ = s.denoise_canvas(kv, prefix_len)
        assert steps_used == expected


def test_adaptive_off_ignores_entropy(tiny_model, device):
    """adaptive=False must run the plain full schedule even with a huge threshold."""
    V = tiny_model.cfg.vocab_size
    s = BlockDiffusionSampler(tiny_model, SamplerConfig(
        n_diffusion_steps=8, adaptive=False, entropy_threshold=1_000_000.0, seed=5))
    kv, prefix_len = s.prefill(torch.randint(0, V, (1, 16), device=device))
    _, steps_used, _ = s.denoise_canvas(kv, prefix_len)
    assert steps_used == 8


def test_gumbel_temperature_draw(tiny_model, device):
    """The Gumbel-max draw matches softmax(logits/tau) sampling: near-argmax at low
    tau, spread across the vocab at high tau."""
    V, L = tiny_model.cfg.vocab_size, tiny_model.cfg.canvas_len
    s = BlockDiffusionSampler(tiny_model, SamplerConfig(n_diffusion_steps=4, seed=6))
    kv, prefix_len = s.prefill(torch.randint(0, V, (1, 16), device=device))
    g = torch.Generator(device=device)
    g.manual_seed(11)
    x = torch.randint(0, V, (1, L), device=device, generator=g)
    committed = torch.zeros(1, L, dtype=torch.bool, device=device)

    _, _, _, _, _, x0_low = s._denoise_step(
        kv, prefix_len, x, committed, None, t=4, tau=0.01, sc_input=None, generator=g)
    logits = s._canvas_step_logits(kv, prefix_len, x, t=4)
    assert torch.equal(x0_low, logits.argmax(-1))     # tau -> 0 collapses to argmax

    counts = torch.zeros(V, device=device)
    for _ in range(200):
        _, _, _, _, _, x0 = s._denoise_step(
            kv, prefix_len, x, committed, None, t=4, tau=20.0, sc_input=None, generator=g)
        counts += torch.bincount(x0[0], minlength=V).float()
    assert (counts > 0).float().mean() > 0.5          # high tau spreads the draw
