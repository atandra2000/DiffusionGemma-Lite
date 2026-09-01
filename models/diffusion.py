"""Uniform-state (D3PM-style) diffusion process: schedule, forward process, loss."""
import math
import torch
import torch.nn.functional as F

def alpha_bar(t: torch.Tensor, T: int) -> torch.Tensor:
    """Cosine schedule ᾱ(t) = cos²(π/2 · t/T) for t ∈ {1..T}; monotone 1→0."""
    return torch.cos(math.pi / 2 * (t.float() / T)) ** 2

def corruption_probs(alpha_bar_t: float, vocab_size: int, clean_token: int) -> torch.Tensor:
    """Forward-process row: ᾱ on the clean token, uniform (1−ᾱ)/V elsewhere; sums to 1."""
    q = torch.full((vocab_size,), (1.0 - alpha_bar_t) / vocab_size)
    q[clean_token] += alpha_bar_t
    return q

def sample_canvas_t(batch, n_canvases, T, device, generator=None):
    """Per-canvas diffusion timesteps, t ∈ {1..T}, shape (batch, n_canvases)."""
    return torch.randint(1, T + 1, (batch, n_canvases), device=device, generator=generator)

def _corrupt_with_alpha(x0, alpha_full, vocab_size, generator=None):
    """One forward-process step: with prob ᾱ keep x0, else draw uniform noise token."""
    noise = torch.randint(0, vocab_size, x0.shape, device=x0.device, generator=generator)
    keep = torch.rand(x0.shape, device=x0.device, generator=generator) < alpha_full.squeeze(-1)
    return torch.where(keep, x0, noise), alpha_full

def q_sample(x0, t, canvas_len, n_diffusion_steps, vocab_size, generator=None):
    """Corrupt each canvas at its own timestep; returns (xt, alpha (B, seq, 1))."""
    a_per_canvas = alpha_bar(t, T=n_diffusion_steps)              # (B, n_canvases)
    a_full = a_per_canvas.repeat_interleave(canvas_len, dim=1)    # (B, seq)
    return _corrupt_with_alpha(x0, a_full.unsqueeze(-1), vocab_size, generator=generator)

def x0_ce_loss(logits, x0):
    """Mean cross-entropy of x0-prediction logits against the clean tokens."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), x0.reshape(-1))
