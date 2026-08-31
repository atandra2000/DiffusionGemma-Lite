"""Sinusoidal canvas-time embedding."""
import math
import torch
import torch.nn as nn

class CanvasTimeEmbedding(nn.Module):
    """Embeds integer canvas timesteps t ∈ {1..T} as d_model vectors."""

    def __init__(self, d_model: int, time_dim: int = 256):
        super().__init__()
        half = time_dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freq", freq)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor, T: int) -> torch.Tensor:
        """t: LongTensor (B, n_canvases) -> FloatTensor (B, n_canvases, d_model)."""
        angles = (t.float() / T).unsqueeze(-1) * self.freq     # (B, n, half)
        time = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, n, time_dim)
        return self.mlp(time)