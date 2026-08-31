"""Self-conditioning: re-embed the model's own softmax prediction, added via zero-init W_sc."""
import torch
import torch.nn as nn


class SelfConditioning(nn.Module):
    """sc_emb = softmax(h @ E.T) @ E through a zero-init linear; identity until trained."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def embed(self, h_norm, embed_weight):
        """(B, T, D) x (V, D) -> (B, T, D): softmax-weighted mean of embedding rows."""
        p = (h_norm @ embed_weight.T).softmax(dim=-1)
        return p @ embed_weight

    def forward(self, h_norm, sc_input, embed_weight):
        """Returns h_norm + proj(sc_input); zero-init makes this exactly h_norm at init."""
        return h_norm + self.proj(sc_input)
