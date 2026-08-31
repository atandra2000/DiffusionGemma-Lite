"""DiffusionGemma: dense block-diffusion denoiser (embed + blocks + canvas time + self-cond)."""
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.block import DenoiseBlock, RMSNorm
from models.mask import build_block_causal_mask
from models.selfcond import SelfConditioning
from models.time_embed import CanvasTimeEmbedding


@dataclass
class DiffusionGemmaConfig:
    """Every key from DESIGN §3 model: block, verbatim (defaults = the 380M config)."""

    vocab_size: int = 50257
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 3072
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    rope_theta: float = 500000.0
    max_seq_len: int = 4096
    attn_impl: str = "sdpa"
    canvas_len: int = 256
    n_diffusion_steps: int = 16
    eval_diffusion_steps: int = 32
    corruption: str = "uniform"
    alpha_schedule: str = "cosine"
    self_conditioning: bool = True
    self_cond_p: float = 0.5
    self_cond_detach: bool = True
    time_embed_dim: int = 256

    @classmethod
    def from_yaml(cls, path):
        """Load the model: sub-dict of a config YAML; training:/data: sections are Phase 4."""
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])


class DiffusionGemma(nn.Module):
    """Uniform-state block-diffusion denoiser: one forward pass denoises every canvas."""

    def __init__(self, cfg: DiffusionGemmaConfig):
        super().__init__()
        assert cfg.attn_impl in ("sdpa", "eager"), f"unknown attn_impl: {cfg.attn_impl!r}"
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            DenoiseBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                         cfg.ffn_dim, cfg.rms_norm_eps, cfg.attn_impl, cfg.rope_theta)
            for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.time_embed = CanvasTimeEmbedding(cfg.d_model, cfg.time_embed_dim)
        self.selfcond = SelfConditioning(cfg.d_model) if cfg.self_conditioning else None
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # h @ E.T
        if cfg.weight_tying:
            self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                if m.weight.device.type != "meta":
                    nn.init.normal_(m.weight, std=self.cfg.init_std)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)
        # tied head shares embed's weight, so selfcond.proj is the only clobbered zero-init
        if self.selfcond is not None:
            nn.init.zeros_(self.selfcond.proj.weight)
            nn.init.zeros_(self.selfcond.proj.bias)

    def _add_canvas_time(self, h, t):
        """t: (B, n_canvases) -> embed added only to that canvas's token slice."""
        B, T, D = h.shape
        n = t.size(1)
        assert n * self.cfg.canvas_len == T, "t canvases must tile the sequence"
        emb = self.time_embed(t, self.cfg.n_diffusion_steps)  # (B, n, D)
        return (h.view(B, n, self.cfg.canvas_len, D) + emb[:, :, None, :]).view(B, T, D)

    def backbone(self, input_ids, t):
        """Token embed + canvas-time embed + blocks -> post-final-norm hidden (B, T, D)."""
        T = input_ids.size(1)
        h = self._add_canvas_time(self.embed(input_ids), t)
        mask = build_block_causal_mask(T, self.cfg.canvas_len, device=input_ids.device)
        positions = torch.arange(T, device=input_ids.device)
        for block in self.blocks:
            h = block(h, mask, positions)
        return self.final_norm(h)

    def _conditioned_hidden(self, h_norm, sc_input):
        """Exactly one W_sc add per path (zero-init proj => identity at init)."""
        if self.selfcond is not None and sc_input is not None:
            return self.selfcond(h_norm, sc_input, self.embed.weight)
        return h_norm

    def head_forward(self, h_norm, sc_input=None):
        """h += W_sc(sc_input) when conditioning, then h @ E.T -> logits (B, T, V)."""
        return self.head(self._conditioned_hidden(h_norm, sc_input))

    def forward(self, input_ids, t, sc_input=None):
        h = self.backbone(input_ids, t)
        return self.head_forward(h, sc_input)

    def final_hidden(self, input_ids, t, sc_input=None):
        """backbone + W_sc(sc_input): INCLUDES the add; chunked-CE loss path reads this."""
        return self._conditioned_hidden(self.backbone(input_ids, t), sc_input)

    def generate(self, prompt_ids, max_new_tokens, n_diffusion_steps=None, adaptive=None):
        raise NotImplementedError("sampler lands in Phase 3 (Task 9)")