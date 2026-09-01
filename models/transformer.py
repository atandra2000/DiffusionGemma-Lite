"""DiffusionGemma: dense block-diffusion denoiser (embed + blocks + canvas time + self-cond)."""
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.block import DenoiseBlock, RMSNorm
from models.mask import build_block_causal_block_mask, build_block_causal_mask
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
        assert cfg.attn_impl in ("sdpa", "eager", "flex"), f"unknown attn_impl: {cfg.attn_impl!r}"
        self.cfg = cfg
        self.grad_ckpt_every = None  # runtime knob: training loop sets from config §3 training:
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

    def _add_canvas_time(self, h, t, positions, time_steps):
        """t: (B, n_span) one entry per canvas the chunk spans; same t for that canvas's tokens."""
        canvas_ids = positions // self.cfg.canvas_len - (positions[0] // self.cfg.canvas_len)
        return h + self.time_embed(t, time_steps)[:, canvas_ids]

    def _build_mask(self, seq_len, device):
        """Block-causal mask in the flavor attn_impl consumes: bool (1,1,T,T) for
        sdpa/eager, BlockMask for flex."""
        if self.cfg.attn_impl == "flex":
            return build_block_causal_block_mask(seq_len, self.cfg.canvas_len, device)
        return build_block_causal_mask(seq_len, self.cfg.canvas_len, device=device)

    def backbone(self, input_ids, t, past_kv=None, positions=None, mask=None,
                 time_steps=None, return_kv=False):
        """Token embed + canvas-time embed + blocks -> post-final-norm hidden (B, T, D).

        Full-sequence path leaves positions/mask/past_kv unset. Decode path: positions
        are absolute, mask is the caller's (1,1,T,T_total) bool, past_kv is per-layer
        (k, v) with roped k; return_kv adds the new per-layer kvs. Under flex the
        decode chunk may pass mask=None: no block mask == full attention, which is
        exactly the all-ones decode-mask semantics."""
        T = input_ids.size(1)
        if positions is None:
            positions = torch.arange(T, device=input_ids.device)
        if mask is None:
            if past_kv is not None:
                assert self.cfg.attn_impl == "flex", \
                    "a past_kv forward needs an explicit decode mask (flex: mask=None = all-visible)"
            else:
                mask = self._build_mask(T, input_ids.device)
        h = self._add_canvas_time(self.embed(input_ids), t, positions,
                                  time_steps or self.cfg.n_diffusion_steps)
        kvs = []
        for i, block in enumerate(self.blocks):
            if self.grad_ckpt_every and i % self.grad_ckpt_every == 0 \
                    and self.training and torch.is_grad_enabled() and past_kv is None:
                h, kv = torch.utils.checkpoint.checkpoint(
                    block, h, mask, positions, return_kv=True, use_reentrant=False)
            else:
                h, kv = block(h, mask, positions,
                              past_kv=None if past_kv is None else past_kv[i], return_kv=True)
            kvs.append(kv)
        h = self.final_norm(h)
        return (h, kvs) if return_kv else h

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
        """Sampler entry point: delegates to inference.generate (sampler owns the KV cache)."""
        from inference.generate import BlockDiffusionSampler, SamplerConfig
        s_cfg = SamplerConfig(
            n_diffusion_steps=n_diffusion_steps or self.cfg.eval_diffusion_steps,
            adaptive=True if adaptive is None else adaptive)
        return BlockDiffusionSampler(self, s_cfg).generate(prompt_ids, max_new_tokens)
