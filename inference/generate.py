"""Block-AR canvas sampler: uniform-state denoising over KV-chained finalized canvases."""
import math
from dataclasses import dataclass
from typing import Optional

import torch

from models.mask import build_canvas_decode_mask


@dataclass
class SamplerConfig:
    """Eval sampler knobs: schedule length, temperature anneal, entropy-bound stop."""

    n_diffusion_steps: int = 32
    adaptive: bool = True
    entropy_threshold: float = 1.0
    stability_steps: int = 2
    temp_start: float = 0.8
    temp_end: float = 0.4
    seed: Optional[int] = None


def _prefix_mask(prompt_len: int, canvas_len: int, device=None) -> torch.Tensor:
    """Prompt rows see all earlier canvases + their own (possibly partial) canvas.

    build_block_causal_mask asserts divisibility, so a partial first canvas is
    built here with the same row semantics."""
    idx = torch.arange(prompt_len, device=device)
    block = idx // canvas_len
    allow = (idx[None, :] < (block * canvas_len)[:, None]) | (block[:, None] == block[None, :])
    return allow.view(1, 1, prompt_len, prompt_len)


class BlockDiffusionSampler:
    """Prefills the prompt once, denoises one canvas at a time, re-encodes each
    finalized canvas into the KV cache (its all-visible last-block view)."""

    def __init__(self, model, cfg: SamplerConfig):
        self.model = model
        self.cfg = cfg
        self._gen = None

    def _generator(self, device):
        if self._gen is None and self.cfg.seed is not None:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self.cfg.seed)
        return self._gen

    def _span_t(self, prefix_len, length, batch, value, device):
        """(B, n_span) time column for the canvases the chunk straddles; one value."""
        L = self.model.cfg.canvas_len
        n_span = (prefix_len + length - 1) // L - prefix_len // L + 1
        return torch.full((batch, n_span), value, dtype=torch.long, device=device)

    def _positions(self, prefix_len, length, device):
        return torch.arange(prefix_len, prefix_len + length, device=device)

    @torch.no_grad()
    def prefill(self, prompt_ids):
        """One forward over the prompt (block-causal among prompt canvases); KV is static."""
        P = prompt_ids.size(1)
        t = self._span_t(0, P, prompt_ids.size(0), 0, prompt_ids.device)
        h, kv = self.model.backbone(prompt_ids, t, mask=_prefix_mask(P, self.model.cfg.canvas_len,
                                                                    prompt_ids.device),
                                    positions=self._positions(0, P, prompt_ids.device),
                                    time_steps=self.cfg.n_diffusion_steps, return_kv=True)
        return kv, P

    @torch.no_grad()
    def _canvas_step_logits(self, past_kv, prefix_len, canvas_xt, t, sc_input=None):
        """One denoise forward over a canvas chunk: logits over canvas rows only."""
        L = self.model.cfg.canvas_len
        mask = build_canvas_decode_mask(prefix_len, L, device=canvas_xt.device)
        tt = self._span_t(prefix_len, L, canvas_xt.size(0), t, canvas_xt.device)
        h, _ = self.model.backbone(canvas_xt, tt, past_kv=past_kv,
                                   positions=self._positions(prefix_len, L, canvas_xt.device),
                                   mask=mask, time_steps=self.cfg.n_diffusion_steps,
                                   return_kv=True)
        return self.model.head_forward(h, sc_input)

    @torch.no_grad()
    def _denoise_step(self, kv, prefix_len, x, committed, prev, t, tau, sc_input, generator):
        """One uniform-state step: forward -> commit unchanged-or-stronger posteriors ->
        re-noise the rest. Returns (x, committed, prev, entropy, sc_next, x0)."""
        V = self.model.cfg.vocab_size
        logits = self._canvas_step_logits(kv, prefix_len, x, t, sc_input)
        p = logits.softmax(-1)
        conf, am = p.max(-1)
        q = p.clamp_min(1e-12)
        entropy = (-(q * q.log()).sum(-1)).mean()
        commit = torch.zeros_like(committed) if prev is None else (am == prev[0]) | (conf >= prev[1])
        old_committed = committed
        new = commit & ~old_committed          # fresh value only for positions not yet frozen
        committed = old_committed | commit
        x0 = am if tau <= 0 else torch.multinomial(
            (logits / tau).softmax(-1).view(-1, V), 1, generator=generator).squeeze(-1).view(x.shape)
        noise = torch.randint(0, V, x.shape, device=x.device, generator=generator)
        x = torch.where(new, x0, torch.where(old_committed, x, noise))
        # sc_next: previous step's posterior re-embedded; None until self-conditioning applies
        sc_next = p @ self.model.embed.weight if self.model.selfcond is not None else None
        return x, committed, (am, conf), entropy, sc_next, x0

    @torch.no_grad()
    def denoise_canvas(self, kv, prefix_len, cfg=None):
        """Denoise one canvas from uniform noise (t=T_eval state). Returns
        (canvas_ids, steps_used, mean-entropy trace)."""
        cfg = cfg or self.cfg
        T_eval = cfg.n_diffusion_steps
        L, V = self.model.cfg.canvas_len, self.model.cfg.vocab_size
        device = kv[0][0].device
        g = self._generator(device)
        x = torch.randint(0, V, (kv[0][0].size(0), L), device=device, generator=g)
        committed = torch.zeros(kv[0][0].size(0), L, dtype=torch.bool, device=device)
        prev, sc, entropies, x0, streak = None, None, [], x, 0
        for k in range(T_eval):
            tau = cfg.temp_start - (cfg.temp_start - cfg.temp_end) / T_eval * k
            x, committed, prev, ent, sc, x0 = self._denoise_step(
                kv, prefix_len, x, committed, prev, T_eval - k, tau, sc, g)
            entropies.append(ent)
            if cfg.adaptive:
                streak = streak + 1 if float(ent) < cfg.entropy_threshold else 0
                if streak >= cfg.stability_steps:
                    break                    # entropy bond: stable low-entropy posterior
        return torch.where(committed, x, x0), len(entropies), entropies

    @torch.no_grad()
    def encode_canvas(self, kv, prefix_len, canvas_ids):
        """Re-encode a finalized canvas (its all-visible last-block view) + KV append."""
        L = self.model.cfg.canvas_len
        assert canvas_ids.size(1) == L, "sampler encodes whole canvases"
        tt = self._span_t(prefix_len, L, canvas_ids.size(0), 0, canvas_ids.device)
        h, kv = self.model.backbone(canvas_ids, tt, past_kv=kv,
                                    positions=self._positions(prefix_len, L, canvas_ids.device),
                                    mask=build_canvas_decode_mask(prefix_len, L,
                                                                  device=canvas_ids.device),
                                    time_steps=self.cfg.n_diffusion_steps, return_kv=True)
        return kv, prefix_len + L

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens):
        """Block-AR decode: prefill once, then denoise + re-encode canvas by canvas."""
        kv, prefix_len = self.prefill(prompt_ids)
        out = prompt_ids
        L = self.model.cfg.canvas_len
        for _ in range((max_new_tokens + L - 1) // L):
            canvas, _, _ = self.denoise_canvas(kv, prefix_len)
            kv, prefix_len = self.encode_canvas(kv, prefix_len, canvas)
            out = torch.cat([out, canvas], dim=1)
        return out[:, : prompt_ids.size(1) + max_new_tokens]
