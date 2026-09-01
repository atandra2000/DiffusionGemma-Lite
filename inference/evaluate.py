"""Headline evaluation harness (DESIGN §4.2): decode forwards, token-forwards,
wall-clock throughput, and held-out x0-NLL.

The AR-baseline side of the headline is analytic: a KV-cached AR decoder spends
exactly 1 forward pass and 1 token-forward per generated token, so
``tokens/forward`` is the checkpoint-free speedup ratio (DESIGN §3 headline).
Wall-clock ``tokens_per_sec`` is measured for our sampler; an AR checkpoint's
tokens/s needs an external baseline run (honest gap in the output when absent).
"""
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from inference.generate import BlockDiffusionSampler, SamplerConfig
from models.diffusion import q_sample, sample_canvas_t
from training.losses import chunked_x0_ce


def parse_baseline(name: str) -> SamplerConfig:
    """'fixed_T16' / 'adaptive_T32' -> SamplerConfig (DESIGN §4.2 row names)."""
    mode, steps = name.split("_T")
    return SamplerConfig(n_diffusion_steps=int(steps), adaptive=(mode == "adaptive"))


class SpeedupEvaluator:
    """Measures the three schedule rows plus the analytic AR baseline row.

    ``tokenizer`` is accepted per the DESIGN §4.2 constructor for future decode
    previews; counting itself is tokenizer-free."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @contextmanager
    def _count_forwards(self):
        """Instrument model.backbone to count forward passes and token-forwards."""
        outer, orig = self, self.model.backbone
        outer._forwards, outer._token_forwards = 0, 0

        def counting(ids, *args, **kwargs):
            outer._forwards += 1
            outer._token_forwards += ids.size(1)
            return orig(ids, *args, **kwargs)

        self.model.backbone = counting
        try:
            yield
        finally:
            self.model.backbone = orig

    @torch.no_grad()
    def _measure(self, n_samples, prompt_tokens, gen_tokens, cfg, wall_clock=True):
        model = self.model
        device = next(model.parameters()).device
        sampler = BlockDiffusionSampler(model, cfg)
        t0 = time.perf_counter()
        with self._count_forwards():
            for _ in range(n_samples):
                prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_tokens), device=device)
                sampler.generate(prompt, gen_tokens)
        seconds = time.perf_counter() - t0 if wall_clock else None
        forwards, token_forwards = self._forwards, self._token_forwards
        row = {
            "forwards": forwards,
            "token_forwards": token_forwards,
            "token_forwards_per_token": token_forwards / (n_samples * gen_tokens),
            "tokens_per_forward": n_samples * gen_tokens / forwards,
            "seconds": seconds,
            "tokens_per_sec": n_samples * gen_tokens / seconds if wall_clock else None,
        }
        return row

    def evaluate(self, n_samples=100, prompt_tokens=64, gen_tokens=1024,
                 baselines=("fixed_T16", "fixed_T32", "adaptive_T32"), wall_clock=True):
        """Run each schedule row; add the analytic AR row and the speedup ratio.

        Returns {"rows": {name -> row dict}, "speedup_vs_ar_tokens_per_forward": {...}}.
        AR row: KV-cached decode = 1 forward / token by construction (no wall-clock
        without an external checkpoint — the honest gap DESIGN §4.2(1) discloses)."""
        rows = {name: self._measure(n_samples, prompt_tokens, gen_tokens,
                                    parse_baseline(name), wall_clock)
                for name in baselines}
        rows["ar_kv_analytic"] = {
            "forwards": n_samples * gen_tokens,
            "token_forwards": n_samples * gen_tokens,
            "token_forwards_per_token": 1.0,
            "tokens_per_forward": 1.0,
            "seconds": None,
            "tokens_per_sec": None,
        }
        speedup = {name: r["tokens_per_forward"] for name, r in rows.items()
                   if name != "ar_kv_analytic"}
        return {"rows": rows, "speedup_vs_ar_tokens_per_forward": speedup}


@torch.no_grad()
def heldout_x0_nll(model, shard_path, n_windows=64, seq_len=None, seed=42):
    """Mean chunked x0-CE over held-out shard windows — the training objective at
    sampled t, no self-cond input (nats/token; directly comparable to an AR
    baseline's CE on the same shard, DESIGN §4.2 quality anchor)."""
    seq_len = seq_len or model.cfg.max_seq_len
    mc = model.cfg
    mm = np.memmap(str(shard_path), dtype=np.uint32, mode="r")
    n = len(mm) // seq_len
    if n == 0:
        raise ValueError(f"shard {shard_path} holds fewer than {seq_len} tokens")
    rng = np.random.default_rng(seed)
    picks = rng.choice(n, size=min(n_windows, n), replace=False)
    device = next(model.parameters()).device
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    for w in picks:
        x0 = torch.from_numpy(np.array(mm[w * seq_len:(w + 1) * seq_len], copy=True)
                              ).long().view(1, -1).to(device)
        t = sample_canvas_t(1, x0.size(1) // mc.canvas_len, mc.n_diffusion_steps, device,
                            generator=gen)
        xt, _ = q_sample(x0, t, canvas_len=mc.canvas_len,
                         n_diffusion_steps=mc.n_diffusion_steps,
                         vocab_size=mc.vocab_size, generator=gen)
        hidden = model.final_hidden(xt, t)
        total += float(chunked_x0_ce(hidden, model.embed.weight, x0))
    return total / len(picks)
