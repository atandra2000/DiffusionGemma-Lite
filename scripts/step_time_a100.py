#!/usr/bin/env python3
"""Step-time / MFU benchmark at the production optimizer-step shape (plan §4.2).

One timed unit = one optimizer step of the production loop: ``grad_accum``
micro-batches (BF16-autocast forward/backward at micro_bs×seq) + grad clip +
fused AdamW.

FLOPs convention: 6·N per token (2·N fwd + 4·N bwd) over unique (untied)
parameters + ~12% self-conditioning pre-pass factor (configs/pretrain_a100_380m.yaml).
Gate: MFU ≥ 33% on an A100 80GB (312 TFLOPS BF16 peak); expect ~35–40%.

Usage:
    python scripts/step_time_a100.py [--compile]   # --compile matches the production loop
    python scripts/step_time_a100.py --tiny        # CPU self-check proxy

On CPU the script prints a tokens/sec proxy and exits 0 (MFU undefined).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import DiffusionGemma, DiffusionGemmaConfig  # noqa: E402
from training.pretrain import Pretrainer, TrainingConfig  # noqa: E402

MFU_GATE = 33.0
A100_BF16_TFLOPS = 312.0


def n_unique_params(model: torch.nn.Module) -> int:
    """Parameter count without double-counting tied weights (embed <-> head)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


def main() -> int:
    p = argparse.ArgumentParser(description="DiffusionGemma-Lite step-time / MFU benchmark (plan §4.2)")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_380m.yaml"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true",
                   help="compile blocks like the production loop (fair MFU, slow warmup)")
    p.add_argument("--tiny", action="store_true",
                   help="Run CPU self-check on a tiny model")
    args = p.parse_args()

    if not torch.cuda.is_available() and not args.tiny:
        print("no CUDA device — MFU is undefined here; run on the A100 pod "
              "(python scripts/step_time_a100.py --compile).")
        return 0

    device = torch.device("cpu" if args.tiny else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if args.tiny:
        model_cfg = DiffusionGemmaConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
            head_dim=16, ffn_dim=128, max_seq_len=128, canvas_len=32,
            n_diffusion_steps=4, time_embed_dim=32, self_conditioning=True,
            attn_impl="sdpa")
        micro_bs = 2
        seq_len = 128
        accum = 1
        grad_clip = 1.0
        grad_ckpt = False
        steps = min(5, args.steps)
        warmup = min(2, args.warmup)
    else:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        t = cfg["training"]
        model_cfg = DiffusionGemmaConfig.from_yaml(args.config)
        micro_bs = t.get("micro_batch_size", 16)
        seq_len = model_cfg.max_seq_len
        accum = t.get("gradient_accumulation_steps", 2)
        grad_clip = t.get("grad_clip", 1.0)
        grad_ckpt = t.get("grad_checkpoint", False)
        steps = args.steps
        warmup = args.warmup

    if device.type == "cpu" and model_cfg.attn_impl == "flex":
        model_cfg.attn_impl = "sdpa"

    train_cfg = TrainingConfig(
        model_config=model_cfg,
        micro_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        total_steps=100,
        warmup_steps=10,
        lr=3.0e-4,
        grad_clip=grad_clip,
        grad_checkpoint=grad_ckpt,
        compile_model=args.compile and device.type == "cuda",
        checkpoint_dir="/tmp/dg_step_time_ckpt")

    trainer = Pretrainer(train_cfg)
    model = trainer.model
    n = n_unique_params(model)

    tokens_per_step = seq_len * micro_bs * accum
    x0 = torch.randint(0, model_cfg.vocab_size, (micro_bs, seq_len), device=device)

    def optimizer_step(step_idx: int) -> None:
        for a in range(accum):
            trainer.train_step(x0, micro_step=step_idx * accum + a)

    # Warmup steps
    for w in range(warmup):
        optimizer_step(w)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for s in range(steps):
            optimizer_step(warmup + s)
        end.record()
        torch.cuda.synchronize()
        sec_per_step = start.elapsed_time(end) / 1000.0 / steps
    else:
        t0 = time.perf_counter()
        for s in range(steps):
            optimizer_step(warmup + s)
        sec_per_step = (time.perf_counter() - t0) / steps

    tps = tokens_per_step / sec_per_step
    sc_factor = 1.0 + (0.5 * 2.0 / 6.0 if model_cfg.self_conditioning else 0.0)  # ~1.167 or ~1.12
    tflops = 6.0 * n * tps * sc_factor / 1e12
    mfu = tflops / A100_BF16_TFLOPS * 100

    print(f"[step_time] shape (ms={micro_bs}, seq={seq_len}, accum={accum}) "
          f"= {tokens_per_step:,} tokens/optimizer-step")
    print(f"[step_time] {sec_per_step:.3f} s/step, {tps:,.0f} tokens/sec")

    if device.type == "cuda":
        print(f"[step_time] {tflops:.1f} TFLOPS (6·N·sc convention)")
        # The 33% gate and the 312 TFLOPS peak are A100 numbers (see the
        # module docstring). On any other card the ratio is not MFU and the
        # floor is meaningless — an RTX A4000 reports ~0% against this scale
        # and the script would fail a perfectly good run. Report and pass.
        gpu_name = torch.cuda.get_device_name(device)
        is_a100 = "A100" in gpu_name
        if is_a100:
            print(f"[step_time] MFU ~{mfu:.1f}% [{'PASS' if mfu >= MFU_GATE else 'FAIL'} >= {MFU_GATE:.0f}%]")
            return 0 if mfu >= MFU_GATE else 1
        print(f"[step_time] MFU ~{mfu:.1f}% (vs A100 peak; no gate on {gpu_name})")
        return 0
    else:
        print(f"[step_time] (CPU proxy) params={n:,}, TFLOPS proxy={tflops:.4f}")
        print("[step_time] PASS (CPU self-check completed)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
