#!/usr/bin/env python3
"""Peak-VRAM microbenchmark for DiffusionGemma-Lite pre-training (plan §4.2).

Runs one steady-state optimizer step at the production shape (micro_bs=16, seq=4096)
or checkpointed shape (micro_bs=8, seq=4096) — fused AdamW, BF16-autocast forward
and chunked x0-CE backward — and reports the CUDA allocator peak.

Gates (DESIGN §4.0 budget table & DIFFUSION.md §5):
    Production (micro_bs=16, seq=4096, no grad-ckpt): peak VRAM < 55.0 GB (fits in 80GB A100)
    Checkpointed (micro_bs=8, seq=4096, grad-ckpt every 3): peak VRAM < 15.0 GB

A warmup step runs first to materialize AdamW's lazily allocated state before the
peak counter is reset; without it the reading misses optimizer moments and passes
vacuously. torch.compile stays off: eager is the conservative allocator upper bound.

Usage:
    python scripts/microbench_a100.py                  # production layout (< 55 GB gate)
    python scripts/microbench_a100.py --checkpointed   # 8x4 + ckpt layout (< 15 GB gate)
    python scripts/microbench_a100.py --tiny           # CPU self-check on tiny model

On a CUDA-less machine the gate is unmeasurable, not violated: the script prints
why and exits 0 (house pattern, cf. LLaMA-3-Lite / HiLS-Attention-Lite).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import DiffusionGemma, DiffusionGemmaConfig  # noqa: E402
from training.pretrain import Pretrainer, TrainingConfig  # noqa: E402
from utils.memory import estimate_model_memory_gb  # noqa: E402


def run_benchmark(config_path: str, checkpointed: bool = False, tiny: bool = False) -> int:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = cfg["training"]

    if tiny:
        model_cfg = DiffusionGemmaConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
            head_dim=16, ffn_dim=128, max_seq_len=128, canvas_len=32,
            n_diffusion_steps=4, time_embed_dim=32, self_conditioning=True,
            attn_impl="sdpa")
        micro_bs = 2
        seq_len = 128
        gate_gb = 5.0
        grad_ckpt = False
    else:
        model_cfg = DiffusionGemmaConfig.from_yaml(config_path)
        if checkpointed:
            micro_bs = 8
            seq_len = model_cfg.max_seq_len
            gate_gb = 15.0
            grad_ckpt = True
        else:
            micro_bs = t.get("micro_batch_size", 16)
            seq_len = model_cfg.max_seq_len
            gate_gb = 55.0
            grad_ckpt = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and model_cfg.attn_impl == "flex":
        model_cfg.attn_impl = "sdpa"

    train_cfg = TrainingConfig(
        model_config=model_cfg,
        micro_batch_size=micro_bs,
        gradient_accumulation_steps=1,
        total_steps=10,
        warmup_steps=2,
        lr=3.0e-4,
        grad_clip=1.0,
        grad_checkpoint=grad_ckpt,
        compile_model=False,
        checkpoint_dir="/tmp/dg_microbench_ckpt")

    trainer = Pretrainer(train_cfg)
    model = trainer.model

    x0 = torch.randint(0, model_cfg.vocab_size, (micro_bs, seq_len), device=device)

    # Warmup step to materialize optimizer state
    trainer.train_step(x0, micro_step=0)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed / measured step
    trainer.train_step(x0, micro_step=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak_reserved = 0.0
        peak_alloc = 0.0

    est = estimate_model_memory_gb(model, seq_len, micro_bs, grad_checkpoint=grad_ckpt)

    print(f"[microbench] layout: micro_bs={micro_bs}, seq={seq_len}, grad_ckpt={grad_ckpt}")
    if device.type == "cuda":
        ok = peak_reserved < gate_gb
        print(f"[microbench] peak VRAM: {peak_reserved:.1f} GB reserved "
              f"({peak_alloc:.1f} GB allocated); estimator predicted {est:.1f} GB")
        print(f"[microbench] {'PASS' if ok else 'FAIL'} < {gate_gb:.1f} GB")
        return 0 if ok else 1
    else:
        print(f"[microbench] (CPU proxy) estimator predicted peak VRAM: {est:.1f} GB (gate < {gate_gb:.1f} GB)")
        print("[microbench] PASS (CPU self-check completed)")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DiffusionGemma-Lite peak-VRAM microbenchmark (plan §4.2)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_380m.yaml"))
    parser.add_argument("--checkpointed", action="store_true",
                        help="Benchmark the 8x4 + checkpoint layout (gate < 15 GB)")
    parser.add_argument("--tiny", action="store_true",
                        help="Run CPU self-check on a tiny model")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.tiny:
        print("no CUDA device — VRAM gates are unmeasurable here; "
              "run on the A100 pod (python scripts/microbench_a100.py).")
        return 0

    return run_benchmark(args.config, checkpointed=args.checkpointed, tiny=args.tiny)


if __name__ == "__main__":
    sys.exit(main())
