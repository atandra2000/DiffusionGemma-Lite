#!/usr/bin/env python3
"""End-to-end GPU/CPU smoke test through the production loop (plan §4.2).

Runs ``training/pretrain.py:Pretrainer`` through real training steps:
    1. Generates synthetic binary shards in a temp directory.
    2. Runs training steps, verifying loss descent.
    3. Verifies checkpoint writing and resume integrity.
    4. Executes a 1-canvas test generation via the Block-AR sampler.

Usage:
    python scripts/e2e_gpu_smoke.py --tiny         # CPU self-check in seconds
    python scripts/e2e_gpu_smoke.py                # A100 run (full config)

Fixed/synthetic data is deliberate: a memorizable stream provides a clear
signal that training loss actually decreases and recovery succeeds.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer import DiffusionGemma, DiffusionGemmaConfig  # noqa: E402
from training.pretrain import Pretrainer, TrainingConfig  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402

GUARD_MARKERS = ("nan-guard", "rolled back", "rollback")


class _GuardCapture(logging.Handler):
    """Capture logs to verify no unwanted stability rollbacks fired during clean smoke."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def create_synthetic_shards(data_dir: Path, vocab_size: int, n_shards: int = 2, tokens: int = 4096, seed: int = 42) -> str:
    """Create flat uint32 shards for the dataloader."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        toks = rng.integers(0, vocab_size, size=tokens, dtype=np.uint32)
        toks.tofile(data_dir / f"shard_{i:05d}.bin")
    return str(data_dir)


def main() -> int:
    p = argparse.ArgumentParser(description="DiffusionGemma-Lite end-to-end smoke test")
    p.add_argument("--config", default=str(ROOT / "configs" / "pretrain_a100_380m.yaml"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--workdir", default=None, help="Directory for checkpoints/shards")
    p.add_argument("--tiny", action="store_true", help="Run fast 2-layer CPU self-check")
    args = p.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="dg_smoke_"))
    data_dir = workdir / "shards"
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    capture = _GuardCapture()
    logging.getLogger("training.pretrain").addHandler(capture)

    if args.tiny:
        model_cfg = DiffusionGemmaConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
            head_dim=16, ffn_dim=128, max_seq_len=128, canvas_len=32,
            n_diffusion_steps=4, time_embed_dim=32, self_conditioning=True,
            attn_impl="sdpa")
        steps = min(15, args.steps)
        warmup = 3
        micro_bs = 2
        accum = 1
        save_interval = 5
    else:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        t = cfg["training"]
        model_cfg = DiffusionGemmaConfig.from_yaml(args.config)
        steps = args.steps
        warmup = min(10, steps // 2)
        micro_bs = t.get("micro_batch_size", 16)
        accum = t.get("gradient_accumulation_steps", 2)
        save_interval = max(5, steps // 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and model_cfg.attn_impl == "flex":
        model_cfg.attn_impl = "sdpa"

    shards_path = create_synthetic_shards(data_dir, model_cfg.vocab_size,
                                          n_shards=4, tokens=max(16384, model_cfg.max_seq_len * micro_bs * 4))

    print(f"[smoke] workdir: {workdir}")
    print(f"[smoke] running {steps} optimizer steps (micro_bs={micro_bs}, accum={accum})...")

    train_cfg = TrainingConfig(
        model_config=model_cfg,
        data_path=shards_path,
        checkpoint_dir=str(ckpt_dir),
        micro_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        total_steps=steps,
        warmup_steps=warmup,
        lr=1.0e-3 if args.tiny else 3.0e-4,
        grad_clip=1.0,
        grad_checkpoint=False,
        compile_model=False,
        save_interval=save_interval,
        log_interval=1000)

    # 1. Run initial training phase
    trainer = Pretrainer(train_cfg)
    trainer.train(max_steps=steps)

    assert len(trainer.loss_history) == steps, \
        f"Expected {steps} losses recorded, got {len(trainer.loss_history)}"
    initial_loss = trainer.loss_history[0]
    final_loss = trainer.loss_history[-1]
    print(f"[smoke] initial loss: {initial_loss:.4f} -> final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, \
        f"Loss did not decrease on memorizable synthetic shards: {initial_loss} -> {final_loss}"

    # 2. Verify checkpoint was created
    ckpt_mgr = CheckpointManager(str(ckpt_dir))
    latest = ckpt_mgr.latest_step()
    assert latest is not None and latest > 0, "Expected at least one valid checkpoint"
    print(f"[smoke] checkpoint saved at step {latest}")

    # 3. Test resume from checkpoint
    resume_cfg = TrainingConfig(
        model_config=model_cfg,
        data_path=shards_path,
        checkpoint_dir=str(ckpt_dir),
        micro_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        total_steps=steps + 2,
        warmup_steps=warmup,
        lr=1.0e-3 if args.tiny else 3.0e-4,
        grad_clip=1.0,
        grad_checkpoint=False,
        compile_model=False,
        save_interval=100,
        log_interval=1000)
    resume_trainer = Pretrainer(resume_cfg)
    resumed_step = resume_trainer.load_checkpoint(latest)
    assert resumed_step == latest, f"Loaded step {resumed_step} != {latest}"
    resume_trainer.train(max_steps=steps + 2)
    assert resume_trainer._opt_steps == steps + 2, "Resume did not complete additional steps"
    print(f"[smoke] successfully resumed from step {latest} to {steps + 2}")

    # 4. Test generation demo with the trained model
    print("[smoke] testing 1-canvas generate demo...")
    prompt = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.canvas_len), device=device)
    gen_out = trainer.model.generate(prompt, max_new_tokens=model_cfg.canvas_len)
    assert gen_out.shape == (1, model_cfg.canvas_len * 2), \
        f"Expected generated shape (1, {model_cfg.canvas_len * 2}), got {gen_out.shape}"
    print(f"[smoke] generate output shape: {tuple(gen_out.shape)} [OK]")

    # 5. Check guards
    fired = [m for m in capture.messages if any(marker in m.lower() for marker in GUARD_MARKERS)]
    assert not fired, f"Unexpected stability guard fired: {fired[:3]}"

    print(f"[smoke] PASS: {steps} steps completed ({initial_loss:.3f} -> {final_loss:.3f}), "
          f"checkpoint/resume verified, generation verified, guards quiet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
