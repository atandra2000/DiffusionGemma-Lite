"""Single-GPU diffusion pre-training loop for DiffusionGemma-Lite.

Per-step sequence (DESIGN §4.1): sample per-canvas t -> uniform corruption on-device
(via ``models/diffusion.py:q_sample``) -> optional no-grad self-conditioning
pre-pass producing a detached sc_input (p=0.5) -> denoiser forward ->
``training/losses.py:chunked_x0_ce`` -> one BF16 step. Data access, optimization,
checkpointing, and NaN recovery stay explicit so a smoke run and a long A100 run
share one execution path.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW

sys.path.append(str(Path(__file__).parent.parent))  # house: allow `python training/pretrain.py`
from data.dataset import build_dataloader
from models.diffusion import q_sample, sample_canvas_t
from models.transformer import DiffusionGemma, DiffusionGemmaConfig
from training.losses import chunked_p_embed, chunked_x0_ce
from utils.checkpoint import CheckpointManager
from utils.logging import TrainingLogger
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> tuple:
    """Return total and trainable parameter counts for logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class TrainingConfig:
    """Runtime settings shared by the CLI and programmatic training entry points."""

    model_config: DiffusionGemmaConfig
    data_path: str = "data/pretrain_chinchilla/shards"
    checkpoint_dir: str = "checkpoints/pretrain_a100"
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    total_steps: int = 61000
    warmup_steps: int = 2000
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_checkpoint: bool = True
    grad_checkpoint_every: int = 3
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    save_interval: int = 4000
    log_interval: int = 50
    nan_guard: bool = True
    nan_guard_max_consecutive: int = 5
    vocab_chunk: int = 8192  # DESIGN §4.0 constant (pipeline-internal, not a yaml key)
    seed: int = 42           # house seed (shared_data convention)


class Pretrainer:
    """Coordinate model setup, mixed precision, diffusion steps, and recovery."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = _DEVICE
        if not torch.cuda.is_available():
            print("[warn] CUDA not available — running on CPU (smoke-testing only).")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        self.ckpt_manager = CheckpointManager(config.checkpoint_dir)
        self.logger = TrainingLogger(
            log_every=config.log_interval, seq_len=config.model_config.max_seq_len,
            batch_size=config.micro_batch_size)
        self._opt_steps = 0
        self._micro_count = 0
        self._nan_streak = 0
        self.loss_history: List[float] = []

        self._log("Initialising DiffusionGemma-Lite...")
        if self.device.type == "cpu" and config.model_config.attn_impl == "flex":
            # flex has no CPU backward kernel; keep CPU smoke runs on the sdpa path
            self._log("[warn] FlexAttention has no CPU backward — smoke run uses attn_impl='sdpa'.")
            config.model_config.attn_impl = "sdpa"
        torch.manual_seed(config.seed)
        raw_model = DiffusionGemma(config.model_config).to(self.device)
        total, trainable = count_parameters(raw_model)
        self._log(f"Parameters: {total:,} total / {trainable:,} trainable")
        estimate = estimate_model_memory_gb(
            raw_model, config.model_config.max_seq_len, config.micro_batch_size,
            grad_checkpoint=config.grad_checkpoint)
        self._log(f"Estimated peak VRAM: {estimate:.1f} GB")
        assert_fits_in_available_gpu(estimate)
        raw_model.grad_ckpt_every = (config.grad_checkpoint_every
                                     if config.grad_checkpoint else None)  # §3 training:
        self.raw_model = raw_model

        if config.compile_model and torch.cuda.is_available() and hasattr(torch, "compile"):
            self._compile_blocks(raw_model)
        self.model = raw_model  # blocks already carry any compiled forwards

        # Tied params (head <-> embed) appear once in parameters(); dedup is compile safety.
        seen, all_params = set(), []
        for p in self.model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                all_params.append(p)
        decay_params = [p for p in all_params if p.dim() >= 2]
        no_decay_params = [p for p in all_params if p.dim() < 2]
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.lr, betas=(config.beta1, config.beta2), fused=self.device.type == "cuda")

        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=config.total_steps - config.warmup_steps,
                                   eta_min=config.lr * config.min_lr_ratio)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine],
                                      milestones=[config.warmup_steps])
        self.amp_dtype = (torch.bfloat16 if torch.cuda.is_available()
                          and torch.cuda.is_bf16_supported() else torch.float32)
        # §4.0 states the CE-chain budget at micro_bs 8; scale the vocab chunk
        # inversely with micro-batch so retained chunk-logit bytes stay constant.
        self.vocab_chunk = max(1024, config.vocab_chunk * 8 // config.micro_batch_size)

    # --- setup helpers ---------------------------------------------------------

    def _compile_blocks(self, model: DiffusionGemma) -> None:
        """Compile only the residual block (DESIGN §4.0); corruption/loss stay eager."""
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", self.config.compile_mode)
        self._log(f"Compiling residual blocks (mode={compile_mode})...")
        for block in model.blocks:
            block.forward = torch.compile(block.forward, mode=compile_mode, fullgraph=False)

    @staticmethod
    def _log(msg: str) -> None:
        print(msg)

    def _amp_context(self):
        if torch.cuda.is_available():
            return autocast("cuda", dtype=self.amp_dtype)
        return autocast("cpu", enabled=False)

    # --- the diffusion training step ------------------------------------------

    def _step_rng(self, step: int) -> torch.Generator:
        """Per-micro-step generator: identical draws before and after a resume."""
        return torch.Generator(device=self.device).manual_seed(
            self.config.seed * 100_003 + int(step))

    def diffusion_loss(self, x0: torch.Tensor, step: int) -> torch.Tensor:
        """Corrupt x0 at per-canvas t, optional detached self-cond pre-pass, chunked x0-CE.

        All draws come from the per-step generator, so a resumed run replays a
        step's (t, corruption, self-cond gate) exactly."""
        mc = self.config.model_config
        model = self.raw_model
        gen = self._step_rng(step)
        n_canvases = x0.size(1) // mc.canvas_len
        with self._amp_context():
            t = sample_canvas_t(x0.size(0), n_canvases, mc.n_diffusion_steps,
                                self.device, generator=gen)
            xt, _ = q_sample(x0, t, canvas_len=mc.canvas_len,
                             n_diffusion_steps=mc.n_diffusion_steps,
                             vocab_size=mc.vocab_size, generator=gen)
            sc_input = None
            if model.selfcond is not None \
                    and torch.rand((), device=self.device, generator=gen) < mc.self_cond_p:
                if mc.self_cond_detach:
                    with torch.no_grad():  # DESIGN §2.4: detached pre-pass, half the double cost
                        sc_input = chunked_p_embed(model.backbone(xt, t), model.embed.weight,
                                                   vocab_chunk=self.vocab_chunk)
                else:
                    sc_input = chunked_p_embed(model.backbone(xt, t), model.embed.weight,
                                               vocab_chunk=self.vocab_chunk)
            return chunked_x0_ce(model.final_hidden(xt, t, sc_input), model.embed.weight, x0,
                                 vocab_chunk=self.vocab_chunk)

    def train_step(self, x0: torch.Tensor, micro_step: int) -> Optional[float]:
        """One micro-step; optimizer updates land at accumulation boundaries."""
        cfg = self.config
        loss = self.diffusion_loss(x0, micro_step)
        loss_value = float(loss.detach())
        is_opt_step = (micro_step + 1) % cfg.gradient_accumulation_steps == 0

        if cfg.nan_guard and not math.isfinite(loss_value):
            self._log(f"[nan-guard] NaN/Inf at micro_step={micro_step}. Skipping backward.")
            self.optimizer.zero_grad(set_to_none=True)
            return None
        (loss / cfg.gradient_accumulation_steps).backward()
        if is_opt_step:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._opt_steps += 1
            self.loss_history.append(loss_value)
        return loss_value

    # --- checkpointing ----------------------------------------------------------

    def save_checkpoint(self, step: int, tag: str = "") -> None:
        """Persist weights, optimizer, scheduler, and progress metadata at a step."""
        extra_meta = {"scheduler": self.scheduler.state_dict(), "opt_steps": self._opt_steps,
                      "tag": tag or f"step_{step}"}
        self.ckpt_manager.save(self.raw_model, self.optimizer, step, extra_meta=extra_meta,
                               state_dict=self.raw_model.state_dict())

    def load_checkpoint(self, step: int) -> int:
        """Restore weights, optimizer, scheduler, and step counters; return the step."""
        meta = self.ckpt_manager.load(self.raw_model, step, device=str(self.device),
                                      optimizer=self.optimizer, strict=False)
        if "scheduler" in meta:
            self.scheduler.load_state_dict(meta["scheduler"])
        self._opt_steps = int(meta.get("opt_steps", step))
        self._micro_count = self._opt_steps * self.config.gradient_accumulation_steps
        self._log(f"Resumed from step {meta.get('step', step)}")
        return meta.get("step", step)

    # --- main loop --------------------------------------------------------------

    def train(self, max_steps: Optional[int] = None, auto_resume: bool = False) -> None:
        """Run optimizer steps until the limit, restoring after NaN streaks."""
        cfg = self.config
        total = cfg.total_steps if max_steps is None else max_steps
        if auto_resume:
            latest = self.ckpt_manager.latest_step()
            if latest is not None:
                self.load_checkpoint(latest)

        self._log(f"Training from step {self._opt_steps} to {total}")
        self.raw_model.train()
        seq_len = cfg.model_config.max_seq_len
        while self._opt_steps < total:
            loader = build_dataloader(cfg.data_path, seq_len, cfg.micro_batch_size,
                                      seed=cfg.seed, offset_batches=self._micro_count,
                                      pin_memory=torch.cuda.is_available())
            if len(loader) == 0:
                raise RuntimeError(
                    f"No complete {seq_len}-token windows for micro_batch_size "
                    f"{cfg.micro_batch_size} in {cfg.data_path}")
            for x0 in loader:
                if self._opt_steps >= total:
                    break
                x0 = x0.to(self.device, non_blocking=True)
                metrics = self.train_step(x0, self._micro_count)
                self._micro_count += 1
                if metrics is None:
                    self._nan_streak += 1
                    if self._nan_streak >= cfg.nan_guard_max_consecutive:
                        latest = self.ckpt_manager.latest_step()
                        if latest is None:
                            raise RuntimeError("NaN/Inf with no checkpoint to restore from")
                        self._log(f"[nan-guard] {self._nan_streak} consecutive NaN/Inf — "
                                  f"restoring checkpoint step {latest}.")
                        self.load_checkpoint(latest)
                    continue
                self._nan_streak = 0
                step = self._opt_steps
                if step % cfg.log_interval == 0:
                    self.logger.log(step, metrics, lr=self.scheduler.get_last_lr()[0])
                if step % cfg.save_interval == 0 and step > 0:
                    self.save_checkpoint(step)
        self.save_checkpoint(self._opt_steps, tag="final")
        self._log("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffusionGemma-Lite pre-training (single GPU)")
    parser.add_argument("--config", type=str, default="configs/pretrain_a100_380m.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--resume", type=int, default=None,
                        help="Checkpoint step to resume from (default: latest)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--dry-run", action="store_true", help="Run 2 steps to verify wiring")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        import yaml
        yaml_cfg = yaml.safe_load(f)
    yam, data_cfg = yaml_cfg.get("training", {}), yaml_cfg.get("data", {})

    config = TrainingConfig(
        model_config=DiffusionGemmaConfig.from_yaml(args.config),
        data_path=args.data_path or data_cfg.get("train_data_path", "data/pretrain_chinchilla/shards"),
        checkpoint_dir=args.checkpoint_dir or yam.get("save_dir", "checkpoints/pretrain_a100"),
        micro_batch_size=yam.get("micro_batch_size", 8),
        gradient_accumulation_steps=yam.get("gradient_accumulation_steps", 4),
        total_steps=2 if args.dry_run else yam.get("total_steps", 61000),
        warmup_steps=yam.get("warmup_steps", 2000),
        lr=yam.get("lr", 3.0e-4),
        min_lr_ratio=yam.get("min_lr_ratio", 0.05),
        weight_decay=yam.get("weight_decay", 0.1),
        beta1=yam.get("beta1", 0.9),
        beta2=yam.get("beta2", 0.95),
        grad_clip=yam.get("grad_clip", 1.0),
        grad_checkpoint=bool(yam.get("grad_checkpoint", True) and not args.no_checkpoint),
        grad_checkpoint_every=yam.get("grad_checkpoint_every", 3),
        compile_model=yam.get("compile", True) and not args.no_compile,
        compile_mode=yam.get("compile_mode", "max-autotune"),
        save_interval=yam.get("save_interval", 4000),
        log_interval=yam.get("log_interval", 50),
        nan_guard=yam.get("nan_guard", True),
        nan_guard_max_consecutive=yam.get("nan_guard_max_consecutive", 5),
    )

    trainer = Pretrainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    trainer.train(auto_resume=args.resume is None)


if __name__ == "__main__":
    main()
