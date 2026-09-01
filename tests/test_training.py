"""Diffusion pretraining loop: LR-schedule shape, 100-step descent, resume determinism."""
import numpy as np
import torch

from models.transformer import DiffusionGemmaConfig
from training.pretrain import Pretrainer, TrainingConfig

_VOCAB = 256


def _model_cfg(**overrides):
    values = dict(vocab_size=_VOCAB, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                  head_dim=16, ffn_dim=128, weight_tying=True, rms_norm_eps=1e-5,
                  init_std=0.02, rope_theta=10000.0, max_seq_len=128,
                  attn_impl="sdpa", canvas_len=32, n_diffusion_steps=16,
                  eval_diffusion_steps=32, corruption="uniform", alpha_schedule="cosine",
                  self_conditioning=True, self_cond_p=0.5, self_cond_detach=True,
                  time_embed_dim=32)
    values.update(overrides)
    return DiffusionGemmaConfig(**values)


def _synthetic_shards(data_dir, n_shards=2, tokens_per_shard=1024, seed=7):
    """Small-range tokens so the tiny model can actually fit them."""
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        rng.integers(0, _VOCAB, size=tokens_per_shard, dtype=np.uint32).tofile(
            data_dir / f"shard_{i:05d}.bin")
    return str(data_dir)


def _trainer(tmp_ckpt_dir, tmp_data_dir, **overrides):
    defaults = dict(total_steps=31, warmup_steps=5, lr=1e-3,
                    gradient_accumulation_steps=1, micro_batch_size=2,
                    log_interval=1000, save_interval=10)
    torch.manual_seed(0)
    return Pretrainer(TrainingConfig(
        model_config=_model_cfg(),
        data_path=str(_synthetic_shards(tmp_data_dir)),
        checkpoint_dir=str(tmp_ckpt_dir),
        **{**defaults, **overrides}))


def test_lr_schedule_shape(tmp_ckpt_dir, tmp_data_dir):
    """Linear warmup ramp, then cosine decay toward min_lr_ratio."""
    trainer = _trainer(tmp_ckpt_dir, tmp_data_dir, warmup_steps=10, total_steps=100, lr=1e-3)
    peak, floor = trainer.config.lr, trainer.config.lr * trainer.config.min_lr_ratio
    lrs = []
    for _ in range(100):
        lrs.append(trainer.scheduler.get_last_lr()[0])
        trainer.scheduler.step()
    assert all(lrs[i] < lrs[i + 1] for i in range(9)), "warmup must ramp linearly"
    assert all(lrs[i] > lrs[i + 1] for i in range(15, 99)), "cosine must decay after warmup"
    assert abs(lrs[99] - floor) < 1e-6
    assert 0 <= lrs[0] < peak


def test_hundred_step_descent(tmp_ckpt_dir, tmp_data_dir):
    """100-step smoke on tiny config with synthetic shards: loss[99] < loss[0]."""
    trainer = _trainer(tmp_ckpt_dir, tmp_data_dir, total_steps=100, warmup_steps=10,
                       lr=1e-3, gradient_accumulation_steps=1, micro_batch_size=2)
    trainer.train()
    assert len(trainer.loss_history) == 100
    assert trainer.loss_history[99] < trainer.loss_history[0]


def test_grad_checkpointing_path(tmp_ckpt_dir, tmp_data_dir):
    """Checkpointed blocks keep the training path intact: finite, descending loss."""
    trainer = _trainer(tmp_ckpt_dir, tmp_data_dir, total_steps=12, warmup_steps=2,
                       lr=2e-3, micro_batch_size=2)
    assert trainer.raw_model.grad_ckpt_every == 3  # default wiring from config
    trainer.raw_model.grad_ckpt_every = 1  # checkpoint every block
    trainer.raw_model.train()
    trainer.train()
    assert trainer.loss_history[-1] < trainer.loss_history[0]


def test_checkpoint_resume_determinism(tmp_ckpt_dir, tmp_data_dir):
    """Resume at step 10 reproduces the uninterrupted run's losses bit-exactly."""
    a = _trainer(tmp_ckpt_dir, tmp_data_dir, total_steps=31, lr=1e-3)
    a.train()  # checkpoints at steps 10, 20, 30 (+ final tag)

    b = _trainer(tmp_ckpt_dir, tmp_data_dir, total_steps=31, lr=1e-3)
    b.load_checkpoint(10)
    b.train()

    assert len(b.loss_history) == 21
    assert b.loss_history == a.loss_history[10:]
