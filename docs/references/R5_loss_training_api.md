# R5 — Loss & training API

> Public symbols of `training/losses.py` and `training/pretrain.py`.
> Semantics: [diffusion-core §8](../concepts/diffusion-core.md) for chunking,
> [training](../training.md) for the loop.

## `training/losses.py`

| symbol | signature | contract |
|---|---|---|
| `chunked_x0_ce` | `(hidden, embed_weight, targets, vocab_chunk: int = 8192)` | x0-CE one vocab chunk at a time; **never** materializes (B, S, V); fp32 logsumexp chunks combine into the global denominator; each chunk's bf16 logits retained for backward — `tests/test_loss.py::test_chunked_equals_eager` (atol 1e-6), `test_chunked_matches_eager_grad_direction`, `test_partial_last_chunk` |
| `chunked_p_embed` | `(h_norm, embed_weight, vocab_chunk: int = 8192)` | self-cond pre-pass `p @ E` chunked, **no_grad only** — running it under grad mode retains every chunk's logits; `test_chunked_p_embed_matches_eager` |
| `_ChunkTerms` | `torch.autograd.Function` | one chunk's fp32 logsumexp + target logit; saves bf16 logits; backward derives softmax from them — the retained-bf16-instead-of-head-recompute trade ([memory-engineering §2](../concepts/memory-engineering.md)) |

`chunked_x0_ce` inputs: `hidden (B, S, d_model)` (from
`DiffusionGemma.final_hidden`), `embed_weight (V, d_model)` (the tied E),
`targets (B, S)` = x0 ids. Returns scalar nats/token.

## `training/pretrain.py`

| symbol | signature | contract |
|---|---|---|
| `count_parameters` | `(model: nn.Module) -> tuple` | `(total, trainable)` — 343,516,160 total at production dims |
| `TrainingConfig` | dataclass, all fields in [R1](R1_config.md) | runtime knobs shared by CLI + programmatic use |
| `Pretrainer.__init__` | `(config: TrainingConfig)` | seeds, device setup (flex→sdpa fallback on CPU), VRAM estimate + guard, optimizer + SequentialLR build |
| `Pretrainer.diffusion_loss` | `(x0, step) -> Tensor` | t → q_sample → optional no-grad sc pre-pass (p=0.5) → `final_hidden` → `chunked_x0_ce`; all draws from `_step_rng(step)` |
| `Pretrainer.train_step` | `(x0, micro_step) -> Optional[float]` | NaN guard → backward(loss/accum) → at boundaries: clip 1.0, step, scheduler, zero; returns loss or None (NaN-skipped) |
| `Pretrainer.save_checkpoint` | `(step: int, tag: str = "") -> None` | weights + optimizer + scheduler + counters |
| `Pretrainer.load_checkpoint` | `(step: int) -> int` | restores all state incl. scheduler; returns the step |
| `Pretrainer.train` | `(max_steps=None, auto_resume=False) -> None` | the main loop: loader rebuild per pass with `offset_batches`, NaN-streak rollback at 5 |
| `main` | `() -> None` | CLI: `--config`, `--resume`, `--no-checkpoint`, `--no-compile`, `--dry-run` |

Per-step generator: `seed * 100_003 + step`
(`training/pretrain.py:Pretrainer._step_rng`) — the determinism keystone
([training §2](../training.md)); pinned fp64 by
`tests/test_training.py::test_checkpoint_resume_determinism`.