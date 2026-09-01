# Training pipeline — the loop, determinism, and recovery

> **Canonical** for the pre-training execution path. `DIFFUSION.md` §4.1
> stays authoritative for the per-step sequence; this page walks the loop,
> the determinism recipe, and the recovery machinery.

**Depends on:** [foundations](concepts/foundations.md) §19–20 ·
[diffusion-core](concepts/diffusion-core.md) §8 ·
[self-conditioning](concepts/self-conditioning.md) §2 ·
[memory-engineering](concepts/memory-engineering.md) §1–4 ·
[data-pipeline](concepts/data-pipeline.md) §1 ·
**Read next:** [inference](inference.md) · [checkpoint-ops](guides/checkpoint-ops.md)

---

## Table of Contents

1. [The anatomy of a step](#1-the-anatomy-of-a-step)
2. [Determinism](#2-determinism)
3. [The optimizer and schedule](#3-the-optimizer-and-schedule)
4. [The NaN guard](#4-the-nan-guard)
5. [Checkpointing](#5-checkpointing)
6. [The A100 run](#6-the-a100-run)
7. [What breaks if you change this](#7-what-breaks-if-you-change-this)
8. [Glossary](#8-glossary)
9. [Interview Q&A](#9-interview-qa)

---

## 1. The anatomy of a step

`training/pretrain.py:Pretrainer.diffusion_loss` is the whole training step —
corrupt, condition, forward, chunked loss:

```python
# training/pretrain.py:Pretrainer.diffusion_loss (structure)
gen = self._step_rng(step)                    # per-micro-step generator (§2)
t = sample_canvas_t(B, n_canvases, T_train, device, generator=gen)
xt, _ = q_sample(x0, t, canvas_len, T, V, generator=gen)
if model.selfcond is not None and rand() < self_cond_p:      # p = 0.5 gate
    with torch.no_grad():                                     # DESIGN §2.4
        sc_input = chunked_p_embed(model.backbone(xt, t), model.embed.weight,
                                   vocab_chunk=self.vocab_chunk)
loss = chunked_x0_ce(model.final_hidden(xt, t, sc_input), model.embed.weight, x0,
                     vocab_chunk=self.vocab_chunk)
```

| stage | symbol | shapes (production) | notes |
|---|---|---|---|
| t draw | `models/diffusion.py:sample_canvas_t` | (B, 16) | one t per canvas, ~ U{1..16} |
| corruption | `models/diffusion.py:q_sample` | (B, 4096) | keep x0 w.p. ᾱ(t), else uniform |
| sc gate | `training/pretrain.py:Pretrainer.diffusion_loss` | scalar | p = 0.5, from the same generator |
| pre-pass | `training/losses.py:chunked_p_embed` | (B, 4096, 1024) | no_grad, detached |
| loss path | `models/transformer.py:DiffusionGemma.final_hidden` | (B, 4096, 1024) | backbone + single W_sc add |
| loss | `training/losses.py:chunked_x0_ce` | scalar | never materializes (B, S, V) |

One micro-step is `train_step`: loss → NaN check →
`(loss / grad_accum).backward()` → at accumulation boundaries: clip, step,
scheduler, zero
(`training/pretrain.py:Pretrainer.train_step`). The
micro/accum split (16×2 shipped, 8×4 in the §4.0 table) lands the same
effective batch — 131,072 tokens/step, 61,000 steps, 7.995B tokens either
way ([memory-engineering §4](concepts/memory-engineering.md)).

## 2. Determinism

Resume determinism is three cooperating mechanisms — and it is *proven*, not
hoped for:

1. **Data order**: the shuffler's permutation is fixed by
   `(seed, n_windows)`; the trainer passes `offset_batches = _micro_count`
   so a resumed run re-enters the same windows
   (`data/dataset.py:ShuffledRangeSampler`,
   [data-pipeline §3](concepts/data-pipeline.md)).
2. **Corruption draws**: every micro-step seeds a fresh generator with
   `seed * 100_003 + step` (`training/pretrain.py:Pretrainer._step_rng`) —
   t, the keep/noise draws, and the p=0.5 self-cond gate all replay exactly
   after a resume (`training/pretrain.py:Pretrainer.diffusion_loss` takes
   *all* randomness from this generator).
3. **Full state restore**: weights + optimizer moments + scheduler + step
   counters ride in the checkpoint
   (`training/pretrain.py:Pretrainer.load_checkpoint`).

Pinned by `tests/test_training.py::test_checkpoint_resume_determinism`
(bit-equal trajectories, NaN-free) and
`test_hundred_step_descent` (100-step loss descent with the real schedule
shape). The seed formula: `42 · 100_003 = 4,200,126` + step — one integer
per (run, step), no global RNG state to drift.

## 3. The optimizer and schedule

AdamW, fused on CUDA, with a tied-weight-safe param split
(`training/pretrain.py:Pretrainer.__init__`): parameters are deduped by
`id()` first (weight tying shares storage —
`tests/test_models.py::test_weight_tying_shared`), then split `dim ≥ 2 →
weight_decay 0.1` vs `dim < 2 → 0` (norms, biases, and the zero-init
`selfcond.proj` get no decay — decaying zero-init weights would drag them
*away* from the zero-init equivalence's starting point).

The schedule is `SequentialLR(LinearLR → CosineAnnealingLR)`
(`training/pretrain.py:Pretrainer.__init__`), verified numerically:

| step | lr |
|---|---|
| 0 | 3.00e-06 (1% of peak) |
| 1,000 | 1.515e-04 |
| 2,000 | 3.000e-04 (peak, end of warmup) |
| 20,000 | 2.394e-04 |
| 40,000 | 9.518e-05 |
| 61,000 | 1.500e-05 (= lr · min_lr_ratio 0.05) |

The shape is pinned by `tests/test_training.py::test_lr_schedule_shape`;
the 100-step descent by `test_hundred_step_descent`. Gradient clipping at
`grad_clip: 1.0` runs *before* `optimizer.step()`, only at accumulation
boundaries — clipping micro-gradients would scale each micro-step's
contribution differently and break the accumulation math.

## 4. The NaN guard

Two layers (`training/pretrain.py:Pretrainer.train_step` +
`Pretrainer.train`):

1. **Per micro-step**: non-finite loss → skip backward, zero grads, return
   `None`. No state update happens on that micro-step.
2. **Streak breaker**: `nan_guard_max_consecutive = 5` consecutive skipped
   micro-steps → restore the latest checkpoint and continue. No checkpoint →
   hard RuntimeError.

The guard exists because a NaN in one micro-batch is usually transient
(corruption drew an extreme t) — one skip is free, a streak means the *model*
is broken, and the correct response is rollback, not descent
(`tests/test_training.py::test_checkpoint_resume_determinism` exercises the
restore path; the guard's parameters come from the config's
`nan_guard`/`nan_guard_max_consecutive`).

## 5. Checkpointing

`utils/checkpoint.py:CheckpointManager` writes atomic checkpoints every
`save_interval = 4000` steps plus the `final` tag — steps 4000…60000 (15) +
final at 61,000 (`training/pretrain.py:Pretrainer.save_checkpoint`). Each
checkpoint holds weights (bf16, 0.69 GB) + AdamW fp32 state (4.12 GB) =
**4.81 GB**, plus scheduler state and `opt_steps` metadata; `latest_step`
skips incomplete writes
(`tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints`).
Resume reconstructs `_micro_count = opt_steps · accumulation_steps`, which
is exactly the data-loader offset (§2). Full ops: [checkpoint-ops](guides/checkpoint-ops.md).

## 6. The A100 run

The shipped config (`configs/pretrain_a100_380m.yaml`), numbers verified:

| quantity | value | derivation |
|---|---|---|
| parameters | 343,516,160 | `tests/test_models.py::test_param_count` (never "380M") |
| tokens | 7,995,392,000 ≈ 8.0B | 61,000 × 131,072 (both 8×4 and 16×2) |
| train FLOPs | 1.6479e19 (6·N·D) + ~12% self-cond = **1.8457e19** | 6·N·D rule |
| time | ~47 h at 35% MFU, ~41 h at 40% | 1.8457e19 / (312e12 · MFU) `[INFERENCE]` |
| peak VRAM | 23.6 GB (8×4+ckpt) / 55.2 GB (16×2) | [memory-engineering §5](concepts/memory-engineering.md) |
| LR | 3e-6 → 3e-4 → 1.5e-5 | §3 table |
| checkpoints | 4.81 GB × 16 (15 interval + final) | §5 |

Both layouts train the same 7.995B tokens in 61,000 steps; the 16×2 layout
drops checkpointing for MFU ([memory-engineering §4](concepts/memory-engineering.md)).
CLI: `--dry-run` (2 steps), `--no-compile`, `--no-checkpoint`, `--resume N`
(`training/pretrain.py:main`). On CPU the trainer auto-falls-back to
`attn_impl='sdpa'` (flex has no CPU backward,
`training/pretrain.py:Pretrainer.__init__`).

## 7. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| global RNG instead of per-step generators | resume replays different corruption; determinism dead | `tests/test_training.py::test_checkpoint_resume_determinism` |
| seed the generator per *epoch* | mid-epoch resume changes (t, corruption) draws | `tests/test_training.py::test_checkpoint_resume_determinism` |
| clip every micro-step | gradients scaled before accumulation; effective clip ≠ 1.0 | `tests/test_training.py::test_hundred_step_descent` |
| decay on `selfcond.proj` (dim 2) | decays the zero-init weights away from 0; zero-init equivalence eroded | `tests/test_self_conditioning.py::test_zero_init_equivalence` |
| `zero_grad` only at boundaries | micro-step grads accumulate across the NaN-skip | `training/pretrain.py:Pretrainer.train_step` NaN path |
| `min_lr_ratio: 0` | cosine floor at exactly 0 — late-training lr spikes on resume rounding | (robustness; schedule shape pinned by `test_lr_schedule_shape`) |
| rebuild the loader without `offset_batches` | data order restarts at window 0 after resume | `tests/test_data.py::test_loader_resumable_offset` |
| scheduler state omitted from the checkpoint | resumed lr ≠ pre-save lr | `tests/test_training.py::test_checkpoint_resume_determinism` |

## 8. Glossary

| symbol | meaning | code |
|---|---|---|
| micro vs opt step | micro-steps accumulate; optimizer steps at boundaries | `training/pretrain.py:Pretrainer.train_step` |
| `_step_rng` | per-micro-step generator `seed·100003 + step` | `training/pretrain.py:Pretrainer._step_rng` |
| `self_cond_p` | probability of the two-pass step (0.5) | config `self_cond_p` |
| `vocab_chunk` | CE chunk width, 8192·8/micro_bs | `training/pretrain.py:TrainingConfig.vocab_chunk` |
| `_nan_streak` | consecutive non-finite micro-steps | `training/pretrain.py:Pretrainer.train` |
| `auto_resume` | load latest checkpoint when no `--resume` given | `training/pretrain.py:Pretrainer.train` |

## 9. Interview Q&A

**Q: Walk me through one training step.**
A: Draw per-canvas t from the step generator
(`training/pretrain.py:Pretrainer._step_rng`), corrupt with `q_sample`
(keep x0 w.p. ᾱ(t) else uniform), optionally (p=0.5) run the no-grad
self-cond pre-pass `chunked_p_embed`, forward `final_hidden`, and compute
`chunked_x0_ce` — all inside autocast, all draws from the step generator.

**Q: How is resume bit-exact?**
A: Three pieces: the data permutation is fixed by (seed, n_windows) with an
offset resume (`data/dataset.py:ShuffledRangeSampler`); every micro-step
seeds its own generator `seed·100003 + step`, so t, corruption, and the
self-cond gate replay exactly; and checkpoints restore optimizer + scheduler
+ counters (`training/pretrain.py:Pretrainer.load_checkpoint`). Pinned fp64:
`tests/test_training.py::test_checkpoint_resume_determinism`.

**Q: Why is the loss divided by grad-accum steps?**
A: Each micro-step's gradients must contribute 1/accum of an optimizer step's
gradient; scaling before backward keeps the effective batch's gradient equal
to a full-batch step
(`training/pretrain.py:Pretrainer.train_step`). Clipping happens once, at
boundaries, on the accumulated gradient.

**Q: What does the NaN guard do that plain clipping doesn't?**
A: Clipping fixes exploding gradients; the NaN guard handles *non-finite*
losses where backward would poison the optimizer state. The guard skips
backward entirely, and after 5 consecutive failures rolls back to the last
checkpoint (`training/pretrain.py:Pretrainer.train`).

**Q: Why does the scheduler restore from the checkpoint?**
A: Cosine position *is* training progress: resuming at step 20,000 with a
step-0 lr would re-warm the model and corrupt the schedule
(`training/pretrain.py:Pretrainer.load_checkpoint` persists scheduler state
in `extra_meta`).

**Q: Why do norms and biases get weight_decay 0?**
A: Decaying a LayerNorm/RMSNorm weight toward zero shrinks the normalize-
and-scale capacity; standard practice, applied via the dim ≥ 2 split
(`training/pretrain.py:Pretrainer.__init__`) — with the extra house rule
that the zero-init `selfcond.proj` also gets no decay so the zero-init
equivalence isn't fought by the optimizer
(`tests/test_self_conditioning.py::test_zero_init_equivalence`).