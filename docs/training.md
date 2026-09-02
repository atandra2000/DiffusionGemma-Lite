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
2. [Worked example: one toy micro-step, end to end](#2-worked-example-one-toy-micro-step-end-to-end)
3. [The chunked loss and the §4.0 memory model](#3-the-chunked-loss-and-the-§40-memory-model)
4. [Determinism](#4-determinism)
5. [The optimizer and the LR schedule](#5-the-optimizer-and-the-lr-schedule)
6. [The NaN guard](#6-the-nan-guard)
7. [Checkpointing](#7-checkpointing)
8. [Instrumentation: logging and the VRAM pre-flight](#8-instrumentation-logging-and-the-vram-pre-flight)
9. [The A100 run](#9-the-a100-run)
10. [What breaks if you change this](#10-what-breaks-if-you-change-this)
11. [Glossary](#11-glossary)
12. [Interview Q&A](#12-interview-qa)

---

## 1. The anatomy of a step

The loop runs on two clocks. A **micro-step** pushes one micro-batch through
corrupt → forward → loss → backward; an **optimizer step** lands only every
`gradient_accumulation_steps` micro-steps, and `total_steps`/`save_interval`/
`log_interval` all count *optimizer* steps. The shipped layout is micro_bs 16 ×
accum 2 (the §4.0 table's 8 × 4 lands the same effective batch): one optimizer
step = 32 windows × 4096 tokens = 131,072 tokens.

`training/pretrain.py:Pretrainer.diffusion_loss` is the whole training objective —
corrupt, condition, forward, chunked loss:

```python
# training/pretrain.py:Pretrainer.diffusion_loss (structure)
gen = self._step_rng(step)                    # per-micro-step generator (§4)
t = sample_canvas_t(B, n_canvases, T_train, device, generator=gen)
xt, _ = q_sample(x0, t, canvas_len, T, V, generator=gen)
if model.selfcond is not None and rand() < self_cond_p:      # p = 0.5 gate
    with torch.no_grad():                                     # DESIGN §2.4
        sc_input = chunked_p_embed(model.backbone(xt, t), model.embed.weight,
                                   vocab_chunk=self.vocab_chunk)
loss = chunked_x0_ce(model.final_hidden(xt, t, sc_input), model.embed.weight, x0,
                     vocab_chunk=self.vocab_chunk)
```

Stage by stage, in production shapes (micro_bs 16, seq 4096, 16 canvases/row,
`d_model` 1024, `V` 50,257):

| stage | symbol | shape (production) | notes |
|---|---|---|---|
| t draw | `models/diffusion.py:sample_canvas_t` | (16, 16) | one t per canvas, ~ U{1..16} |
| corruption | `models/diffusion.py:q_sample` | (16, 4096) | keep x0 w.p. ᾱ(t), else uniform |
| sc gate | `training/pretrain.py:Pretrainer.diffusion_loss` | scalar | p = 0.5, from the same generator |
| pre-pass | `training/losses.py:chunked_p_embed` | (16, 4096, 1024) | no_grad, detached |
| loss path | `models/transformer.py:DiffusionGemma.final_hidden` | (16, 4096, 1024) | backbone + single W_sc add |
| loss | `training/losses.py:chunked_x0_ce` | scalar | never materializes (16, 4096, 50257) |

Why each stage is shaped the way it is:

- **Per-canvas `t`, not per-sequence `t`.** A 16-canvas row carries 16
  independent corruption levels, so one micro-batch samples the t-marginal 256
  times instead of 32 — heavy-noise regimes get trained many times per step,
  at zero extra cost.
- **Single-step corruption, not an iterated chain.** `q_sample` draws
  `xt | x0, t` directly from the closed form `q(xt|x0)` (row
  `models/diffusion.py:corruption_probs`: ᾱ on the clean token,
  (1−ᾱ)/V elsewhere) — iterating the Markov chain would give the *same*
  marginal at T times the draw cost.
- **Targets are x0, not xt.** The sampler's commit rule needs the per-position
  posterior `p(x0 | xt, t)` in one softmax; an xt-head would force trajectory
  marginalization at eval. (The planning docs' `xt` is a known typo — Ruling 19.)
- **The self-cond gate draws from the same generator** — its *position in the
  draw stream* is part of the determinism contract (§4).
- **Everything under autocast.** `self._amp_context()` wraps the whole
  computation in bf16 autocast on CUDA
  (`training/pretrain.py:Pretrainer._amp_context`); parameters stay fp32, and
  bf16's fp32-wide exponent range needs no loss scaler (fp16 would need
  `GradScaler`).

One micro-step is `train_step`: loss → NaN check →
`(loss / grad_accum).backward()` → at boundaries: clip, step, scheduler, zero
(`training/pretrain.py:Pretrainer.train_step`). The boundary test is
`(micro_step + 1) % accumulation_steps == 0` on the *global* micro counter —
which is why resume reconstructs the phase from `_opt_steps` alone (§4).

## 2. Worked example: one toy micro-step, end to end

Toy config (small enough to hand-check every shape): micro_bs B = 2,
seq_len 512, canvas_len 256 → 2 canvases per row; train-time T = 8; toy vocab
V = 100; `d_model` 64; `vocab_chunk` 40; accum 2; micro-step index 17.

**The generator.** `_step_rng(17)` builds a fresh `torch.Generator` seeded
`42 · 100_003 + 17 = 4,200,143` (`training/pretrain.py:Pretrainer._step_rng`);
every draw of this micro-step — the t's, the keep/uniform corruption draws,
the self-cond coin — comes out of this one sequential stream.

**The schedule.** `models/diffusion.py:alpha_bar` gives ᾱ(t) = cos²(π/2 · t/8);
at T = 8 the eight levels are (production T = 16 spans the same 1 → 0 range,
ᾱ(1) = 0.990 → ᾱ(16) = 0):

| t | ᾱ(t) | expected kept of 256 tokens | regime learned |
|---|---|---|---|
| 1 | 0.962 | 246 | light correction |
| 2 | 0.854 | 218 | light correction |
| 3 | 0.691 | 177 | moderate repair |
| 4 | 0.500 | 128 | half-noise |
| 5 | 0.309 | 79 | heavy repair |
| 6 | 0.146 | 38 | mostly synthesis |
| 7 | 0.038 | 10 | near-pure synthesis |
| 8 | 0.000 | 0 | unconditional canvas |

**Draw t.** Suppose the generator yields t = (3, 7) for row 0's canvases and
(5, 1) for row 1's. `q_sample` broadcasts ᾱ across each canvas and draws
keep/uniform per token: (0,0) keeps ~177 real tokens, (0,1) ~10, (1,0) ~79,
(1,1) ~246 — two rows covered four corruption levels, the per-canvas t
variance-reduction choice at work (§8.3 of DIFFUSION.md).

**Self-cond gate.** The next stream draw is u ~ U(0,1); suppose u = 0.23 <
0.5, so the gate fires. `model.backbone(xt, t)` runs once under `no_grad`;
`chunked_p_embed` re-embeds its posteriors — chunks [0,40), [40,80), [80,100),
each contributing exp(chunk_logits − lse) @ E[chunk] — into a (2, 512, 64)
fp32 tensor that enters the loss pass **detached**: an input constructor, not
a gradient path (DESIGN §2.4).

**Loss and backward.** `model.final_hidden(xt, t, sc_input)` gives
(2, 512, 64). `chunked_x0_ce` runs three `_ChunkTerms` calls — bf16 logits
(2, 512, 40), an fp32 logsumexp, a masked target gather per chunk — whose
lse's combine per position into the global denominator while the gathers sum
to exactly one target logit (§3); the result is mean CE over 1024 positions.
**Sanity anchor:** at initialization the logits are near-uniform, so lse ≈
ln V — the toy model's first losses sit near ln 100 ≈ 4.61, the production
model's near ln 50,257 ≈ 10.82; a first-logged loss far below that means
something leaks the answer. `train_step` then scales by 1/accum —
`(loss / 2).backward()` — because two micro-batches must jointly equal one
full-batch gradient at the next boundary; `_ChunkTerms.backward` rebuilds
each chunk's softmax from the *saved* bf16 logits (§3). No optimizer step
here; micro-step 18 finishes the window.

## 3. The chunked loss and the §4.0 memory model

### 3.1 The math

Per position, the x0 cross-entropy is `L = lse(z) − z_y` with `z` the
full-vocab logits and `y` the clean token. The chunk trick is pure log-sum-exp
associativity — partition the vocabulary into chunks C_j and

```
lse(z) = log Σ_v exp z_v = log Σ_j exp(lse_j)      # lse_j = lse over chunk j
```

`training/losses.py:chunked_x0_ce` computes each `lse_j` in fp32 from bf16
chunk logits and takes one final logsumexp over the stack — algebraically
identical to the full-vocab lse. The target logit is assembled the same way:
each chunk gathers its target logit where the target is in-range and zeros
elsewhere, and the parts sum to exactly one nonzero term
(`tests/test_loss.py::test_partial_last_chunk` covers the ragged last chunk).
The eager reference is `models/diffusion.py:x0_ce_loss`; parity is pinned at
atol 1e-6 by `tests/test_loss.py::test_chunked_equals_eager`.

The gradient of `L` w.r.t. the logits is the classic
`∂L/∂z_v = p_v − 1[v = y]` (scaled by 1/(B·S) by the trailing `.mean()`).
`training/losses.py:_ChunkTerms` implements exactly this split in backward:
`p · grad_lse` plus the target term scatter-added into the target column —
out-of-chunk targets arrive with zeroed `grad_tgt`, so the clamped scatter
only adds zeros.

### 3.2 Why the bf16 logits are *kept*

The naive step materializes full-vocab logits `(B, S, V)` — 8·4096·50257·4 ≈
6.6 GB of fp32 at micro_bs 8. Two cheaper designs exist: **checkpoint the
head GEMM** (`torch.utils.checkpoint` — retain nothing, but re-run the logits
GEMM in every backward), or **retain the chunk logits in bf16** (what ships):
each chunk's 2-byte logits stay alive for backward, `_ChunkTerms.backward`
derives the softmax from them, and the head GEMM runs exactly once per step.
The retained chain is B·S·V·2 ≈ 3.3 GB at micro_bs 8
(`tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate`) and
buys back that wasted recompute; `chunked_x0_ce` stays differentiable
end-to-end (`tests/test_loss.py::test_chunked_loss_differentiable`).

### 3.3 Chunk sizing

`training/pretrain.py:Pretrainer.__init__` sets
`vocab_chunk = max(1024, 8192 · 8 // micro_bs)` — 8192 at micro_bs 8, 4096 at
the shipped 16. What the inverse scaling pins constant is the *transient* fp32
chunk, B·S·chunk·4 (8·4096·8192·4 ≈ 1.07 GB at either micro-batch); the
retained bf16 chain, B·S·V·2, scales with the micro-batch like any activation
(3.3 GB at 8×4, 6.6 GB at 16×2) — bytes the 16×2 layout spends to drop
gradient checkpointing entirely
([memory-engineering §4](concepts/memory-engineering.md)).

### 3.4 The self-cond pre-pass hazard

Re-embedding the posterior (`p @ E`) needs the full-vocab softmax; naively
materialized at (8, 4096, 50257) fp32 it alone breaks the §4.0 budget.
`training/losses.py:chunked_p_embed` splits it — per chunk, fp32 logits →
`exp(chunk_logits − lse)` weights → weighted sum against E[chunk] — and the
per-chunk weights sum to the global softmax because each chunk is normalized
by the shared global lse. It runs under `torch.no_grad()`; under grad mode it
would retain *every* chunk's logits, which is why the docstring forbids it as
a differentiable path (the eager reference is
`models/selfcond.py:SelfConditioning`; parity:
`tests/test_loss.py::test_chunked_p_embed_matches_eager`).

### 3.5 The estimator and the pre-flight guard

`utils/memory.py:estimate_model_memory_gb` encodes the §4.0 budget as four
terms plus overhead (per-parameter byte counts, N = 343,516,160):

| term | bytes | production value |
|---|---|---|
| parameters | 4·N (fp32 master) | ~1.4 GB |
| optimizer | 12·N (moments + fp32 master — conservative) | ~4.1 GB |
| activations | ckpt on: boundaries only, n_layers·S·B·d_model·2; off: + SwiGLU n_layers·3·S·B·ffn·2 | ~30 GB with the shipped 16×2 no-ckpt layout |
| CE chain | B·S·V·2 retained + B·S·chunk·4 transient | 6.6 + 1.1 GB at 16×2 |
| overhead | min(13.7, max(2.0, 0.17·total)) | ~13.6 GB on an 80 GB A100 |

The 12-byte optimizer line is a *budget*, not a measurement: the real fused
AdamW state_dict is two fp32 moments (8 B/param) because the fp32 parameters
are their own master copy. The estimate is monotone in batch
(`tests/test_utils.py::test_memory_estimator_monotone_in_batch`), printed at
startup, and turned into a hard pre-flight raise by
`utils/memory.py:assert_fits_in_available_gpu` when it leaves less than a
2 GB margin; a failed probe logs a warning instead of failing silently
(`tests/test_utils.py::test_gpu_guard_noops_without_cuda`).

## 4. Determinism

Resume determinism is three cooperating mechanisms — *proven*, not hoped for:

1. **Data order**: the shuffler's permutation is fixed by `(seed, n_windows)`;
   the trainer passes `offset_batches = _micro_count` so a resumed run
   re-enters the same windows (`data/dataset.py:ShuffledRangeSampler`,
   [data-pipeline §3](concepts/data-pipeline.md)). The permutation is
   `np.random.default_rng(seed).permutation(n_windows)` — a function of the
   seed and window count only, never of *how far* training got, so "re-enter
   at position k" is exact. `build_dataloader` converts batches to windows
   (`offset = offset_batches · batch_size`) and the offset wraps mod
   `n_windows`, so the sampler stays finite and long runs cycle the same
   permutation; the trainer rebuilds the loader on every outer `while`
   iteration with the current `_micro_count`
   (`training/pretrain.py:Pretrainer.train`), stitching resumed runs onto the
   same stream. Windows are flat seq_len-token slices
   (`data/dataset.py:ShardWindows`) — no +1 AR shift: diffusion reconstructs
   x0 *everywhere*.
2. **Corruption draws**: every micro-step seeds a fresh generator with
   `seed * 100_003 + step` (`training/pretrain.py:Pretrainer._step_rng`) —
   t, the keep/noise draws, and the p=0.5 self-cond gate all replay exactly
   after a resume (`training/pretrain.py:Pretrainer.diffusion_loss` takes
   *all* randomness from this generator). This beats a global RNG cursor on
   every axis: no device-resident state to save or drift, the draw for step k
   depends only on k, and no per-epoch reseeding exists that a mid-epoch
   resume could desynchronize; seed 42 gives 4,200,126 + step — one distinct
   integer per (run, micro-step). The draw *order* inside `diffusion_loss` is
   itself part of the contract: t, then corruption, then the gate —
   reordering the draws changes which steps get self-conditioning.
3. **Full state restore**: weights + optimizer moments + scheduler + step
   counters ride in the checkpoint
   (`training/pretrain.py:Pretrainer.load_checkpoint`). `_opt_steps` comes
   from `extra_meta["opt_steps"]`; `_micro_count` is reconstructed as
   `_opt_steps · accumulation_steps` — exactly the data-loader offset *and*
   exactly the accumulation-boundary phase, so both clocks restart in step.

Pinned by `tests/test_training.py::test_checkpoint_resume_determinism`
(bit-equal trajectories, NaN-free) and `test_hundred_step_descent` (100-step
loss descent with the real schedule shape). One honest caveat (SDD ledger
Minor 3): NaN-guard skips advance `_micro_count` without an optimizer step,
and a rollback rewinds counters while the data iterator keeps streaming —
bit-equality holds for NaN-free runs, exactly what the test asserts.

## 5. The optimizer and the LR schedule

AdamW, fused on CUDA, with a tied-weight-safe param split
(`training/pretrain.py:Pretrainer.__init__`): parameters are deduped by
`id()` first (weight tying shares storage —
`tests/test_models.py::test_weight_tying_shared`), then split `dim ≥ 2 →
weight_decay 0.1` vs `dim < 2 → 0` (norms, biases, and the zero-init
`selfcond.proj` get no decay — decaying zero-init weights would drag them
*away* from the zero-init equivalence's starting point). The dedup is compile
safety — `parameters()` yields a tied tensor once today, but a refactor
surfacing the alias twice would apply two AdamW updates to one shared
storage. Fused AdamW runs one kernel per param group;
`fused=self.device.type == "cuda"` falls back to the unfused path on CPU.

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

The cosine leg is `lr(s) = η_min + (η_peak − η_min)/2 · (1 + cos(π·(s − w)/(S − w)))`
with w = 2,000, S = 61,000, η_min = 3e-4·0.05. Check s = 20,000:
(s−w)/(S−w) = 18,000/59,000 = 0.3051, cos(π·0.3051) ≈ 0.575, so lr ≈
1.5e-5 + 2.85e-4 · 0.7875 ≈ 2.394e-4 — the table row is the formula, not a
measurement. Warmup avoids large immediate updates while the Adam moments are
still cold. The 5% floor (`min_lr_ratio 0.05`) keeps the cosine tail away
from exact-zero lr, where resume rounding can spike.

The shape is pinned by `tests/test_training.py::test_lr_schedule_shape`; the
100-step descent by `test_hundred_step_descent`. Gradient clipping at
`grad_clip: 1.0` runs *before* `optimizer.step()`, only at accumulation
boundaries — clipping micro-gradients would scale each micro-step's
contribution differently and break the accumulation math.

## 6. The NaN guard

Two layers (`training/pretrain.py:Pretrainer.train_step` +
`Pretrainer.train`):

1. **Per micro-step**: non-finite loss → skip backward, zero grads, return
   `None`. No state update happens on that micro-step.
2. **Streak breaker**: `nan_guard_max_consecutive = 5` consecutive skipped
   micro-steps → restore the latest checkpoint and continue. No checkpoint →
   hard RuntimeError.

Both halves are needed; the details are load-bearing:

- **`zero_grad(set_to_none=True)` in the NaN path discards the whole
  accumulation window**, not just the poisoned micro-batch: non-finite
  gradients would have entered the running sum, and the window's other
  micro-batches cannot be salvaged. (This is also why the guard checks the
  loss *before* `backward()`.)
- **The optimizer state is never touched by a guarded step** — no `step()`,
  no `scheduler.step()`, no counters advance except `_micro_count`. That is
  what makes the skip "free": Adam's m/v are exactly as they were before the
  bad micro-batch.
- **The trigger is the loss value only.** A finite-loss / infinite-gradient
  step would pass the check and poison the moments through
  `optimizer.step()`; the guard covers loss-side non-finiteness — the observed
  failure mode, not every conceivable one.
- **Why rollback after 5, not descent anyway**: one NaN is almost always
  transient — the corruption draw landed on an extreme t. Five in a row means
  the *model* is broken (diverging weights); go back to the last known-good
  state and let the (different) data continue. `load_checkpoint` restores the
  optimizer moments too, so the rollback is true time-travel, not a
  weight-only rewind
  (`tests/test_training.py::test_checkpoint_resume_determinism` exercises the
  restore path; the guard's parameters come from the config's
  `nan_guard`/`nan_guard_max_consecutive`).

## 7. Checkpointing

`utils/checkpoint.py:CheckpointManager` writes **three files per step**:

| file | contents | size (production) |
|---|---|---|
| `model_step_N.safetensors` | weights, fp32 (parameters are fp32 masters; bf16 is the autocast compute dtype, not a storage dtype) | 343,516,160 · 4 B ≈ 1.37 GB |
| `optim_step_N.pt` | AdamW `exp_avg` + `exp_avg_sq`, fp32 | 8 B/param ≈ 2.75 GB |
| `meta_step_N.json` | step, scheduler state, `opt_steps`, tag | ~KB |

≈ **4.1 GB per checkpoint**; at `save_interval = 4000` plus the `final` tag
that is steps 4000…60000 + final at 61,000 → 16 checkpoints ≈ 66 GB of disk
(`training/pretrain.py:Pretrainer.save_checkpoint`).

Two mechanics matter:

- **Crash-resilience by completeness, not atomicity.** The manager writes
  directly — weights, then optimizer, then meta — with no rename dance. A
  crash mid-save leaves a prefix of the three files, and
  `utils/checkpoint.py:CheckpointManager.latest_step` only returns a step
  whose *all three* files exist
  (`utils/checkpoint.py:CheckpointManager._checkpoint_complete`; pinned by
  `tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints`) —
  weights are written first, so incompleteness is always detectable.
- **Tied weights need one clone.** Weight tying means `embed.weight` and the
  LM head share storage, and safetensors refuses aliased tensors.
  `utils/checkpoint.py:CheckpointManager.save` clones only tensors whose
  `data_ptr` was already seen — untied checkpoints stay zero-copy, and the
  tied alias is materialized exactly once.

`Pretrainer.save_checkpoint` stashes the scheduler state dict, `_opt_steps`,
and a tag in `extra_meta` — the metadata *is* the resume contract (§4 item 3).
On load (`utils/checkpoint.py:CheckpointManager.load`), weights restore with
`strict=False` so a key mismatch between code versions logs instead of
crashing; a missing optimizer file warns and resumes with a *fresh* optimizer.
Resume reconstructs `_micro_count = opt_steps · accumulation_steps`, exactly
the data-loader offset (§4). Full ops: [checkpoint-ops](guides/checkpoint-ops.md).

## 8. Instrumentation: logging and the VRAM pre-flight

`utils/logging.py:TrainingLogger` is the console/WandB face of the loop:

- The trainer calls `TrainingLogger.log` when `opt_step % log_interval == 0`
  (50), passing that boundary micro-step's loss. The internal loss window
  exists for finer cadences; with `log_every == log_interval` it holds exactly
  one loss, so the printed `loss=` is that boundary micro-step's value, not a
  50-step average.
- `ppl = exp(avg_loss)` — the x0-posterior perplexity at the mixed train-time
  t-marginal; *not* comparable to an AR token perplexity (the comparable
  quantity is the held-out x0-CE of §7 in DIFFUSION.md).
- `tps` counts `log_every · seq_len · micro_batch_size / elapsed` on the
  **micro**-batch basis, so at the shipped 16×2 the printed figure reads half
  the true token throughput (two micro-batches per optimizer step).
- WandB is opt-in by environment: `WANDB_PROJECT` initializes a run (named
  from `WANDB_RUN_NAME`); without it — or without wandb installed — logging
  stays local (`utils/logging.py:TrainingLogger.log` forwards the same four
  metrics). The §3.5 estimator and pre-flight raise run once at startup —
  the cheapest place to discover a config that cannot fit the GPU.

## 9. The A100 run

The shipped config (`configs/pretrain_a100_380m.yaml` via
`models/transformer.py:DiffusionGemmaConfig.from_yaml`), numbers verified:

| quantity | value | derivation |
|---|---|---|
| parameters | 343,516,160 | `tests/test_models.py::test_param_count` (never "380M") |
| tokens | 7,995,392,000 ≈ 8.0B | 61,000 × 131,072 (both 8×4 and 16×2) |
| train FLOPs | 1.6479e19 (6·N·D) + ~12% self-cond = **1.8457e19** | 6·N·D rule |
| time | ~47 h at 35% MFU, ~41 h at 40% | 1.8457e19 / (312e12 · MFU) `[INFERENCE]` |
| peak VRAM | 23.6 GB (8×4+ckpt) / 55.2 GB (16×2) | [memory-engineering §5](concepts/memory-engineering.md) |
| LR | 3e-6 → 3e-4 → 1.5e-5 | §5 table |
| checkpoints | 4.1 GB × 16 (15 interval + final) ≈ 66 GB | §7 |

CLI (`training/pretrain.py:main`):

- `--dry-run` / `--no-compile` — cap `total_steps` at 2 to verify wiring;
  skip `torch.compile`.
- `--no-checkpoint` — **despite the name, this disables *gradient*
  checkpointing** (the `grad_checkpoint` knob), not checkpoint saving;
  checkpoints still land every 4000 steps. The VRAM-for-MFU trade is the
  config default (`grad_checkpoint: false`), so the flag flips *back* to the
  checkpointed layout on a smaller GPU
  (`tests/test_training.py::test_grad_checkpointing_path`).
- `--resume N` — explicit step; with no `--resume`, `train` auto-resumes from
  the latest complete checkpoint (`training/pretrain.py:Pretrainer.train`).

Two environment notes. On CPU the trainer auto-falls-back to
`attn_impl='sdpa'` (flex has no CPU backward,
`training/pretrain.py:Pretrainer.__init__`). And the compile scope is narrow
by design: `training/pretrain.py:Pretrainer._compile_blocks` compiles only the
residual blocks, in-place, on CUDA — corruption, loss, and sampler stay eager
so numerics are reproducible and the chunk loop is never traced.

## 10. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| global RNG instead of per-step generators | resume replays different corruption; determinism dead | `tests/test_training.py::test_checkpoint_resume_determinism` |
| seed the generator per *epoch* | mid-epoch resume changes (t, corruption) draws | `tests/test_training.py::test_checkpoint_resume_determinism` |
| reorder the draws inside `diffusion_loss` | the self-cond gate samples a different coin; which steps see sc changes | `tests/test_training.py::test_checkpoint_resume_determinism` |
| clip every micro-step | gradients scaled before accumulation; effective clip ≠ 1.0 | `tests/test_training.py::test_hundred_step_descent` |
| decay on `selfcond.proj` (dim 2) | decays the zero-init weights away from 0; zero-init equivalence eroded | `tests/test_self_conditioning.py::test_zero_init_equivalence` |
| `zero_grad` only at boundaries | micro-step grads accumulate across the NaN-skip | `training/pretrain.py:Pretrainer.train_step` NaN path |
| `min_lr_ratio: 0` | cosine floor at exactly 0 — late-training lr spikes on resume rounding | (robustness; schedule shape pinned by `test_lr_schedule_shape`) |
| rebuild the loader without `offset_batches` | data order restarts at window 0 after resume | `tests/test_data.py::test_loader_resumable_offset` |
| scheduler state omitted from the checkpoint | resumed lr ≠ pre-save lr | `tests/test_training.py::test_checkpoint_resume_determinism` |
| materialize full-vocab logits again | +6.6 GB fp32 CE chain at micro_bs 8 — the §4.0 budget dies | (no test; `estimate_model_memory_gb` reports it) |
| use `chunked_p_embed` as a differentiable path | every chunk's logits retained under grad — pre-pass memory blows the budget | `training/losses.py:chunked_p_embed` docstring contract |
| treat `--no-checkpoint` as "stop saving checkpoints" | nothing stops; you silently turned gradient checkpointing *on* | (documentation; §9 CLI) |
| drop the alias clone in `CheckpointManager.save` | safetensors raises on the tied embed/head storage | `tests/test_utils.py::test_checkpoint_roundtrip_resume_equivalence` |

## 11. Glossary

| symbol | meaning | code |
|---|---|---|
| micro vs opt step | micro-steps accumulate; optimizer steps at boundaries | `training/pretrain.py:Pretrainer.train_step` |
| `_step_rng` | per-micro-step generator `seed·100003 + step` | `training/pretrain.py:Pretrainer._step_rng` |
| `self_cond_p` | probability of the two-pass step (0.5) | config `self_cond_p` |
| `vocab_chunk` | CE chunk width, 8192·8/micro_bs | `training/pretrain.py:TrainingConfig.vocab_chunk` |
| `_nan_streak` / `auto_resume` | consecutive non-finite micro-streak counter; auto-load latest checkpoint when no `--resume` | `training/pretrain.py:Pretrainer.train` |
| `_opt_steps` / `_micro_count` | optimizer-step and global micro-step counters; resume derives the latter as `opt_steps · accum` | `training/pretrain.py:Pretrainer.load_checkpoint` |
| three-file checkpoint | weights safetensors + optim pt + meta json; resumable only when all exist | `utils/checkpoint.py:CheckpointManager` |
| `offset_batches` | micro-step count handed to the loader as the resume offset (in windows: × batch size) | `data/dataset.py:build_dataloader` |

## 12. Interview Q&A

**Q: Walk me through one training step.**
A: Draw per-canvas t from the step generator
(`training/pretrain.py:Pretrainer._step_rng`), corrupt with `q_sample`
(keep x0 w.p. ᾱ(t) else uniform), optionally (p=0.5) run the no-grad
self-cond pre-pass `chunked_p_embed`, forward `final_hidden`, compute
`chunked_x0_ce` — all inside autocast, all draws from the step generator.

**Q: How is resume bit-exact?**
A: Three pieces: the data permutation is fixed by (seed, n_windows) with an
offset resume (`data/dataset.py:ShuffledRangeSampler`); every micro-step seeds
its own generator `seed·100003 + step`, so t, corruption, and the self-cond
gate replay exactly; checkpoints restore optimizer + scheduler + counters
(`training/pretrain.py:Pretrainer.load_checkpoint`). Pinned fp64:
`tests/test_training.py::test_checkpoint_resume_determinism`.

**Q: Why is the loss divided by grad-accum steps?**
A: Each micro-step's gradients must contribute 1/accum of an optimizer step's
gradient; scaling before backward keeps the effective batch's gradient equal
to a full-batch step, and clipping happens once at boundaries on the
accumulated gradient (`training/pretrain.py:Pretrainer.train_step`).

**Q: What does the NaN guard do that plain clipping doesn't?**
A: Clipping fixes exploding gradients; the NaN guard handles *non-finite*
losses whose backward would poison the optimizer state — it skips backward
entirely and, after 5 consecutive failures, rolls back to the last checkpoint
(`training/pretrain.py:Pretrainer.train`).

**Q: Why does the scheduler restore from the checkpoint?**
A: Cosine position *is* training progress: resuming at step 20,000 with a
step-0 lr would re-warm the model and corrupt the schedule
(`training/pretrain.py:Pretrainer.load_checkpoint` persists scheduler state
in `extra_meta`).

**Q: Why do norms and biases get weight_decay 0?**
A: Decaying a LayerNorm/RMSNorm weight toward zero shrinks the normalize-
and-scale capacity; standard practice, applied via the dim ≥ 2 split
(`training/pretrain.py:Pretrainer.__init__`) — with the house rule that the
zero-init `selfcond.proj` also gets no decay so the zero-init equivalence
isn't fought by the optimizer
(`tests/test_self_conditioning.py::test_zero_init_equivalence`).