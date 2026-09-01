# Concept: the diffusion core (uniform state, x0-prediction, chunked loss)

> **Canonical** for the forward process, the cosine schedule, x0-parameterization,
> and the chunked production loss. `DIFFUSION.md` §1/§5 stays authoritative for
> rulings; this page teaches them.

**Depends on:** [foundations](foundations.md) §2–§3, §6 · **Read next:**
[block-causal-attention](block-causal-attention.md) · [sampler](sampler.md)

---

## Table of Contents

1. [The forward process](#1-the-forward-process)
2. [The cosine schedule, numerically](#2-the-cosine-schedule-numerically)
3. [The corruption distribution q(xt|x0), shape by shape](#3-the-corruption-distribution-qxtx0-shape-by-shape)
4. [Per-canvas t: the variance argument](#4-per-canvas-t-the-variance-argument)
5. [x0-parameterization (and the recorded typo)](#5-x0-parameterization-and-the-recorded-typo)
6. [A worked corruption pass (tiny scale)](#6-a-worked-corruption-pass-tiny-scale)
7. [The one-table shape summary](#7-the-one-table-shape-summary)
8. [The chunked-CE memory story](#8-the-chunked-ce-memory-story)
9. [Time conditioning](#9-time-conditioning)
10. [What breaks if you change this](#10-what-breaks-if-you-change-this)
11. [Glossary](#11-glossary)
12. [Interview Q&A](#12-interview-qa)

---

## 1. The forward process

### 1.1 The transition matrix view

D3PM defines corruption as a Markov chain over token *types*: a clean token
x0 transitions to a noisy token xt by drawing from row x0 of a transition
matrix Q_t. This repo uses the **uniform-state** instance:

```
ᾱ(t) = cos²(π/2 · t/T)                       # models/diffusion.py:alpha_bar
q(xt | x0):  keep x0        with prob ᾱ(t)
             uniform token  with prob 1 − ᾱ(t)   # models/diffusion.py:q_sample
```

The full row — `ᾱ` on the clean token, `(1−ᾱ)/V` on every other token — is
`models/diffusion.py:corruption_probs`. It sums to 1 because
`ᾱ + (V−1)·(1−ᾱ)/V = ᾱ + (1−ᾱ) = 1`.

The noise state is a **valid vocabulary token**, not `<mask>` and not
Gaussian noise. Because *any* sequence is a legal state at any corruption
level, the sampler can commit partial answers and re-noise the rest
([foundations §3.2](foundations.md)).

### 1.2 Single-step corruption (why the chain is never iterated)

The forward chain is Markov with a uniform-state kernel, so the t-step
marginal has a closed form: it is exactly the one-step row above with ᾱ(t).
Training therefore never iterates — `models/diffusion.py:q_sample` draws
`xt | x0, t` directly:

```python
a_per_canvas = alpha_bar(t, T)                     # (B, n_canvases)
a_full = a_per_canvas.repeat_interleave(canvas_len, dim=1)   # (B, seq)
noise = randint(0, V, x0.shape)                    # uniform token draws
keep  = rand(x0.shape) < a_full                    # per-position keep mask
xt    = where(keep, x0, noise)
```

Shapes: `x0: (B, seq)`; `t: (B, n_canvases)` → `ᾱ: (B, n_canvases)` →
broadcast to `(B, seq)` by `repeat_interleave(canvas_len)`; both random draws
consume the *step generator* (resume determinism,
`training/pretrain.py:Pretrainer._step_rng`). The test suite pins the
endpoints (`ᾱ(T)=0 ⇒ xt is a pure uniform draw`) and the marginals:
`tests/test_diffusion.py::test_q_sample_zero_alpha_is_uniform_draw`,
`test_q_sample_identity_when_alpha_one`, `test_forward_process_marginals`.

### 1.3 Keep-probability vs the schedule (a subtlety)

ᾱ(t) is the *keep* probability, but the probability a position still *holds*
its original token is slightly higher: a uniform redraw can land on the clean
token by chance (`1/V` chance per corrupted position):

```
P(xt = x0) = ᾱ(t) + (1 − ᾱ(t))/V
```

At `t = 8` (V = 50,257): `0.5 + 0.5/50257 = 0.50001` — effectively ᾱ, but the
distinction matters when you reason about what the model can infer from a
position: **nothing**. A noisy position is indistinguishable from a kept one;
the model's only signal is the canvas's joint structure.

## 2. The cosine schedule, numerically

`ᾱ(t) = cos²(π/2 · t/T)`, monotone from ≈1 to ≈0
(`tests/test_diffusion.py::test_alpha_bar_monotone_and_bounded`,
`test_final_step_pure_noise`). Train-time `T = 16`
(`configs/pretrain_a100_380m.yaml` `n_diffusion_steps`):

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ᾱ(t) | .9904 | .9619 | .9157 | .8536 | .7778 | .6913 | .5975 | .5000 | .4025 | .3087 | .2222 | .1464 | .0843 | .0381 | .0096 | .0000 |

What the model sees at each level (expected kept tokens per 256-token canvas):

| t | ᾱ | kept of 256 | regime |
|---|---|---|---|
| 1 | 0.9904 | 253.5 | light correction |
| 4 | 0.8536 | 218.5 | local repair |
| 8 | 0.5000 | 128.0 | half noise — hardest regime |
| 12 | 0.1464 | 37.5 | mostly synthesis |
| 16 | 0.0000 | 0.0 | unconditional canvas synthesis |

The cosine shape concentrates training mass near the *clean* end (ᾱ stays
high for longer than a linear schedule would) while still reaching exact 0 at
`t = T` — the model gets abundant easy cases, a smooth middle, and a
guaranteed pure-noise terminal state for unconditional synthesis.

## 3. The corruption distribution q(xt|x0), shape by shape

One micro-batch of the §4.0 layout (micro_bs 8, seq 4,096 = 16 canvases of
256):

| tensor | shape | note |
|---|---|---|
| `x0` | (8, 4096) | clean windows from `data/dataset.py:ShardWindows` (no +1 shift) |
| `t` | (8, 16) | one `t ~ U{1..16}` per canvas (`models/diffusion.py:sample_canvas_t`) |
| `ᾱ(t)` | (8, 16) | cosine table lookup |
| `ᾱ` broadcast | (8, 4096) | `repeat_interleave(canvas_len, dim=1)` |
| `keep` mask | (8, 4096) | `rand < ᾱ` |
| `noise` | (8, 4096) | `randint(0, 50257)` |
| `xt` | (8, 4096) | `where(keep, x0, noise)` |

Total random draws per micro-batch: `t` (128 ints), `keep` (32,768 floats),
`noise` (32,768 ints) — all from the step generator, which is what makes
resume bit-exact ([training.md](../training.md) §Resume determinism).

## 4. Per-canvas t: the variance argument

Each canvas draws its own `t ~ U{1..T}` (`models/diffusion.py:sample_canvas_t`).
The alternatives, and why each loses:

- **per-token t** — 4,096 corruption levels per row (maximum variance
  reduction) but it destroys the canvas as a unit: neighboring tokens at
  wildly different noise levels make the bidirectional-within-canvas
  structure meaningless, and the time embedding becomes per-token.
- **per-sequence t** (all 16 canvases share one t) — wastes the batch:
  16 canvases × B rows would cover only T distinct levels, sampled
  independently per row.
- **per-canvas t (the design)** — every canvas denoises coherently at its own
  level; one micro-batch row of 16 canvases covers 16 levels of the
  t-marginal — a free 16-point quadrature over the schedule, 16× the variance
  reduction of per-sequence t at the same compute.

`models/diffusion.py:sample_canvas_t` draws `t ~ U{1..T}` per canvas at train
time; the sampler, by contrast, uses a *fixed descending* schedule
`t = T_eval, T_eval−1, ..., 1` ([sampler](sampler.md)) — train-time covers
the marginal, eval-time walks it.

## 5. x0-parameterization (and the recorded typo)

The denoiser predicts the **clean tokens** directly; the loss is CE against
`x0` at corrupted input `xt`:

- eager reference: `models/diffusion.py:x0_ce_loss` (test-only),
- production: `training/losses.py:chunked_x0_ce` — never materializes the
  full `(B, T, V)` logits tensor (§8).

The softmax over the head **is** the x̂0 posterior the sampler needs to commit
and re-noise (`DIFFUSION.md` §1.2). The plan's `xt` wording is a recorded typo
(Ruling 19).

Why x0-parameterization: the sampler needs a per-position posterior over the
vocabulary to (a) commit high-confidence tokens and (b) re-noise the rest.
An x0-prediction head gives exactly that posterior `p(x0 | xt, t)` in one
softmax — no marginalization over trajectories is needed at any point.

### 5.1 Parameterization alternatives and why x0 wins here

| parameterization | the network outputs | sampler needs |
|---|---|---|
| **x0 (this repo)** | `p(x0 \| xt, t)` directly — the posterior | exactly this: commit from it, temperature-draw from it, re-embed it |
| noise/edge (REPA-style) | the corruption that was applied | must invert the corruption to get a posterior — extra step, no benefit here |
| score/edge (D3PM) | per-step transition probabilities | needs trajectory marginalization for commit decisions |

The decisive argument is the sampler ([sampler.md](sampler.md)): commit rule,
temperature annealing, and self-conditioning all consume `p̂0 = softmax(logits)`
directly. Any other parameterization would have to reconstruct that posterior
first.

### 5.2 The eager reference

`models/diffusion.py:x0_ce_loss` is a two-line wrapper over
`F.cross_entropy` — deliberately naive, materializing full logits. It exists
as the ground-truth twin for `training/losses.py:chunked_x0_ce` (pinned equal
at `atol=1e-6` for loss and grads, `tests/test_loss.py`) and for
`SelfConditioning.embed` (the exact `softmax(h @ E.T) @ E`). Do not delete
them as "dead code" — they are the regression oracles (AGENTS.md §2).

## 6. A worked corruption pass (tiny scale)

Canvas of 8 tokens, toy vocab V = 12 (ids 0–11), `T = 16`, `t = 5` →
`ᾱ(5) = 0.7778`:

```
x0      = [ 3,  7,    1,  9,  4,   11,   2,   6]
draw k  = [ T,  T,     F,  T,  T,    F,   T,   T]   (keep ~ ᾱ = 0.7778)
noise   = [ —,  —,    10,  —,  —,     5,   —,   —]  (uniform over 0..11)
xt      = [ 3,  7,    10,  9,  4,    5,   2,   6]
```

The model's head outputs a `(8, 12)` posterior; CE scores every row against
x0 — the two corrupted positions (idx 2, 5) against their originals, the kept
positions against themselves (a free "copy this" signal at rate ᾱ). Note that
nothing about positions 2 or 5 "looks" corrupted — token 10 is as legal as
token 1. Take-away: even at ᾱ = 0.78 the model must resolve the corrupted
positions *and* not disturb the kept ones, using joint structure alone.

## 7. The one-table shape summary

| stage | shape | note |
|---|---|---|
| windows | (B, 4096) | 16 canvases, flat uint32→long (`data/dataset.py:ShardWindows`) |
| `t` | (B, 16) | per-canvas, `~U{1..16}` (`models/diffusion.py:sample_canvas_t`) |
| `xt` | (B, 4096) | `q_sample` output |
| time embed | (B, 16, 1024) | added per canvas (`models/time_embed.py:CanvasTimeEmbedding`) |
| hidden | (B, 4096, 1024) | 24 blocks, block-causal mask (1,1,4096,4096) |
| logits (eager) | (B, 4096, 50257) | **never materialized in training** |
| chunk logits | (B, 4096, 8192) | one chunk at a time (`training/losses.py:chunked_x0_ce`) |
| loss | scalar | CE vs x0 |

## 8. The chunked-CE memory story

The naive training step materializes logits `(B, T, V)` = 6.6 GB fp32 at
micro_bs 8 / seq 4096 / V 50,257 — see
[memory-engineering](memory-engineering.md) for the full byte budget and
[foundations §15](foundations.md) for the from-scratch walkthrough. The
mechanics that matter here:

- `training/losses.py:chunked_x0_ce` computes `hidden @ E[chunk].T` for one
  `vocab_chunk`-wide slice at a time (runtime chunk
  `8192 · 8 / micro_bs`, `training/pretrain.py:Pretrainer.__init__`).
- Each chunk's **bf16 logits are retained for backward** by
  `training/losses.py:_ChunkTerms` (the previous `torch.utils.checkpoint`
  scheme re-ran the head GEMM on every backward; the retained-bf16 chain buys
  back one full head GEMM per step).
- Per-chunk fp32 logsumexp → global logsumexp → masked target-logit gather;
  equivalence to eager pinned at `atol=1e-6` for loss and gradients
  (`tests/test_loss.py::test_chunked_equals_eager`,
  `test_chunked_matches_eager_grad_direction`); partial last chunk pinned by
  `tests/test_loss.py::test_partial_last_chunk`.
- The self-cond pre-pass needs `p @ E` without gradients —
  `training/losses.py:chunked_p_embed` (eager twin:
  `models/selfcond.py:SelfConditioning.embed`).

Memory bounds are encoded by `utils/memory.py:estimate_model_memory_gb`
(§4.0 table; ~33 GB of retained activations with grad-checkpointing off) and
enforced pre-flight by `utils/memory.py:assert_fits_in_available_gpu`.

## 9. Time conditioning

`models/time_embed.py:CanvasTimeEmbedding` embeds the normalized time `t/T`
per canvas (sinusoidal → 256-dim → SiLU MLP → d_model) and adds it to the
token embeddings. Train-time forwards use the model's train `T`; eval forwards
pass `time_steps=SamplerConfig.n_diffusion_steps` so `t/T ∈ (0,1]` at any eval
schedule (`DIFFUSION.md` §4.3). `t=0` (prompt, finalized canvases) is
T-independent.

## 10. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| target `xt` instead of x0 | the model learns to reproduce its own corrupted input; the sampler's posterior is meaningless | `tests/test_loss.py` (targets are x0 by construction) + sampler tests fail |
| per-token t instead of per-canvas | time embedding no longer constant per canvas; the `canvas_ids` broadcast breaks | `tests/test_time_embed.py::test_time_embed_shape_bounds_distinctness` |
| per-sequence t | 16× less coverage of the t-marginal per batch (no test pins this — it is a *quality* argument, not an invariant) | — |
| BF16 logsumexp boundaries | loss shifts beyond the 1e-6 pin | `tests/test_loss.py::test_chunked_equals_eager` |
| learned noise schedule | `alpha_bar` closed form lost; q_sample single-draw identity gone | `tests/test_diffusion.py::test_alpha_bar_monotone_and_bounded` |
| absorbing `<mask>` state instead of uniform | commit-and-renoise semantics collapse (masked positions are *marked*, not hidden) | the whole sampler test suite |

## 11. Glossary

| symbol | meaning | code |
|---|---|---|
| `ᾱ(t)` | cosine schedule / forward-process keep-prob | `models/diffusion.py:alpha_bar` |
| `q(xt\|x0)` | forward-process row | `models/diffusion.py:corruption_probs` |
| `t`, `T` | canvas timestep / step count (train 16, eval ≤32) | `models/diffusion.py:sample_canvas_t`, `SamplerConfig.n_diffusion_steps` |
| `x0`/`xt` | clean / corrupted canvas | `models/diffusion.py:q_sample` |
| `lse` | log-sum-exp (fp32 in every chunk boundary) | `training/losses.py:_ChunkTerms` |
| `K` | vocab chunk (8192·8/micro_bs) | `training/pretrain.py:Pretrainer.__init__` |

## 12. Interview Q&A

**Q: Why per-canvas timesteps instead of per-sequence?**
A: Variance reduction across the t-marginal: one micro-batch row of 16
canvases sees 16 corruption levels
(`models/diffusion.py:sample_canvas_t`) instead of one, so every gradient
step covers the whole difficulty curriculum. Per-token t would break the
canvas as a unit; per-sequence t wastes the batch.

**Q: Why is the target x0 and not xt?**
A: The plan text's `xt` is a recorded typo (Ruling 19). The sampler needs the
posterior over *clean* tokens to commit and re-noise; x0-prediction provides
it in one softmax
(`models/diffusion.py:x0_ce_loss` reference, `training/losses.py:chunked_x0_ce`
production).

**Q: What is the memory hazard in the loss, exactly?**
A: Full-vocab logits `(8, 4096, 50257)` fp32 = 6.6 GB. The production loss
`training/losses.py:chunked_x0_ce` computes the head GEMM one 8192-token
vocab chunk at a time, keeps each chunk's bf16 logits for backward (~3.3 GB
total, trading for one head-GEMM recompute per step), and combines per-chunk
fp32 logsumexps into a global lse — equal to eager CE at `atol=1e-6`
(`tests/test_loss.py`).

**Q: Why a cosine schedule?**
A: Closed form (`models/diffusion.py:alpha_bar`), monotone 1→0, no learnable
parameters, and it front-loads easy denoising cases while guaranteeing
`ᾱ(T) = 0` exactly (pure-noise terminal state,
`tests/test_diffusion.py::test_final_step_pure_noise`).

**Q: If the noise state is just random tokens, how does the model know what
to denoise?**
A: It doesn't get a signal — that is the point. Uniform-state corruption
means every position is a legal token at every level, so the model must learn
the data distribution itself, at every corruption level. The per-canvas time
embedding (`models/time_embed.py:CanvasTimeEmbedding`) tells it *how much* to
trust the input.
