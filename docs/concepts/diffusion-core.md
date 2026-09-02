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
x0 transitions to a noisy token xt by drawing from row x0 of a `V × V`
transition matrix `Q_t`. Every choice of `Q_t` is a different "noise model" —
absorbing-mask (MDLM), categorical with a learned stationary distribution,
Gaussian in embedding space — and the choice determines what the sampler is
allowed to do at generation time. This repo uses the **uniform-state**
instance (`configs/pretrain_a100_380m.yaml` `corruption: uniform`):

```
ᾱ(t) = cos²(π/2 · t/T)                       # models/diffusion.py:alpha_bar
q(xt | x0):  keep x0        with prob ᾱ(t)
             uniform token  with prob 1 − ᾱ(t)   # models/diffusion.py:q_sample
```

Written as a matrix, the kernel is a scalar mix of the identity and the
uniform distribution `U = (1/V)·𝟙𝟙ᵀ`:

```
Q_t = ᾱ(t) · I + (1 − ᾱ(t)) · U
```

Read row `r` of that product: the diagonal entry is `ᾱ + (1−ᾱ)/V` and every
off-diagonal entry is `(1−ᾱ)/V`. The concrete row the code builds —
`(1−ᾱ)/V` on **every** token, plus `ᾱ` added on top of the clean token — is
`models/diffusion.py:corruption_probs`, which is exactly this matrix row:

```
q[xt = x0]  = ᾱ + (1−ᾱ)/V          # kept, or redrawn onto x0 by chance
q[xt = j≠x0] = (1−ᾱ)/V             # uniform mass on every other token
```

The row sums to 1 because `ᾱ + V·(1−ᾱ)/V = ᾱ + (1−ᾱ) = 1` — every one of the
`V` entries carries the uniform share, and the clean entry gets the extra `ᾱ`
on top. (Stating it as "`ᾱ + (V−1)·(1−ᾱ)/V`" misses the uniform share the
clean token also receives; the code's `q[clean_token] += alpha_bar_t` is the
ground truth.)

The noise state is a **valid vocabulary token**, not `<mask>` and not Gaussian
noise. Because *any* sequence is a legal state at any corruption level, the
sampler can commit partial answers and re-noise the rest
([foundations §3.2](foundations.md), [sampler](sampler.md)) — with a `<mask>`
state, "not yet answered" is *marked* on the sequence, and re-noising a
committed token would re-introduce a symbol the training distribution never
placed mid-answer. With uniform noise, an uncommitted position is just another
plausible token — indistinguishable, in the input, from a committed one
(§1.4 develops this).

### 1.2 Why the chain is never iterated: the closed-form marginal

The forward chain is Markov, so training could iterate it step by step. It
never does, and the reason is algebra, not convenience. Because `U` is a
projection (`U² = U`, since `𝟙𝟙ᵀ𝟙𝟙ᵀ = V·𝟙𝟙ᵀ`) and `UI = IU = U`, powers and
products of uniform-state kernels stay in the same two-parameter family:

```
Q_s Q_r = (ᾱ_s ᾱ_r) I + (1 − ᾱ_s ᾱ_r) U          # multiply out; U² = U collapses it
Q_t ··· Q_1 = (∏_s ᾱ_s) I + (1 − ∏_s ᾱ_s) U
```

Induction generalizes the two-step case: the t-step marginal is *the same shape
of row* with the cumulative keep-probability `∏_s ᾱ_s`. The code's `alpha_bar`
is defined directly as that cumulative quantity — `cos²(π/2 · t/T)` *is* `ᾱ(t)`,
not a per-step rate — so the marginal `q(xt | x0)` equals the one-step row
exactly. Drawing `xt | x0, t` in one shot is not an approximation of iterating
the chain; it is the chain, collapsed:

```python
a_per_canvas = alpha_bar(t, T)                     # (B, n_canvases)
a_full = a_per_canvas.repeat_interleave(canvas_len, dim=1)   # (B, seq)
noise = randint(0, V, x0.shape)                    # uniform token draws
keep  = rand(x0.shape) < a_full                    # per-position keep mask
xt    = where(keep, x0, noise)
```

That is `models/diffusion.py:q_sample` delegating to
`models/diffusion.py:_corrupt_with_alpha`. Shapes: `x0: (B, seq)`;
`t: (B, n_canvases)` → `ᾱ: (B, n_canvases)` → broadcast to `(B, seq)` by
`repeat_interleave(canvas_len)`; both random draws consume the *step generator*
(resume determinism, `training/pretrain.py:Pretrainer._step_rng`). The test
suite pins the endpoints (`ᾱ(T)=0 ⇒ xt` is a pure uniform draw) and the
marginals: `tests/test_diffusion.py::test_q_sample_zero_alpha_is_uniform_draw`,
`test_q_sample_identity_when_alpha_one`, `test_forward_process_marginals`.

The practical payoff is cost: one `randint` + one `rand` + one `where` over
`(B, seq)` per training step, independent of `T`. A naive chain implementation
would spend `T` draws and `T` passes to reach the same distribution — and
would need a *schedule* of per-step rates whose product reproduces `ᾱ(t)`,
i.e. a second hyperparameter surface.

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

This is worth restating as the *information-theoretic* contract of the
corruption. For a corrupted position, the channel output is independent of
x0 — zero mutual information, token by token. Whatever the denoiser recovers
about position `i`, it recovers from the *other* positions of the canvas (and
the causal context of earlier canvases), never from position `i` itself. That
is why this is a generative model and not a denoising autoencoder with a
corruption-detection trick: the task at every `t` is literal
conditional-distribution modeling `p(x0_i | x0_{<canvas}, xt_{canvas∖i})`,
exactly the skill the sampler later exploits when it commits a token and asks
the model to fill the rest.

### 1.4 Why uniform state beats an absorbing mask (the design-choice argument)

The absorbing alternative — `q(xt|x0)` sends each position to `<mask>` with
probability `1−ᾱ` (MDLM/BERT-style) — has one genuine advantage: masked
positions are *labeled*, so the model always knows which slots need filling,
and the objective is closer to standard MLM pretraining. It loses on three
counts that matter for this project:

1. **Commit-and-renoise needs unlabeled states.** The sampler (§4 of
   DIFFUSION.md) freezes confident positions and redraws the rest every step.
   Under uniform corruption a redrawn position is a legal token, so the
   in-flight canvas is always on the model's training manifold. Under an
   absorbing mask the committed/uncommitted split is visible structure the
   training corruption never produced at a matching rate — off-manifold input.
2. **The terminal state doubles as an unconditional prior.** At `t=T`,
   `ᾱ=0` exactly, and the input is uniform noise over the vocabulary — the
   model's conditional `p(x0 | pure noise)` is its unconditional canvas
   distribution, and generation starts there. An absorbing model's
   intermediate eval states (partially live canvases) are likewise
   off-manifold.
3. **No sentinel-token bookkeeping.** The vocab is GPT-2 BPE (50,257) shared
   across the portfolio for parity; a mask state would either consume a real
   token id or grow the embedding. Uniform state needs neither.

The cost is honest and stated in §1.3: the model gets no per-position
corruption signal, so it must learn the full conditional distribution at
every level instead of "copy the unmasked, guess the masked".

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

### 2.1 Why cos² and not linear

Put the two side by side at T = 16 (`linear ᾱ = 1 − t/16`):

| t | cos² ᾱ | linear ᾱ | what the gap means |
|---|---|---|---|
| 1 | 0.9904 | 0.9375 | cosine keeps near-clean cases *cleaner* |
| 4 | 0.8536 | 0.7500 | more training mass on structure-preserving repair |
| 8 | 0.5000 | 0.5000 | identical at the midpoint by construction |
| 12 | 0.1464 | 0.2500 | cosine goes *deeper* into noise late |
| 15 | 0.0096 | 0.0625 | second-to-last step is already nearly pure noise |
| 16 | 0.0000 | 0.0000 | both hit exact 0 |

The cosine shape concentrates training mass near the *clean* end (ᾱ stays
high for longer than a linear schedule would) while still reaching exact 0 at
`t = T` — the model gets abundant easy cases, a smooth middle, and a
guaranteed pure-noise terminal state for unconditional synthesis. The slope
tells the same story: `dᾱ/dt = −(π/2T)·sin(πt/T)` is zero at both endpoints
and maximal at `t = T/2` (≈0.098/step at T=16, vs the linear schedule's
constant 0.0625). Corruption is *accelerated through the middle* and gentle at
the ends — the ends are where the model needs the most examples (light
correction is the regime generation finishes in, and `t=T` defines the
generation prior), while the middle is where the hard joint-structure learning
happens and benefits from dense sampling.

### 2.2 Why a closed form at all

Three reasons the schedule is one line of math
(`models/diffusion.py:alpha_bar`) rather than a learned or table-interpolated
curve:

1. **The single-draw identity (§1.2) needs a cumulative ᾱ with a known value
   at every integer t.** A learned schedule would have to be constrained
   monotone and evaluated at every t anyway; the closed form removes the
   constraint machinery.
2. **`ᾱ(T) = 0` exactly.** `cos²(π/2) = 0` in floating point to the last bit,
   so the terminal training state is *pure* uniform noise, not "mostly noise".
   This matters twice: the unconditional-prior interpretation of the final
   level is exact, and the sampler's first step starts from `torch.randint`
   noise at the same level it was trained to denoise from
   (`tests/test_diffusion.py::test_final_step_pure_noise`).
3. **Eval-time interpolation for free.** Training only ever draws
   `t/T ∈ {1/16, …, 16/16}`; eval walks `t/T ∈ {k/32}`. Both the schedule and
   the time embedding (§9) are functions of the *normalized* time, so every
   eval level is a point on the same trained curve — no schedule re-fit, no
   embedding re-normalization.

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

At the production training layout (micro_bs 16 × accum 2, the 80 GB A100
config), every row of that table doubles its batch dimension: `x0/xt` are
(16, 4096), `t` is (16, 16), and the random-draw counts double to 65,536
floats + 65,536 ints + 256 ints per micro-batch. Nothing else changes — the
corruption is embarrassingly per-position and its memory footprint
(`3 × 16·4096` int/bool/long tensors, well under 1 MB) is noise compared to
the loss's logits story (§8).

Note what is *absent* from the table: no mask tensor, no special noise-token
id, no attention change for corrupted positions. The corruption is fully
expressed in the token ids and one per-canvas scalar `t` that the time
embedding will consume (§9). The transformer never knows which positions were
corrupted — by construction it *cannot* know (§1.3).

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

The variance claim, made precise. Let `L(t)` be the expected per-canvas loss
at corruption level `t`; the training signal we want is `E_{t~U{1..T}}[L(t)]`.

- Under **per-sequence t**, each row contributes one draw: the micro-batch
  loss averages `B` iid samples of `L(t)` → the t-sampling component of
  gradient noise scales as `Var[L]/B`.
- Under **per-canvas t**, each row's loss is `(1/16)·Σ_c L(t_c)` with the
  `t_c` iid → `Var[L]/16` per row, i.e. `Var[L]/(16·B)` for the batch. At the
  production effective batch (32 rows) that is 512 independent t-draws per
  optimizer step instead of 32 — a 16× variance reduction on the
  curriculum-sampling component of gradient noise, at zero extra compute.

The honest caveat: canvases within a row share a document context
(`data/dataset.py:ShardWindows` cuts flat 4096-token windows), so the 16 draws
are not fully independent — the reduction is real but bounded by the
within-row correlation. The direction of the argument is unchanged.

There is a second, quieter benefit: **canvas coherence**. Within a canvas,
every token shares one ᾱ, so "how noisy is this canvas" is a single scalar the
time embedding can express, and the bidirectional attention inside the canvas
operates on a homogeneous corruption level — which is what makes the eval-time
sampler's per-canvas uniform schedule a *trained* regime rather than an
interpolation.

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
(Ruling 19): training targets are x0, the clean token distribution — an early
planning doc's "xt" is not a semantic choice the code made, it is a typo the
ledger recorded.

Why x0-parameterization: the sampler needs a per-position posterior over the
vocabulary to (a) commit high-confidence tokens and (b) re-noise the rest.
An x0-prediction head gives exactly that posterior `p(x0 | xt, t)` in one
softmax — no marginalization over trajectories is needed at any point.

### 5.1 Parameterization alternatives and why x0 wins here

In the continuous-diffusion world the parameterization question ("predict the
noise ε, predict x0, or predict the score") is genuinely open — the three are
reparameterizations of each other with different loss weightings. In
*discrete* diffusion they are different objects with different costs:

| parameterization | the network outputs | sampler needs |
|---|---|---|
| **x0 (this repo)** | `p(x0 \| xt, t)` directly — the posterior | exactly this: commit from it, temperature-draw from it, re-embed it |
| noise/edge (REPA-style) | the corruption that was applied | must invert the corruption to get a posterior — extra step, no benefit here |
| score/edge (D3PM) | per-step transition probabilities | needs trajectory marginalization for commit decisions |

The decisive argument is the sampler ([sampler.md](sampler.md)): commit rule,
temperature annealing, and self-conditioning all consume `p̂0 = softmax(logits)`
directly. Any other parameterization would have to reconstruct that posterior
first.

There is also a structural reason an "xt-prediction" objective cannot work
*even ignoring the sampler*: per §1.3 a corrupted position's value carries
zero information about x0, so the best achievable xt-predictor is the trivial
copying one — the loss degenerates to a copy task, the gradient collapses to
"learn ᾱ", and the head's softmax becomes a corruption simulator, not a
posterior. x0 is the only parameterization whose target carries information
the input does not already contain.

Where the objective sits in the D3PM variational bound: the discrete-diffusion
ELBO decomposes over timesteps into KL terms between the forward posterior and
the model's reverse prediction, plus a reconstruction term. Each KL term is
dominated by the model's belief about the *clean* tokens given the noisy input;
the simplified surrogate this repo trains — mean CE of `pθ(x0 | xt, t)` against
x0 at uniformly sampled `t` — optimizes that dominant term at every corruption
level with weight 1, which is what the per-canvas-t marginal coverage of §4 is
designed to sample. The bound's remaining structure (exact reverse transitions)
is consumed by the sampler's commit-and-renoise loop rather than by an explicit
reverse kernel.

### 5.2 The eager reference

`models/diffusion.py:x0_ce_loss` is a two-line wrapper over
`F.cross_entropy` — deliberately naive, materializing full logits. It exists
as the ground-truth twin for `training/losses.py:chunked_x0_ce` (pinned equal
at `atol=1e-6` for loss and grads, `tests/test_loss.py`) and for
`SelfConditioning.embed` (the exact `softmax(h @ E.T) @ E`). Do not delete
them as "dead code" — they are the regression oracles (AGENTS.md §2).

The full path the production loss serves, end to end:
`models/transformer.py:DiffusionGemma.backbone` produces the post-final-norm
hidden state; `models/transformer.py:DiffusionGemma.final_hidden` adds the
self-conditioning projection (exactly once — see [foundations](foundations.md)
and `DIFFUSION.md` §3.1's routing invariant);
`training/losses.py:chunked_x0_ce` scores that hidden state against x0. The
head itself (`h @ E.T`, tied to the embedding) is never run at full vocab width
during training — §8 is about why.

## 6. A worked corruption pass (tiny scale)

Canvas of 8 tokens, toy vocab V = 12 (ids 0–11), `T = 16`, `t = 5` →
`ᾱ(5) = 0.7778`:

```
x0      = [ 3,  7,    1,  9,  4,   11,   2,   6]
draw k  = [ T,  T,     F,  T,  T,    F,   T,   T]   (keep ~ ᾱ = 0.7778)
noise   = [ —,  —,    10,  —,  —,     5,   —,   —]  (uniform over 0..11)
xt      = [ 3,  7,    10,  9,  4,    5,   2,   6]
```

Step through the mechanics:

1. **The draw.** `keep = rand < 0.7778` per position; positions 2 and 5 failed
   the draw. Their replacements are uniform over *all 12 ids* — including, with
   probability 1/12 each, their originals. Expected corrupted count for this
   canvas: `8 · (1 − 0.7778) = 1.78`; this draw got 2.
2. **What the model receives.** `xt = [3, 7, 10, 9, 4, 5, 2, 6]` and the scalar
   time signal for t=5 (§9). Nothing about positions 2 or 5 "looks" corrupted —
   token 10 is as legal as token 1. The model can score position 2's original
   (token 1) above token 10 only because `[3, 7, ?, 9, 4, ?, 2, 6]` is a more
   plausible canvas with 1 than with 10 *in context* — the bidirectional
   attention inside the canvas is what makes "?" and the context negotiate.
3. **The loss.** The head outputs a `(8, 12)` posterior; CE scores every row
   against x0 — the two corrupted positions (idx 2, 5) against their originals,
   the kept positions against themselves (a free "copy this" signal at rate ᾱ).
   If the model puts 0.6 on token 1 at position 2, that row costs
   `−ln 0.6 = 0.51` nats; a kept position where the model correctly concentrates
   on the visible token costs ~0. The kept positions are a free, dense "don't
   disturb what you can see" curriculum — 6 of 8 rows here — exactly the regime
   ᾱ near 1 trains (§2).
4. **The scale-up.** At production scale the same pass runs over (16, 4096)
   tokens: ~8,192 corrupted positions per micro-batch on average
   (`E[1−ᾱ] ≈ 0.5` over the t-marginal), each scored against the full
   50,257-token vocab by the chunked loss of §8.

Take-away: even at ᾱ = 0.78 the model must resolve the corrupted positions
*and* not disturb the kept ones, using joint structure alone.

## 7. The one-table shape summary

Toy scale (B=2, one canvas of L=8, V=12, d_model=64), then production. The
toy row exists so every shape can be checked by eye; the production rows are
what the memory budget (§8) is computed from.

| stage | toy | production (§4.0, micro_bs 8) | production (16×2 layout) |
|---|---|---|---|
| windows | (2, 8) | (8, 4096) | (16, 4096) |
| `t` | (2, 1) | (8, 16) | (16, 16) |
| `xt` | (2, 8) | (8, 4096) | (16, 4096) |
| time embed | (2, 1, 64) | (8, 16, 1024) | (16, 16, 1024) |
| hidden | (2, 8, 64) | (8, 4096, 1024) | (16, 4096, 1024) |
| full logits (never in training) | (2, 8, 12) | (8, 4096, 50257) | (16, 4096, 50257) |
| chunk logits | (2, 8, 12) | (8, 4096, 8192) bf16 | (16, 4096, 4096) bf16 |
| loss | scalar | scalar | scalar |

Production notes on the rows that differ from toy intuition:

- **windows**: flat uint32→long windows, `seq_len` tokens with **no +1 AR shift**
  (`data/dataset.py:ShardWindows`) — diffusion reconstructs x0 everywhere, so
  there is no next-token shift to manage.
- **time embed**: `(B, n_canvases, d_model)`, broadcast to tokens per canvas by
  `models/transformer.py:DiffusionGemma._add_canvas_time`; toy has one canvas
  per row, production has 16.
- **full logits**: (16, 4096, 50257) would be 13.2 GB fp32 / 6.6 GB bf16 *per
  micro-batch for the forward alone* — the reason
  `models/transformer.py:DiffusionGemma.head_forward` is bypassed by the training
  loss path entirely (§8).
- **chunk logits**: chunk width `8192·8/micro_bs` — 8192 at micro_bs 8, 4096 at
  micro_bs 16 — so the retained-bytes budget stays constant as the batch grows
  (`training/pretrain.py:Pretrainer.__init__`).

## 8. The chunked-CE memory story

The naive training step materializes logits `(B, T, V)` = 6.6 GB fp32 at
micro_bs 8 / seq 4096 / V 50,257 — see
[memory-engineering](memory-engineering.md) for the full byte budget and
[foundations §15](foundations.md) for the from-scratch walkthrough. This
section derives the numbers and the mechanics.

### 8.1 The naive step's bytes, exactly

The head is `h @ E.T` with E the tied `(50257, 1024)` embedding. One
micro-batch of logits at §4.0 scale:

```
(8, 4096, 50257) fp32 = 8 · 4096 · 50257 · 4 B ≈ 6.59 GB
```

and the CE backward needs that tensor *plus* its log-softmax retained in the
autograd graph, so the naive chain costs roughly 2× the forward figure in
retained activations — before counting the transient fp32 buffers the CE
kernel allocates. Under bf16 autocast the GEMM output halves (3.3 GB), but
`F.cross_entropy` internally upcasts, so the fp32 copy still exists. The
backbone's ~33 GB of SwiGLU and boundary activations are intrinsic to the model;
the full-vocab logits chain is not — it exists only because the CE must be
evaluated over all 50,257 tokens, which is exactly what §8.2 removes.

### 8.2 The chunked computation

`training/losses.py:chunked_x0_ce` computes `hidden @ E[chunk].T` for one
`vocab_chunk`-wide slice at a time (runtime chunk
`8192 · 8 / micro_bs`, `training/pretrain.py:Pretrainer.__init__`).

Per chunk `c` (rows `c0:c1` of the embedding):

1. **GEMM in bf16**: `logits_c = hidden @ E_c.T` — `(B, seq, chunk_w)`.
2. **fp32 normalization inside the chunk**: `lse_c = logsumexp(logits_c.float(), −1)`
   and `tgt_c = gather(logits_c.float(), target − c0)` — both by
   `training/losses.py:_ChunkTerms.forward`, which **saves the bf16 logits** for backward.
3. **Masked contribution**: only the chunk containing a token's target contributes
   its logit; others contribute a masked zero.

The two identities that make the assembly exact:

```
lse(x over V)  = logsumexp([lse(x over c0), lse(x over c1), …])     # log-sum-exp of chunk lses
target_logit   = Σ_c in_chunk_c · gather(logits_c, target − c0)      # exactly one chunk contributes
loss           = mean(lse − target_logit)                            # = −log softmax[target]
```

The first is associativity of sum-of-exps (log Σ_c e^{lse_c} = log Σ_v e^{l_v});
the second is that the in-chunk masks partition the vocabulary. The result
equals the eager CE at `atol=1e-6` for loss *and* gradients
(`tests/test_loss.py::test_chunked_equals_eager`,
`test_chunked_matches_eager_grad_direction`); the partial last chunk — at
chunk 8192, V = 50,257 = 6·8192 + 1,105, so 7 chunks with a 1,105-wide tail —
is pinned by `tests/test_loss.py::test_partial_last_chunk`. The tail's gather
indices are clamped into range inside `_ChunkTerms`, then zeroed by the
caller's in-chunk mask before the sum, so the clamp never leaks gradient.

### 8.3 The backward, and why bf16 logits are retained

`training/losses.py:_ChunkTerms.backward` rebuilds the softmax from the saved
bf16 logits and emits both gradient slices:

```
p        = softmax(saved bf16 logits)            # re-derived, bit-identical to a GEMM recompute
g_logits = p · grad_lse + onehot(target) · grad_tgt
g_hidden = g_logits(bf16) @ E_c                  # into the backbone
g_weight = g_logitsᵀ @ hidden                    # into the tied embedding
```

Because `d lse/d logits = softmax(logits)` and `d target_logit/d logits =
onehot(target)`, the two gradient terms compose to exactly
`softmax − onehot` scaled — the standard CE gradient — split across the
chunk. Re-deriving `p` from the saved bf16 logits is bit-identical to what a
`torch.utils.checkpoint` scheme would get by re-running the same bf16 GEMM in
backward; retaining the bf16 chain therefore buys back **one full head-GEMM
forward per step** at the cost of ~2 bytes/element retained. The bytes:

| item | micro_bs 8 (chunk 8192) | micro_bs 16 (chunk 4096) |
|---|---|---|
| retained bf16 logits, all chunks | `8·4096·50257·2 ≈ 3.3 GB` | `16·4096·50257·2 ≈ 6.6 GB` |
| transient fp32 chunk (freed per chunk) | `8·4096·8192·4 ≈ 1.07 GB` | `16·4096·4096·4 ≈ 1.07 GB` |
| naive fp32 logits (avoided) | 6.6 GB | 13.2 GB |

Note the retained total is chunk-width-invariant (every element is retained
exactly once); chunk width only sets the *transient* peak and the GEMM tile
size. `training/pretrain.py:Pretrainer` scales `vocab_chunk` inversely with
micro-batch (`8192·8/micro_bs`) so the retained bytes stay at the micro_bs=8
budget *per unit of effective batch* while the transient chunk never grows.

### 8.4 The self-cond pre-pass has its own hazard

The self-conditioning pre-pass needs `p @ E` — a full-vocab softmax times the
embedding — without gradients (`DIFFUSION.md` §3.2). Under grad mode that would
retain every chunk's logits; the pre-pass instead runs
`training/losses.py:chunked_p_embed` under `no_grad` (eager twin:
`models/selfcond.py:SelfConditioning.embed`). It makes two sweeps over the
vocab: sweep 1 accumulates per-chunk fp32 logsumexps into a global `lse`; sweep
2 computes `(logits − lse).exp() @ E_c` chunk by chunk, accumulating into a
`(B, seq, d_model)` buffer. Nothing of width `V` is ever alive except one
transient chunk. Its cost is FLOPs, not memory — the config budgets it at ~12%
of the training step's `6·N·D` (`configs/pretrain_a100_380m.yaml` header).

Memory bounds for the whole step are encoded by
`utils/memory.py:estimate_model_memory_gb` (§4.0 table; ~33 GB of retained
activations with grad-checkpointing off) and enforced pre-flight by
`utils/memory.py:assert_fits_in_available_gpu`.

## 9. Time conditioning

`models/time_embed.py:CanvasTimeEmbedding` embeds the normalized time `t/T` per
canvas (sinusoidal → 256-dim → SiLU MLP → d_model) and adds it to the token
embeddings. Train-time forwards use the model's train `T`; eval forwards pass
`time_steps=SamplerConfig.n_diffusion_steps` so `t/T ∈ (0,1]` at any eval
schedule (`DIFFUSION.md` §4.3). `t=0` (prompt, finalized canvases) is
T-independent.

### 9.1 The sinusoid, derived

`models/time_embed.py:CanvasTimeEmbedding.forward` builds, for frequency
index `j ∈ [0, 128)`:

```
freq_j  = 10000^(−j/128)                    # geometric frequency ladder
angle_j = (t/T) · freq_j                    # normalized time times frequency
input   = [sin(angle_0..127), cos(angle_0..127)]   # 256-dim
```

Because the argument is the *normalized* time `t/T ∈ (0,1]`, each frequency
contributes a smooth, monotone-ish feature over the corruption range: the
fastest component (`j=0`) swings its full sine arc across `t/T ∈ (0,1]`, the
slowest (`j=127`, period ≈ 600 in normalized units) barely moves and acts as
an almost-constant offset. The MLP (`Linear 256→1024, SiLU, Linear
1024→1024`) mixes these into a learned per-level code; no output
nonlinearity, so the embedding surface stays smooth in `t/T` — which is what
lets eval at `T_eval ≤ 32` land between trained levels (§2.2's point 3).

Three properties do real work:

1. **Normalization makes the embedding schedule-independent.** The function
   embedded is `t ↦ embed(t/T)`, not `t ↦ embed(t)`. Train saw `t/T ∈
   {k/16}`; eval walks `{k/32}` — the same curve, more finely sampled. If the
   embedding took raw `t`, eval at T=32 would feed the model time codes from
   a range it half never saw.
2. **`t=0` is a constant, T-independently.** All angles are 0, so the input
   is `(sin 0…0, cos 0…1)` — the same 256-vector for any `T`. Prompt and
   finalized canvases enter through `t=0` and therefore get one fixed
   "clean/conditioning" embedding regardless of train or eval schedule.
3. **Per-canvas constancy.** The embedding is computed once per canvas and
   broadcast to its 256 tokens: `models/transformer.py:DiffusionGemma._add_canvas_time`
   maps absolute positions to span-local canvas ids (`positions // canvas_len −
   offset`) and indexes `time_embed(t)[:, canvas_ids]`, so a decode chunk that
   starts mid-sequence still gets one time vector per canvas it spans, aligned
   with the `t` entries the caller passes. The backbone applies this at the
   embedding, before the blocks
   (`models/transformer.py:DiffusionGemma.backbone`).

Why sinusoidal + MLP rather than a learned `nn.Embedding(T)`: a learned table
would be pinned to integer levels of one specific `T` — the eval schedule
(`T ≤ 32`, `configs/pretrain_a100_380m.yaml` `eval_diffusion_steps`) would
need either its own table or interpolation the table was never trained to
support. The sinusoid of normalized time is one function covering all
schedules; the MLP learns *what each level means* from the data.

## 10. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| target `xt` instead of x0 | the model learns to reproduce its own corrupted input; the sampler's posterior is meaningless | `tests/test_loss.py` (targets are x0 by construction) + sampler tests fail |
| per-token t instead of per-canvas | time embedding no longer constant per canvas; the `canvas_ids` broadcast breaks | `tests/test_time_embed.py::test_time_embed_shape_bounds_distinctness` |
| per-sequence t | 16× less coverage of the t-marginal per batch (no test pins this — it is a *quality* argument, not an invariant) | — |
| BF16 logsumexp boundaries | loss shifts beyond the 1e-6 pin | `tests/test_loss.py::test_chunked_equals_eager` |
| learned noise schedule | `alpha_bar` closed form lost; q_sample single-draw identity gone | `tests/test_diffusion.py::test_alpha_bar_monotone_and_bounded` |
| absorbing `<mask>` state instead of uniform | commit-and-renoise semantics collapse (masked positions are *marked*, not hidden) | the whole sampler test suite |
| eval forwards without `time_steps=SamplerConfig.n_diffusion_steps` | `t/T` computed against the train `T=16`; eval levels beyond 16 normalize above 1 — an input regime never trained | sampler + time-embed tests |
| drop the retained bf16 logits in `_ChunkTerms` | backward re-runs every chunk GEMM (checkpoint-style) — correct but slower; the memory win survives, the MFU win does not | `tests/test_loss.py` still passes (equivalence unaffected) |
| iterate the forward chain step-by-step | same marginal (§1.2 proves it) but T× the random draws, breaking resume-determinism replay order | `tests/test_training.py::test_checkpoint_resume_determinism` |

## 11. Glossary

| symbol | meaning | code |
|---|---|---|
| `ᾱ(t)` | cosine schedule / forward-process keep-prob | `models/diffusion.py:alpha_bar` |
| `q(xt\|x0)` | forward-process row | `models/diffusion.py:corruption_probs` |
| `t`, `T` | canvas timestep / step count (train 16, eval ≤32) | `models/diffusion.py:sample_canvas_t`, `SamplerConfig.n_diffusion_steps` |
| `x0`/`xt` | clean / corrupted canvas | `models/diffusion.py:q_sample` |
| `x̂0` | the model's clean-token posterior, `softmax(head)` | `models/transformer.py:DiffusionGemma.head_forward` |
| `p@E` | posterior re-embedded as tokens (`p @ E`), the self-cond input | `training/losses.py:chunked_p_embed` |
| `lse` | log-sum-exp (fp32 in every chunk boundary) | `training/losses.py:_ChunkTerms` |
| `K` | vocab chunk (8192·8/micro_bs) | `training/pretrain.py:Pretrainer.__init__` |

## 12. Interview Q&A

**Q: Why per-canvas timesteps instead of per-sequence?**
A: Variance reduction across the t-marginal: one micro-batch row of 16
canvases sees 16 corruption levels (`models/diffusion.py:sample_canvas_t`)
instead of one, so every gradient step covers the whole difficulty curriculum
— 512 independent t-draws per optimizer step at the production effective
batch instead of 32. Per-token t would break the canvas as a unit;
per-sequence t wastes the batch.

**Q: Why is the target x0 and not xt?**
A: The plan text's `xt` is a recorded typo (Ruling 19). The sampler needs the
posterior over *clean* tokens to commit and re-noise; x0-prediction provides
it in one softmax (`models/diffusion.py:x0_ce_loss` reference,
`training/losses.py:chunked_x0_ce` production). Deeper: an xt-predictor's
target carries no information the input lacks (corrupted positions are
independent of x0), so the objective degenerates to a copy task; x0 is the
only target that forces actual conditional modeling.

**Q: What is the memory hazard in the loss, exactly?**
A: Full-vocab logits `(8, 4096, 50257)` fp32 = 6.6 GB. The production loss
`training/losses.py:chunked_x0_ce` computes the head GEMM one 8192-token vocab
chunk at a time, keeps each chunk's bf16 logits for backward (~3.3 GB total,
trading for one head-GEMM recompute per step), and combines per-chunk fp32
logsumexps into a global lse — equal to eager CE at `atol=1e-6`
(`tests/test_loss.py`).

**Q: Why a cosine schedule?**
A: Closed form (`models/diffusion.py:alpha_bar`), monotone 1→0, no learnable
parameters, and it front-loads easy denoising cases while guaranteeing
`ᾱ(T) = 0` exactly (pure-noise terminal state,
`tests/test_diffusion.py::test_final_step_pure_noise`). The exact zero is
load-bearing twice: the final training level is the unconditional prior the
sampler generates from, and the single-draw corruption identity (§1.2) needs
a cumulative ᾱ that is known in closed form at every t.

**Q: If the noise state is just random tokens, how does the model know what
to denoise?**
A: It doesn't get a signal — that is the point. Uniform-state corruption
means every position is a legal token at every level, so the model must learn
the data distribution itself, at every corruption level. The per-canvas time
embedding (`models/time_embed.py:CanvasTimeEmbedding`) tells it *how much* to
trust the input.

**Q: How can the loss be backpropagated if the logits are never materialized?**
A: Per chunk. `training/losses.py:_ChunkTerms` saves that chunk's bf16 logits
in forward and rebuilds the softmax from them in backward, emitting
`g_hidden = g_logits @ E_c` and `g_weight = g_logitsᵀ @ hidden`; the global
logsumexp chain routes gradient through every chunk's `lse` term, and the
target-logit term contributes `onehot(target)` only in the chunk that owns the
target. The composition is exactly the eager CE gradient, pinned at `atol=1e-6`
(`tests/test_loss.py::test_chunked_matches_eager_grad_direction`).

**Q: Why is the time embedding normalized to t/T instead of raw t?**
A: The sinusoid of normalized time
(`models/time_embed.py:CanvasTimeEmbedding.forward`) puts train's `{k/16}` and
eval's `{k/32}` on the same curve, and `t=0` collapses to a T-independent
constant — a learned `nn.Embedding(T)` is pinned to one T's integer levels.