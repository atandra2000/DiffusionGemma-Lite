# Concept: self-conditioning — the model seeing its own previous guess

> **Canonical** for the zero-init equivalence, the exactly-once routing
> invariant, the detached pre-pass, and cross-step eval conditioning.
> `DIFFUSION.md` §3 stays authoritative for rulings; this page teaches them.

**Depends on:** [foundations](foundations.md) §5.7 ·
[diffusion-core](diffusion-core.md) §5 · **Read next:** [sampler](sampler.md)

---

## Table of Contents

1. [The idea](#1-the-idea)
2. [The two-pass training step](#2-the-two-pass-training-step)
3. [Zero-init equivalence (the bit-exact proof)](#3-zero-init-equivalence-the-bit-exact-proof)
4. [What sc actually is](#4-what-sc-actually-is)
5. [Routing: exactly one add per path](#5-routing-exactly-one-add-per-path)
6. [The detached pre-pass (and its memory hazard)](#6-the-detached-pre-pass)
7. [Cross-step conditioning at eval](#7-cross-step-conditioning-at-eval)
8. [What breaks if you change this](#8-what-breaks-if-you-change-this)
9. [Glossary](#9-glossary)
10. [Interview Q&A](#10-interview-qa)

---

## 1. The idea

Diffusion transformers denoise better when they can see their own previous
guess. During training, with probability `p = 0.5` per step
(`self_cond_p: 0.5`), a step becomes **two-pass**:

```
pass 1 (no grad):  ĥ = f(xt, t)              → sc = p̂ @ E   (posterior re-embedded)
pass 2 (grad):     h  = f(xt, t, sc = detach(sc))
```

The conditioning input is the previous posterior re-embedded through the
embedding matrix `E` — a `d_model`-shaped summary of *what the model currently
believes*, not a one-hot guess. At eval the conditioning is cross-step: each
denoising step receives the previous step's posterior re-embedded (§7), which
is what makes iterative commit-and-renoise coherent.

## 2. The two-pass training step

`training/pretrain.py:Pretrainer.diffusion_loss`:

```
p = 0.5 draw (from the step generator):
    fail → single pass, no conditioning
    pass → 1. no_grad forward on (xt, t)         → final hidden ĥ
           2. sc = chunked_p_embed(ĥ, E)          # posterior re-embedded, no grad
           3. forward WITH sc_input (detached)    → chunked_x0_ce vs x0
```

Cost: expected FLOPs ≈ 1.12× a no-sc run (p=0.5 × a no-grad forward ≈ half a
full pass, plus the chunked `p @ E`). The +12% in the config's FLOP budget
is this line.

## 3. Zero-init equivalence (the bit-exact proof)

`models/selfcond.py:SelfConditioning` is a linear `d_model → d_model`
projection whose weight *and* bias are initialized to exactly zero. At init:

```
f(xt, t, sc) = f(xt, t) + W_sc·sc + b_sc = f(xt, t) + 0 = f(xt, t)   (bit-exact)
```

Pinned fp64 with `atol=0, rtol=0`:
`tests/test_self_conditioning.py::test_zero_init_equivalence`. Why it
matters:

1. **Training starts exactly at the no-self-cond model** — the conditioning
   is *learned*, never injected. Step 0's loss equals the no-sc model's loss
   digit-for-digit.
2. **Naive init would break it silently**: the generic init pass
   (`models/transformer.py:DiffusionGemma._init_weights`) re-initializes every
   `nn.Linear` with `init_std 0.02` — including `selfcond.proj` — so
   `_init_weights` re-zeros it *after* the generic pass. Note the tied head
   shares `embed`'s weight, so `selfcond.proj` is the *only* zero-init tensor
   the re-zero must restore (`models/transformer.py:DiffusionGemma._init_weights`
   comment).

The bit-exact test also pins *why* the add is a projection and not the raw
`sc`: at init `proj(sc) = 0` regardless of `sc`, so the model is identical to
the no-sc model *even though sc is nonzero* — the conditioning path is off
until trained otherwise.

## 4. What sc actually is

`models/selfcond.py:SelfConditioning.embed`:

```
p = softmax(h_norm @ Eᵀ)          # (B, T, V) — the x̂0 posterior
sc = p @ E                        # (B, T, 1024) — softmax-weighted mean of E rows
```

`sc` is a `d_model`-shaped **summary of what the model currently believes**:
a probability-weighted average of embedding rows. Softness is the point —
a one-hot argmax guess would destroy gradient information about *near
misses*; the posterior keeps the full distribution. Pinned:
`tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean`.

Shapes: `h_norm: (B, T, D)`; `p: (B, T, V)` — the full-vocab hazard again,
which is why the training pre-pass uses the chunked path (§4).

## 5. Routing: exactly one add per path

The add must happen **exactly once per path** (SDD Ruling 6):

| path | symbol | role |
|---|---|---|
| loss path | `models/transformer.py:DiffusionGemma.final_hidden` | backbone + the single W_sc add; `training/losses.py:chunked_x0_ce` reads this |
| head path | `models/transformer.py:DiffusionGemma.head_forward` | same single add, then LM head `h @ Eᵀ` |

`models/transformer.py:DiffusionGemma._conditioned_hidden` is the single
choke point both paths call. Never compose `final_hidden` with
`head_forward` — that would add the conditioning twice; the two-step overfit
test and the wiring tests catch it
(`tests/test_models.py::test_two_step_overfit`).

## 6. The detached pre-pass

Pass 1 runs under `torch.no_grad()` and its output enters pass 2 **detached**
(DESIGN §2.4): the pre-pass constructs an *input*, it is not a second
gradient path. Gradient flows only through pass 2.

The hazard: pass 1's `p @ E` is a full-vocab softmax — at
`(8, 4096, 50257)` fp32 that alone exceeds the §4.0 memory budget. The
production pre-pass therefore uses `training/losses.py:chunked_p_embed` (same
chunked-logsumexp trick as the loss, under `no_grad`; eager twin:
`models/selfcond.py:SelfConditioning.embed`). Under grad mode
`chunked_p_embed` would retain every chunk's logits — never use it as a
differentiable path (its docstring warns exactly this).

## 7. Cross-step conditioning at eval

At eval the conditioning is **cross-step**: step `k`'s input conditioning is
step `k−1`'s posterior re-embedded — computed inside the sampler
(`inference/generate.py:BlockDiffusionSampler._denoise_step`,
`sc_next = p @ E`), always on after canvas step 1. The mechanism that makes
commit-and-renoise coherent: each step knows what the previous step believed,
so sharpening (commit) and re-noising (uncommitted redraws) compose instead
of fighting ([sampler](sampler.md) §3).

Training ↔ eval symmetry: training shows the model its *own no-grad guess* as
conditioning ~50% of steps; eval always shows the previous step's guess. The
zero-init equivalence is what makes "always on at eval" safe: had the model
been trained with conditioning, the sampler's step-1 (no sc yet) forward
would be off-distribution — with zero-init it is exactly the model's
no-conditioning mode.

## 8. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| nonzero init on `selfcond.proj` | training no longer starts as the no-sc model; the zero-init identity dies | `tests/test_self_conditioning.py::test_zero_init_equivalence` (atol=0, fp64) |
| add W_sc twice (compose `final_hidden` + `head_forward`) | double conditioning on one path; loss skewed | `tests/test_models.py::test_two_step_overfit` + wiring tests |
| detach removed from the pre-pass | second gradient path → silent memory blow-up + different training objective | `tests/test_self_conditioning.py::test_sc_grad_flow` (single path) |
| `p = 0` | sc never trained; eval cross-step conditioning sees an untrained input distribution | `tests/test_self_conditioning.py::test_sc_grad_flow` |
| full-vocab softmax in the pre-pass | +6.6 GB fp32 spike (the hazard §6) | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |

## 9. Glossary

| symbol | meaning | code |
|---|---|---|
| `sc` / `sc_input` | self-conditioning input: previous posterior re-embedded | `models/selfcond.py:SelfConditioning` |
| `p̂ @ E` | posterior re-embedding (softmax-weighted mean of E rows) | `models/selfcond.py:SelfConditioning.embed` |
| `W_sc` | zero-init projection `d_model → d_model` | `models/selfcond.py:SelfConditioning.proj` |
| `final_hidden` | backbone + the single W_sc add (loss path) | `models/transformer.py:DiffusionGemma.final_hidden` |
| `head_forward` | same single add, then LM head | `models/transformer.py:DiffusionGemma.head_forward` |
| `sc_next` | eval's cross-step conditioning (prev step's `p @ E`) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |

## 10. Interview Q&A

**Q: What exactly is the self-conditioning input?**
A: The model's own x̂0 posterior re-embedded through the (tied) embedding
matrix — `softmax(h @ Eᵀ) @ E` (`models/selfcond.py:SelfConditioning.embed`)
— a d_model-shaped summary of current belief, added to the final hidden state
through a zero-init projection
(`tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean` pins
the softmax-weighted-mean semantics).

**Q: Why zero-init the projection?**
A: So the model starts bit-for-bit identical to a no-self-cond model
(fp64, `atol=0`:
`tests/test_self_conditioning.py::test_zero_init_equivalence`) and *learns*
how much to use the conditioning. Naive init would inject random noise into
the residual stream from step 0.

**Q: Where does the ~12% FLOP overhead come from?**
A: The p = 0.5 two-pass steps: pass 1 is a full no-grad forward
(`training/pretrain.py:Pretrainer.diffusion_loss`) plus the chunked
`p @ E`. Expected overhead ≈ 0.5 × (forward share) ≈ 12% of 6·N·D.

**Q: Why is the pre-pass detached?**
A: DESIGN §2.4: the pre-pass constructs an input; making it a second gradient
path would double backward cost and change the objective (gradients through
the model's own guess). `training/pretrain.py:Pretrainer.diffusion_loss` runs
it under `no_grad` when `self_cond_detach: true`.

**Q: What happens at eval step 1, before any posterior exists?**
A: `sc` is `None` — the first denoise step runs unconditioned
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`); from step 2 on,
`sc_next = p̂ₖ₋₁ @ E` is always on
(`inference/generate.py:BlockDiffusionSampler._denoise_step`).

**Q: What's the one invariant you'd flag in code review here?**
A: Exactly one W_sc add per path (SDD Ruling 6): `final_hidden` already
includes it, `head_forward` applies the same single add — composing the two
double-conditions. The two-step overfit test exists because this exact bug is
easy to write.