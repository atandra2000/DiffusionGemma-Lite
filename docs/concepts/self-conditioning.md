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
5. [Worked example](#5-worked-example)
6. [Routing: exactly one add per path](#6-routing-exactly-one-add-per-path)
7. [The detached pre-pass (and its memory hazard)](#7-the-detached-pre-pass)
8. [Cross-step conditioning at eval](#8-cross-step-conditioning-at-eval)
9. [What breaks if you change this](#9-what-breaks-if-you-change-this)
10. [Glossary](#10-glossary)
11. [Interview Q&A](#11-interview-qa)

---

## 1. The idea

Diffusion transformers denoise better when they can see their own previous
guess: asking a x0-parameterized denoiser to *refine an answer it has already
formed* is strictly easier than deriving the full posterior from noise, because
the second view spends its capacity fixing the first view's mistakes. During
training, with probability `p = 0.5` per step (`self_cond_p: 0.5` in
`configs/pretrain_a100_380m.yaml`), a step becomes **two-pass**:

```
pass 1 (no grad):  ĥ = f(xt, t)              → sc = p̂ @ E   (posterior re-embedded)
pass 2 (grad):     h  = f(xt, t, sc = detach(sc))
```

The conditioning input is the previous posterior re-embedded through the
embedding matrix `E` — a `d_model`-shaped summary of *what the model currently
believes*, not a one-hot guess. At eval the conditioning is cross-step: each
denoising step receives the previous step's posterior re-embedded (§8), which
is what makes iterative commit-and-renoise coherent. Formally the denoiser's
signature changes from `f(xt, t)` to `f(xt, t, sc)`; everything in this
chapter is about the three choices hiding in that one line — what to condition
on (the x̂0 posterior re-embedded through `E`,
`models/selfcond.py:SelfConditioning.embed`), how it enters the network (a
zero-init projection added to the post-final-norm hidden state,
`models/selfcond.py:SelfConditioning.proj`), and when it exists during
training (a gated p=0.5, detached pre-pass,
`training/pretrain.py:Pretrainer.diffusion_loss`) — each spelled out below
against the alternative it beats.

### 1.1 Why not plain conditioning (the alternatives, one by one)

**Feed back the sampled token (argmax one-hot).** The argmax destroys the
*margin* information: a `[dog: 0.51, cat: 0.49]` position would present as a
confident "dog", while the posterior shows pass 2 the tie is real. This is
mechanical, not just statistical — the eval commit rule ([sampler](sampler.md)
§3) compares consecutive posteriors ("unchanged-or-stronger"), meaningful only
if the conditioning carries distributional information.

**Feed back raw logits or the posterior itself** (a `(V,)`-wide vector per
position). A `V → d_model` projection would add ~51M parameters; logits are
unnormalized so the input's scale drifts with training; and logit-space
distances are anchored to nothing — token geometry already lives in `E`, which
the re-embedding reuses for free.

**Re-embed through `E`** (chosen). `sc = p @ E` is a convex combination of
embedding rows: it lands in the space the residual stream occupies, is bounded
by the table's geometry, keeps the full distribution's mass, and with weight
tying costs zero new vocabulary parameters (§4 semantics, §5 worked example).

### 1.2 Why not two-pass *sampling*

The naive way to get the "refine your guess" benefit at eval is to run every
diffusion step twice: one forward to form a guess, a second conditioned on it.
That doubles eval cost per step — and eval FLOPs are the point of this project
(see [sampler](sampler.md)). Cross-step conditioning gets the same benefit with
**one forward per step**: step `k`'s conditioning is step `k−1`'s
already-computed posterior, a by-product of the step the sampler had to run
anyway (`inference/generate.py:BlockDiffusionSampler._denoise_step` emits
`sc_next = p @ E`); the extra work is one `(B, L, V) @ (V, D)` GEMM — noise
next to a full transformer forward. Only half of training steps pay the
two-pass cost (§2), and eval gets conditioning essentially for free (§8).

## 2. The two-pass training step

`training/pretrain.py:Pretrainer.diffusion_loss` is the only place training
conditioning is constructed:

```
p = 0.5 draw (from the step generator):
    fail → single pass, no conditioning
    pass → 1. no_grad forward on (xt, t)         → final hidden ĥ
           2. sc = chunked_p_embed(ĥ, E)          # posterior re-embedded, no grad
           3. forward WITH sc_input (detached)    → chunked_x0_ce vs x0
```

Reading the actual call sequence — `t = sample_canvas_t(...)` and
`xt = q_sample(...)` corrupt `x0` as in any step
([diffusion-core](diffusion-core.md) §3), then:

1. The gate `torch.rand(()) < mc.self_cond_p` is drawn **from the same
   per-micro-step generator** (`training/pretrain.py:Pretrainer._step_rng`,
   seeded `seed·100_003 + micro_count`) that produced `t` and the corruption —
   a resumed run must replay it, so the gate is part of the
   resume-determinism contract pinned by
   `tests/test_training.py::test_checkpoint_resume_determinism`.
2. On the winning branch, pass 1 runs `model.backbone(xt, t)` under
   `torch.no_grad()` — the **backbone only**: the post-final-norm hidden
   `ĥ_norm` of shape `(B, T, D)`. No LM head, no loss.
3. `sc_input = chunked_p_embed(ĥ_norm, model.embed.weight)` re-embeds the
   posterior (§7 explains why it must be the chunked variant).
4. Pass 2 runs `model.final_hidden(xt, t, sc_input)` — the backbone again,
   with gradients — and `training/losses.py:chunked_x0_ce` scores the
   conditioned hidden against the clean `x0`. Pass 1 is grad-free throughout;
   only pass 2 contributes to the loss.

Two properties deserve explicit statement. **The model learns both modes:**
because the gate is 50/50, roughly half of all gradient updates flow through
the *unconditioned* signature `f(xt, t)` and half through the conditioned one —
the gate is dropout on the conditioning *channel*, preventing co-adaptation
and making eval's always-on conditioning an in-distribution request (and at
eval step 1 there is no previous posterior yet, §8). **Pass 2's input is the
model's own no-grad guess at the same `(xt, t)`:** training never shows a
stale guess from another noise level or sequence — both passes see the
identical `(xt, t)`, the closest rehearsal of the eval-time cross-step setting
while keeping the objective exactly the x0-CE.

Cost: expected FLOPs ≈ 1.12× a no-sc run (p=0.5 × a no-grad forward ≈ half a
full pass, plus the chunked `p @ E`); this is the +12% line in the config's
FLOP budget (`configs/pretrain_a100_380m.yaml`: `~1.85e19 train FLOPs
(6·N·D + ~12% self-cond)`). The asymmetry that makes it cheap: pass 1 runs
under `no_grad`, so it pays forward FLOPs only and retains no activations —
the memory bill is the chunked re-embed's transient working set (§7), not a
second training step.

## 3. Zero-init equivalence (the bit-exact proof)

`models/selfcond.py:SelfConditioning` is a linear `d_model → d_model`
projection whose weight *and* bias are initialized to exactly zero
(`nn.init.zeros_` on both in `models/selfcond.py:SelfConditioning.__init__`).
`models/selfcond.py:SelfConditioning.forward` is
`h_out = h_norm + proj(sc_input) = h_norm + W_sc·sc + b_sc`, so at init:

```
f(xt, t, sc) = f(xt, t) + W_sc·sc + b_sc = f(xt, t) + 0 = f(xt, t)   (bit-exact)
```

Pinned fp64 with `atol=0, rtol=0`:
`tests/test_self_conditioning.py::test_zero_init_equivalence`. "Bit-exact" is
doing real work there — zero tolerance, not float-noise tolerance, because the
claim is an algebraic identity (multiplication by an exact zero matrix), not a
numerical coincidence. Why it matters:

1. **Training starts exactly at the no-self-cond model** — the conditioning
   is *learned*, never injected. Step 0's loss equals the no-sc model's loss
   digit-for-digit, so a with/without comparison starts from a common initial
   condition and early divergence is attributable to the conditioning pathway
   actually learning.
2. **Naive init would break it silently**: the generic init pass
   (`models/transformer.py:DiffusionGemma._init_weights`) re-initializes every
   `nn.Linear` with `init_std 0.02` — including `selfcond.proj` — so
   `_init_weights` re-zeros it *after* the generic pass. Note the tied head
   shares `embed`'s weight, so `selfcond.proj` is the *only* zero-init tensor
   the re-zero must restore (`models/transformer.py:DiffusionGemma._init_weights`
   comment). This is the classic module-ordering trap: `__init__` zeroes
   correctly, a later whole-model pass clobbers it, nothing fails — the model
   just quietly stops being the no-sc model at step 0.

The test also pins *why* the add is a projection and not the raw `sc`: at
init `proj(sc) = 0` regardless of `sc`, so the model is identical to the no-sc
model *even though sc is nonzero* — the conditioning path is off until trained
otherwise. Adding `sc_input` directly would give `h_norm + sc ≠ h_norm` and
the two modes would differ from step 0.

### 3.1 Why zero beats the other inits

| init | behavior at step 0 | why it loses |
|---|---|---|
| default `nn.Linear` init (post generic pass) | random `d_model`-wide perturbation of every position, scaled by `sc` | injects noise straight into the residual stream before the head |
| small random (std 1e-3) | same, smaller | still breaks the identity; "how much to use sc" answered by a random coin flip, not gradient |
| identity + small noise | sc pathway immediately live | random *direction* in the residual stream; early training spends capacity removing noise it never asked for |
| **zero** (chosen) | exact identity | the pathway is a no-op until the loss gradient gives `W_sc` a direction |

A subtler payoff is specific to this architecture: the p=0.5 gate mixes two
signatures from step 0, and with zero-init they are *identical* at
initialization — the gate cannot manufacture a train/eval mismatch, whereas
any nonzero init makes conditioned and unconditioned two different random
functions, and eval's unconditioned step-1 forward (§8) starts off the
manifold training was carving.

## 4. What sc actually is

`models/selfcond.py:SelfConditioning.embed`:

```
p = softmax(h_norm @ Eᵀ)          # (B, T, V) — the x̂0 posterior
sc = p @ E                        # (B, T, 1024) — softmax-weighted mean of E rows
```

Unrolled, `sc_i = Σ_v p_i[v] · E[v]` — per position, a probability-weighted
average of vocabulary embedding rows: a `d_model`-shaped **summary of what the
model currently believes**, a point in the convex hull of the embedding table.
Pinned: `tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean`.
Softness is the point — a one-hot argmax destroys near-miss information
(§1.1); the posterior keeps it, and `[dog: 0.51, cat: 0.49]` re-embeds to
almost the midpoint of `E[dog]` and `E[cat]`, exactly what pass 2 needs to
break the tie. Because `p` is a probability vector, `sc` is bounded: softmax
saturation sends it toward an ordinary token embedding — a regime the residual
stream already handles — never toward a blow-up.

Two details of the definition are load-bearing. **Why `h_norm`, the
post-final-norm hidden:** the conditioning is added to `h_norm` *after*
`final_norm` (§6), so the summary is computed from the same normalized stream
it will be added to; from a pre-norm hidden, the posterior's sharpness (hence
`sc`'s effective temperature) would depend on the pre-norm magnitude the norm
exists to control. **Why through `E`, the tied embedding:** with weight tying
(`models/transformer.py:DiffusionGemma` ties `head.weight` to `embed.weight`),
`E` is input geometry, output geometry, and LM head at once — `sc` is
interpretable as a *soft token sequence* the model already knows how to read,
with no new `V × D` table for zero new information.

Shapes, both regimes:

| tensor | toy (tests) | production (train) | bytes at production (fp32) |
|---|---|---|---|
| `h_norm` | `(B, 32, 64)` | `(B, 4096, 1024)` | 8·4096·1024·4 ≈ 134 MB (micro_bs 8) |
| `h_norm @ Eᵀ` (logits) | `(B, 32, 256)` | `(B, 4096, 50257)` | 8·4096·50257·4 ≈ 6.6 GB |
| `p` | `(B, 32, 256)` | `(B, 4096, 50257)` | 6.6 GB (same as logits) |
| `sc` | `(B, 32, 64)` | `(B, 4096, 1024)` | ≈ 134 MB |
| `W_sc` | `64 × 64` | `1024 × 1024` | 1,049,600 params ≈ 4.2 MB fp32 |

The full-vocab `(B, T, V)` hazard in the middle rows is the subject of §7: in
training it is never materialized; it exists only in the eager reference
`models/selfcond.py:SelfConditioning.embed`.

## 5. Worked example

### 5.1 A hand-computable posterior re-embedding

Shrink the vocabulary to three tokens `{a, b, c}` and `d_model = 2`:

```
E[a] = [ 1,  0]        E[b] = [0,  1]        E[c] = [-1, -1]
```

Suppose pass 1 produced, at one position, a posterior `p = [0.5, 0.3, 0.2]`
over `(a, b, c)`. Then:

```
sc = 0.5·[1,0] + 0.3·[0,1] + 0.2·[-1,-1] = [0.5 − 0.2, 0.3 − 0.2] = [0.3, 0.1]
```

Checks against the formula: sharpen the posterior to `[1, 0, 0]` and
`sc → E[a] = [1, 0]` — a confident guess re-embeds to the token's ordinary
embedding. At `[0.5, 0.3, 0.2]` the summary sits between `E[a]` and `E[b]`
with a small `E[c]` pull — the *uncertainty* is visible in the vector, where a
one-hot feed would show certainty and erase the tie. And `sc` always lies
inside the triangle spanned by the three rows (the convex hull): the
conditioning channel has no scale of its own to get wrong. Then
`h_out = h_norm + W_sc·sc + b_sc`: at init `W_sc = b_sc = 0` and the whole
computation is invisible; after training, `W_sc` has learned which directions
of belief-summary to inject.

### 5.2 The tiny-model two-pass step, shape by shape

The test-suite model (`tests/conftest.py::tiny_cfg`: vocab 256, d_model 64,
n_layers 2, canvas_len 32) runs the full mechanism at desk-checkable sizes.
Batch `B = 2`, one `T = 32` canvas: `x0, xt (2, 32)` → backbone out
`ĥ_norm (2, 32, 64)` → logits `(2, 32, 256)` → posterior `p (2, 32, 256)` →
re-embed `sc (2, 32, 64)` → projection `(2, 32, 64)` → pass-2 output
`h_out (2, 32, 64)` → CE against `x0 (2, 32)` — every tensor fits on one
line, so the whole two-pass step can be checked by eye against §2.

### 5.3 The same step at production scale

Production (`configs/pretrain_a100_380m.yaml`: d_model 1024, vocab 50257,
seq 4096, canvas_len 256, train T=16; micro_bs 8 in the §4.0 memory budget):

| step | shape | note |
|---|---|---|
| `x0`, `xt` | `(8, 4096)` | 16 canvases per row, each with its own `t` |
| `ĥ_norm` | `(8, 4096, 1024)` | from `model.backbone(xt, t)` under `no_grad` |
| full-vocab logits | `(8, 4096, 50257)` | **never materialized** — the §7 hazard |
| `p`, per chunk | `(8, 4096, 8192)` | one vocab chunk of `chunked_p_embed` at a time |
| `sc` | `(8, 4096, 1024)` | the only tensor the pre-pass retains |
| pass-2 hidden | `(8, 4096, 1024)` | `model.final_hidden(xt, t, sc_input)` |
| CE | chunks of `(8, 4096, 8192)` bf16 logits | `training/losses.py:chunked_x0_ce` |

The comparison that motivates the chunked apparatus: the full
`(8, 4096, 50257)` fp32 posterior costs 6.6 GB per tensor; the chunked path's
peak is one `(8, 4096, 8192)` fp32 chunk (≈ 1.07 GB) plus a `(8, 4096, 1024)`
accumulator. The pre-pass buys a `d_model`-shaped summary and never pays for
the vocabulary dimension in residence.

## 6. Routing: exactly one add per path

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

Why two paths exist at all: the loss path wants the *hidden* (the chunked CE
consumes `hidden @ E[chunk].T`, not logits), while the sampler wants *logits*
(`inference/generate.py:BlockDiffusionSampler._canvas_step_logits` calls
`head_forward` on the backbone output). Both need the conditioning — a loss
scored on unconditioned hidden while eval runs conditioned logits would train
one function and deploy a different one — so the add lives in shared code and
each path invokes it exactly once. Two deliberate non-designs: **`backbone`
never adds sc** (`models/transformer.py:DiffusionGemma.backbone` returns the
plain post-final-norm hidden; conditioning is applied by the caller through
the choke point, so pass 1 of the training step gets an unconditioned result
for free — the unconditioned mode is not a flag, it is simply "don't call the
choke point with an sc_input"), and **one add at the readout, not per-layer
injection**. Injecting `sc` into every block's input (the way `t` is added at
the embedding) would put 24 copies of the add on every path — one per
`DenoiseBlock` — with 24 chances to double-add or skip and 24 tensors to
zero-init. Here conditioning acts as a *readout refinement*: the blocks have
already computed an answer; `sc` nudges the final representation by what a
first look at the problem suggested. One choke point, one invariant, one
zero-init tensor, one place for the exactly-once test to police.

## 7. The detached pre-pass (and its memory hazard)

Pass 1 runs under `torch.no_grad()` and its output enters pass 2 **detached**
(DESIGN §2.4): the pre-pass constructs an *input*, not a second gradient path —
gradient flows only through pass 2. The gate is `self_cond_detach: true`;
`training/pretrain.py:Pretrainer.diffusion_loss` branches on it (the `False`
branch exists for experiments and shows what you would be paying).
Why detach, concretely: **objective, not just cost** — with gradients through
pass 1 the model could reduce loss by sharpening its own pass-1 guess, a
self-referential term where the model trains on its own untrained opinion;
detaching keeps the objective the plain x0-CE of DESIGN §2.4, computed once.
**Backward cost** — a differentiable two-pass chain doubles the backward, the
thing §1.2 refused to pay at eval, re-introduced at train time. **Memory** —
the chain would retain pass 1's activations *and* every chunk's logits for
backward; the docstring of `training/losses.py:chunked_p_embed` warns exactly
this: under grad mode it retains every chunk's logits, so it must never be a
differentiable path.

The hazard, quantified: pass 1's `p @ E` wants a full-vocab softmax — at
`(8, 4096, 50257)` fp32 that is ≈ 6.6 GB *per tensor* (logits and softmax
alike), alone enough to break the §4.0 memory budget before pass 2's own CE
chain is counted. The production pre-pass therefore uses
`training/losses.py:chunked_p_embed` — same chunked-logsumexp idea as the
loss, under `no_grad`; the eager twin is
`models/selfcond.py:SelfConditioning.embed`, kept as the reference the tests
pin (`tests/test_loss.py::test_chunked_p_embed_matches_eager`).

How the chunked version avoids the full-vocab tensor, in three moves:

1. **Logsumexp in pieces.** Per vocab chunk `E[c:c+8192]`, compute the chunk
   logits `(B, T, 8192)` fp32 and their per-row logsumexp; a global `logsumexp`
   over the stacked per-chunk values gives the exact full-vocab denominator —
   softmax assembled from pieces, never holding all `50257` logits at once.
2. **Accumulate the re-embedding.** Per chunk, `exp(chunk_logits − lse)` (the
   chunk's slice of `p`) multiplies `E[chunk]` and adds into a `(B, T, D)`
   accumulator; the sum over chunks *is* `p @ E`, exactly.
3. **Retain nothing.** Under `no_grad` no chunk's logits are saved; the peak
   working set is one chunk's fp32 logits (≈ 1.07 GB at micro_bs 8,
   `vocab_chunk = 8192`) plus the accumulator. The chunk size scales inversely
   with micro-batch (`training/pretrain.py:Pretrainer.__init__`,
   `8192·8/micro_bs`) so the transient bytes stay at the micro_bs=8 budget
   line regardless of batch.

`tests/test_loss.py::test_chunked_p_embed_matches_eager` pins that the chunked
result equals the eager `p @ E`; `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate`
pins the memory accounting that justifies it.

## 8. Cross-step conditioning at eval

At eval the conditioning is **cross-step**: step `k`'s input conditioning is
step `k−1`'s posterior re-embedded (`sc_next = p @ E`, computed inside
`inference/generate.py:BlockDiffusionSampler._denoise_step`), always on after
canvas step 1 — each step knows what the previous step believed, so sharpening
(commit) and re-noising (uncommitted redraws) compose instead of fighting
([sampler](sampler.md) §3). The per-step data flow inside
`inference/generate.py:BlockDiffusionSampler.denoise_canvas`:

```
x = randint(V, (B, L));  committed = zeros;  sc = None
for k in 1..T_eval:
    logits = _canvas_step_logits(kv, prefix_len, x, t=T_eval−k, sc_input=sc)
    p      = softmax(logits)                       # this step's posterior
    ... commit unchanged-or-stronger, redraw the rest pure uniform ...
    sc     = p @ E                                 # sc_next: handed to step k+1
```

Three things to notice. **The conditioning is computed from the posterior, not
the noisy state:** at step `k+1` the canvas has been partly redrawn to pure
uniform noise, but `sc` still encodes what step `k` *believed*, because it was
built from `p` and frozen before the redraw — re-noising destroys the
token-level evidence but not the belief-level summary, so the next step
re-derives the answer instead of starting over. Had the conditioning been the
noisy `xt` itself, the redraw would erase the memory every step. **It is
nearly free:** `sc_next = p @ E` reuses `p`, already computed for the commit
rule and entropy trace — one extra `(B, L, V) @ (V, D)` GEMM per step against
a full transformer forward; the eval path uses the eager inline product, not
`chunked_p_embed`, because the eval batch is a few rows of one 256-token
canvas (`(B, 256, 50257)` fp32 is tens of MB) — the pre-pass hazard is a
training-scale problem that only exists at `(B, 4096, V)`. **Step 1 runs with
`sc = None`:** `inference/generate.py:BlockDiffusionSampler.denoise_canvas`
initializes `sc = None`, and
`models/transformer.py:DiffusionGemma._conditioned_hidden` treats a `None`
sc_input as "skip the add" — the first step is the plain unconditioned
forward; no previous posterior exists and none is fabricated.

Training ↔ eval symmetry: training shows the model its *own no-grad guess* as
conditioning ~50% of steps; eval always shows the previous step's guess. The
zero-init equivalence is what makes "always on at eval" safe: had the model
been trained *only* with conditioning, the sampler's step-1 (no sc yet) forward
would be off-distribution — with the 50/50 gate it is exactly the model's
no-conditioning mode, exercised on half of all training steps. One honest
distribution gap remains, accepted deliberately: at training the conditioned
pass sees `sc` from a no-grad pass at the *same* `(xt, t)`; at eval it sees
`sc` from the previous step at a slightly different `x` (after
commit-and-redraw) and neighboring `t` — one denoise step of drift. Closing it
fully would mean simulating the sampler's commit rule inside training, the
complexity the one-forward-per-step eval design exists to avoid; what the
design buys instead is one forward per step at eval *and* one shared code path
(`models/selfcond.py:SelfConditioning`) between train and eval.

## 9. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| nonzero init on `selfcond.proj` | training no longer starts as the no-sc model; the zero-init identity dies | `tests/test_self_conditioning.py::test_zero_init_equivalence` (atol=0, fp64) |
| generic init clobbers `proj` after module init (re-zero removed from `models/transformer.py:DiffusionGemma._init_weights`) | silent: `proj` carries `init_std 0.02` noise; conditioning on garbage from step 0 | same zero-init test — it builds the full model, not the module alone |
| add W_sc twice (compose `final_hidden` + `head_forward`) | double conditioning on one path; loss skewed | `tests/test_models.py::test_two_step_overfit` + wiring tests |
| detach removed from the pre-pass | second gradient path → silent memory blow-up + different training objective | `tests/test_self_conditioning.py::test_sc_grad_flow` (single path) |
| `p = 0` | sc never trained; eval cross-step conditioning sees an untrained input distribution | `tests/test_self_conditioning.py::test_sc_grad_flow` |
| `p = 1` | unconditioned mode untrained; eval step 1 (`sc = None`) off-distribution | same — the 50/50 gate is what both endpoints rely on |
| full-vocab softmax in the pre-pass | +6.6 GB fp32 spike (the hazard §7) | `tests/test_utils.py::test_chunked_ce_term_bounds_memory_estimate` |
| conditioning computed from pre-final-norm hidden | posterior scale drifts with pre-norm magnitude; sc temperature uncontrolled | `tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean` (pins the `h_norm`-based semantics) |
| eval computes `sc` from the noisy `x` instead of `p` | commit-and-renoise loses its memory: redraws erase the previous step's belief | `tests/test_sampler.py::test_commit_rule_monotone` (commit semantics depend on posterior-derived state) |

## 10. Glossary

| symbol | meaning | code |
|---|---|---|
| `sc` / `sc_input` | self-conditioning input: previous posterior re-embedded | `models/selfcond.py:SelfConditioning` |
| `p̂ @ E` | posterior re-embedding (softmax-weighted mean of E rows) | `models/selfcond.py:SelfConditioning.embed` |
| `W_sc` | zero-init projection `d_model → d_model` | `models/selfcond.py:SelfConditioning.proj` |
| `self_cond_p` | probability a training step becomes two-pass (0.5) | `models/transformer.py:DiffusionGemmaConfig` |
| `self_cond_detach` | pass 1 under `no_grad` (true in production) | `training/pretrain.py:Pretrainer.diffusion_loss` |
| `final_hidden` | backbone + the single W_sc add (loss path) | `models/transformer.py:DiffusionGemma.final_hidden` |
| `head_forward` | same single add, then LM head | `models/transformer.py:DiffusionGemma.head_forward` |
| `_conditioned_hidden` | the exactly-once choke point both paths share | `models/transformer.py:DiffusionGemma._conditioned_hidden` |
| `chunked_p_embed` | memory-bounded `p @ E` for the training pre-pass | `training/losses.py:chunked_p_embed` |
| `sc_next` | eval's cross-step conditioning (prev step's `p @ E`) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |

## 11. Interview Q&A

**Q: What exactly is the self-conditioning input?**
A: The model's own x̂0 posterior re-embedded through the (tied) embedding
matrix — `softmax(h @ Eᵀ) @ E` (`models/selfcond.py:SelfConditioning.embed`)
— a d_model-shaped summary of current belief, added to the final hidden state
through a zero-init projection
(`tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean` pins
the softmax-weighted-mean semantics). Concretely a convex combination of
embedding rows: same space as the residual stream, bounded by the table's
geometry, and — unlike an argmax one-hot — preserving near-miss information.

**Q: Why zero-init the projection?**
A: So the model starts bit-for-bit identical to a no-self-cond model
(fp64, `atol=0`:
`tests/test_self_conditioning.py::test_zero_init_equivalence`) and *learns*
how much to use the conditioning; naive init would inject random noise into
the residual stream from step 0. It also makes the two training modes
identical at init, so the p=0.5 gate cannot manufacture an early train/eval
mismatch.

**Q: Where does the ~12% FLOP overhead come from?**
A: The p = 0.5 two-pass steps: pass 1 is a full no-grad forward
(`training/pretrain.py:Pretrainer.diffusion_loss`) plus the chunked
`p @ E`. Expected overhead ≈ 0.5 × (forward share) ≈ 12% of 6·N·D — the line
booked in `configs/pretrain_a100_380m.yaml`'s `6·N·D + ~12% self-cond`
budget. FLOP overhead only: under `no_grad` the pre-pass retains no
activations, so the memory bill is the re-embed's transient working set.

**Q: Why is the pre-pass detached?**
A: DESIGN §2.4: the pre-pass constructs an input; making it a second gradient
path would double backward cost and change the objective (gradients through
the model's own guess). `training/pretrain.py:Pretrainer.diffusion_loss` runs
it under `no_grad` when `self_cond_detach: true`; a differentiable
`chunked_p_embed` would also retain every vocab chunk's logits — the exact
hazard its `no_grad` design exists to avoid.

**Q: What happens at eval step 1, before any posterior exists?**
A: `sc` is `None` — the first denoise step runs unconditioned
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`); from step 2 on,
`sc_next = p̂ₖ₋₁ @ E` is always on
(`inference/generate.py:BlockDiffusionSampler._denoise_step`). "Always on" is
safe because half of training ran unconditioned — the model has a trained mode
for exactly that input signature.

**Q: Why re-embed through `E` instead of feeding the logits or a one-hot?**
A: (1) Information: the posterior keeps margins between near-misses; a one-hot
erases them, raw logits are unnormalized and drift. (2) Space: `p @ E` is a
convex combination of embedding rows, arriving in the geometry the model
already reads; a `V → d_model` projection would add ~51M parameters to
re-learn what `E` encodes. (3) Cost: with weight tying, `E` is free.

**Q: What's the one invariant you'd flag in code review here?**
A: Exactly one W_sc add per path (SDD Ruling 6): `final_hidden` already
includes it, `head_forward` applies the same single add — composing the two
double-conditions. The two-step overfit test exists because this exact bug is
easy to write. And a close second: the two mechanisms (training two-pass, eval
cross-step) are only sound together — cross-step conditioning is
in-distribution because the 50/50 two-pass training taught the conditioned
mode, and eval step 1's unconditioned forward is safe because the same
training exercised the unconditioned mode. Remove either half and the other
degrades silently.