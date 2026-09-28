# Concept: block diffusion — the uniform-state formulation

> **Audience: beginner.** The compact, self-contained statement of the
> corruption process this repo implements: what "uniform state" means, why the
> sequence is split into canvases, and what the denoiser is trained to output.
> `DIFFUSION.md` §1/§2 stays authoritative for rulings; the full derivation
> lives in [diffusion-core](diffusion-core.md).

**Depends on:** [foundations](foundations.md) §2–§4 (or just "what a
transformer LM is") · **Read next:** [canvas-denoising](canvas-denoising.md)

---

## 1. The corruption process

Training needs noisy inputs. This repo corrupts a clean canvas `x0`
(256 GPT-2 BPE tokens) in one closed-form step, at a per-canvas timestep
`t ∈ {1..T}`:

```
# verified — the exact rule implemented by models/diffusion.py
ᾱ(t) = cos²(π/2 · t/T)                       # cosine schedule, monotone 1 → 0
q(xt | x0): keep each x0 token with prob ᾱ(t)
            else replace it with a UNIFORM random vocab token
```

- Schedule: `models/diffusion.py:alpha_bar`. At `t=1`, `ᾱ ≈ 1` (nearly
  clean); at `t=T`, `ᾱ ≈ 0` (nearly pure noise). Every level in between is
  trained, because `models/diffusion.py:sample_canvas_t` draws each canvas its
  own `t ~ U{1..T}` per sequence.
- The one-step draw is `models/diffusion.py:q_sample`: with probability `ᾱ`
  keep the token, else draw uniform. It never iterates a Markov chain,
  because `q(xt | x0)` already has this closed form — the chain is summed out.
- The per-token probability row behind that draw is
  `models/diffusion.py:corruption_probs`: mass `ᾱ` on the clean token,
  `(1−ᾱ)/V` on every other token of the `V = 50,257` vocab. It sums to 1.

## 2. What "uniform state" means

"Uniform" names the **noise distribution**, not the attention pattern. The
noise state is a *random valid token* drawn uniformly from the whole
vocabulary — not a special `<mask>` symbol and not Gaussian static. Three
consequences:

1. Every position always carries a plausible-looking token, so the model never
   sees an out-of-vocabulary placeholder and the embedding table does double
   duty for clean and noisy inputs.
2. The corruption is **symmetric over tokens**: no token is privileged the way
   `<mask>` is in BERT-style absorbing schemes. The design comparison table
   (uniform vs `[MASK]` absorbing state) is in `DIFFUSION.md` §8.3.
3. Commit-and-renoise decoding works: at any step the sampler may keep some
   positions and redraw pure uniform elsewhere
   (`inference/generate.py:BlockDiffusionSampler.denoise_canvas`) — the state
   after that redraw is just another legal `xt`, the same family training saw.
   The sampler redraws the uniform distribution itself; `q_sample` is
   train-time only.

## 3. Why blocks — the canvas structure

A full-sequence diffusion LM denoises all 4096 training positions jointly,
which is powerful but gives no left-to-right structure for generation. This
repo factors the sequence into `canvas_len = 256`-token **canvases**
(`models/transformer.py:DiffusionGemmaConfig.canvas_len`) and makes the
factorization explicit in attention:
`models/mask.py:build_block_causal_mask` allows **causal** attention across
canvases (canvas `b` sees only canvases `0..b−1`) and **bidirectional**
attention within a canvas. One dense transformer therefore does two jobs at
once: a conditional LM across canvases, a parallel denoiser within one.

Why this is the sweet spot for a single-GPU budget: generation advances 256
tokens per block-autoregressive step (one denoise pass per canvas), while the
causal-across-canvases structure keeps every canvas a valid conditional
continuation of the finalized prefix. The noise level is also injected **per
canvas** through `models/time_embed.py:CanvasTimeEmbedding`, so one training
row of 16 canvases covers 16 corruption levels at once — a variance-reduction
choice over a single shared `t` (per `DIFFUSION.md` §8.3).

## 4. What the denoiser outputs: x̂0

The network is trained to predict the **clean tokens** `x0` directly from
`(xt, t)` — x0-parameterization — with cross-entropy against `x0`
(`models/diffusion.py:x0_ce_loss` is the eager reference;
`training/losses.py:chunked_x0_ce` is the production loss). The softmax over
that output *is* the posterior `p(x0 | xt, t)` the sampler needs: it gives a
per-position distribution to commit from, with no trajectory marginalization.
The plan document's "predict xt" wording is a recorded typo (Ruling 19,
`DIFFUSION.md` §1.2). Generation itself is the reverse walk: start a canvas at
pure uniform (`t = T`), apply the denoiser `T_eval` times, commit as you go —
that loop is the subject of [canvas-denoising](canvas-denoising.md).

## 5. Goes deeper

- Full derivation, transition-matrix view, schedule numerics:
  [diffusion-core](diffusion-core.md).
- Lineage (AR → diffusion → D3PM → block-AR) and the worked forward pass:
  [foundations](foundations.md) §2–§4.
- Rulings and deltas vs upstream practice: `DIFFUSION.md` §1, §8.
- Property pins: `tests/test_diffusion.py::test_forward_process_marginals`,
  `test_final_step_pure_noise`, `test_q_sample_identity_when_alpha_one`.
