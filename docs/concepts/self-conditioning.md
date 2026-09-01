# Concept: self-conditioning

> Full context: [`DIFFUSION.md`](../../DIFFUSION.md) §3. This page is the
> concept-level walkthrough.

## The idea

Denoising improves when the model can see its own previous guess. During
training, with probability `p = 0.5`, a step is made two-pass:

```
pass 1 (no grad):  ĥ = f(xt, t)              → sc = p̂ @ E   (posterior re-embedded)
pass 2 (grad):     h  = f(xt, t, sc = detach(sc))
```

The conditioning input is the previous posterior re-embedded through the
embedding matrix — a `d_model`-shaped summary of *what the model currently
believes*, not a one-hot guess.

## Zero-init equivalence

`models/selfcond.py:SelfConditioning` is a **zero-initialized** linear proj
whose output is added to the normalized hidden state. At init the projection
outputs exactly zero, so

```
f(xt, t, sc) == f(xt, t)   bit-for-bit (fp64, atol=0, rtol=0)
```

pinned by `tests/test_self_conditioning.py::test_zero_init_equivalence`.
Training therefore starts exactly at the no-self-cond model and learns how
much to use the conditioning. Naive init loops would silently break this —
`models/transformer.py:DiffusionGemma`'s `_init_weights` re-zeros the proj
after generic init.

## Routing invariant (exactly one add per path)

- `models/transformer.py:DiffusionGemma.final_hidden` = backbone + the W_sc add — **the loss
  path** (`training/losses.py:chunked_x0_ce` reads this).
- `models/transformer.py:DiffusionGemma.head_forward` = same single add, then LM head.

Never compose the two — that would add the conditioning twice (SDD Ruling 6;
the two-step overfit and wiring tests would catch it).

## The detached pre-pass

Pass 1 runs under `torch.no_grad()` and enters pass 2 **detached** (DESIGN
§2.4): the pre-pass constructs an input, it is not a second gradient path.
Memory-wise the pre-pass's `p @ E` is the full-vocab hazard — handled by
`training/losses.py:chunked_p_embed` under `no_grad`.

## Cross-step conditioning at eval

During sampling, the conditioning is **cross-step**: step `k` receives
`sc_next = p̂ₖ₋₁ @ E` — the previous step's posterior re-embedded — always on
after canvas step 1 (`inference/generate.py:BlockDiffusionSampler._denoise_step`). This is the
mechanism that makes iterative commit-and-renoise coherent: each step knows
what the previous step believed, so sharpening (commit) and re-noising
(uncommitted positions) stay coherent with the frozen prefix.
