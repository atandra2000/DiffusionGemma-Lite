# Concept: the self-conditioning mechanism in one page

> **Audience: intermediate.** The mechanism, compactly: what the self-cond
> input *is*, the zero-init that makes it a no-op at birth, exactly when it is
> fed (train gate vs eval cross-step), and the train/test symmetry. The full
> treatment — alternatives considered, worked examples, routing derivations —
> is the canonical [self-conditioning](self-conditioning.md); `DIFFUSION.md`
> §3 stays authoritative.

**Depends on:** [block-diffusion](block-diffusion.md) §4 ·
**Read next:** [self-conditioning](self-conditioning.md) for the deep dive

---

## 1. What the conditioning input is

The model denoises better when it can see its own previous guess. The guess
is re-expressed in embedding space: `sc = p̂ @ E`, the softmax-weighted mean
of embedding rows under the model's own posterior.
`models/selfcond.py:SelfConditioning.embed` computes exactly that
(`softmax(h_norm @ E.T) @ E`), and
`tests/test_loss.py::test_chunked_p_embed_matches_eager` pins the chunked
training-path equivalent, `training/losses.py:chunked_p_embed`, to it.

## 2. Zero-init: the model is born without conditioning

`models/selfcond.py:SelfConditioning` wraps the input in a `d_model →
d_model` linear (`SelfConditioning.proj`) whose weight and bias are zeroed at
init, and `SelfConditioning.forward` returns `h_norm + proj(sc_input)` —
which is *exactly* `h_norm` at initialization, so the conditioned and
unconditioned models are the same model on day one and training *learns* how
much to trust the input. This is pinned bit-for-bit in fp64 by
`tests/test_self_conditioning.py::test_zero_init_equivalence` with
`atol=0, rtol=0`; the same zeroing is re-applied after weight init by
`models/transformer.py:DiffusionGemma._init_weights` so weight tying cannot
clobber it.

Routing matters as much as the zero: the add happens **exactly once per
path**. `models/transformer.py:DiffusionGemma.final_hidden` (backbone + the
single add) is the loss path; `models/transformer.py:DiffusionGemma.head_forward`
applies the same single add before the LM head. Never compose the two
(Ruling 6).

## 3. When the input is fed

**Training** (`training/pretrain.py:Pretrainer.diffusion_loss`): with
probability `self_cond_p = 0.5` per micro-step
(`models/transformer.py:DiffusionGemmaConfig.self_cond_p`) a *pre-pass* runs
under `torch.no_grad()`, computes `sc_input = chunked_p_embed(...)` from the
unconditioned hidden state, and the gradient-bearing pass runs with that
`sc_input` **detached** — the pre-pass is an input constructor, not a second
gradient path. All draws (the gate coin included) come from
`training/pretrain.py:Pretrainer._step_rng`, so resumed runs replay a step's
conditioning bit-for-bit.

**Evaluation** (`inference/generate.py:BlockDiffusionSampler._denoise_step`):
conditioning is **cross-step** — each denoise step's input is the *previous*
step's posterior re-embedded (`sc_next = p @ E`), always on from canvas
step 2 onward. The model literally conditions on its last guess while
refining ([canvas-denoising](canvas-denoising.md) §2, step 4).

## 4. Train/test symmetry

Both regimes push the model's own posterior through the *same* single
`W_sc` add at the *same* place (post-final-norm hidden, pre-head), so the
only asymmetries are the sampling of the input, not its form or path:

| | training | eval denoise |
|---|---|---|
| source of `p̂` | pass 1 of the same step (unconditioned) | previous step's posterior |
| gradient | detached (`no_grad` pre-pass) | whole loop under `no_grad` |
| on/off | Bernoulli gate at `self_cond_p = 0.5` | always on after step 1 |

The gate trains the model to be useful *with or without* `sc_input`, which
is what makes the always-on eval behavior and the step-1 (no-input) behavior
two points on one learned function rather than two modes.

## 5. Why a weighted mean, not the argmax token

Re-embedding the *sampled* token would make the conditioning input
non-differentiable and noisy; `p̂ @ E` instead returns a convex combination of
embedding rows with weights summing to 1 — a smooth relaxation of "re-embed
your best guess" that keeps a gradient path into the posterior that produced
it (`tests/test_self_conditioning.py::test_embed_is_softmax_weighted_mean`
pins the weights' positivity and unit sum). It also degrades gracefully: as
the posterior sharpens, the mean approaches the single embedding of the
argmax token — the test's saturation probe drives a posterior one-hot and
recovers exactly that row. At eval the sampler computes the same object
inline as `sc_next` in
`inference/generate.py:BlockDiffusionSampler._denoise_step`, so training and
inference feed the model the same *kind* of input, not merely the same shape.

## 6. Goes deeper

Alternatives rejected (plain concatenation, two-pass sampling), the
saturation property of the posterior mean, and the memory hazard that
motivates `training/losses.py:chunked_p_embed`:
[self-conditioning](self-conditioning.md) §1, §4, §7.
