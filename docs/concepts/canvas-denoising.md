# Concept: canvas denoising — one 256-token canvas, step by step

> **Audience: intermediate.** How a single canvas is generated in parallel:
> the starting state, the per-step semantics (posterior → commit → re-noise),
> the mask and visibility story, and how the canvas joins the KV cache.
> `DIFFUSION.md` §4 stays authoritative for rulings; the sampler-system view
> (temperature, entropy bond, FLOPs) lives in [sampler](sampler.md).

**Depends on:** [block-diffusion](block-diffusion.md) ·
**Read next:** [sampler](sampler.md), [self-conditioning-mechanism](self-conditioning-mechanism.md)

---

## 1. The canvas's initial state

Generation advances one canvas at a time
(`inference/generate.py:BlockDiffusionSampler.generate`): prefill the prompt
once, then per canvas — denoise, commit, re-encode. The canvas starts as
**pure uniform noise**: `denoise_canvas` draws every one of the 256 positions
uniformly from the 50,257-token vocab (`torch.randint`). This is deliberately
*not* `models/diffusion.py:q_sample` — there is no `x0` to corrupt at
generation time; the sampler redraws the `t = T` marginal itself (Ruling 17).

## 2. The per-step semantics

Each of the `T_eval = SamplerConfig.n_diffusion_steps` steps (default 32,
`inference/generate.py:SamplerConfig`) runs the same four beats in
`inference/generate.py:BlockDiffusionSampler._denoise_step`:

```
# illustrative — the step's control flow; the code is the contract
p      = softmax(logits)                          # x̂0 posterior, all 256 rows at once
commit = (am == prev_am) | (conf >= prev_conf)    # unchanged-or-stronger rule
new    = commit & ~old_committed                  # fresh commits only: frozen stays frozen
x      = where(new, x̂0, where(old_committed, x, uniform_noise))
```

1. **One forward, 256 rows.** The whole canvas passes through the backbone in
   one forward (`inference/generate.py:BlockDiffusionSampler._canvas_step_logits`),
   attended under `models/mask.py:build_canvas_decode_mask` — an **all-ones**
   view over the finalized prefix plus the in-flight canvas. Nothing is
   masked *out*: the canvas's visibility is total, its *content* is what is
   still noisy (Rulings 15–17, `DIFFUSION.md` §2.2).
2. **x0 prediction, used as a posterior.** The model is an x0-predictor
   ([block-diffusion](block-diffusion.md) §4), so the softmax at each position
   is `p(x0 | xt, t)` — the posterior the commit rule consumes.
3. **Commit, never overwrite.** A position commits when its posterior mode is
   unchanged since last step or its confidence grew. Once committed it is
   frozen for the canvas's lifetime — `new = commit & ~old_committed` — and
   uncommitted positions are redrawn pure uniform. Monotonicity is pinned by
   `tests/test_sampler.py::test_commit_rule_monotone`.
4. **Self-conditioning input for the next step.** The step hands its
   posterior (`p @ E`, re-embedded) forward as the next step's conditioning
   input — the cross-step mechanism in
   [self-conditioning-mechanism](self-conditioning-mechanism.md) §3.

Time is injected per canvas at the *schedule* value: step `k` runs with
`t = T_eval − k` while `time_steps` stays at `T_eval`, keeping `t/T ∈ (0,1]`
in the range `models/time_embed.py:CanvasTimeEmbedding` was normalized to
(Ruling 16).

## 3. Endgame and bookkeeping

- The returned canvas is `where(committed, x, x̂0)`: every position ends at
  its best posterior guess even if it never triggered the commit rule.
- With `adaptive=True` (the default) the loop may stop early when mean
  posterior entropy stays below `entropy_threshold` for `stability_steps`
  consecutive steps; `adaptive=False` is bit-identical to the fixed schedule
  (`tests/test_sampler.py::test_adaptive_off_ignores_entropy`). The threshold
  semantics live in [sampler](sampler.md) §4.
- The finalized canvas is then re-encoded by
  `inference/generate.py:BlockDiffusionSampler.encode_canvas` under the same
  all-ones decode view, appending its KV to the cache. The cache therefore
  grows once per canvas (256 tokens per block-AR step), and the chained
  decode must equal a fresh single-shot block-causal forward — pinned
  bit-exactly in fp64 by
  `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`.
- Cost accounting: a fixed schedule spends `1 + n_canvases·(T_eval + 1)`
  token-forwards per generation, i.e. `L/(T_eval+1) ≈ 7.8` tokens per forward
  at `L = 256`, `T_eval = 32` (derived in `DIFFUSION.md` §7).

## 4. Why this schedule is coherent

The commit rule is a conservative consensus filter, not free-form sampling:
a token only freezes when the model's belief about it held steady or
strengthened, so early mistakes require *sustained* evidence to become
permanent. Combined with re-noising the remainder (which keeps every
intermediate state inside the trained corruption family —
[block-diffusion](block-diffusion.md) §2) and the entropy bond (which stops
spending forwards once the canvas has settled), the loop turns one dense
forward pass into 256 tokens of conditioned text. What breaks if you perturb
each piece: [sampler](sampler.md) §6 and `DIFFUSION.md` §4.
