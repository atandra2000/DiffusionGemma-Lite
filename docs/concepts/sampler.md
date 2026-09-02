# Concept: the sampler — commit-and-renoise over KV-chained canvases

> **Canonical** for the decode loop, the commit rule, temperature annealing,
> the entropy bond, and FLOP accounting. `DIFFUSION.md` §4 stays authoritative
> for final rulings; this page derives the loop from first principles.

**Depends on:** [foundations](foundations.md) §4, §7 ·
[diffusion-core](diffusion-core.md) §4–5 ·
[block-causal-attention](block-causal-attention.md) §6 ·
[self-conditioning](self-conditioning.md) §7 ·
**Read next:** [training](../training.md) · [inference](../inference.md) ·
[sampler-tuning guide](../guides/sampler-tuning.md) ·
[R6 API reference](../references/R6_sampler_eval_api.md)

---

## Table of Contents

1. [The decode loop, end to end](#1-the-decode-loop-end-to-end)
2. [The commit rule, derived](#2-the-commit-rule-derived)
3. [Temperature annealing and the Gumbel draw](#3-temperature-annealing-and-the-gumbel-draw)
4. [The entropy bond (adaptive stopping)](#4-the-entropy-bond-adaptive-stopping)
5. [FLOP accounting and the AR baseline](#5-flop-accounting-and-the-ar-baseline)
6. [Worked example: one canvas at toy scale](#6-worked-example-one-canvas-at-toy-scale)
7. [What breaks if you change this](#7-what-breaks-if-you-change-this)
8. [Glossary](#8-glossary)
9. [Interview Q&A](#9-interview-qa)

---

## 1. The decode loop, end to end

The sampler is the eval-time twin of the training-time block-AR factorization:
text is produced **canvas by canvas** — a 256-token window is denoised in
parallel from uniform noise over a small schedule, frozen, re-encoded into the
KV cache, and the next canvas is conditioned on everything finalized so far.
One forward refines 256 tokens instead of 1; every mechanism in this chapter
exists to keep that ratio without wrecking quality.

`inference/generate.py:BlockDiffusionSampler.generate` is four moving parts in
a fixed rhythm — prefill **once**, then per canvas: denoise → encode → slice
into the output buffer:

```python
# inference/generate.py:BlockDiffusionSampler.generate (structure)
kv, prefix_len = self.prefill(prompt_ids)          # 1 forward over the prompt
for _ in range(ceil(max_new_tokens / L)):          # one iteration per canvas
    canvas, _, _ = self.denoise_canvas(kv, prefix_len)   # T_eval (+streak) forwards
    kv, prefix_len = self.encode_canvas(kv, prefix_len, canvas)  # 1 forward + KV append
    out[:, filled:filled+n] = canvas[:, :n]         # preallocated buffer, no cat
```

| stage | symbol | forwards | what it does |
|---|---|---|---|
| prefill | `inference/generate.py:BlockDiffusionSampler.prefill` | 1 | prompt through the block-causal backbone; KV cache becomes *static* |
| denoise | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` | T_eval (+ early stop) | uniform noise → clean canvas; cache untouched, mask = all-ones |
| encode | `inference/generate.py:BlockDiffusionSampler.encode_canvas` | 1 | finalized canvas re-encoded; per-layer `cat(past_kv, new_kv)` |

### 1.1 Shapes: toy and production

All sampler tensors are batch-shaped `(B, …)`; the tables below use `B = 1`.
"Toy" is the repo's test fixture (`tests/conftest.py::tiny_cfg`: 2 layers,
d_model 64, 4Q/2KV heads of dim 16); "production" is
`models/transformer.py:DiffusionGemmaConfig` at the A100 settings (24 layers,
d_model 1024, GQA 16Q/4KV, head_dim 64).

| quantity | toy | production |
|---|---|---|
| canvas length `L` | 32 | 256 |
| vocab `V` | 256 | 50,257 (GPT-2 BPE) |
| d_model | 64 | 1024 |
| layers × (heads) | 2 × (4Q/2KV, head_dim 16) | 24 × (16Q/4KV, head_dim 64) |
| `max_seq_len` | 128 = 4 canvases | 4096 = 16 canvases |
| eval schedule `T_eval` | up to 32 | 32 (`eval_diffusion_steps`) |
| KV per layer, per cached token | 2 · 2 · 16 = 64 floats | 2 · 4 · 64 = 512 floats |
| full KV cache at `max_seq_len` | ~0.5 MB | ~0.2 GB fp32 (half in bf16) |
| logits per denoise step | (1, 32, 256) ≈ 32 KB | (1, 256, 50257) ≈ 51.5 MB fp32 |

The last row is why the sampler stays eager and unchunked at eval: one
`(1, 256, 50257)` logits tensor plus its posterior `p` and clamped twin `q`
is ~150 MB of transient fp32 — trivial at inference batch sizes, so the
training-time full-vocab hazards that forced
`training/losses.py:chunked_x0_ce` and `training/losses.py:chunked_p_embed`
(see [memory-engineering](memory-engineering.md)) do not bind here.

### 1.2 Prefill: one forward, partial-canvas mask, static cache

The prompt is finalized content, so it enters at `t=0` — and at `t=0` the
canvas-time embedding is T-independent by construction
([diffusion-core §5](diffusion-core.md)), exactly like the re-encoded canvases
later. The prefill's one wrinkle: prompts are rarely a multiple of 256 tokens,
but `models/mask.py:build_block_causal_mask` asserts
`seq_len % canvas_len == 0`. The sampler therefore builds its own
partial-canvas mask with *identical row semantics* —
`inference/generate.py:_prefix_mask` computes, for prompt row `i` and key `k`,
`allow = (k < floor(i/L)·L) | (floor(k/L) == floor(i/L))` over the prompt
length only — so a 3-token prompt is one fully-bidirectional 3-row block
rather than a padded 256-token canvas. Under the flex path the sampler skips
the inline tensor entirely: `models/mask.py:build_block_causal_block_mask`
handles a partial first canvas natively (its `mask_mod` rule is per-element).

After prefill, `kv` (per-layer `(k, v)` with k already roped — see
[attention internals](block-causal-attention.md)) is **static**: no prompt
token is ever reprocessed, and positions are absolute
(`inference/generate.py:BlockDiffusionSampler._positions`), so RoPE phase
continues across canvases with no offset bookkeeping.

### 1.3 The canvas is a position window, not a mask block

`denoise_canvas` always works on exactly `L` fresh positions
`[prefix_len, prefix_len + L)`. Those positions do **not** have to line up
with a mask block: a 3-token prompt means the first generated canvas is
positions 3–6, which straddles block 0's last slot and block 1's first three.
Two pieces of plumbing make straddling a non-event:
`inference/generate.py:BlockDiffusionSampler._span_t` computes `n_span` —
the number of mask-blocks the chunk touches — and returns a `(B, n_span)`
time column with the *same* `t` in every entry, because the time embedding is
indexed per canvas (`models/transformer.py:DiffusionGemma.backbone` gathers
it by `positions // canvas_len`); and the decode mask is all-ones regardless
of alignment — `models/mask.py:build_canvas_decode_mask` returns an all-ones
`(1, 1, L, prefix_len + L)` view: finalized prefix fully visible, in-flight
canvas fully visible to itself. There is no zero-masking of the canvas
against itself; the *content* is what's noisy, not its visibility. Under flex
the sampler passes `mask=None` instead — no block mask *is* full attention,
the same semantics without building the tensor
(`inference/generate.py:BlockDiffusionSampler._decode_mask`).

### 1.4 Encode: one forward per canvas, and the cache cadence

Once the canvas is frozen, `inference/generate.py:BlockDiffusionSampler.encode_canvas`
re-encodes the finalized tokens at `t=0` under the same all-ones decode view
and appends the new per-layer `(k, v)` to the cache via
`models/attention.py:DenoiseAttention._roped_qkv`'s `torch.cat`. The KV cache
is written **once per canvas** (256 tokens per write), not once per token —
that is the entire speedup story (§5). Two properties worth stating exactly:
the cache shape is **identical to AR's** at equal sequence length (all tokens
are eventually encoded; the win is *fewer forward passes*, not a smaller
cache), and the encode forward is **not optional** — denoise steps run with
`x` still noisy under the in-flight decode view, so their KV states are
noise-conditioned, and reusing them would break the fp64 chaining
equivalence (`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).
The frozen canvas gets its own forward.

### 1.5 Output: preallocated buffer, remainder slicing

`generate` allocates the full output `(B, prompt + max_new_tokens)` once and
writes each canvas with `out[:, filled:filled+n] = canvas[:, :n]`, where
`n = min(L, total - filled)` handles the final partial canvas — no per-canvas
`torch.cat`, so an O(total) copy per canvas disappears. Why one encode forward
is *necessary* rather than an optimization target: the next canvas's denoise
forwards attend over the finalized prefix through the KV cache, and those
states must equal what a fresh single-shot block-causal forward over
`prompt + finalized canvas` would write — enforced bit-exactly in fp64 by
`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`. Every
design choice below (commit semantics, all-ones re-encode, `t=0` encode)
exists to keep that equivalence.

## 2. The commit rule, derived

One denoise step (`inference/generate.py:BlockDiffusionSampler._denoise_step`):

```
logits = f(x, t, sc)             # (B, L, V)
p      = softmax(logits)         # posterior over the clean canvas x̂0
conf, am = p.max(-1)             # per-position confidence and argmax
commit = (am == am_prev) | (conf >= conf_prev)
```

### 2.1 Why commit at all

Uniform-state diffusion defines a reverse process where each step re-draws
uncommitted positions. Without commitment, every step would re-sample **all**
256 positions from scratch: tokens the model already settled would be exposed
to another round of uniform re-noising and could regress. Committing freezes a
position once the model is convinced, so refinement accumulates — the
per-step budget goes to the unresolved positions only.

The deeper reason it is *legal*: any sampler state — some positions holding
final values, the rest uniform random tokens — is distributionally a
`models/diffusion.py:q_sample` draw. Training corruption keeps `x0` with
probability `ᾱ(t)` and replaces the rest with uniform tokens
(`models/diffusion.py:corruption_probs`); a canvas with fraction `f`
committed looks exactly like a corruption at `ᾱ ≈ fraction-kept`. Because the
model is trained at **every** `t` in `{1..T}` (`models/diffusion.py:sample_canvas_t`
samples per canvas), every state the sampler visits is in-distribution. The
sampler even walks the diagonal honestly: it starts at `t = T_eval`, where
`models/diffusion.py:alpha_bar` gives `ᾱ(T) ≈ 0` (the model expects
near-pure noise — and `x` *is* pure `torch.randint` noise), and anneals `t`
down to 1 as the canvas fills in. The noise is drawn directly rather than via
`q_sample`, which corrupts a known `x0` and is train-time only.

### 2.2 Which positions are "good enough to freeze"

The design question: which positions survive the next re-noising? The rule
that falls out of the block-AR factorization is **unchanged-or-stronger**:
freeze a position when its argmax is stable across steps
(`am == am_prev`) *or* when its posterior mass has grown
(`conf >= conf_prev`). Both clauses are monotone predicates on the posterior
sequence — pinned by `tests/test_sampler.py::test_commit_rule_monotone`.

Why "unchanged-or-stronger" rather than a bare confidence threshold: the
model's confidence is not calibrated across timesteps (early steps see pure
noise, late steps see nearly-clean canvases) — at `t = T_eval` *every*
posterior is near-uniform and a 0.3 threshold would freeze argmax coin-flips
immediately. The relative rule uses only the *comparison between consecutive
steps*, which is well-defined at every `t`: "this position's belief did not
get worse." Its natural failure mode works in our favor: a position whose
argmax keeps flipping never satisfies the stability clause and keeps getting
re-noised until the annealing temperature (§3) sharpens the posterior enough
for a mode to persist.

Step 0 is special-cased by absence of history: `prev is None` →
`commit = all-False` (asserted by
`tests/test_sampler.py::test_commit_rule_monotone`) — every position gets one
free exploration step before anything can freeze.

### 2.3 Once frozen, always frozen — value semantics vs mask semantics

The committed set only grows (`committed |= commit`) — and the value written
is the *step's fresh posterior only for positions not yet frozen*
(`new = commit & ~old_committed`):

```python
x = torch.where(new, x0, torch.where(old_committed, x, noise))
#                  ↑ fresh commit    ↑ already frozen: keep   ↑ else re-noise
```

Three-way routing, exactly one branch per position — the sampler-side twin of
self-conditioning's exactly-once add. The distinction between `commit` (the
predicate fires) and `new` (the value is rewritten) is load-bearing: a frozen
position can satisfy the commit predicate again on later steps, but `new`
masks it out, so its token is never touched again. This is what
`tests/test_sampler.py::test_commit_rule_monotone` pins verbatim:
`xs[2][cs[1]] == xs[1][cs[1]]` — committed values frozen byte-for-byte across
steps.

The honest cost: the rule has **no undo**. A wrong commit is frozen for the
rest of the canvas. Two mechanisms bound the damage: the annealed Gumbel draw
(§3) delays freezing while `tau` is high, and the `am == am_prev` clause
demands the mode survive a *full re-noise of everything else* — the model
re-derives it from a fresh random context. What does *not* protect you:
committed-but-wrong tokens bias the next forward's context (the state is then
slightly off the `q_sample` marginal, whose "clean" tokens are true `x0`) —
a real, accepted skew, and the reason the entropy bond (§4) and a stricter
`entropy_threshold` are the quality levers, not the commit rule.

### 2.4 The final `torch.where`

The returned canvas is `torch.where(committed, x, x0)`: every position ends
with its best posterior guess even if it never "committed" — uncommitted
positions hold their last uniform re-noise in `x`, and returning that would
ship pure noise tokens. `x0 = am` unless the temperature draw (§3) overrides
it, so at `tau > 0` the *drawn* token, not the argmax, is what a fresh commit
takes. The final `where` runs after the loop, once, in
`inference/generate.py:BlockDiffusionSampler.denoise_canvas`.

Outcome table for a position, per step:

| state | predicate | value written |
|---|---|---|
| fresh commit | `commit & ~old_committed` | this step's `x0` (drawn or argmax) |
| already frozen | `old_committed` | previous value, untouched |
| uncommitted | neither | fresh `torch.randint` uniform token |
| never committed (end of canvas) | `~committed` at return | last `x0` via the final `torch.where` |

## 3. Temperature annealing and the Gumbel draw

Temperature is linear-annealed per step `k` of `T_eval`
(`SamplerConfig.temp_start=0.8 → temp_end=0.4`,
`inference/generate.py:SamplerConfig`):

```
tau_k = temp_start - (temp_start - temp_end) / T_eval * k
# T_eval=32: tau = 0.8 → 0.7875 (k=1) → 0.6 (k=16) → 0.4125 (k=31)
```

Note the literal formula never reaches `temp_end`: the last step is
`k = T_eval - 1`, so `tau` stops one increment short (0.4125, not 0.4).
Recorded upstream as a deferred minor; at toy scale (§6, `T_eval=3`) the gap
is visible as `0.8 → 0.667 → 0.533`.

### 3.1 The Gumbel-max draw, derived

When `tau > 0`, the draw is a **Gumbel-max trick**: sample from
`softmax(logits / tau)` as one elementwise op + argmax —

```python
gumbel = -log(-log(U(0,1)))
x0 = (log(q) / tau + gumbel).argmax(-1)
```

Why this samples `softmax(logits/tau)`: the Gumbel–max theorem says that for
probabilities `π`, `argmax_i(log π_i + G_i)` with `G_i ~ Gumbel(0,1)`
returns `i` with probability exactly `π_i`. With temperature we want
`π ∝ p^{1/tau} = softmax(logits/tau)`, i.e. `log π_i = logits_i/tau + const`.
The code computes `q.log() / tau + gumbel` where `q = p.clamp_min(1e-12)`;
since `log p_i = logits_i − logsumexp(logits)` and the lse term is
**constant across the row**, dividing by `tau` leaves it a per-row constant,
and argmax is shift-invariant per row — so the lse subtraction is skipped
entirely. That is the "scores are shift-invariant, no lse needed" comment in
`inference/generate.py:BlockDiffusionSampler._denoise_step`, and it is also
why the same clamped `q` serves both the entropy (§4) and the draw: one
`clamp_min(1e-12)`, two uses. The clamp exists because `log(0) = −inf` would
poison the Gumbel scores; at `1e-12` the log is ≈ −27.6, finite, and the
entropy bias is ~`1e-12 · 27.6` per clamped entry — negligible.

The `tau > 0` guard is the degenerate case: at `tau = 0` the draw collapses
to `x0 = am`, the plain argmax (`tests/test_sampler.py::test_gumbel_temperature_draw`
asserts `x0_low == logits.argmax(-1)` at `tau=0.01`). The anneal therefore
walks the sampler from exploration to exploitation *within* each canvas:
at `tau = 0.8` the draw frequently disagrees with the argmax (the same test
shows `tau=20` spreading draws over >50% of the vocab), while by the last
steps only genuinely competing tokens can win.

### 3.2 Why Gumbel-max and not multinomial

The old path was `torch.multinomial` over `(rows × 50k)` — correct, but a
slow serial kernel at that width. The Gumbel form is one `torch.rand` +
log + add + argmax, elementwise over the same `(B, L, V)` tensors the softmax
just produced, no per-row sampling loop. Equivalence is pinned statistically
by `tests/test_sampler.py::test_gumbel_temperature_draw` (near-argmax at low
`tau`, vocab-wide spread at high `tau`).

### 3.3 Randomness plumbing, and why anneal *within* the canvas

All stochasticity — the initial `torch.randint`, the Gumbel `U`, the re-noise
`randint` — flows through one `torch.Generator`
(`inference/generate.py:BlockDiffusionSampler._generator`), seeded once per
sampler when `SamplerConfig.seed` is set: reproducible decode is a config
knob, not a global-RNG discipline.

Why anneal at all? A fixed `tau` per canvas is the obvious alternative, and
it fails at both ends. Fixed high `tau` keeps drawing sample-y tokens even
when the posterior is settled — the final canvas is noisier than the model's
own belief. Fixed `tau → 0` is greedy-with-renoise: the first step's argmax
freezes immediately (`am == am_prev` is trivially satisfiable by coincidence)
and cascades — exactly the failure the tuning guide warns about for
`temp_start=0`. The linear ramp matches the commit rule's needs: exploration
while the canvas is uninformed, exploitation once the context is mostly
frozen.

## 4. The entropy bond (adaptive stopping)

Fixed schedules always spend T_eval steps per canvas. The adaptive path
(`SamplerConfig.adaptive=True`, `entropy_threshold=1.0`,
`stability_steps=2`) stops early when the canvas's posterior is *settled*:

```
entropy_k = mean over canvas rows of −Σ p·log p     # (B,) per step
if entropy < 1.0 for `stability_steps` consecutive steps: break
```

The signal is computed every step anyway —
`inference/generate.py:BlockDiffusionSampler._denoise_step` returns it as
part of its step tuple — but the *stopping* logic in
`inference/generate.py:BlockDiffusionSampler.denoise_canvas` reads it only
when `adaptive=True`. The streak resets to 0 on any step at or above
threshold; two consecutive sub-threshold steps are required to fire.

### 4.1 Why entropy and not mean confidence

Mean max-confidence saturates. Consider a canvas of 256 positions where 250
are essentially decided (max-prob 0.99, residual tail spread over the vocab)
and 6 are still nearly undecided (max-prob 0.30):

- mean max-confidence = (250·0.99 + 6·0.30)/256 ≈ **0.974** — statistically
  indistinguishable from the fully-settled canvas's 0.99 at any threshold you
  would dare to set. Confidence lives in `[0, 1]`, so its dynamic range
  compresses to a few percent exactly in the regime the bond must
  discriminate.
- mean entropy = (250·0.164 + 6·8.19)/256 ≈ **0.35 nats**, versus 0.16 for
  the settled canvas — a 2× separation on a scale that runs from 0 to
  `ln V ≈ 10.82` nats. Entropy integrates the *entire* posterior per
  position — the residual 1% tail over 50k tokens still counts — so hard
  positions move the mean by multiples, not by percent.

At the other anchor point, a uniform posterior over V = 50,257 has entropy
`ln V ≈ 10.825` nats; the 1.0-nat threshold reads as "effectively ~e^1 ≈ 2.7
tokens of remaining uncertainty per position" (entropy in nats is
log-perplexity).

### 4.2 The inert-at-init caveat

The entropy bond is **inert on untrained weights**: an untrained head is
near-uniform, mean entropy sits at ≈ `ln V` ≈ 10.8 nats (≈ 5.5 on the tiny
fixture), far above any sane threshold — so the bond never fires and the
"adaptive" schedule degrades to exactly the fixed schedule. The mechanism
tests therefore do not rely on a trained model:
`tests/test_sampler.py::test_adaptive_stopping_fires` injects a converged
head (per-position one-hot logits) and asserts the bond fires at
`steps_used == 2` — one step under threshold plus the `stability_steps`
confirmation — out of `T_eval=8`;
`tests/test_sampler.py::test_adaptive_threshold_calibratable` proves the
threshold is calibratable by construction on the *untrained* model (at
threshold 0.1 the full schedule runs, at threshold 10⁶ it stops at exactly
2 steps); and `tests/test_sampler.py::test_adaptive_off_ignores_entropy` pins
the contract — with `adaptive=False` even a threshold of 10⁶ never shortens
the schedule, the fallback being bit-identical to the fixed sampler
(`tests/test_sampler.py::test_fixed_sampler_runs_full_schedule` exercises the
same path end to end).

The FLOP win is bounded and one-directional: adaptive ≤ fixed token-forwards
per canvas — `tests/test_inference.py::test_flop_counter_adaptive_le_fixed`.
The floor is `stability_steps` denoise steps + 1 encode: at production scale
the bond can cut a canvas to 3 forwards — `tokens/forward` up to 256/3 ≈ 85×
asymptotically — though in practice the bond fires mid-schedule, not at
step 2.

### 4.3 A closed rule beats a learned halter

Upstream adaptive-compute methods spend RL or distillation budget to learn
*when to stop* (a learned halting policy). This is a closed rule: the
posterior's own entropy *is* the signal — no extra training phase, no reward
model, no halting head. The trade-off is that the threshold is a
hyperparameter, not learned: too low and the bond never fires (you paid for
adaptive and got fixed — provably identical, per the test above); too high
and it fires on garbage. Calibration guidance lives in the
[sampler-tuning guide](../guides/sampler-tuning.md).

## 5. FLOP accounting and the AR baseline

`inference/evaluate.py:SpeedupEvaluator` counts **forwards** and
**token-forwards** by instrumenting `model.backbone`
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`): a context manager
swaps the backbone for a closure that bumps `outer._forwards` and adds
`ids.size(1)` to `outer._token_forwards`, then restores it in `finally`. The
headline metric needs no checkpoint:

- AR + KV cache: **1 forward and 1 token-forward per token** — by
  construction, so `tokens/forward = 1.0` analytically (the honest gap:
  AR wall-clock needs an external checkpoint, disclosed as `seconds=None`,
  `inference/evaluate.py:SpeedupEvaluator.evaluate`).
- Block-AR: `(T_eval + 1)` denoise+encode forwards per canvas plus one
  prefill → `forwards = 1 + n_canvases·(T_eval + 1)` for a run.

### 5.1 What the counters actually count

Three scope decisions matter when reading the numbers. First, the numerator
of `tokens_per_forward` is generated tokens only (`n_samples · gen_tokens`)
while `forwards` includes the prefill — so the measured ratio is the
*end-to-end* figure, slightly below the per-canvas asymptote `L/(T_eval+1)`,
and the prefill amortizes away as `gen_tokens` grows. Second,
**token-forwards count incoming chunk lengths only**: a decode forward
contributes `L` (the canvas), not `prefix_len + L` — attention over the
cached prefix is served from KV, not reprocessed, so the proxy measures
forward passes × new tokens, robust to sampler internals. Third, **the vocab
head is not counted** — the wrapper instruments `model.backbone`, so
`head_forward`'s `h @ E.T` GEMM and the sampler's `p @ E` re-embedding are
outside the denominator; the omission is identical on both sides of any
comparison, including the analytic AR row.

### 5.2 The rows

Row names parse via `inference/evaluate.py:parse_baseline`
(`"fixed_T16"` → `SamplerConfig(n_diffusion_steps=16, adaptive=False)`).

| schedule | forwards per canvas | tokens/forward (asymptote) | vs AR | row name |
|---|---|---|---|---|
| fixed T=16 | 16 denoise + 1 encode = 17 | 256/17 = **15.06×** | measured row `fixed_T16` |
| fixed T=32 | 33 | 7.76× | `fixed_T32` |
| adaptive T=32 | ≤ 33 (entropy bond) | ≥ 7.76× | `adaptive_T32` |
| AR (analytic) | 1 per token | 1.0× | `ar_kv_analytic` |

Concretely, 1,024 generated tokens = 4 canvases: fixed T=16 spends
`1 + 4·17 = 69` forwards (1 prefill + 68 denoise/encode) vs 1,024 for AR —
**14.8× fewer forwards end-to-end**, approaching the 15.06× asymptote as the
prefill amortizes; at T=32 it is 133 forwards, 7.70× vs the 7.76× asymptote.
`tests/test_inference.py::test_evaluator_produces_three_rows` pins the exact
row structure *and* the formula (`forwards == 1 + canvases·(T+1)`,
`tokens_per_forward == gen/(1 + canvases·(T+1))` on the tiny fixture);
`tests/test_inference.py::test_flop_counter_adaptive_le_fixed` pins
adaptive ≤ fixed in both directions (forwards down ⇒ tokens/forward up).

The per-canvas asymptote is the number to quote at production scale; at test
scale it inverts — on the tiny fixture (`L=32`, `T_eval=32`) the fixed
schedule yields `32/33 < 1` tokens per forward, so **never quote tiny-model
eval numbers** (the same run that demonstrates the harness is a bad
throughput datapoint).

### 5.3 The honest gaps and the quality anchor

- **AR wall-clock** (`tokens_per_sec`) is measured for this sampler only; the
  AR row ships `seconds=None` — it needs an external baseline run on the A100
  pod. `inference/evaluate.py:SpeedupEvaluator._measure` wraps the whole
  `generate` call (prefill included) in `time.perf_counter`, so the block
  rows' `tokens_per_sec` is honest end-to-end decode throughput.
- **Quality** is anchored separately:
  `inference/evaluate.py:heldout_x0_nll` computes the training objective
  (chunked x0-CE via `training/losses.py:chunked_x0_ce` at sampled `t`, no
  self-cond input) on a held-out shard, in nats/token, directly comparable to
  an AR baseline's CE; the ±5% acceptance verdict is
  `scripts/loss_parity_eval.py --ar-nll <AR CE>`. Throughput at acceptable
  quality is the headline — neither number alone is.

### 5.4 Self-conditioning interacts cleanly

`sc_next = p @ E` is computed from the *previous* step's posterior inside
`inference/generate.py:BlockDiffusionSampler._denoise_step` — no extra
transformer forward, only the `p @ E` re-embedding (one `(L, V) @ (V, D)`
GEMM per step, the eval twin of `training/losses.py:chunked_p_embed`, which
exists because the *training-time* full-vocab product would blow the §4.0
memory budget). Two precision points: the conditioning vector is built from
the **posterior** `p`, not the temperatured draw — the next step sees what
the model *believed*, not the sample it happened to commit. And step 1 of
each canvas has no previous posterior and runs unconditioned; zero-init
equivalence (`models/selfcond.py:SelfConditioning`) makes that exactly the
no-sc mode ([self-conditioning §3](self-conditioning.md)). From step 2 on,
`models/transformer.py:DiffusionGemma.head_forward` applies the single
`W_sc` add before the LM head — never composed with the loss path's add
(`models/transformer.py:DiffusionGemma.final_hidden`).

## 6. Worked example: one canvas at toy scale

Dims shrunk to hand-traceable size: `L = 4`, `V = 8`, `T_eval = 3`,
`temp_start = 0.8`, `temp_end = 0.4`, prompt = 3 tokens `[5, 2, 7]`
(a partial first canvas, handled by `inference/generate.py:_prefix_mask`),
`prefix_len = 3`, batch 1. Posterior rows are made-up but the
commit/renoise/anneal logic is exactly
`inference/generate.py:BlockDiffusionSampler._denoise_step`'s. Prefill runs
one forward over the 3 prompt tokens (all-rows-visible: they share one
partial block), the KV goes static, and the first generated canvas is
positions 3–6 — straddling block 0's last slot and block 1's first three, so
`inference/generate.py:BlockDiffusionSampler._span_t` returns a `(1, 2)` time
column (two straddled blocks, same `t`); attention sees prefix + canvas under
the all-ones decode view.

```
init   x = [3, 7, 1, 5]  (uniform draws, V=8)   committed = [F, F, F, F]
       prev = None, sc = None

step k=0, tau = 0.8 - 0.4/3·0 = 0.800, t = T_eval - 0 = 3:
  p (rows) →  am = [2, 0, 5, 1]   conf = [.31, .28, .30, .26]   H = 2.10
  prev = None → commit = [F,F,F,F]        (no history, no commits)
  new = [F,F,F,F]; committed stays [F,F,F,F]
  tau=0.8 Gumbel draw → x0 = [2, 0, 5, 6]   (pos3's draw flipped 1 → 6: exploration)
  no fresh commits, no frozen tokens → ALL positions re-noise:
      x = [6, 2, 4, 0]   (fresh uniform randint)
  sc_next = p @ E           (conditioning for step k=1)
  entropies = [2.10]

step k=1, tau = 0.8 - 0.4/3·1 = 0.667, t = 2:
  forward(x=[6,2,4,0], sc) → am = [2, 0, 3, 0]  conf = [.55, .41, .22, .24]  H = 1.40
  commit = (am==prev_am) | (conf>=prev_conf)
      pos0: 2==2 → T        pos1: 0==0 → T
      pos2: 3≠5, .22<.30 → F    pos3: 0≠1, .24<.26 → F
  new = [T,T,F,F]; committed = [T,T,F,F]
  Gumbel draw at tau=0.667 → x0 = [2, 0, 3, 0]   (matches argmax now)
  x = [2, 0, 4, 6]        (pos0,1 take x0; pos2,3 re-noise)
  H = 1.40 > 1.0 → streak = 0
  entropies = [2.10, 1.40]

step k=2, tau = 0.8 - 0.4/3·2 = 0.533, t = 1:
  am = [2, 0, 5, 3]  conf = [.71, .66, .55, .33]  H = 0.80
  commit: pos0: 2==2 → T; pos1: 0==0 → T
          pos2: 5≠3 but .55≥.22 → T (strength)
          pos3: 3≠0 but .33≥.24 → T (strength)
  new = commit & ~committed = [F,T,T,T]   ← pos0 commits AGAIN but is not rewritten
  committed = [T,T,T,T]
  x = [2, 0, 5, 3]        (pos0 keeps its frozen 2; pos1←0; pos2←5; pos3←3)
  H = 0.80 < 1.0 → streak = 1 < 2 → bond does not fire; loop ends (k=2 was last)
  entropies = [2.10, 1.40, 0.80]

return torch.where(committed, x, x0) → [2, 0, 5, 3]   steps_used = 3
```

Then `inference/generate.py:BlockDiffusionSampler.encode_canvas` spends one
forward re-encoding `[2, 0, 5, 3]` at `t=0` and appends its KV
(`prefix_len 3 → 7`); the second canvas repeats identically; `generate`
slices both into the preallocated `(1, 11)` output buffer.

Take-aways, each visible in the trace: positions 0–1 froze on **stability**
(`am` unchanged), positions 2–3 on **strength** (`conf` grew) — both clauses
earning their keep. Position 0 satisfied the commit predicate again at `k=2`
but was **not rewritten** (`new = commit & ~committed`): once frozen, always
frozen. Had a position never committed, the final
`torch.where(committed, x, x0)` would return its last posterior draw, not its
last random re-noise. The entropy trace never shows two consecutive steps
under 1.0, so the bond does not fire at `T_eval=3` — with `T_eval=32` and a
canvas that settles at `k=9`, the same streak logic exits at
`k+stability_steps`. And `tau` ends at 0.533, not 0.4 — the schedule's last
step stops one increment short of `temp_end` (the deferred minor, visible at
toy scale).

At production scale the same loop runs 256 positions with `tau` annealed
0.8 → 0.4125 over ≤ 32 steps and the entropy bond cutting the schedule short
when the posterior settles (`tests/test_sampler.py::test_commit_rule_monotone`
pins the rule's monotonicity on exactly this pattern).

## 7. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| commit rule reads only `conf >= conf_prev` (drop the argmax clause) | stable-but-dipping posteriors re-noised; quality drops | `tests/test_sampler.py::test_commit_rule_monotone` |
| committed set can shrink (`=` instead of `\|=`) | frozen tokens get overwritten by later noise; KV chain diverges from training distribution | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| temperature draw on committed positions | frozen tokens mutate; commit semantics dead | `tests/test_sampler.py::test_commit_rule_monotone` |
| drop the fp64 chaining equivalence | sampler drifts from the single-shot distribution (subtle quality decay) | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| `multinomial` instead of Gumbel-max | correct but serial-kernel slow over (rows × 50k) | `tests/test_sampler.py::test_gumbel_temperature_draw` (equivalence) |
| entropy read when `adaptive=False` | non-deterministic step count vs fixed schedule | `tests/test_sampler.py::test_adaptive_off_ignores_entropy` |
| re-noise from `x0` instead of uniform noise | committed-quality bias compounds; state leaves the `q_sample` marginal the model was trained on (off-policy) | `tests/test_diffusion.py::test_q_sample_zero_alpha_is_uniform_draw` / `test_q_sample_identity_when_alpha_one` family |
| per-token KV appends (AR-style) | cache-write cadence ×256; the block-AR win gone | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (still passes; throughput regresses) |
| off-by-one in the streak (`stability_steps` ±1, or no reset on threshold breach) | bond fires one step early/late or stops on a single dip | `tests/test_sampler.py::test_adaptive_stopping_fires` (exact `steps_used == 2`), `test_adaptive_threshold_calibratable` (exact counts at both extremes) |
| encode at the last denoise `t` instead of `t=0` | finalized tokens enter with a nonzero time embedding, off the training contract (finalized content enters at `t=0`) | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (single-shot reference uses `t=0` for the finalized canvas) |
| prompt prefill through `build_block_causal_mask` with padded canvas | partial prompt gets phantom key slots; wrong prefix KV | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (chained logits must equal the single-shot forward) |

## 8. Glossary

| symbol | meaning | code |
|---|---|---|
| `kv` / `past_kv` | per-layer (k, v); k already roped; grows once per canvas | `inference/generate.py:BlockDiffusionSampler.encode_canvas` |
| `committed` | (B, L) bool: position frozen against re-noising | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `am`, `conf` | posterior argmax and its max prob | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `commit` | unchanged-or-stronger predicate (am==prev \| conf≥prev) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `new` | fresh-commit mask: `commit & ~old_committed` — value rewritten only here | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `prev` | previous step's `(am, conf)` pair; `None` at step 0 | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `x0` | this step's drawn tokens (argmax, or the Gumbel draw when `tau>0`) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `q` | `p.clamp_min(1e-12)` — shared by the entropy and the draw | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `tau_k` | linear-annealed temperature, 0.8 → 0.4 | `inference/generate.py:SamplerConfig` |
| `sc_next` | previous step's posterior re-embedded (`p @ E`) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `entropy` | mean row entropy of `p`, the bond signal | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `streak` | consecutive steps under `entropy_threshold`; resets on breach | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| `steps_used` | denoise steps actually executed for a canvas | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| `_span_t` | (B, n_span) time column covering every mask-block the chunk straddles | `inference/generate.py:BlockDiffusionSampler._span_t` |
| forwards / token-forwards | backbone calls / incoming tokens through them | `inference/evaluate.py:SpeedupEvaluator._count_forwards` |
| tokens/forward | L/(T_eval+1) per-canvas asymptote; end-to-end incl. prefill | `inference/evaluate.py:SpeedupEvaluator` |
| `parse_baseline` | row name → `SamplerConfig` | `inference/evaluate.py:parse_baseline` |

## 9. Interview Q&A

**Q: Why is the sampler called "commit-and-renoise"?**
A: Each step forwards the whole canvas, then partitions positions three ways
(`inference/generate.py:BlockDiffusionSampler._denoise_step`): fresh commits
take the new posterior draw, already-frozen positions keep their value, and
everything else is redrawn from uniform noise. The committed set only grows.

**Q: Derive the commit rule.**
A: Confidence is not calibrated across t, so an absolute cutoff misfreezes.
The well-defined signal at every t is the *change* between consecutive
posteriors: commit when the argmax is unchanged (`am == am_prev`) or the
confidence grew (`conf >= conf_prev`) — both monotone clauses
(`tests/test_sampler.py::test_commit_rule_monotone`). Freshness is masked by
`new = commit & ~old_committed` so a frozen token is never rewritten.

**Q: Why is committing legal at all — doesn't it take the model off-policy?**
A: The reverse: any state of committed tokens plus uniform noise *is* a
`models/diffusion.py:q_sample` draw at the matching `ᾱ` — keep-with-prob-ᾱ,
else uniform — and the model is trained at every `t`
(`models/diffusion.py:sample_canvas_t`). The one genuine skew is a
committed-but-wrong token (training's clean tokens are true `x0`); the rule
bounds it by requiring a mode to survive a full re-noise before freezing.

**Q: Why Gumbel-max instead of multinomial?**
A: `multinomial` over (rows × 50,257) is a slow serial kernel; the Gumbel
trick samples `softmax(logits/tau)` as one elementwise op + argmax, reusing
the entropy's clamped `q` and needing no lse subtraction — `log p` differs
from `logits/tau` by a per-row constant, and argmax is shift-invariant
(`tests/test_sampler.py::test_gumbel_temperature_draw`).

**Q: Why does the entropy bond use entropy rather than max-confidence?**
A: Dynamic range. Mean confidence compresses toward 1.0 — 250 settled
positions plus 6 garbage ones still average 0.974 — while mean entropy
separates the same two canvases by ~2× (0.16 vs 0.35 nats) because it
integrates the whole posterior, residual tail included. Threshold 1.0 nat ≈
2.7 effective tokens of uncertainty, vs `ln 50257 ≈ 10.8` at pure uniform,
with `stability_steps=2` consecutive requirement
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`).

**Q: What happens if you enable the entropy bond on an untrained model?**
A: Nothing. A near-uniform head pins the mean entropy at ≈ `ln V` nats, far
above the threshold, so the streak never accumulates and the schedule runs
fixed — the bond is inert at init by construction, and the mechanism is
proven instead by injecting a converged head
(`tests/test_sampler.py::test_adaptive_stopping_fires`).

**Q: What exactly does the speedup metric measure, checkpoint-free?**
A: Token-forwards per generated token: AR = 1.0 by construction; block-AR =
`1 + n_canvases·(T_eval+1)` forwards for `n_canvases·L` tokens — 256/17 ≈
15.06× per canvas at T=16, ≈14.8× end-to-end once the prefill is counted
(`inference/evaluate.py:SpeedupEvaluator.evaluate`). Wall-clock AR needs an
external checkpoint — the disclosed gap (`seconds=None`).

**Q: What does the FLOP proxy not count?**
A: Attention over the cached prefix (decode forwards contribute only the new
`L` tokens) and the vocab head (`h @ E.T` — the wrapper instruments
`model.backbone` only). Both omissions are symmetric across every row being
compared, which is what makes the *ratio* meaningful.

**Q: How does the sampler stay consistent with training-time conditioning?**
A: `sc_next = p @ E` from the previous step's *posterior* (not the
temperatured draw), matching the eval cross-step contract; step 1 has no
posterior and runs unconditioned, which zero-init equivalence makes exactly
the no-sc mode ([self-conditioning](self-conditioning.md) §3).

**Q: Why is the encode step a separate forward — couldn't the last denoise
step's KV be reused?**
A: Denoise steps run under the in-flight canvas's all-ones decode view with
`x` still noisy; `encode_canvas` runs *after* freezing, re-encoding the final
tokens at `t=0` — reusing noisy-step KV would cache noise-conditioned states
and break the fp64 chaining equivalence
(`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).

**Q: Why does the sampler start from `torch.randint` and not `q_sample`?**
A: `q_sample` corrupts a *known* `x0`; the sampler has none — it must
synthesize from the noise state itself, so it draws the noise distribution
directly and enters at `t = T_eval`, where `ᾱ(T) ≈ 0` makes pure uniform
exactly what the model expects as input.

**Q: Why does the decode loop stay eager instead of `torch.compile`?**
A: The KV cache grows once per canvas, so a decode step's past length changes
every iteration; graph-capturing the step needs a statically preallocated
cache first (then `mode="reduce-overhead"`). The accepted ceiling is the
per-step past-append `cat` (~0.1 ms/copy at a 4k prefix), noted inline in
`inference/generate.py:BlockDiffusionSampler._canvas_step_logits`.