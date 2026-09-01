# Concept: the sampler — commit-and-renoise over KV-chained canvases

> **Canonical** for the decode loop, the commit rule, temperature annealing,
> the entropy bond, and FLOP accounting. `DIFFUSION.md` §5 stays authoritative
> for final rulings; this page derives the loop from first principles.

**Depends on:** [foundations](foundations.md) §4, §7 ·
[diffusion-core](diffusion-core.md) §4–5 ·
[block-causal-attention](block-causal-attention.md) §6 ·
[self-conditioning](self-conditioning.md) §7 ·
**Read next:** [training](../training.md) · [inference](../inference.md)

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

`inference/generate.py:BlockDiffusionSampler.generate` is four moving parts
in a fixed rhythm — prefill **once**, then per canvas: denoise → encode →
slice into the output buffer:

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

The KV cache is written **once per canvas** (256 tokens per write), not once
per token: that is the entire speedup story (§5). The prefill handles a
partial first canvas via `inference/generate.py:_prefix_mask` (flex path:
`models/mask.py:build_block_causal_block_mask` handles it natively).

## 2. The commit rule, derived

One denoise step (`inference/generate.py:BlockDiffusionSampler._denoise_step`):

```
logits = f(x, t, sc)             # (B, L, V)
p      = softmax(logits)         # posterior over the clean canvas x̂0
conf, am = p.max(-1)             # per-position confidence and argmax
commit = (am == am_prev) | (conf >= conf_prev)
```

**Derivation.** Uniform-state diffusion defines a reverse process where each
step re-draws uncommitted positions; committing an already-good position
protects it from later re-noising. The design question: which positions are
"good enough to freeze"? The rule that falls out of the block-AR
factorization is **unchanged-or-stronger**: freeze a position when its argmax
is stable across steps (`am == am_prev`) *or* when its posterior mass has
grown (`conf >= conf_prev`). Both clauses are monotone predicates on the
posterior sequence — pinned by
`tests/test_sampler.py::test_commit_rule_monotone`.

Why "unchanged-or-stronger" rather than a bare confidence threshold: the
model's confidence is not calibrated across timesteps (early steps see pure
noise, late steps see nearly-clean canvases), so an absolute cutoff freezes
the wrong things early. The relative rule uses only the *comparison between
consecutive steps*, which is well-defined at every t.

The committed set only grows (`committed |= commit`) — and the value written
is the *step's fresh posterior only for positions not yet frozen*
(`new = commit & ~old_committed`):

```python
x = torch.where(new, x0, torch.where(old_committed, x, noise))
#                  ↑ fresh commit    ↑ already frozen: keep   ↑ else re-noise
```

Three-way routing, exactly one branch per position — the sampler-side twin of
self-conditioning's exactly-once add. `x0 = am` unless temperature draws
(§3); the final `torch.where(committed, x, x0)` returns the last posterior
for never-committed positions rather than their last random noise.

## 3. Temperature annealing and the Gumbel draw

Temperature is linear-annealed per step `k` of `T_eval`
(`SamplerConfig.temp_start=0.8 → temp_end=0.4`,
`inference/generate.py:SamplerConfig`):

```
tau_k = temp_start - (temp_start - temp_end) / T_eval * k
# T_eval=32: tau = 0.8 → 0.7875 (k=1) → 0.6 (k=16) → 0.4125 (k=31)
```

When `tau > 0`, the draw is a **Gumbel-max trick**: sample from
`softmax(logits / tau)` as one elementwise op + argmax —

```python
gumbel = -log(-log(U(0,1)))
x0 = (log(q) / tau + gumbel).argmax(-1)
```

`q = p.clamp_min(1e-12)` is shared between the entropy and the draw (one
clamp, two uses); scores are shift-invariant so no lse subtraction is
needed. Replacing the old `multinomial` path (a slow serial kernel over
`rows × 50k`) with elementwise+argmax is a deliberate kernel choice — pinned
equivalent by `tests/test_sampler.py::test_gumbel_temperature_draw`.
Low `tau` sharpens the draw toward the argmax; at `tau → 0` the draw becomes
the argmax itself, so annealing 0.8 → 0.4 walks the sampler from exploration
to exploitation *within* each canvas.

## 4. The entropy bond (adaptive stopping)

Fixed schedules always spend T_eval steps per canvas. The adaptive path
(`SamplerConfig.adaptive=True`, `entropy_threshold=1.0`,
`stability_steps=2`) stops early when the canvas's posterior is *settled*:

```
entropy_k = mean over canvas rows of −Σ p·log p     # (B,) per step
if entropy < 1.0 for `stability_steps` consecutive steps: break
```

Why entropy and not confidence: mean max-confidence saturates (many easy
positions at 0.99) long before the hard positions resolve; mean entropy keeps
counting the residual uncertainty of the hard positions. A uniform posterior
over V = 50,257 has entropy `ln V ≈ 10.825` nats; the 1.0-nat threshold means
"effectively ~e^1 ≈ 2.7 tokens of remaining uncertainty per position on
average". `streak` resets on any step above threshold — two consecutive
low-entropy steps are required
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`).

Pinned: `tests/test_sampler.py::test_adaptive_stopping_fires`,
`test_adaptive_threshold_calibratable` (threshold reachable by construction),
`test_adaptive_off_ignores_entropy` (adaptive=False runs the full schedule
and never reads entropy). The FLOP win is bounded: adaptive ≤ fixed
token-forwards per canvas — `tests/test_inference.py::test_flop_counter_adaptive_le_fixed`.

## 5. FLOP accounting and the AR baseline

`inference/evaluate.py:SpeedupEvaluator` counts **forwards** and
**token-forwards** by instrumenting `model.backbone`
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`). The headline
metric needs no checkpoint:

- AR + KV cache: **1 forward and 1 token-forward per token** — by
  construction, so `tokens/forward = 1.0` analytically (the honest gap:
  AR wall-clock needs an external checkpoint, disclosed as `seconds=None`,
  `inference/evaluate.py:SpeedupEvaluator.evaluate`).
- Block-AR: `(T_eval + 1)` forwards per canvas → `tokens/forward = L/(T_eval+1)`.

| schedule | forwards/canvas | tokens/forward | vs AR |
|---|---|---|---|
| fixed T=16 | 16 denoise + 1 encode = 17 | 256/17 = **15.06×** | measured row `fixed_T16` |
| fixed T=32 | 33 | 7.76× | `fixed_T32` |
| adaptive T=32 | ≤ 33 (entropy bond) | ≥ 7.76× | `adaptive_T32` |
| AR (analytic) | 1 per token | 1.0× | `ar_kv_analytic` |

Concretely, 1,024 generated tokens = 4 canvases: 68 forwards at T=16 vs
1,024 for AR — **15.1× fewer forwards**
(`tests/test_inference.py::test_flop_counter_adaptive_le_fixed` pins
adaptive ≤ fixed; `tests/test_inference.py::test_evaluator_produces_three_rows`
pins the row structure). Wall-clock `tokens_per_sec` is measured for this
sampler; AR's requires an external baseline run — the disclosed gap.

Self-conditioning interacts cleanly: `sc_next = p @ E` is computed from the
*previous* step's posterior inside
`inference/generate.py:BlockDiffusionSampler._denoise_step` — no extra
forward, only the `p @ E` re-embedding ([self-conditioning §7](self-conditioning.md)).

## 6. Worked example: one canvas at toy scale

Toy dims: `L = 4`, `V = 8`, `T_eval = 3`, prompt = 4 tokens (one finalized
canvas already encoded), `prefix_len = 4`. Posterior rows are made-up but
commit/renoise/anneal logic is exactly `_denoise_step`'s.

```
init   x = [7, 7, 7, 7]  (uniform noise)   committed = [F, F, F, F]

step k=0, tau=0.8, t = T_eval - 0 = 3:
  p (rows) = [  [.., .., .., ..]  am=[2, 0, 5, 1]  conf=[.31, .28, .30, .26] ]
  prev = None → commit = all-False        (no previous posterior yet)
  tau=0.8 Gumbel draw → x0 = [2, 0, 5, 6]
  all positions re-noise: x = noise
  sc = None → after step: sc_next = p @ E

step k=1, tau=0.7875, t = 2:
  am = [2, 0, 5, 1]  conf = [.55, .41, .30, .29]
  commit = (am==prev_am) | (conf>=prev_conf) = [T, T, F, T]
  new = [T, T, F, F]   → positions 0,1 get x0; 2,3 re-noise
  committed = [T, T, F, F]

step k=1, entropy = 2.1 nats → above 1.0 → keep going
step k=2, t=1, tau=0.6:
  am = [2, 0, 5, 3]  conf = [.71, .66, .55, .33]
  commit = [T, T, T, T]  (am stable: 2==2, 0==0; conf grown: .30→.30? no—
  pos2 am==prev am [5==5] → T; pos3 conf .33 ≥ .29 → T)
  committed = [T, T, T, T]

final: torch.where(committed, x, x0) → x (all frozen) = [2, 0, 5, 1]
steps_used = 3 of T_eval=32 (a real early exit would need the entropy bond;
this trace freezes by commit-rule, not by the bond)
```

Take-away: positions 0–1 froze on *stability* (`am` unchanged), position 3 on
*strength* (`conf` grew), and each frozen position is immune to all later
re-noise draws. At production scale the same loop runs 256 positions with
`tau` annealed 0.8 → 0.4 over ≤ 32 steps and the entropy bond cutting the
schedule short when the posterior settles
(`tests/test_sampler.py::test_commit_rule_monotone` pins the rule's
monotonicity on exactly this pattern).

## 7. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| commit rule reads only `conf >= conf_prev` (drop the argmax clause) | stable-but-dipping posteriors re-noised; quality drops | `tests/test_sampler.py::test_commit_rule_monotone` |
| committed set can shrink (`=` instead of `\|=`) | frozen tokens get overwritten by later noise; KV chain diverges from training distribution | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| temperature draw on committed positions | frozen tokens mutate; commit semantics dead | `tests/test_sampler.py::test_commit_rule_monotone` |
| drop the fp64 chaining equivalence | sampler drifts from the single-shot distribution (subtle quality decay) | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| `multinomial` instead of Gumbel-max | correct but serial-kernel slow over (rows × 50k) | `tests/test_sampler.py::test_gumbel_temperature_draw` (equivalence) |
| entropy read when `adaptive=False` | non-deterministic step count vs fixed schedule | `tests/test_sampler.py::test_adaptive_off_ignores_entropy` |
| re-noise from `x0` instead of uniform noise | committed-quality bias compounds; off-policy vs training corruption | `tests/test_diffusion.py::test_q_sample_endpoints` family |
| per-token KV appends (AR-style) | cache-write cadence ×256; the block-AR win gone | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` (still passes; throughput regresses) |

## 8. Glossary

| symbol | meaning | code |
|---|---|---|
| `kv` / `past_kv` | per-layer (k, v); k already roped; grows once per canvas | `inference/generate.py:BlockDiffusionSampler.encode_canvas` |
| `committed` | (B, L) bool: position frozen against re-noising | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `am`, `conf` | posterior argmax and its max prob | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `commit` | unchanged-or-stronger predicate (am==prev \| conf≥prev) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `tau_k` | linear-annealed temperature, 0.8 → 0.4 | `inference/generate.py:SamplerConfig` |
| `sc_next` | previous step's posterior re-embedded (`p @ E`) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `entropy` | mean row entropy of `p`, the bond signal | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `streak` | consecutive steps under `entropy_threshold` | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| tokens/forward | L/(T_eval+1) fixed; ≥ for adaptive | `inference/evaluate.py:SpeedupEvaluator` |

## 9. Interview Q&A

**Q: Why is the sampler called "commit-and-renoise"?**
A: Each step forwards the whole canvas, then partitions positions three ways
(`inference/generate.py:BlockDiffusionSampler._denoise_step`): fresh commits
take the new posterior argmax, already-frozen positions keep their value, and
everything else is redrawn from uniform noise. The committed set only grows.

**Q: Derive the commit rule.**
A: Confidence is not calibrated across t, so an absolute cutoff misfreezes.
The well-defined signal at every t is the *change* between consecutive
posteriors: commit when the argmax is unchanged (`am == am_prev`) or the
confidence grew (`conf >= conf_prev`) — both monotone clauses
(`tests/test_sampler.py::test_commit_rule_monotone`).

**Q: Why Gumbel-max instead of multinomial?**
A: `multinomial` over (rows × 50,257) is a slow serial kernel; the Gumbel
trick samples `softmax(logits/tau)` as one elementwise op + argmax, reusing
the entropy's clamped `q` and needing no lse subtraction
(`tests/test_sampler.py::test_gumbel_temperature_draw`).

**Q: Why does the entropy bond use entropy rather than max-confidence?**
A: Mean confidence saturates on easy positions while hard ones stay
unresolved; mean entropy keeps measuring residual uncertainty. Threshold 1.0
nat ≈ 2.7 effective tokens of uncertainty, vs `ln 50257 ≈ 10.8` at pure
uniform — with `stability_steps=2` consecutive requirement
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`).

**Q: What exactly does the speedup metric measure, checkpoint-free?**
A: Token-forwards per generated token: AR = 1.0 by construction; block-AR =
(T_eval+1)/L per token → 256/17 ≈ 15.1× at T=16
(`inference/evaluate.py:SpeedupEvaluator.evaluate`). Wall-clock AR needs an
external checkpoint — the disclosed gap (`seconds=None`).

**Q: How does the sampler stay consistent with training-time conditioning?**
A: `sc_next = p @ E` from the previous step's posterior
(`inference/generate.py:BlockDiffusionSampler._denoise_step`), matching the
eval cross-step contract; step 1 has no posterior and runs unconditioned,
which zero-init equivalence makes exactly the no-sc mode
([self-conditioning](self-conditioning.md) §3).

**Q: Why is the encode step a separate forward — couldn't the last denoise
step's KV be reused?**
A: Denoise steps run under the in-flight canvas's all-ones decode view with
`x` still noisy; `encode_canvas` runs *after* freezing, re-encoding the final
tokens — reusing noisy-step KV would cache noise-conditioned states and break
the fp64 chaining equivalence
(`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).