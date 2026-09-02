# Inference pipeline — the eval harness, the metrics, and the honest gaps

> **Canonical** for the evaluation protocol. `DIFFUSION.md` §4.2 stays
> authoritative for the eval design; this page walks the harness, the
> speedup metric, the quality anchor, and — explicitly — what is *not*
> measured yet.

**Depends on:** [sampler](concepts/sampler.md) §5 ·
[diffusion-core](concepts/diffusion-core.md) §5 ·
**Read next:** [benchmarking](guides/benchmarking.md) · [sampler-tuning](guides/sampler-tuning.md)

---

## Table of Contents

1. [The harness](#1-the-harness)
2. [The headline metric: tokens per forward](#2-the-headline-metric)
3. [Quality: held-out x0-NLL](#3-quality-held-out-x0-nll)
4. [Honest gaps](#4-honest-gaps)
5. [Worked example: reading a row at toy scale](#5-worked-example-reading-a-row-at-toy-scale)
6. [What breaks if you change this](#6-what-breaks-if-you-change-this)
7. [Glossary](#7-glossary)
8. [Interview Q&A](#8-interview-qa)

---

## 1. The harness

`inference/evaluate.py:SpeedupEvaluator` measures the three sampler schedules
plus the analytic AR row in one call:

```python
# inference/evaluate.py:SpeedupEvaluator.evaluate (structure)
rows = {name: self._measure(n, prompt, gen, parse_baseline(name), wall_clock)
        for name in ("fixed_T16", "fixed_T32", "adaptive_T32")}
rows["ar_kv_analytic"] = {..., "tokens_per_forward": 1.0, "seconds": None}
```

### 1.1 What one call produces

`evaluate(n_samples=100, prompt_tokens=64, gen_tokens=1024)` returns
`{"rows": {name -> row}, "speedup_vs_ar_tokens_per_forward": {name -> ratio}}`
— the speedup dict just restates each measured row's `tokens_per_forward`
against the AR row's 1.0. Every measured row carries: `forwards` (backbone
call count, an exact int), `token_forwards` (sum of `ids.size(1)` over those
calls), the ratios `tokens_per_forward = n_samples · gen_tokens / forwards`
and `token_forwards_per_token = token_forwards / (n_samples · gen_tokens)`,
and the wall-clock pair `seconds` / `tokens_per_sec`. `seconds` is
`time.perf_counter()` around the sampling loop inside
`inference/evaluate.py:SpeedupEvaluator._measure` — while the AR row, which
has no checkpoint to run, carries `seconds=None` and `tokens_per_sec=None`
forever (the subject of §4).

### 1.2 From row name to sampler config

`parse_baseline("adaptive_T32")` → `SamplerConfig(n_diffusion_steps=32,
adaptive=True)` (`inference/evaluate.py:parse_baseline`). The string contract
is `mode_T<steps>`: split on `"_T"`, parse the tail as the schedule length,
set `adaptive = (mode == "adaptive")`. Everything else keeps the defaults of
`inference/generate.py:SamplerConfig` — the Gumbel temperature anneal
(`temp_start=0.8` → `temp_end=0.4`), the entropy bond
(`entropy_threshold=1.0`, `stability_steps=2`), the optional seed. A row name
is a complete reproducible recipe: `fixed_T16` is "16 denoise steps, ignore
entropy"; `adaptive_T32` is "up to 32, stop on the entropy bond".

### 1.3 Counting by instrumenting the backbone

Counting works by **instrumenting the backbone**:
`inference/evaluate.py:SpeedupEvaluator._count_forwards` is a context manager
that swaps `model.backbone` for a wrapper:

```python
# inference/evaluate.py:SpeedupEvaluator._count_forwards (structure)
def counting(ids, *args, **kwargs):
    outer._forwards += 1
    outer._token_forwards += ids.size(1)
    return orig(ids, *args, **kwargs)
self.model.backbone = counting      # restored in a finally block
```

Nothing inside the model or sampler changes: every call site that would
invoke `models/transformer.py:DiffusionGemma.backbone` now goes through the
wrapper, which tallies one forward and `ids.size(1)` token-forwards, then
delegates unchanged. This beats an in-sampler counter, which encodes one
particular decomposition of generate into steps and must be updated in
lockstep whenever the loop changes; the wrap counts *what actually reached
the transformer*, and the `finally` restore means an exception cannot leave
a counting shim installed.

### 1.4 Getting weights into the evaluator

The evaluator is model-agnostic — `inference/evaluate.py:SpeedupEvaluator.__init__`
takes any `DiffusionGemma` (plus an optional tokenizer, see §4). Load a
trained checkpoint with `utils/checkpoint.py:CheckpointManager` first:

- `utils/checkpoint.py:CheckpointManager.latest_step` returns the newest step
  that is *resumable*: `utils/checkpoint.py:CheckpointManager._checkpoint_complete`
  requires all three files (`model_step_N.safetensors`, `optim_step_N.pt`,
  `meta_step_N.json`), so a weights-only leftover from a crashed save is
  never selected.
- `utils/checkpoint.py:CheckpointManager.load` restores weights with
  `strict=False` reporting (missing/unexpected keys logged, raised under
  `strict=True`), so an architecture-drifted checkpoint fails loudly instead
  of silently benchmarking a half-loaded model.

The same three-file discipline that protects training resume protects
evaluation: you cannot benchmark a checkpoint that was mid-write.

## 2. The headline metric

`tokens_per_forward = n_samples · gen_tokens / forwards` — checkpoint-free
by construction:

- **AR + KV cache**: 1 forward per token → `tokens/forward = 1.0` exactly.
  No AR checkpoint is needed to state this — it is what KV-cache decode
  *is* (`inference/evaluate.py:SpeedupEvaluator.evaluate` writes the
  `ar_kv_analytic` row with `seconds=None`).
- **Block-AR**: `(T_eval + 1)` forwards per canvas →
  `tokens/forward = L / (T_eval + 1)`.

### 2.1 The per-canvas identity

Derive the block-AR number from the decode loop
(`inference/generate.py:BlockDiffusionSampler.generate`): each finalized
canvas costs `T_eval` denoise forwards
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`) plus exactly
one re-encode forward (`inference/generate.py:BlockDiffusionSampler.encode_canvas`),
and yields `L = canvas_len = 256` tokens:

```
tokens per forward = L / (T_eval + 1)
T_eval = 16 → 256 / 17 ≈ 15.1
T_eval = 32 → 256 / 33 ≈ 7.8
```

The `+1` is the re-encode: chaining costs one extra full-canvas forward per
block, paid so the next canvas attends to finalized KV. Halving `T_eval`
does not double the ratio because the encode forward is fixed — which is
also why adaptive stopping (which trims only denoise steps) helps the
fixed-32 row proportionally more. One prefill forward per sample
(`inference/generate.py:BlockDiffusionSampler.prefill`) sits outside this
identity; the harness charges it too, so a *measured* row lands slightly
below the identity (§5).

### 2.2 Why token-forwards is the FLOP proxy

`tokens_per_forward` and `token_forwards_per_token` are different numbers and
both are reported. A backbone forward over `n` tokens costs roughly `O(n)`
work, so the hardware-honest unit is **token-forwards** — tokens pushed
through the backbone, summed over all calls
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`). The headline is
stated per *forward* because the structural claim is "one forward refines
256 tokens instead of 1": the fixed overhead of a forward is what AR pays
per token and block-AR pays once per canvas. The two metrics bracket the truth:

- `tokens_per_forward` = the structural parallelism win (the headline).
- `token_forwards_per_token` = the honest compute accounting: at
  `T_eval=16` each canvas passes through the backbone 17 times, so the model
  does ~17× AR's token-level work for the same token — the win is *fewer
  serial dependency barriers and fewer forward launches*, not less total
  compute. `tests/test_inference.py::test_flop_counter_adaptive_le_fixed`
  pins adaptive ≤ fixed on both.

### 2.3 The row table

| row | forwards | tokens/forward | measured? |
|---|---|---|---|
| `fixed_T16` | 17/canvas | 15.06× | forwards + wall-clock |
| `fixed_T32` | 33 | 7.76× | forwards + wall-clock |
| `adaptive_T32` | ≤ 33 (entropy bond) | ≥ 7.76× | forwards + wall-clock |
| `ar_kv_analytic` | 1/token | 1.0× | analytic only |

Per-schedule detail at production dims (256-token canvases, 1,024-token
generations = 4 canvases): 68 denoise+encode forwards at T=16 (69 with the
prefill) vs 1,024 for AR — ≈15.1× fewer forwards on the per-canvas identity,
≈14.8× once the prefill is amortized; token_forwards/token = 17.0 at T=16,
33.0 at T=32 (`tests/test_inference.py::test_flop_counter_adaptive_le_fixed`
pins adaptive ≤ fixed; `test_evaluator_produces_three_rows` pins the row set
and the exact formula `1 + n_canvases·(T_eval + 1)`).

**Do not quote toy-model numbers as the headline.** At a 32-token toy canvas
with `T_eval=32`, forwards per sample `= 1 + 2·33 = 67` for 64 generated
tokens → tokens/forward ≈ 0.96 < 1.0: a canvas too small to amortize its
denoise+encode cost looks *slower than AR* — a statement about canvas size,
not the method (`inference/evaluate.py:SpeedupEvaluator._measure` reports it
faithfully; the identity needs production `L = 256`).

## 3. Quality: held-out x0-NLL

`inference/evaluate.py:heldout_x0_nll` is the quality anchor — the training
objective on held-out windows:

```
per window:  t ~ U{1..T} per canvas (seeded rng)
             xt = q_sample(x0, t)                # corruption, eval protocol
             hidden = model.final_hidden(xt, t)  # NO sc input
             NLL_w = chunked_x0_ce(hidden, E, x0)     # nats/token
return mean over windows
```

### 3.1 Why this number is comparable to an AR baseline

The AR comparison is direct: x0-CE (nats/token) on the same shard is the
same unit as an AR baseline's CE — that parity *is* the quality anchor
(DESIGN §4.2). Spelled out: an AR model's CE is `−(1/N) Σ log p(xᵢ | x<ᵢ)`
over next-token predictions; the diffusion objective is `−log p(x0 | xt, t)`
over every position of every canvas (diffusion reconstructs x0 everywhere —
no AR shift). Averaged at sampled `t`, both estimate the same quantity —
expected bits-per-token of the data under the model's distribution over
clean text — in the same unit over the same tokenizer and shard, so the
acceptance verdict (+5% at most, via `scripts/loss_parity_eval.py --ar-nll
<AR CE>`) compares like with like. And the mechanics match training exactly:
`inference/evaluate.py:heldout_x0_nll` draws per-canvas `t` with
`models/diffusion.py:sample_canvas_t`, corrupts with `models/diffusion.py:q_sample`
(single-step closed form — never the iterated chain), and scores with
`training/losses.py:chunked_x0_ce` — the production chunked loss itself, so
the eval number comes from the same kernel that trained the model.

### 3.2 The window procedure, with shapes

`seq_len` defaults to `model.cfg.max_seq_len` (4096 production = 16 canvases;
the toy fixture uses 128 = 4 canvases of 32). The shard is a flat uint32
memmap windowed into `n = len(mm) // seq_len` disjoint windows; a shard too
short for one window raises
(`tests/test_inference.py::test_heldout_x0_nll_short_shard_raises`). Per
window: `t` is `(1, n_canvases)` — one draw per canvas, matching the
per-canvas time conditioning of training — `xt` is `(1, seq_len)`, and
`hidden = model.final_hidden(xt, t)` is `(1, seq_len, d_model)`; the final
NLL is the unweighted mean over windows of the per-window chunked CE.

### 3.3 Why no self-conditioning input

No self-cond input at eval-time NLL: the metric conditions the model exactly
as the training corruption step does, minus the sc input — matching the
loss-parity protocol (`inference/evaluate.py:heldout_x0_nll`; sc-off is also
why the number is directly comparable to a no-sc AR baseline). Concretely,
`final_hidden` is called without `sc_input`, so
`models/selfcond.py:SelfConditioning` never fires: conditioning on the
model's own no-grad posterior would change the objective from "likelihood of
the data" to "likelihood given my last guess", and only the former has an AR
counterpart. The sampler, by contrast, *does* use cross-step self-conditioning
during decode — a generation-time refinement, not the measured likelihood.

### 3.4 Determinism

Deterministic: fixed seed for window picks *and* corruption draws —
`np.random.default_rng(seed)` chooses the windows and a separate
`torch.Generator` seeded with the same value drives every `t` draw and
corruption mask, so the metric is bit-reproducible across runs
(`tests/test_inference.py::test_heldout_x0_nll_finite`); that is what makes
the +5% parity verdict a controlled comparison rather than run-to-run
variance. One caveat on an *untrained* model: the posterior is near-uniform,
so the NLL sits near `ln(V) ≈ 10.8` nats/token and the sampler's entropy
bond never fires (§5.3) — the quality anchor and the speedup rows are
independent measurements, and the harness reports both.

## 4. Honest gaps

The harness discloses what it does *not* measure (DESIGN §4.2):

| gap | why | code |
|---|---|---|
| AR wall-clock tokens/s | needs a trained external AR checkpoint; without it the row is `seconds=None`, analytically 1 forward/token | `inference/evaluate.py:SpeedupEvaluator.evaluate` |
| decode preview / text output | tokenizer accepted but unused — counting is tokenizer-free | `inference/evaluate.py:SpeedupEvaluator.__init__` |
| quality-vs-steps curves | single NLL point per shard, not a sweep | `inference/evaluate.py:heldout_x0_nll` |
| sample-quality metrics (perplexity beyond x0-NLL, MAUVE, human eval) | out of harness scope; not yet built | — |

Each gap has a reason, not just an absence:

- **AR wall-clock** is a *blocked* measurement, not missing work: the
  baseline is a separate LLaMA-3-Lite model that must be trained on the same
  data first, and printing an estimate would fabricate the one number the
  speedup claim leans on — so the harness encodes the absence as `None`. The
  CLI (`scripts/speedup_eval.py`, `--flops-only`) prints the gap when inputs
  are absent.
- **Decode previews** need only plumbing: the constructor already accepts
  the tokenizer, and `models/transformer.py:DiffusionGemma.generate` is the
  one-call entry point that returns ids.
- **Quality-vs-steps curves** would answer "does adaptive stopping hurt
  quality?" directly; today there is one NLL point per shard, and the
  `inference/generate.py:SamplerConfig` knobs are calibrated by unit tests,
  not a sweep.
- **Sample-quality metrics** are a different project layer; x0-NLL is a
  *likelihood* anchor and deliberately not a fluency metric.

The speedup claim survives because it is *counting*, not timing: forwards
and token-forwards are exact integers from the instrumented loop
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`).

## 5. Worked example: reading a row at toy scale

Tiny fixture model (`V=256, d=64, L=32, max_seq_len=128`),
`T_eval=4`, `n_samples=1`, prompt 64 tokens (2 canvases), `gen_tokens=64`
(2 canvases). The harness wraps the whole `generate` call, so the prefill is
included:

```
prefill                     = 1 forward over 64 tokens
per canvas: 4 denoise + 1 encode = 5 forwards × 32 tokens
canvases                    = 2
forwards       = 1 + 2 × 5                          = 11
token_forwards = 64 + 2 × (5 × 32)                  = 384
tokens generated                                    = 64

tokens_per_forward        = 64 / 11       ≈ 5.8
token_forwards_per_token  = 384 / 64      = 6.0
ar_kv_analytic row        = 1.0 (by construction)
speedup (tokens/forward)  ≈ 5.8 × AR
```

Take-away: the first metric charges the prompt canvas once via prefill and
counts generated tokens over backbone forwards; the second counts *all*
tokens pushed through the backbone (including re-denoising passes over
committed positions) — the headline is the first, the honest cost accounting
is the second (`inference/evaluate.py:SpeedupEvaluator._measure` returns
both). `tests/test_inference.py::test_evaluator_produces_three_rows` pins
the formula `forwards == 1 + 2·(T+1)`, so a decode-loop change that adds or
drops a forward per canvas fails the test, not just the prose.

### 5.1 Production walkthrough: one canvas, end to end

`T_eval=16`, prompt of 64 tokens, canvas `L=256`, `V=50257`. Trace one
block-AR step of `inference/generate.py:BlockDiffusionSampler.generate`:

1. **Prefill** — one forward over the 64 prompt tokens under the
   partial-canvas block-causal mask (`inference/generate.py:_prefix_mask`
   under sdpa; `models/mask.py:build_block_causal_block_mask` handles the
   partial first canvas natively under flex). All prompt canvases enter at
   `t=0`, with `time_steps = SamplerConfig.n_diffusion_steps` so normalized
   time `t/T` stays in the trained range
   (`inference/generate.py:BlockDiffusionSampler.prefill`). KV cache is
   populated for positions 0..63 and static.
2. **Denoise step k** (of ≤16) — the in-flight canvas starts as pure uniform
   noise over `V` (`torch.randint`, *not* `models/diffusion.py:q_sample` —
   the sampler redraws the noise distribution itself; `q_sample` is
   train-time corruption only). One forward
   (`inference/generate.py:BlockDiffusionSampler._canvas_step_logits`) runs
   the 256-token canvas chunk against the cached prefix under the
   all-visible decode mask (§5.2), at `t = T_eval − k` and temperature
   `τ(k) = temp_start − (temp_start − temp_end)/T_eval · k` (0.8 → 0.4).
   The draw is a **Gumbel-max**: `argmax(log p / τ + gumbel_noise)` ≡
   sampling `softmax(logits/τ)` in one elementwise pass + argmax — multinomial
   over `(256, 50257)` rows is a slow serial kernel, and Gumbel scores are
   shift-invariant, so no logsumexp is needed
   (`inference/generate.py:BlockDiffusionSampler._denoise_step`).
3. **Commit rule** — a position commits when its argmax token is unchanged
   from the previous step *or* its confidence got stronger; committed
   positions are never overwritten (`new = commit & ~old_committed`,
   monotone by construction); uncommitted positions are redrawn pure
   uniform. The returned canvas is `where(committed, x, x0)` — every
   position ends at its best posterior guess even if it never committed.
4. **Entropy bond** — after each step, mean per-position entropy
   `H[p(·|position)]` is compared to `entropy_threshold=1.0`; two
   consecutive steps below it stop early. Adaptive never spends more
   forwards than fixed
   (`tests/test_inference.py::test_flop_counter_adaptive_le_fixed`).
5. **Re-encode** — the finalized 256 tokens are re-encoded in one forward
   with the **all-ones decode mask**: the finalized canvas is encoded as its
   fully-visible last-block view and appended to the KV cache
   (`inference/generate.py:BlockDiffusionSampler.encode_canvas`); then
   `prefix_len += 256` and the next canvas attends to it causally. The
   cached KV equals a fresh single-shot block-causal forward bit-exactly
   (`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).

For a 1,024-token generation: 4 canvases × (≤16 denoise + 1 encode) + 1
prefill = 69 forwards, 1,024 tokens → measured tokens/forward ≈ 14.8.

### 5.2 Why the re-encode mask is all-ones

The commit rule re-encodes the canvas with an **all-ones** mask
(`models/mask.py:build_canvas_decode_mask` returns all-visible; under the
flex path `inference/generate.py:BlockDiffusionSampler._decode_mask` passes
`mask=None`, which under flex *is* full attention — the same semantics
without building the tensor). This is not an oversight: the canvas's
*content* is noisy but its *visibility* is complete — the model must see all
256 in-flight positions bidirectionally while denoising, plus the finalized
prefix. A zero-masking variant would prevent the denoise pass from
coordinating tokens within the canvas, destroying the parallelism the design
buys; the mask is trivially permissive because noising, not masking, is the
information bottleneck.

### 5.3 Reading `adaptive_T32` on untrained weights

The Gumbel sampler with temperature annealing drives the decode loop; the
entropy bond (adaptive stopping) is **inert on untrained weights**: with a
randomly initialized model the posterior is near-uniform
(`H ≈ ln 50257 ≈ 10.8` nats ≫ the `1.0` threshold), the
`stability_steps=2` streak never accumulates, and `adaptive_T32` runs the
full schedule — `forwards == fixed_T32` exactly, never more
(`tests/test_sampler.py::test_adaptive_off_ignores_entropy` pins the
fixed-fallback equivalence). Do not read an adaptive row equal to the fixed
row as "the bond is broken": before the posterior sharpens there is nothing
for the bond to detect, and the gap between the rows is itself a
training-progress signal.

## 6. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| count forwards inside the sampler (not via backbone wrap) | double-counts encode+denoise differently; rows incomparable | `tests/test_inference.py::test_evaluator_produces_three_rows` |
| AR row wall-clock without a real checkpoint | fabricated baseline — the exact gap §4 forbids | (disclosure contract, `ar_kv_analytic.seconds=None`) |
| NLL with self-cond input on | eval ≠ training-corruption parity; number not comparable to AR CE | `tests/test_inference.py::test_heldout_x0_nll_finite` |
| `heldout_x0_nll` on the training shard | leakage; quality metric meaningless | (protocol; shard path caller-provided) |
| `parse_baseline` name format change | config rows fail to parse | `tests/test_inference.py::test_parse_baseline` |
| re-encode canvases under a zero/hiding mask | in-flight coordination lost; KV diverges from single-shot forward | `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot` |
| eval forwards with `time_steps` left at train-time T | `t/T` mis-normalized for eval schedules; time embedding out of range | (protocol: `time_steps=SamplerConfig.n_diffusion_steps` on every forward) |
| drop the `+1` encode from the forward formula | headline overstated by `L/(T)` vs `L/(T+1)`; row set no longer matches | `tests/test_inference.py::test_evaluator_produces_three_rows` |
| let committed positions be overwritten mid-schedule | commit rule no longer monotone; late steps destroy early tokens | `tests/test_sampler.py::test_commit_rule_monotone` |
| skip the `torch.Generator` seeding in `heldout_x0_nll` | corruption draws vary per run; parity verdict uncontrolled | `tests/test_inference.py::test_heldout_x0_nll_finite` |

## 7. Glossary

| symbol | meaning | code |
|---|---|---|
| `forwards` | backbone call count (denoise + encode + prefill) | `inference/evaluate.py:SpeedupEvaluator._count_forwards` |
| `token_forwards` | sum of sequence lengths over forwards | same |
| `tokens_per_forward` | generated tokens / backbone forwards | `inference/evaluate.py:SpeedupEvaluator.evaluate` |
| `token_forwards_per_token` | total tokens through the backbone / generated tokens | `inference/evaluate.py:SpeedupEvaluator._measure` |
| `heldout_x0_nll` | chunked x0-CE on held-out windows, no sc | `inference/evaluate.py:heldout_x0_nll` |
| `parse_baseline` | `"fixed_T16"` → `SamplerConfig` | `inference/evaluate.py:parse_baseline` |
| `ar_kv_analytic` | AR row: 1 forward/token by construction | `inference/evaluate.py:SpeedupEvaluator.evaluate` |
| `SamplerConfig` | eval knobs: steps, adaptive, entropy bond, anneal, seed | `inference/generate.py:SamplerConfig` |
| `denoise_canvas` | ≤ T_eval uniform-state steps from pure noise → canvas ids | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| `encode_canvas` | re-encode finalized canvas (all-visible view) + KV append | `inference/generate.py:BlockDiffusionSampler.encode_canvas` |
| `prefill` | one prompt forward; static KV, partial-canvas mask | `inference/generate.py:BlockDiffusionSampler.prefill` |
| `build_canvas_decode_mask` | all-ones decode view: prefix + own canvas fully visible | `models/mask.py:build_canvas_decode_mask` |
| `entropy bond` | stop after `stability_steps` steps under `entropy_threshold` | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| `latest_step` | newest fully-written (3-file) checkpoint step | `utils/checkpoint.py:CheckpointManager.latest_step` |

## 8. Interview Q&A

**Q: What exactly does the 15.06× number measure?**
A: Generated tokens per backbone forward at T=16: 256 tokens per canvas /
17 forwards (16 denoise + 1 encode) vs AR's exactly-1 — the ratio is 15.06
(`inference/evaluate.py:SpeedupEvaluator.evaluate`). Checkpoint-free: forwards
are counted by instrumenting `model.backbone`. The measured row is a touch
lower (the one-time prefill is charged to the run); the identity is the
per-canvas structural claim.

**Q: Why is the AR baseline "analytic"?**
A: A KV-cached AR decoder performs exactly 1 forward per generated token by
construction — measuring it would need an external trained checkpoint, so
the harness states the row analytically and leaves wall-clock as the
disclosed gap (`inference/evaluate.py:SpeedupEvaluator.evaluate`).

**Q: Why is x0-NLL the quality anchor, not perplexity of generated text?**
A: It is the training objective evaluated on held-out data at sampled t —
directly comparable to an AR model's CE on the same shard, checkpoint-free,
and deterministic under the fixed seed
(`inference/evaluate.py:heldout_x0_nll`). Sample-quality metrics
(MAUVE, human eval) are the disclosed gap.

**Q: Why no self-conditioning input in the eval NLL?**
A: The quality anchor compares against an AR baseline's CE on the same
tokens; conditioning on the model's own no-grad guess would change the
objective being measured
(`inference/evaluate.py:heldout_x0_nll` — no `sc_input` in the call).

**Q: Why does adaptive stopping look for entropy rather than confidence?**
A: Mean max-probability saturates early — a few confident tokens pin the
mean while the canvas is unresolved — whereas mean per-position entropy
keeps measuring unresolved mass, with a calibratable threshold
(`inference/generate.py:BlockDiffusionSampler.denoise_canvas`). On untrained
weights the bond is inert (entropy ≈ ln V), so the adaptive row degrades
gracefully to the fixed row.

**Q: What would you add first to strengthen this harness?**
A: An AR wall-clock row from a real checkpoint (the one `None` in the
output), then decode previews — both are constructor-ready
(`SpeedupEvaluator.__init__` already accepts the tokenizer).