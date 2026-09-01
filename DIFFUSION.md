# DIFFUSION.md — DiffusionGemma-Lite

> The authoritative technical document for this repo: the diffusion math, the
> block-causal mask derivation, self-conditioning, the sampler, the training
> recipe, and how this recipe differs from upstream diffusion-LM practice.
> Every code citation uses `file.py:Symbol` anchors; `tests/test_doc_refs.py`
> fails CI if any anchor stops resolving.

---

## 0. The one-paragraph summary

DiffusionGemma-Lite is a **from-scratch, single-GPU, Chinchilla-optimal
discrete-diffusion LM**. It generates text **block by block**: a 256-token
"canvas" is denoised in parallel from uniform noise over a small number of
diffusion steps, committed, re-encoded into the KV cache, and the next canvas
is conditioned on everything finalized so far (block-autoregressive, "block-AR").
The win is **decode throughput at matched quality**: one forward pass refines
256 tokens instead of 1, and entropy-bounded adaptive stopping cuts eval FLOPs
further. Everything is pure PyTorch — no diffusion library, no custom CUDA.

---

## 1. Uniform-state diffusion (the forward process)

### 1.1 The process

We use the D3PM-style **uniform-state** corruption. For a clean canvas `x0`
(256 GPT-2 BPE tokens) and a per-canvas timestep `t ∈ {1..T}`:

```
ᾱ(t) = cos²(π/2 · t/T)                     # cosine schedule, monotone 1 → 0
q(xt | x0):  keep x0 with prob ᾱ(t)
             replace with a UNIFORM random token with prob 1 − ᾱ(t)
```

- Schedule: `models/diffusion.py:alpha_bar` — a single closed form; no
  learnable noise schedule.
- Forward-process row: `models/diffusion.py:corruption_probs` — the row is
  `ᾱ` on the clean token, `(1−ᾱ)/V` on every other token; it sums to 1.
- Per-canvas `t` sampling: `models/diffusion.py:sample_canvas_t` — each canvas
  in a sequence gets its own `t ~ U{1..T}` (train-time `T=16`,
  `configs/pretrain_a100_380m.yaml` `n_diffusion_steps`).
- Corruption: `models/diffusion.py:q_sample` — one **single-step** corruption
  (draw `xt | x0, t` directly from `q(xt|x0)`); we never iterate the Markov
  chain step-by-step because `q(xt|x0)` has the closed form above.

Two properties matter:

1. **`ᾱ(1) ≈ 1`** — at `t=1` the sequence is nearly clean; the model learns
   light correction. **`ᾱ(T) ≈ 0`** — at `t=T` the sequence is nearly pure
   uniform noise over the 50,257-token vocab; the model learns unconditional
   canvas synthesis. Every corruption level in between is trained.
2. **The state space is the vocabulary itself** — the "noise" is random valid
   tokens, not `<mask>` or Gaussian noise. This is what lets the sampler
   commit partial answers and re-noise the rest (§4).

### 1.2 What the model predicts: x0, not xt

The denoiser is trained to predict the **clean tokens** `x0` directly from
`(xt, t)`, with cross-entropy against `x0`:

- Training loss: `training/losses.py:chunked_x0_ce` (production) with the
  eager reference `models/diffusion.py:x0_ce_loss`.
- The plan text says `chunked_x0_ce(hidden, E, xt)` — that is a **known plan
  typo**; targets are x0 (this is recorded as Ruling 19 in the SDD ledger).

Why x0-parameterization: the sampler needs a per-position posterior over the
vocabulary to (a) commit high-confidence tokens and (b) re-noise the rest.
An x0-prediction head gives exactly that posterior `p(x0 | xt, t)` in one
softmax — no marginalization over trajectories is needed at any point.

### 1.3 Per-canvas time conditioning

`t` is injected per canvas through `models/time_embed.py:CanvasTimeEmbedding`
(sinusoidal embedding of the **normalized** time `t/T`, projected to
`time_embed_dim=256` and added to each token embedding within its canvas).
During training the model's `time_steps` argument stays at the train-time `T`;
at eval the sampler passes `time_steps=SamplerConfig.n_diffusion_steps` so
`t/T ∈ (0,1]` stays correctly normalized for eval schedules `T ≤ 32`
(`inference/generate.py:BlockDiffusionSampler.prefill`, `inference/generate.py:BlockDiffusionSampler._canvas_step_logits`).
At `t=0` (prompt / finalized canvases during prefill and re-encode) the
embedding is T-independent by construction.

---

## 2. Block-causal attention (the load-bearing primitive)

### 2.1 The mask

`models/mask.py:build_block_causal_mask` builds a boolean `(1, 1, T, T)` mask
where, for query position `i` and key position `k`:

```
allow[i, k] = (k < floor(i / L) · L)        # strictly earlier canvases: causal
            | (floor(k / L) == floor(i / L)) # own canvas: fully bidirectional
```

with `L = canvas_len = 256`. So:

- **Across canvases:** standard causal attention — canvas `b` sees canvases
  `0..b−1` and nothing later. This is what makes the final sequence a valid
  left-to-right conditional factorization.
- **Within a canvas:** full bidirectional attention. The 256 positions of a
  canvas denoise *together* — this is the parallelism the whole project buys.

The mask is the contract that lets one dense transformer do two jobs: a
**denoiser** within a canvas (bidirectional) and a **conditional LM** across
canvases (causal). It is asserted divisible (`seq_len % canvas_len == 0`);
the sampler's prompt prefill relaxes this with an inline partial-canvas mask
with identical row semantics (`inference/generate.py:_prefix_mask`).

### 2.2 Decode-time masks

During canvas decoding the in-flight canvas is the *last* block of the
attention window, and everything before it is finalized and frozen.
`models/mask.py:build_canvas_decode_mask(prefix_len, canvas_len)` returns an
**all-ones** `(1, 1, L, prefix_len + L)` view: the finalized prefix is fully
visible and the in-flight canvas sees itself fully. There is no zero-masking
of the canvas against itself — the mask is trivially permissive because the
*content* of the canvas is what's still noisy, not its visibility. (This
supersedes DESIGN §2.5's earlier "zero-mask row over itself" wording; the
binding sentence is the last one of §2.5: "canvas = last block in the mask".)

### 2.3 Attention implementation

- Production path: `models/mask.py:block_causal_sdpa_attention` →
  `F.scaled_dot_product_attention` with the boolean mask.
- Ground-truth path: `models/mask.py:eager_block_causal_attention` — an
  explicit O(T²) scores→mask→softmax→@v mirror, kept (do not "clean it up")
  because `tests/test_attention.py` transplants weights between the eager
  branch and SDPA to prove they agree.
- Heads: GQA `16` query / `4` KV heads, `head_dim=64`; KV heads are tiled to
  the query head count *before* the SDPA helper (SDPA requires equal head
  counts) — `models/attention.py:DenoiseAttention`.
- Positions: RoPE in the canonical GPT-NeoX/LLaMA rotate-half form
  (`models/attention.py:apply_rope`). A regression property test pins that
  RoPE preserves per-position norms and that `q·k` depends only on `m − n`.
  Note the consequence: a uniform +c shift of all positions leaves attention
  unchanged — the plan's original "shifted positions → different attention"
  test could only pass against a buggy interleaved-frequency rope, so the
  test asserts relative spacing instead (SDD Ruling 12).

---

## 3. Self-conditioning

### 3.1 The mechanism

Diffusion transformers denoise better when they can see their own previous
guess. At training time, with probability `p = 0.5` per step, we run the
model once **without** self-cond input, re-embed its predicted posterior at
the input, and run it again:

```
pass 1 (no grad):  ĥ = f(xt, t)                 → sc = p̂ @ E   (posterior re-embedded)
pass 2 (grad):     h  = f(xt, t, sc = detach(sc))
```

- Module: `models/selfcond.py:SelfConditioning` — a zero-init linear proj
  from `d_model` back to `d_model`, added to the normalized hidden state.
- Zero-init equivalence: at init the projection outputs exactly 0, so
  `f(xt, t, sc) == f(xt, t)` bit-for-bit in fp64 — pinned by
  `tests/test_self_conditioning.py::test_zero_init_equivalence` with
  `atol=0, rtol=0`. Training therefore starts from the no-sc model and
  *learns* how much to use the conditioning.
- Routing invariant (SDD Ruling 6): the add happens **exactly once per
  path**. `models/transformer.py:DiffusionGemma.final_hidden` = backbone + the `W_sc` add
  and is the loss path; `models/transformer.py:DiffusionGemma.head_forward` applies the same
  single add before the LM head. Never compose the two.
- Detachment (DESIGN §2.4): pass 1 runs under `torch.no_grad()` and its
  output enters pass 2 **detached** — the pre-pass is an input constructor,
  not a second gradient path.

### 3.2 The memory reason for `training/losses.py:chunked_p_embed`

Re-embedding the posterior means materializing `p @ E` — a full-vocab softmax
at `(8, 4096, 50257)` fp32 would blow the §4.0 memory budget by itself.
`chunked_p_embed` computes the same product one vocab chunk at a time under
`no_grad` for the self-cond pre-pass, and the sampler reuses the same idea
inline (`inference/generate.py:BlockDiffusionSampler._denoise_step` computes `sc_next = p @ E` for
the cross-step conditioning at eval).

### 3.3 Attribution

At eval, self-conditioning is **cross-step**: step `k`'s input conditioning is
step `k−1`'s posterior re-embedded (`sc_next = p̂ₖ₋₁ @ E`), always on after
canvas step 1. This is how the model "sees its previous guess" during
iterative refinement — the mechanism that makes commit-and-renoise coherent
(the next step knows what the previous step believed).

---

## 4. The block-AR sampler

### 4.1 The decode loop

`inference/generate.py:BlockDiffusionSampler.generate`:

```
prefill(prompt)                      # one forward; KV cache is static
for each canvas:
    denoise_canvas(kv, prefix_len)   # T steps of uniform denoising (parallel over 256 tokens)
    encode_canvas(kv, prefix_len)    # re-encode the finalized canvas + KV append
```

- `inference/generate.py:BlockDiffusionSampler.prefill` — one forward over the prompt under
  `inference/generate.py:_prefix_mask`'s partial-canvas block-causal mask.
- `inference/generate.py:BlockDiffusionSampler.denoise_canvas` — starts from pure uniform noise
  (`torch.randint`, **not** `q_sample`: the sampler redraws the noise
  distribution itself; `q_sample` is train-time corruption only, SDD Ruling 17).
- `inference/generate.py:BlockDiffusionSampler.encode_canvas` — re-encodes the finalized canvas
  under the all-ones `models/mask.py:build_canvas_decode_mask` and appends its
  KV to the cache. The cached KV must equal what a fresh single-shot
  block-causal forward would write — enforced bit-exactly (fp64) by
  `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`.

The KV cache grows **once per finalized canvas** (256 tokens per block-AR
step), not once per token. At equal sequence length the cache shape is
identical to AR (all tokens are eventually encoded) — the win is *fewer
forward passes*, not a smaller cache.

### 4.2 The commit rule

Inside a denoise step (`inference/generate.py:BlockDiffusionSampler._denoise_step`):

```
p      = softmax(logits)                        # x̂0 posterior
commit = (am == prev_am) | (conf >= prev_conf)  # unchanged-or-stronger
new    = commit & ~old_committed                # never overwrite a frozen token
x      = where(new, x̂0_draw, where(old_committed, x, uniform_noise))
```

1. A position commits when its posterior mode is unchanged from the previous
   step **or** its confidence got stronger — the "unchanged-or-stronger"
   posterior rule of DESIGN §2.5.
2. Committed positions are **never overwritten** (`new = commit &
   ~old_committed`): once frozen, always frozen. Monotonicity is pinned by
   `tests/test_sampler.py::test_commit_rule_monotone`.
3. Uncommitted positions are redrawn **pure uniform** every step.
4. The x̂0 draw at step `k` uses temperature `τ(k)` annealed linearly from
   `temp_start` to `temp_end` across the schedule (matching DESIGN §2.5's
   literal formula; note τ never exactly reaches `temp_end` on the last
   step — recorded as a deferred minor).
5. The returned canvas is `where(committed, x, x̂0_draw)`: every position ends
   with its best posterior guess even if it never "committed".

### 4.3 Eval-time time normalization

The sampler passes `time_steps = SamplerConfig.n_diffusion_steps` through
every forward (`inference/generate.py:BlockDiffusionSampler.prefill`,
`inference/generate.py:BlockDiffusionSampler._canvas_step_logits`,
`inference/generate.py:BlockDiffusionSampler.encode_canvas`), so eval at `T_eval ≤ 32` keeps
`t/T ∈ (0,1]` in the range the time embedding was normalized to. Prompt and
finalized canvases enter at `t=0`. (SDD Ruling 16; refines DESIGN §2.5's
silence on eval normalization.)

### 4.4 Entropy-bounded adaptive stopping

The fixed schedule always spends `T` steps per canvas. The adaptive variant
stops early when the posterior has settled:

```
entropy_t = mean over canvas positions of H[p(·|position)]
if entropy_t < entropy_threshold for stability_steps consecutive steps: stop
```

- Implementation: the tail of `inference/generate.py:BlockDiffusionSampler.denoise_canvas`.
- Contract: with `adaptive=False` the loop ignores entropy and runs the full
  schedule — the fallback is bit-identical to the fixed sampler
  (`tests/test_sampler.py::test_adaptive_off_ignores_entropy`).
- Why entropy, not confidence: mean max-probability saturates early (a few
  very confident tokens pin the mean), while per-position entropy keeps
  measuring unresolved mass. The threshold is calibratable —
  `tests/test_sampler.py` proves some threshold separates "stop at step 2"
  from "run the full schedule".
- FLOP accounting: `inference/evaluate.py:SpeedupEvaluator` counts
  token-forwards; adaptive never spends more forwards than fixed
  (`tests/test_inference.py::test_flop_counter_adaptive_le_fixed`).

---

## 5. The §4.0 memory story (why chunked-CE exists)

The naive training step materializes full-vocab logits `(B, T, V)`:

| term (micro_bs=8, seq=4096, V=50,257) | naive | chunked (`vocab_chunk=8192`) |
|---|---|---|
| fp32 logits/CE chain | ~6.6 GB | **~1.1 GB** (one chunk alive at a time) |
| params + AdamW fp32 state | ~5.5 GB | ~5.5 GB |
| boundary activations (grad-ckpt every 3) | ~1.6 GB | ~1.6 GB |

- `training/losses.py:chunked_x0_ce` computes `hidden @ E[chunk].T` one chunk
  at a time; each chunk's logits are computed under
  `torch.utils.checkpoint(use_reentrant=False)` so only **one chunk's fp32
  autograd chain is alive** at a time; the loss combines per-chunk fp32
  logsumexp → global logsumexp → target-logit gather, so the result equals the
  eager CE to 0.0 max abs diff at production vocab (verified in review;
  `tests/test_loss.py::test_chunked_equals_eager` pins `atol=1e-6`).
- The self-cond pre-pass has its own full-vocab hazard (`p @ E`), handled by
  `training/losses.py:chunked_p_embed` (§3.2).
- `utils/memory.py:estimate_model_memory_gb` encodes the §4.0 table and is
  monotone in batch; `utils/memory.py:assert_fits_in_available_gpu` turns the
  estimate into a pre-flight raise (logged, never silent, if the probe fails).
- `torch.compile` scope: per-block in-place on CUDA only
  (`training/pretrain.py:Pretrainer._compile_blocks`); corruption, loss, and
  sampler stay eager so numerics are reproducible.

## 6. Training recipe

`training/pretrain.py:Pretrainer` + `training/pretrain.py:TrainingConfig`,
driven by `configs/pretrain_a100_380m.yaml` via
`models/transformer.py:DiffusionGemmaConfig.from_yaml`:

| knob | value | note |
|---|---|---|
| steps | 61,000 optimizer steps | ~8.0B tokens / (8 · 4 · 4096) — Chinchilla-optimal for ~343.5M params |
| batch | micro_bs 8 × grad_accum 4 | `total_steps`/`save_interval`/`log_interval` count **optimizer** steps |
| optimizer | AdamW lr 3e-4, β (0.9, 0.95), wd 0.1 | linear warmup 2000 → cosine to 5% (`min_lr_ratio`) |
| precision | BF16 autocast + TF32 | `training/pretrain.py:Pretrainer` |
| grad | clip 1.0, checkpoint every 3 layers | `grad_ckpt_every` |
| NaN guard | 5 consecutive → rollback | `utils/checkpoint.py:CheckpointManager` |
| checkpoints | every 4,000 steps, 3 files each | weights safetensors + optim + meta; a step is resumable only when all three exist |

**Resume determinism** (pinned bitwise by
`tests/test_training.py::test_checkpoint_resume_determinism`): every draw
(per-canvas `t`, corruption mask, self-cond gate) comes from a per-micro-step
`torch.Generator` seeded `seed·100_003 + micro_count`
(`training/pretrain.py:Pretrainer._step_rng`), and the data order is replayed
by `data/dataset.py:ShuffledRangeSampler` with `offset_batches` (offset wraps
mod `n_windows`, so the sampler stays finite). On resume `_micro_count`
restores as `opt_steps × grad_accum`. Caveat: NaN-guard skips advance the
micro count without an optimizer step, so bit-equality holds for NaN-free
runs (SDD ledger Minor 3).

**Data**: `data/prepare_data.py:main` delegates to the workspace
`shared_data` pipeline (GPT-2 BPE, 50M-token uint32 shards, no
cross-document boundaries, seed 42) and pins `LLM_DATA_ROOT` to
`data/pretrain_chinchilla` — the pack stage runs as a subprocess that only
honors that env var. `data/dataset.py:ShardWindows` windows the shards flat
(`seq_len` tokens, **no** +1 AR shift — diffusion reconstructs x0 everywhere),
and `data/dataset.py:build_dataloader` wraps them in the resumable shuffler.

**Params**: the mandated architecture (GQA 16Q/4KV head_dim 64 + weight
tying) pins exactly **343,516,160 ≈ 343.5M** parameters — not the "~380M" in
early planning docs, whose param table counted full-MHA K/V at 1024 (SDD
Ruling 13). README carries the honest number.

## 7. Evaluation

`inference/evaluate.py:SpeedupEvaluator.evaluate(...)` produces the DESIGN
§4.2 rows:

| row | meaning | forwards |
|---|---|---|
| `fixed_T16` / `fixed_T32` | full schedule | exactly `1 + n_canvases·(T+1)` (1 prefill + per canvas: T denoise + 1 encode) |
| `adaptive_T32` | entropy-bounded stop | ≤ fixed (pinned by test) |
| `ar_kv_analytic` | KV-cached AR decode | 1 forward/token **by construction** |

- The FLOP proxy is **token-forwards** (tokens through the backbone), counted
  by instrumenting `model.backbone` — robust to sampler internals.
- `tokens/forward` of a fixed schedule is `L/(T+1)`: at the production
  canvas L=256, `fixed_T32` ≈ 7.8 tokens/forward even before adaptive
  stopping. **Do not** quote tiny-model eval numbers (canvas 32 → <1).
- Quality anchor: `inference/evaluate.py:heldout_x0_nll` — the training
  objective (chunked x0-CE at sampled t, no self-cond input) on a held-out
  shard, in nats/token, directly comparable to an AR baseline's CE; the +5%
  acceptance verdict is `scripts/loss_parity_eval.py --ar-nll <AR CE>`.
- CLI: `scripts/speedup_eval.py` (with `--flops-only`) and
  `scripts/loss_parity_eval.py`. Honest gaps: AR wall-clock tokens/s and the
  parity verdict need the external LLaMA-3-Lite baseline run and a trained
  checkpoint (A100 pod); both scripts print the gap when inputs are absent.
- `models/transformer.py:DiffusionGemma.generate` is the one-call entry point
  that delegates to the sampler.

## 8. Recipe deltas vs upstream

How this repo's recipe differs from the upstream discrete-diffusion-LM
literature and from the portfolio's other Lites. Each delta is a *choice*, and
the alternative is named.

### 8.1 From-scratch pretrain vs SFT / SD·RL reuse

Upstream block-diffusion work (e.g. LLaDA-style and Block-Diffusion-style
models) typically starts from a **pretrained AR backbone** and teaches it the
diffusion objective — an SFT-style phase — or distills from an AR teacher
with self-distillation/RL. This repo does **neither**: the denoiser is
initialized from scratch (`init_std 0.02`, `selfcond.proj` re-zeroed by
`models/transformer.py:DiffusionGemma`'s `_init_weights`) and trained on the
8.0B-token mixture with the diffusion objective only.

Why: the portfolio's headline is a *from-scratch* control — every Lite is
initialized, trained, and measured on identical data/tokenizer/harness. A
distilled model inherits an AR teacher's losses and cannot separate "what the
diffusion objective teaches" from "what the teacher already knew". The cost
is honest: we expect a loss gap vs an AR model at equal budget, and the
measured number is disclosed (`scripts/loss_parity_eval.py`), never hidden —
the headline is throughput *at acceptable quality* (DESIGN §4.2(3)).

### 8.2 Entropy rule vs RL distillation for adaptive compute

Upstream adaptive-length/adaptive-compute methods spend RL or distillation
budget to learn *when to stop* (a learned halting policy). This repo's
adaptive stopping is a **closed rule**: stop when the mean posterior entropy
over the canvas stays below a threshold for `stability_steps` consecutive
steps (`inference/generate.py:BlockDiffusionSampler.denoise_canvas`). No extra training phase, no
reward model, no halting head — the posterior's own entropy *is* the signal,
because the model was trained to sharpen it toward the x0 posterior at every
`tau`. The trade-off: the threshold is a hyperparameter (calibratable in
`tests/test_sampler.py`), not learned; a badly calibrated threshold either
fires early (garbage tokens) or never (fixed-schedule cost — and the fallback
is provably identical to the fixed sampler).

### 8.3 Other deltas from the D3PM/MDLM lineage

| choice | here | common upstream |
|---|---|---|
| noise state | uniform over the **whole vocab** (D3PM uniform) | `[MASK]` absorbing state (MDLM/BERT-style) |
| parameterization | direct **x0** prediction | x0 *or* edge/score parameterizations |
| schedule | single cosine `ᾱ(t) = cos²(π/2 · t/T)` (`models/diffusion.py:alpha_bar`) | learned or linear schedules |
| time granularity | **per-canvas** `t` (every 256-token canvas gets its own t) | per-sequence t |
| factorization | block-AR across canvases (causal), bidirectional within | full-sequence diffusion, no causal structure |
| KV cache | grows once per finalized **canvas** | per-token (AR) or none (full diffusion) |
| corpus | 8.0B tokens, GPT-2 BPE, `shared_data` mixture, no cross-doc boundaries | often OpenWebText/other mixes |

The per-canvas `t` is a deliberate variance-reduction choice: a sequence of
16 canvases sees 16 different corruption levels in one micro-batch row,
covering more of the `t`-marginal per step than a single shared `t`.

## 9. Repository map

```
models/       mask.py (masks), diffusion.py (process), attention.py, selfcond.py,
              time_embed.py, block.py, transformer.py (model + config)
training/     losses.py (chunked-CE), pretrain.py (loop)
data/         prepare_data.py (shared_data shim), dataset.py (ShardWindows)
inference/    generate.py (sampler), evaluate.py (headline harness)
utils/        checkpoint.py, logging.py, memory.py
scripts/      speedup_eval.py, loss_parity_eval.py, check_docs.py, build_docs_html.py
docs/         concepts/ + guides/ + references/ (doc map: docs/README.md)
```

SDD decision ledger: `.superpowers/sdd/EXECUTION-PLAN-diffusiongemma-lite/progress.md`
(rulings 1–26); the durable spec is
`../llm-research/{DESIGN,EXECUTION-PLAN}-diffusiongemma-lite.md` outside the repo.