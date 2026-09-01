# Foundations & Architecture — DiffusionGemma-Lite

> **Canonical** for the from-scratch primitives every other chapter assumes.
> Educational textbook chapter: it builds the language model from first
> principles, derives every number in the config, and ends with practice
> problems, a glossary, and interview Q&A. Every code claim is anchored to a
> real symbol (`file.py:Symbol`); where a number is an estimate rather than a
> measurement it is tagged `[INFERENCE]`.

**Depends on:** nothing — this is the entry point · **Read next:**
[diffusion-core](diffusion-core.md) · [block-causal-attention](block-causal-attention.md) ·
[self-conditioning](self-conditioning.md) · [sampler](sampler.md)

---

## Table of Contents

1. [The generation problem: why token-by-token decoding is slow](#1-the-generation-problem-why-token-by-token-decoding-is-slow)
2. [Diffusion in 90 seconds](#2-diffusion-in-90-seconds)
3. [Discrete diffusion: D3PM and the uniform state](#3-discrete-diffusion-d3pm-and-the-uniform-state)
4. [Block-AR: the design this repo commits to](#4-block-ar-the-design-this-repo-commits-to)
5. [The transformer primitives, from scratch](#5-the-transformer-primitives-from-scratch)
6. [The x0 objective](#6-the-x0-objective)
7. [Chinchilla accounting: 343,516,160 params and 8.0B tokens](#7-chinchilla-accounting)
8. [Worked example: one full forward pass at test-tiny scale](#8-worked-example-one-full-forward-pass-at-test-tiny-scale)
9. [Training dynamics and optimization](#9-training-dynamics-and-optimization)
10. [What the model must learn at every corruption level](#10-what-the-model-must-learn-at-every-corruption-level)
11. [Practice problems (with answers)](#11-practice-problems-with-answers)
12. [Glossary](#12-glossary)
13. [Load-bearing invariants](#13-load-bearing-invariants)
14. [Interview Q&A](#14-interview-qa)
15. [The memory story: why chunked cross-entropy exists](#15-the-memory-story-why-chunked-cross-entropy-exists)
16. [The data pipeline (from-scratch view)](#16-the-data-pipeline-from-scratch-view)
17. [Evaluation: what "faster" and "as good" mean](#17-evaluation-what-faster-and-as-good-mean)
18. [The recipe deltas](#18-the-recipe-deltas-why-this-differs-from-the-upstream-literature)
19. [Training dynamics (advanced)](#19-training-dynamics-advanced-determinism-and-the-nan-guard)
20. [References](#20-references)

---

## 1. The generation problem: why token-by-token decoding is slow

### 1.1 Autoregressive decoding, one token at a time

Every model in this portfolio's generation family — LLaMA-3-Lite,
DeepSeek-v3-Lite, Mamba-3-Lite, GPT2 — produces text the same way: one token
per forward pass, each conditioned on all previous tokens:

```
p(x) = p(x1) · p(x2 | x1) · p(x3 | x1..2) · ...
```

This *left-to-right factorization* is the defining property of an
autoregressive (AR) language model. It is also its speed ceiling. To generate
`n` tokens the model must run `n` sequential forward passes; pass `k` cannot
start before pass `k−1` has produced token `k`. At production batch sizes the
per-pass cost is dominated by loading weights from HBM, not by arithmetic —
so generating 1,024 tokens costs 1,024 weight-loading round trips no matter
how fast the matmuls are.

### 1.2 The KV cache (and what it costs)

Caching each layer's keys and values removes all *recomputation* from decode:
pass `k` processes only the new token and attends over the cache. What remains
is sequential latency — one token of progress per forward pass, forever. A
KV-cached AR decoder therefore spends **exactly 1 forward and 1 token-forward
per generated token**. That analytic fact (`inference/evaluate.py:SpeedupEvaluator`
encodes it as the `ar_kv_analytic` row) is the baseline this project races
against.

The cache itself has a size, and it depends on the head layout. For a model
with `n_layers`, `n_kv_heads`, `head_dim`, storing keys and values in BF16:

```
cache bytes/token = 2 (k and v) × n_layers × n_kv_heads × head_dim × 2 bytes
```

DiffusionGemma-Lite uses GQA with 4 KV heads (§7.3), so one token of context
costs `2 · 24 · 4 · 64 · 2 = 24,576 B ≈ 24 KB`; a full-MHA model with 16 KV
heads at the same width would pay 96 KB/token — 4× more. At the 4,096-token
context that is a 100.7 MB cache (402.7 MB under MHA). GQA does not change the
*number* of cache writes; it shrinks each one. The block-AR design (§4) attacks
the other axis: how many writes and forwards a sequence needs at all.

### 1.3 The throughput ladder

| decode scheme | forwards for n new tokens | tokens per forward |
|---|---|---|
| naive AR (recompute everything) | n · (growing prefix) | ≪ 1 |
| KV-cached AR (all siblings) | exactly n | exactly 1 |
| **block-AR diffusion (this repo)** | n/L · (T+1) for canvas L, schedule T | L/(T+1) |
| block-AR + adaptive stopping | ≤ n/L · (T+1) | ≥ fixed schedule |

At the production canvas `L = 256`: `fixed_T16` yields 256/17 ≈ **15.1
tokens/forward**, `fixed_T32` yields 256/33 ≈ **7.8 tokens/forward**, and
entropy-bounded adaptive stopping only cuts forwards further
(`tests/test_inference.py::test_flop_counter_adaptive_le_fixed` pins adaptive ≤
fixed). Generating 1,024 tokens with `fixed_T32` costs `4 canvases · 33 = 132`
forwards instead of 1,024. The rest of this document builds the machine that
makes those forwards *good* — a transformer that can denoise 256
independently-corrupted positions in parallel while still reading everything
finalized to its left.

---

## 2. Diffusion in 90 seconds

> **Primer (assumed background).** Diffusion models learn to *undo* a gradual
> corruption process. Training corrupts data slightly (many noise levels); the
> network learns to predict the clean data from any corrupted state; sampling
> starts from pure noise and repeatedly applies the learned cleanup. You need
> exactly three ideas: a **forward process** (data → noise), a **denoiser
> network** trained at all noise levels, and a **sampling loop** that runs the
> denoiser iteratively. Nothing here requires Gaussian diffusion specifics.

### 2.1 The forward and reverse processes

In continuous diffusion (DDPM) a clean image `x0` is progressively blurred by
adding Gaussian noise of increasing magnitude. The forward process is a
Markov chain `x0 → x1 → ... → xT` after which `xT` is (approximately) pure
noise. A network `f(xt, t)` is trained to predict the clean signal `x0` (or
equivalently the noise) from any intermediate `xt`; generation runs the chain
backwards from a fresh `xT`, one learned denoising step at a time.

The two properties that matter — and that the discrete version in §3
reproduces exactly — are:

1. **Closed-form corruption.** The state after *any* number of steps has a
   closed form, so training corrupts `x0 → xt` in a single draw instead of
   stepping the chain. (`models/diffusion.py:q_sample` is precisely this
   single-draw corruption.)
2. **A schedule parameter** `ᾱ(t) ∈ (0, 1]` that interpolates between "almost
   clean" (`t = 1`) and "pure noise" (`t = T`). Every intermediate level is a
   training case; the schedule decides how much of the *easy* (low-noise) and
   *hard* (high-noise) work the model sees.

### 2.2 Why text breaks Gaussian diffusion

Pixels are continuous: averaging two images yields an image. Tokens are
categorical: there is no meaningful "halfway between token 464 and token
9,822". The fix is to make the corruption process **categorical** — the noise
state must itself be a sequence of valid tokens. That is the D3PM insight, and
it is why this project's forward process (§3) is a *token replacement* rule,
not an additive-noise rule.

---

## 3. Discrete diffusion: D3PM and the uniform state

### 3.1 The D3PM formulation

D3PM (Discrete Denoising Diffusion Probabilistic Models) replaces the Gaussian
kernel with a **transition matrix** `Q_t`: each token jumps to another token
type with probabilities given by row `x0` of `Q_t`. The schedule scalar
`ᾱ(t)` controls how much of the row's probability mass stays on the clean
token. Two families dominate:

| family | noise state | row of Q_t | lineage |
|---|---|---|---|
| **absorbing (mask) diffusion** | a dedicated `<mask>` token | `<mask>` is absorbing: once masked, always masked | BERT-style pretraining, MDLM, SEDD |
| **uniform-state diffusion** | *every vocabulary token* is a possible noise value | `ᾱ(t)` on the clean token, `(1−ᾱ)/V` on all others | D3PM uniform, this repo |

`models/diffusion.py:corruption_probs` builds exactly that row:
`ᾱ(t)` on the clean token, `(1−ᾱ)/V` on every other token (it sums to 1). The
repo's schedule is the closed-form cosine `ᾱ(t) = cos²(π/2 · t/T)`
(`models/diffusion.py:alpha_bar`) — monotone from ≈1 to ≈0, no learnable
schedule parameters.

### 3.2 Why uniform-token noise (the load-bearing choice)

With an absorbing `MASK` state, a partially generated sequence carries an
obvious "hole pattern" — the model sees where the unknowns are. With
uniform-state noise, **the corrupted canvas is statistically indistinguishable
from any other token sequence**: every position holds a legal token at every
corruption level. Three consequences drive the whole design:

1. **Commit-and-renoise works.** The sampler can freeze high-confidence
   tokens and redraw the rest as *uniform random tokens* — the frozen text
   stays a valid model input at every step (§4.3, [sampler](sampler.md)).
2. **The model cannot lean on mask positions.** It must genuinely denoise:
   at `t = T` the canvas is uniform noise over 50,257 tokens and the model
   performs unconditional canvas synthesis; at `t = 1` it performs light
   correction (`ᾱ(1) = 0.9904` for `T = 16` — see the table below).
3. **Loss targets stay dense.** Every position is scored against its clean
   token every step (`data/dataset.py:ShardWindows` windows carry **no +1
   next-token shift** — diffusion reconstructs `x0` everywhere).

### 3.3 The cosine schedule, numerically

`ᾱ(t) = cos²(π/2 · t/T)` with train-time `T = 16`
(`configs/pretrain_a100_380m.yaml` `n_diffusion_steps: 16`):

| t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ᾱ(t) | .9904 | .9619 | .9157 | .8536 | .7778 | .6913 | .5975 | .5000 | .4025 | .3087 | .2222 | .1464 | .0843 | .0381 | .0096 | .0000 |

Read the table as a curriculum: `t = 1` leaves 99% of tokens intact (the model
learns light correction), `t = 8` is a coin flip per token, `t = 16` is
effectively a fresh uniform draw (`ᾱ(16) = cos²(π/2) = 0` exactly — pinned by
`tests/test_diffusion.py::test_final_step_pure_noise`). Because the
*forward-process row* (`models/diffusion.py:corruption_probs`) puts `ᾱ` on the
clean token and `(1−ᾱ)/V` elsewhere, the probability a given token survives
corruption at `t = 8` is `0.5 + 0.5/50257 ≈ 0.50001` — the clean token is only
special by a `1/V` sliver. The model must learn to recover structure from a
canvas that is half random vocabulary tokens.

Per-canvas `t` (each 256-token canvas draws its own `t ~ U{1..T}`,
`models/diffusion.py:sample_canvas_t`) is a variance-reduction choice: one
micro-batch row of 16 canvases covers 16 corruption levels of the `t`-marginal
instead of one. The derivation lives in [diffusion-core](diffusion-core.md).

---

## 4. Block-AR: the design this repo commits to

### 4.1 The spectrum of text-generation designs

| scheme | parallelism | quality structure | cache growth |
|---|---|---|---|
| token-AR (GPT-style) | 1 token/forward | exact left-to-right factorization | per token |
| full-sequence diffusion | all tokens at once | no causal structure; fixed-length canvas | none, but no conditioning chain |
| **block-AR diffusion (here)** | **L tokens/forward** | **causal across canvases, bidirectional within** | **once per finalized canvas** |

Full diffusion has no notion of "everything so far is final" — it cannot
extend a sequence conditionally. Token-AR is maximally sequential. Block-AR is
the hybrid: the sequence is a chain of **canvases** (L = 256 tokens each);
within a canvas, all positions denoise together in parallel; across canvases,
generation is strictly left-to-right. The conditional factorization

```
p(canvas_0, ..., canvas_15) = p(canvas_0) · p(canvas_1 | canvas_0) · ...
```

is *exactly* as valid as token-AR's, but each factor is a 256-token joint the
denoiser models internally — one forward pass refines a whole canvas.

### 4.2 The mask that does both jobs

One dense transformer plays denoiser and conditional LM simultaneously
because attention is governed by
`models/mask.py:build_block_causal_mask`: for query position `i` and key
position `k`,

```
allow[i, k] = (k < floor(i/L) · L)  |  (floor(k/L) == floor(i/L))
             strictly earlier canvases   own canvas, all of it
```

Concretely (seq 8, canvas 4; `1` = attend, `.` = masked):

```
        k: 0 1 2 3 4 5 6 7
q0 (c0) |  1 1 1 1 . . . .
q1 (c0) |  1 1 1 1 . . . .
q2 (c0) |  1 1 1 1 . . . .
q3 (c0) |  1 1 1 1 . . . .
q4 (c1) |  1 1 1 1 1 1 1 1
q5 (c1) |  1 1 1 1 1 1 1 1
q6 (c1) |  1 1 1 1 1 1 1 1
q7 (c1) |  1 1 1 1 1 1 1 1
```

Canvas 0's rows see only canvas 0 (bidirectional); canvas 1's rows see
everything. Verbatim semantics are pinned by `tests/test_mask.py`
(`test_block_causal_mask_matches_manual`, `test_mask_causal_across_canvases`,
`test_mask_bidirectional_within_canvas`). The mask makes one transformer a
bidirectional denoiser *and* a valid left-to-right LM — the full derivation,
the FlexAttention BlockMask variant, and the decode-time all-ones view are in
[block-causal-attention](block-causal-attention.md).

### 4.3 The generation loop

`inference/generate.py:BlockDiffusionSampler.generate`:

```
prefill(prompt)                      # 1 forward; KV cache established
for each canvas:
    denoise_canvas(kv, prefix_len)   # T_eval steps, 256 tokens in parallel
    encode_canvas(kv, prefix_len)    # re-encode finalized canvas, append KV
```

Each canvas starts as **pure uniform noise** (`torch.randint` over the vocab —
`inference/generate.py:BlockDiffusionSampler.denoise_canvas`; the sampler
redraws its own noise, `models/diffusion.py:q_sample` is train-time-only, SDD
Ruling 17). Over `T_eval` steps the model's posterior `p(x0 | xt, t)` commits
confident tokens (never overwriting them) and re-noises the rest. When the
canvas finalizes, it is re-encoded into the KV cache and the next canvas
conditions on it. The contract that this chained cache equals a fresh
single-shot forward — **bit-exact in fp64** — is
`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`.

The KV cache therefore grows **once per finalized canvas (256 tokens), not
once per token**. At equal context length its shape is identical to AR's
(every token is eventually encoded) — the win is **33× fewer forwards**
(1,024 tokens: 132 forwards at `fixed_T32`, §1.3) and entropy-bounded early
stopping on top ([sampler](sampler.md)).

### 4.4 Why not start from a pretrained AR backbone?

Upstream block-diffusion work (LLaDA, Block-Diffusion) typically SFTs a
diffusion objective onto a pretrained AR teacher. This repo trains **from
scratch on the diffusion objective only** — every portfolio Lite shares the
same data, tokenizer, and budget, so a distilled model could not separate
"what the objective teaches" from "what the teacher knew". The cost is
disclosed, never hidden: a loss gap vs AR at equal budget is expected and is
measured by the parity harness (`inference/evaluate.py:heldout_x0_nll`,
verdict via `scripts/loss_parity_eval.py`). Recipe deltas are §8 of
[`DIFFUSION.md`](../../DIFFUSION.md).

---

## 5. The transformer primitives, from scratch

The denoiser (`models/transformer.py:DiffusionGemma`) is a Gemma-class dense
transformer with one exotic input (per-canvas time) and one exotic add
(self-conditioning). Everything else is standard, and this section derives
each piece from scratch.

```
GPT-2 BPE tokens (V = 50,257)
    │
    ▼
Embedding E (50257 × 1024)  ← weight-tied with the LM head
    │  + per-canvas time embedding (sinusoid of t/T → 256 → d_model)
    ▼
24 × DenoiseBlock:
    RMSNorm → GQA attention (16Q/4KV, head_dim 64, RoPE, block-causal mask)
    RMSNorm → SwiGLU FFN (ffn_dim 3072)
    │
    ▼
final RMSNorm → (+ zero-init self-conditioning add) → head: h @ Eᵀ
```

### 5.1 Embeddings and weight tying

`models/transformer.py:DiffusionGemma` projects token ids through
`nn.Embedding(V, d_model)`. The LM head is `nn.Linear(d_model, V, bias=False)`
— mathematically `h @ Eᵀ` — and with `weight_tying: true` the head's weight
**is** the embedding matrix (`self.head.weight = self.embed.weight`,
`models/transformer.py:DiffusionGemma.__init__`). Sharing saves 51.4M
parameters (the single largest tensor in the model) and is part of the
canonical 343,516,160 count (§7). Shared storage is pinned by
`tests/test_models.py::test_weight_tying_shared`.

Shapes: `E ∈ R^{50257 × 1024}`; an input `(B, T)` of token ids becomes
`h ∈ R^{B × T × 1024}`.

### 5.2 RMSNorm and pre-norm residuals

> **Primer.** Normalization keeps activations in a stable range so deep
> stacks train. Modern LLMs use **RMSNorm** (Gemma, LLaMA): rescale each
> vector by its root-mean-square, then apply a learned per-dimension scale.
> No mean subtraction (that is LayerNorm), no bias. **Pre-norm** means the
> norm sits *inside* each residual branch, so the residual stream is an
> unnormalized additive highway — gradients flow through `+` untouched.

`models/block.py:RMSNorm`:

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) · γ        γ learned, init 1; eps = 1e-5
```

with `F.rms_norm` as the fused stdlib kernel of the manual formula. Each
`models/block.py:DenoiseBlock` is

```
h ← h + Attention(RMSNorm(h))       # attention sublayer (pre-norm)
h ← h + SwiGLU(RMSNorm(h))          # FFN sublayer (pre-norm)
```

and a final `RMSNorm` (`models/transformer.py:DiffusionGemma.backbone`) runs
before the head. Test coverage: `tests/test_models.py::test_grad_flow_all_params`
(every parameter receives gradient), `tests/test_models.py::test_forward_shapes`.

### 5.3 Grouped-query attention (GQA)

Multi-head attention gives every head its own key/value projections: 16
heads × head_dim 64 → each cached token costs `2 · 24 · 16 · 64 · 2 B = 96 KB`
(§1.2). GQA shares each KV head across a *group* of query heads:

- **MHA**: 16 query heads, 16 KV heads. Expressive; heavy cache.
- **MQA**: 16 query heads, 1 KV head. Tiny cache; quality regression.
- **GQA (here)**: 16 query heads, **4 KV heads** (`n_heads=16`,
  `n_kv_heads=4`, `head_dim=64` → 4 query heads share each KV head). The
  middle of the trade: cache drops 4× to 24 KB/token with no measured quality
  loss at this scale.

`models/attention.py:DenoiseAttention` projects `q` to 16 heads but `k`/`v`
to only 4; the SDPA/Flex kernels consume the untiled KV directly
(`enable_gqa=True` in `models/mask.py:block_causal_sdpa_attention` and
`models/mask.py:flex_block_causal_attention`) — no `repeat_interleave`
expansion, which is precisely what the eager ground-truth twin does by hand
(`models/block.py:DenoiseBlock._attention`). Layout is pinned by
`tests/test_attention.py::test_gqa_kv_heads`.

| tensor | shape (B=1) | note |
|---|---|---|
| `q` | (16, T, 64) | one row per query head |
| `k`, `v` | (4, T, 64) | shared per group; **cached** |
| `attn out` | (16, T, 64) → reshape → out_proj | back to d_model |

### 5.4 RoPE — rotary position embeddings

> **Primer.** Attention is permutation-invariant: swap two tokens and, without
> position information, the output swaps identically. RoPE injects position by
> *rotating* each head's query/key vectors by an angle proportional to their
> absolute position, using a different frequency per 2-D plane. After
> rotation, the dot product `q_m · k_n` depends only on the **relative**
> offset `m − n` — which is what attention should care about — and each
> vector's norm is unchanged (a rotation preserves length).

Canonical GPT-NeoX / LLaMA **rotate-half** form: for head_dim `d_h`, split the
head into `half = d_h/2`; dimension pair `(i, i + half)` rotates in its own
plane by angle `m · θ^(−2i/d_h)`, where `m` is the absolute position and
`θ = rope_theta = 500,000` (`configs/pretrain_a100_380m.yaml`).

```
inv_freq_i = theta^(−2i / d_h),  i ∈ {0 .. half−1}
R(m) applied to x:  x'_{i}        = x_i · cos(m·f_i) − x_{i+half} · sin(m·f_i)
                    x'_{i+half}   = x_{i+half} · cos(m·f_i) + x_i · sin(m·f_i)
with f_i = theta^(−2i/d_h)                       # models/attention.py:apply_rope
```

**Worked example** (`head_dim = 4`, `θ = 10000` → `inv_freq = [1, 0.01]`;
verified against `models/attention.py:apply_rope`):

Take `q = [1.0, 0.5, −0.3, 2.0]` at position `m = 3`. Angles:
`(3·1, 3·0.01) = (3.0, 0.03)`.

```
q'0 = q0·cos(3.0) − q2·sin(3.0) = 1·(−0.9900) − (−0.3)(0.1411) = −0.9477
q'1 = q1·cos(0.03) − q3·sin(0.03) = 0.5·0.9996 − 2·0.0300 = 0.4398
q'2 = q2·cos(3.0) + q0·sin(3.0) = −0.3·(−0.9900) + 1·0.1411 = 0.4381
q'3 = q3·cos(0.03) + q1·sin(0.03) = 2·0.9996 + 0.5·0.0300 = 2.0141

rope(q, m=3) = [−0.9477, 0.4398, 0.4381, 2.0141]
```

Norm preservation and relativity, both measured against the real
implementation:

```
‖q‖ = 2.310844  →  ‖rope(q, 3)‖ = 2.310844          (norms preserved)
q_m · k_n = f(m − n):   q3·k5 = −1.027842 = q1·k3    (both f(−2))
```

Consequences, all pinned by
`tests/test_attention.py::test_rope_preserves_norms_and_relative_position`:

- A **uniform +c shift of every position leaves attention unchanged** (every
  `q·k` product depends only on differences). The plan's original
  "shifted positions → different attention" test could only pass against a
  buggy interleaved-frequency RoPE, so the test asserts relative spacing
  instead — SDD Ruling 12.
- Per-position norms never change, so attention logits stay scale-stable.

Production detail: `models/attention.py:DenoiseAttention` precomputes fp32
`cos`/`sin` tables for all `max_seq_len` positions at init
(`rope_cos`/`rope_sin` buffers, `models/attention.py:DenoiseAttention._roped_qkv`
indexes them per forward) — the trig was previously recomputed in every one of
the 24 layers on every forward. `models/attention.py:apply_rope` remains the
reference implementation the property tests pin. RoPE is applied to `q` and
the **untiled** `k` (4 heads), *before* the cache append — cached keys are
already roped (`models/attention.py:DenoiseAttention._roped_qkv`).

### 5.5 SwiGLU feed-forward

Each block's second sublayer (`models/block.py:DenoiseBlock.forward`):

```
gate, up = W13 · RMSNorm(h)            # W13: (1024 → 6144), one fused GEMM
h ← h + W2 · (SiLU(gate) ⊙ up)         # W2: 3072 → 1024
```

`SiLU(g) · u` is the **gated** variant of the standard FFN: instead of
`W2·act(x)`, the layer learns *which* features to amplify (`gate`) separately
from the features themselves (`up`). SwiGLU is the LLaMA/Gemma family
convention and needs `ffn_dim = 3072 = 3 · d_model` split as 6,144 = 2 × 3,072
for the fused gate/up projection (`models/block.py:DenoiseBlock.w13`).

### 5.6 Per-canvas time conditioning

The denoiser must know *how corrupted* each canvas is. Time enters as a
sinusoidal embedding of the **normalized** time `t/T`, projected to
`time_embed_dim = 256` and MLP-mapped to `d_model`
(`models/time_embed.py:CanvasTimeEmbedding`), then added to every token
embedding of its canvas
(`models/transformer.py:DiffusionGemma._add_canvas_time`):

```
angles = (t/T) · freq          # (B, n_canvases, 128)
time   = [sin(angles) | cos(angles)]     # (B, n_canvases, 256)
Δ      = MLP(time)              # (B, n_canvases, d_model), SiLU MLP
h      = h + Δ[canvas_ids]      # same vector broadcast over the canvas's 256 tokens
```

- Train-time forwards normalize with the model's train `T = 16`; **eval
  forwards pass `time_steps = SamplerConfig.n_diffusion_steps`** so
  `t/T ∈ (0,1]` stays in-range for any eval schedule `T ≤ 32` (SDD Ruling 16;
  `inference/generate.py:BlockDiffusionSampler.prefill` and
  `inference/generate.py:BlockDiffusionSampler._canvas_step_logits` both pass
  it).
- `t = 0` (prompt prefill, finalized-canvas re-encode) is T-independent by
  construction — `t=0` means "this content is not being denoised".

Shape trace: `t: (B, n_canvases) → (B, n_canvases, d_model)`; the embedding is
added per canvas, not per token.

### 5.7 Self-conditioning, primer level

When `self_conditioning: true`, the model can also receive `sc_input` — its
own previous posterior re-embedded through `E` — added to the final hidden
state through a **zero-initialized** projection
(`models/selfcond.py:SelfConditioning`). Zero init means the add is exactly
zero at initialization: the model starts bit-for-bit identical to a
no-self-cond model and *learns* how much to use the conditioning (fp64
bit-exactness pinned at `atol=0, rtol=0` by
`tests/test_self_conditioning.py::test_zero_init_equivalence`). Full
mechanism, the exactly-once routing invariant, and the detached pre-pass:
[self-conditioning](self-conditioning.md).

### 5.8 The three attention implementations

The same masked attention exists three times, deliberately
(`models/mask.py`):

| path | symbol | role |
|---|---|---|
| production (A100 config) | `models/mask.py:flex_block_causal_attention` | FlexAttention fused block-sparse kernel over a canvas-sized `BlockMask` |
| portable fallback | `models/mask.py:block_causal_sdpa_attention` | `F.scaled_dot_product_attention` with the bool mask, `enable_gqa` |
| ground truth | `models/mask.py:eager_block_causal_attention` | explicit O(T²) scores→mask→softmax→@V; the test oracle |

The three are proven equal by weight-transplant tests
(`tests/test_attention.py::test_attention_matches_eager`,
`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
`tests/test_models.py::test_flex_attn_impl_matches_sdpa`). The eager twin is
deliberate duplication — never consolidate it (project AGENTS.md §2).

---

## 6. The x0 objective

The model is trained to output, at every position, a posterior over the
**clean token** given `(xt, t)` — *x0-prediction*:

```
loss = CE(head(backbone(xt, t)), x0)      # cross-entropy against clean tokens
```

- Production loss: `training/losses.py:chunked_x0_ce` (never materializes the
  full `(B, T, V)` logits — [memory-engineering](memory-engineering.md)).
- Eager reference: `models/diffusion.py:x0_ce_loss` (test oracle).
- The softmax over the head **is** the x̂0 posterior the sampler needs to
  commit and re-noise; no marginalization over trajectories is ever needed.

> **Recorded typo (Ruling 19).** The plan text says the loss targets `xt`;
> that is a typo. Targets are `x0` — sampler semantics depend on it. Do not
> "fix" the code to match the plan (`SKILLS.md` Pitfalls).

Why x0 and not the noise (or an edge/score parameterization): the sampler
needs a per-position posterior over the vocabulary in **one softmax** — to
commit confident tokens, to anneal-temperature redraws (Gumbel-max over
`log p / τ`), and to re-embed the posterior as self-conditioning input. An x0
head hands all three the same object. Alternatives (edge/score
parameterization) would require trajectory marginalization the sampler has no
use for. Full derivation and the corruption-marginal math:
[diffusion-core](diffusion-core.md).

---

## 7. Chinchilla accounting

### 7.1 The parameter count, digit-for-digit

`V = 50,257`, `d = 1024`, `n_layers = 24`, GQA `16Q/4KV`, `head_dim = 64`,
`ffn_dim = 3072`, weight tying, `time_embed_dim = 256`:

| component | expression | params |
|---|---|---|
| embedding (tied head) | `50257 × 1024` | 51,463,168 |
| q_proj | `1024·1024 + 1024` | 1,049,600 |
| k_proj | `256·1024 + 256` | 262,400 |
| v_proj | `256·1024 + 256` | 262,400 |
| out_proj | `1024·1024 + 1024` | 1,049,600 |
| w13 (fused SwiGLU) | `1024·6144 + 6144` | 6,297,600 |
| w2 | `3072·1024 + 1024` | 3,146,752 |
| 2 × RMSNorm | `2 × 1024` | 2,048 |
| **per layer** | | **12,070,400** |
| 24 layers | `24 × 12,070,400` | 289,689,600 |
| time MLP | `(256·1024+1024) + (1024·1024+1024)` | 1,312,768 |
| selfcond proj | `1024·1024 + 1024` | 1,049,600 |
| final norm | `1024` | 1,024 |
| **total** | | **343,516,160** |

`tests/test_models.py::test_param_count` pins the digit-for-digit count at
production config; `training/pretrain.py:count_parameters` prints it at
startup. **343.5M, not ~380M**: the early planning docs' table counted
full-MHA K/V at 1024-wide (16 KV heads instead of GQA's 4); with the mandated
GQA + tying the honest total is 343,516,160 (SDD Ruling 13). The config
*filename* still says "380m" — both sides of that mismatch are documented, do
not "fix" one without the other.

### 7.2 The token budget

Chinchilla-optimal compute puts tokens `D ≈ 20 × params`. At 343.5M:

```
D = 8.0B tokens  →  D/N = 23.3  (slightly above the 20× rule of thumb)
```

Batch accounting: micro_bs 16 × grad_accum 2 × seq 4,096 =
131,072 tokens per optimizer step
(`configs/pretrain_a100_380m.yaml`):

```
total_steps = 8.0e9 / 131,072 ≈ 61,036  →  configured 61,000
61,000 × 131,072 = 7.9954e9 ≈ 8.0B tokens ✓
```

(The `TrainingConfig` dataclass defaults are micro 8 × accum 4 — the same
131,072 tokens/optimizer-step; the A100 yaml uses 16 × 2 instead, dropping
gradient checkpointing for MFU — §9.)

### 7.2 FLOPs and wall-clock `[INFERENCE]`

```
train FLOPs ≈ 6 · N · D = 6 · 3.4352e8 · 8.0e9 = 1.649e19
+ ~12% self-cond double-pass          ≈ 1.847e19 FLOPs
```

On an A100 80GB SXM (312 TFLOPS dense BF16) at 35–40% MFU:

```
1.847e19 / (0.35 · 312e12) ≈ 47.0 hours
```

→ the README's "~40–50 h" window. These are budget estimates `[INFERENCE]`
— no training run has happened yet; the A100 session (Task 15) owns the
measurements. MFU = model FLOPs per second ÷ accelerator peak; the +12%
self-cond overhead comes from the p=0.5 two-pass steps (pass 1 is no-grad but
still spends FLOPs).

### 7.3 Why these widths

- `d_model = 1024`, 24 layers: Gemma-class dense shape; every layer 12.07M
  params, ~35% of them in SwiGLU (`w13` 6.30M + `w2` 3.15M of 12.07M).
- GQA 16/4: 4× KV-cache saving (24 KB/token vs 96 KB, §5.3) at negligible
  quality cost at this scale.
- `head_dim = 64`: 16 heads × 64 = 1024 exactly (standard).
- `ffn_dim = 3072 = 3·d_model`: the SwiGLU convention (the gated projection
  doubles the intermediate, so the FFN GEMM is `1024 → 6144 → 1024`).
- `rope_theta = 500,000`: a large base keeps low-frequency components slow
  over the 4,096-token context.
- `vocab 50,257`: GPT-2 BPE, the shared_data portfolio default
  (sibling parity).

---

## 8. Worked example: one full forward pass at test-tiny scale

Production dims (V = 50,257, d = 1024) are unreadable by hand. The test
suite's own fixture (`tests/conftest.py:tiny_cfg`) is the smallest config that
exercises every component:

```
V=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
ffn_dim=128, canvas_len=32, time_embed_dim=32, weight_tying=True
→ 101,888 parameters (verified by instantiation; see Practice Problem 6)
```

Trace one training forward (`B = 1`, `seq = 64`, i.e. **2 canvases of 32**;
`t = (7, 13)` — per-canvas timesteps out of `T = 16`):

```
step                     shape                  note
──────────────────────   ─────────────────────  ─────────────────────────────
input_ids                (1, 64)                x0 windows: 2 clean canvases
q_sample                 (1, 64)                xt: canvas 0 kept w.p. ᾱ(8)=.50,
                                                canvas 1 kept w.p. ᾱ(5)=.78
                                                (models/diffusion.py:q_sample)
embed(xt)                (1, 64, 64)            E rows (weight-tied with head)
+ CanvasTimeEmbedding    (1, 2, 64) → broadcast sin/cos of t/T → MLP → add
backbone block 0         (1, 64, 64)            RMSNorm → GQA(4Q/2KV, hd 16)
                                                block-causal mask (1,1,64,64)
                                                RoPE at abs positions 0..63
block 1                  (1, 64, 64)            same, fresh weights
final RMSNorm            (1, 64, 64)
+ W_sc(sc)               (1, 64, 64)            zero-init ⇒ adds 0 at init
head: h @ Eᵀ             (1, 64, 256)           logits (tiny V!)
chunked_x0_ce            scalar                 vs clean x0 (1, 64)
```

Two things to notice:

1. **The loss target is `x0`, not `xt`** — every position is scored against
   the token that was *corrupted away*, at every position simultaneously
   (there is no autoregressive shift;
   `tests/test_smoke.py::test_tiny_forward_backward` runs exactly this path).
2. **One forward trains 16 corruption levels at once** in the production
   batch: each of the 16 canvases drew its own `t` (§3.3). At tiny scale the
   same mechanics hold with 2 canvases and 2 timesteps.

The sampler at eval repeats the same forward shape but with
`time_steps = SamplerConfig.n_diffusion_steps` and, per canvas, `T_eval`
denoise forwards + 1 encode forward — the arithmetic in §1.3.

---

## 9. Training dynamics and optimization

The full loop is `training/pretrain.py:Pretrainer` (deep dive:
[training.md](../training.md)); the from-scratch essentials:

**Optimizer.** AdamW, β = (0.9, 0.95), weight decay 0.1 (decoupled — decay
applies to matrices only; norms/biases are excluded via the dim < 2 split in
`training/pretrain.py:Pretrainer.__init__`). LR 3e-4 with 2,000-step linear
warmup, then cosine decay to 5% (`min_lr_ratio`). Warmup exists because at
init the loss surface is dominated by randomly-corrupted tokens; large early
steps would amplify gradient noise.

**Precision.** BF16 autocast for the forward/backward; TF32 allowed for fp32
matmuls. BF16 (8 exponent bits, like fp32) avoids fp16's overflow/underflow
cliffs on the logsumexp boundaries — the chunked-CE keeps fp32 accumulation
exactly where the softmax normalizes (`training/losses.py:_ChunkTerms`).

**Gradient flow.** clip_grad_norm at 1.0 after gradient accumulation (2
micro-steps of 16×4096 per optimizer step). No gradient checkpointing in the
production config — the VRAM-for-MFU trade (§7 and
[memory-engineering](memory-engineering.md)).

**Determinism.** Every random draw inside a training step (per-canvas `t`,
corruption mask, self-cond gate) comes from a per-micro-step generator seeded
`seed·100_003 + micro_count` (`training/pretrain.py:Pretrainer._step_rng`);
data order is a fixed permutation replayed from an offset
(`data/dataset.py:ShuffledRangeSampler`). Resume is therefore **bit-exact**
for NaN-free runs — pinned by
`tests/test_training.py::test_checkpoint_resume_determinism`.

**Stability net.** NaN guard: 5 consecutive non-finite micro-steps → rollback
to the last complete checkpoint (`utils/checkpoint.py:CheckpointManager`,
3-file atomic checkpoints; `tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints`).

**The loss curve you should expect.** At init, `CE ≈ ln(50,257) ≈ 10.82`
nats (uniform posterior — the model knows nothing). The self-cond zero-init
equivalence means step 0 is *exactly* the no-self-cond model's loss; the
conditioning is learned, not present. Loss at `t` near `T` (pure noise) starts
highest; per-canvas `t` mixing means each logged step averages all difficulty
levels. (No measured training curve exists yet — the 8.0B run is pending;
anything beyond these structural claims would be `[INFERENCE]`.)

---

## 10. What the model must learn at every corruption level

The `t`-marginal is a curriculum the model sees *every step* (per-canvas `t`):

| regime | ᾱ(t) | the sub-task |
|---|---|---|
| light correction | ≈ 1 (t=1: 0.9904) | copy input, fix the few corruptions |
| mid denoising | 0.3–0.8 (t=5–11) | local coherence: predict tokens from noisy context |
| canvas synthesis | ≈ 0 (t≥14) | unconditional generation from uniform noise |

This is why uniform-state beats `MASK`-style noise for this design: there is
no free "where are the holes" signal, so the model must learn real
distributional knowledge at every corruption level — which is exactly what
makes the sampler's commit-and-renoise loop meaningful
([diffusion-core](diffusion-core.md) derives the marginals;
[sampler](sampler.md) exploits them).

---

## 11. Practice problems (with answers)

Work these by hand; every number is computable from the config
(`configs/pretrain_a100_380m.yaml`). Answers computed, not estimated.

**P1.** Compute `ᾱ(5)` and `ᾱ(12)` for `T = 16`.
→ `cos²(π/2·5/16) = 0.7778`; `cos²(π/2·12/16) = 0.1464` (§3.3 table).

**P2.** With `V = 50,257`, what is the probability a *specific* clean token
survives corruption at `t = 8`?
→ `ᾱ(8) + (1−ᾱ(8))/V = 0.5 + 0.5/50257 = 0.50001`. The keep-prob is ᾱ plus a
`1/V` sliver from "corrupted *into* the same token".

**P3.** A canvas of 256 tokens at `t = 4` — how many tokens do you expect to
survive?
→ `ᾱ(4) = 0.8536` → `256 × 0.8536 ≈ 218.5` kept; ≈ 37.5 replaced by uniform
draws (each draw still has `1/V` chance of matching the original).

**P4.** Compute the parameter count of a half-scale sibling: `V = 50,257`,
`d = 512`, `n_layers = 12`, GQA 16Q/4KV `head_dim 64`, `ffn_dim 1536`,
weight-tied, `time_embed_dim 128`, self-cond on.
→ embed `50,257·512 = 25,731,584`; per layer: q `512·1024+1024 = 525,312`,
k+v `2·(256·512+256) = 262,656`, out `512·1024+512 = 524,800`, w13
`512·3072+3072 = 1,575,936`, w2 `1536·512+512 = 786,944`, norms `2·512 = 1,024` →
per-layer `3,676,672`; 12 layers → `44,120,064`; time MLP
`(128·512+512)+(512·512+512) = 328,704`; self-cond `512·512+512 = 262,656`;
final norm `512`. **Total 70,443,520** — verified by instantiating the model;
same structure as §7.1 (embed + layers + time + sc + final norm).

**P5.** The DESIGN §4.0 layout is micro_bs 8 × accum 4 × seq 4096. How many
optimizer steps for 8.0B tokens, and how does the A100 config (16 × 2) keep
the same count?
→ `8·4·4096 = 131,072` tokens/optimizer step → `8e9/131,072 ≈ 61,035` steps.
The A100 config's `16·2` is the same product, so `total_steps = 61,000`
serves both layouts (§7.1).

**P6.** Compute the tiny test model's parameter count (§8 config: V=256, d=64,
2 layers, 4Q/2KV hd 16, ffn 128, time_dim 32, tying, self-cond).
→ embed `256·64 = 16,384`; per layer: q `64·64+64 = 4,160`, k+v
`2·(32·64+32) = 4,160`, out `64·64+64 = 4,160`, w13 `64·256+256 = 16,640`,
w2 `128·64+64 = 8,256`, norms `2·64 = 128` → per-layer `37,504`; ×2 = `75,008`;
time MLP `(32·64+64)+(64·64+64) = 6,272`; self-cond `64·64+64 = 4,160`;
final norm `64`. **Total 101,888** — verified by instantiating the model.
(This is the model `tests/conftest.py:tiny_cfg` builds.)

**P7.** KV cache bytes for 8,192 context tokens (bf16, GQA 4KV, hd 64, 24
layers)?
→ `2 · 24 · 4 · 64 · 2 B = 24,576 B/token` → `24,576 · 8,192 ≈ 201 MB`. Full
MHA would be 804 MB (§1.2).

**P8.** (a) `tokens/forward` for `fixed_T16` at canvas 256? (b) At canvas
512? (c) How many forwards to generate 1,024 tokens with `fixed_T32`, and how
many would KV-AR spend?
→ (a) `256/17 = 15.06`. (b) `512/17 = 30.12` — doubling canvas doubles the
parallel win (and costs bidirectional attention across a longer block).
(c) `4 canvases × 33 = 132` vs AR's 1,024 (§1.3).

**P9.** What is the model's minimum possible eval entropy at `t = T` on an
*untrained* model, and what does that mean for adaptive stopping?
→ Posterior ≈ uniform over V → entropy `≈ ln(50,257) = 10.82` nats, far above
`entropy_threshold = 1.0` — so adaptive stopping never fires on random
weights (`inference/evaluate.py` rows only diverge post-training; project
AGENTS.md §5).

**P10.** Why does the dataloader not shift windows by +1 like every AR repo?
→ The loss reconstructs `x0` at *every* position from corrupted input; there
is no next-token target chain to align (`data/dataset.py:ShardWindows`, §3.2).

---

## 12. Glossary

| symbol | meaning | code |
|---|---|---|
| `ᾱ(t)` | cosine schedule, keep-probability of the clean token | `models/diffusion.py:alpha_bar` |
| `t` | per-canvas corruption timestep, `~ U{1..T}` | `models/diffusion.py:sample_canvas_t` |
| `T` | diffusion steps (train 16 / eval ≤ 32) | config `n_diffusion_steps` / `SamplerConfig.n_diffusion_steps` |
| `L` | canvas length (256) | config `canvas_len` |
| `x0` / `xt` | clean canvas / corrupted canvas | `models/diffusion.py:q_sample` |
| `x̂0` | model's posterior over clean tokens (head softmax) | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| `E` | embedding matrix, weight-tied with head (50257 × 1024) | `models/transformer.py:DiffusionGemma` |
| `sc` | self-conditioning input (posterior re-embedded) | `models/selfcond.py:SelfConditioning` |
| `V` | vocab size (50,257) | config `vocab_size` |
| GQA | grouped-query attention (16Q/4KV here) | `models/attention.py:DenoiseAttention` |
| MFU | model FLOPs per second ÷ peak | benchmarking guide (Phase 5) |

## 13. Load-bearing invariants

Every claim in this chapter is pinned by a test; the full map lives in
`.docs-authoring/PLAN.md`. The five that would most embarrass you in an
interview if broken:

| invariant | pinned by |
|---|---|
| mask semantics (causal across, bidirectional within) | `tests/test_mask.py` |
| GQA layout + untiled fast paths ≡ eager twin | `tests/test_attention.py`, `tests/test_models.py` |
| RoPE norms + relative-position identity | `tests/test_attention.py::test_rope_preserves_norms_and_relative_position` |
| chunked-CE ≡ eager CE (loss and grads, atol 1e-6) | `tests/test_loss.py::test_chunked_equals_eager` |
| param count = 343,516,160 | `tests/test_models.py::test_param_count` |

## 14. Interview Q&A

**Q: Why discrete diffusion rather than just making AR faster (speculative
decoding, MoE decode)?**
A: Those accelerate AR but keep the 1-token-per-forward *serial* structure;
block diffusion changes the structure — one forward refines 256 tokens
(`fixed_T32` ≈ 7.8 tokens/forward at production settings vs exactly 1 for
KV-AR), and adaptive stopping cuts eval FLOPs further. It is a different
point on the parallelism/quality curve, not an AR optimization.

**Q: What does "uniform-state" mean and why not mask diffusion?**
A: Corruption replaces tokens with uniformly sampled *valid vocabulary
tokens* (`models/diffusion.py:corruption_probs`) instead of an absorbing
mask. Any sequence is then a legal state at any corruption level, which is
what lets the sampler freeze confident tokens and redraw the rest
(`inference/generate.py:BlockDiffusionSampler._denoise_step`); with  the
noise state itself would leak the hole pattern.

**Q: Walk me through the KV cache arithmetic. Why 24 KB/token?**
A: `2 tensors (k,v) × 24 layers × 4 KV heads × 64 head_dim × 2 bytes (BF16) =
24,576 B/token`; GQA shares 4 KV heads across 16 query heads, so it is 4×
smaller than MHA's 96 KB/token at the same width. At 4,096 context that is a
100.7 MB cache.

**Q: Your docs say 343.5M but the config file is named 380m — which is it?**
A: 343,516,160, digit-for-digit (`tests/test_models.py::test_param_count`).
The planning table counted full-MHA K/V; the mandated GQA 16Q/4KV + weight
tying pins 343.5M. Both facts are recorded (SDD Ruling 13) — renaming the
config without the history would hide a real lesson.

**Q: Why is the loss cross-entropy against x0 and not against xt?**
A: The plan's wording is a recorded typo (Ruling 19); the target is the clean
tokens. x0-parameterization gives the sampler exactly the per-position
posterior it needs to commit and re-noise in one softmax — no trajectory
marginalization.

**Q: Where does the 12% compute overhead come from?**
A: Self-conditioning: with p = 0.5 a training step runs the model twice (the
first pass under `no_grad` to build the conditioning input
(`training/losses.py:chunked_p_embed`)). Expected FLOPs are therefore
`6·N·D · (1 + 0.5·(forward cost share)) ≈ 1.12 × 6·N·D` → 1.847e19 total
`[INFERENCE]`.

**Q: What is the hardest invariant to keep in the sampler?**
A: That KV-chained decode equals a fresh single-shot block-causal forward —
bit-exact in fp64 (`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).
Any attention/KV plumbing change re-runs it; it is the difference between a
block-AR model and a subtly inconsistent one.

---

## 15. The memory story: why chunked cross-entropy exists

> Deep dive with the full byte budget: [memory-engineering](memory-engineering.md).
> This section is the from-scratch version.

### 15.1 The hazard

The head produces logits `(B, T, V)`. At the §4.0 training layout
(micro_bs 8, seq 4,096, V = 50,257) the full fp32 logits tensor is

```
8 × 4096 × 50,257 × 4 B = 6.6 GB
```

— larger than many GPUs' *total* VRAM, held *just to compute a softmax*. The
naive step is dead on arrival; the fix is chunking over vocabulary.

### 15.2 The chunked algorithm

`training/losses.py:chunked_x0_ce` splits the vocab into chunks of
`K = 8192 · 8 / micro_bs` (`training/pretrain.py:Pretrainer.__init__` scales
K inversely with micro-batch so retained bytes stay constant):

```
for each vocab chunk c (hidden @ E[c:c+K].T):        # (B, T, K) bf16 logits
    lse_c  = logsumexp(chunk fp32 logits)            # fp32 boundary INSIDE the chunk
    tgt_c  = chunk's target logit, masked to in-chunk targets
global lse = logsumexp(all chunk lse's)               # fp32, vector over (B, T)
loss     = mean(global_lse − target_logit)            # identical to eager CE
```

The identity `CE = lse − target_logit` holds because logsumexp over chunks
concatenated equals logsumexp over the whole vocab. Equivalence to the eager
reference `models/diffusion.py:x0_ce_loss` is pinned at `atol=1e-6` for both
loss and gradients (`tests/test_loss.py::test_chunked_equals_eager`,
`test_chunked_matches_eager_grad_direction`).

`training/losses.py:_ChunkTerms` (an `autograd.Function`) retains each chunk's
**bf16** logits for backward and re-derives the softmax from them — trading
~2 bytes/element of retained bf16 chunks (~3.3 GB) for skipping a full
head-GEMM recompute per backward. The previous scheme (`torch.utils.checkpoint`)
re-ran the head GEMM every backward; the retained-bf16 chain buys that back.

The self-cond pre-pass has the same hazard (`p @ E` needs a full-vocab
softmax) — `training/losses.py:chunked_p_embed` computes it chunk-by-chunk
under `no_grad` (`models/selfcond.py:SelfConditioning.embed` is the eager
twin).

### 15.3 Why the fp32 boundaries are sacred

Softmax numerics need more dynamic range than BF16's mantissa gives: per-chunk
logsumexp and the global combination stay fp32 (`training/losses.py:chunked_x0_ce`).
Collapsing them to BF16 shifts loss values and would break the
`atol=1e-6` equivalence pin (`tests/test_loss.py`). Do not "simplify".

---

## 16. The data pipeline (from-scratch view)

The universal portfolio pipeline (`LLM/shared_data/`) downloads, cleans,
dedups, and packs the 7-source mixture into 50M-token uint32 shards; this
repo consumes it through a shim (`data/prepare_data.py:main`) that pins
`LLM_DATA_ROOT` to `data/pretrain_chinchilla` — the pack subprocess honors
only that env var, and consumers read `data/pretrain_chinchilla/shards/`
(pinned by `tests/test_data.py::test_producer_consumer_shard_path_wiring`).

Training windows (`data/dataset.py:ShardWindows`):

- **Flat windows, no +1 shift** — diffusion reconstructs `x0` at every
  position (§3.2), so a window is just `seq_len` consecutive tokens
  (`seq_len = max_seq_len = 4096` = 16 canvases).
- **No cross-document boundaries inside a window** — the pack contract;
  windows never straddle documents.
- **Deterministic resumable shuffle** — `data/dataset.py:ShuffledRangeSampler`
  fixes a permutation from (seed, n_windows) and replays from
  `offset_batches × batch_size`; offsets wrap mod n_windows so long runs
  cycle deterministically (pinned: `tests/test_data.py::test_loader_seed42_repeatability`,
  `test_loader_resumable_offset`).

Full pipeline: [data-pipeline](data-pipeline.md).

---

## 17. Evaluation: what "faster" and "as good" mean

The headline is **decode throughput at acceptable quality**, and both halves
are measured — or disclosed as unmeasured:

- **Forwards/token** (`inference/evaluate.py:SpeedupEvaluator.evaluate`):
  instruments `model.backbone` to count forwards and token-forwards for each
  schedule row (`fixed_T16`, `fixed_T32`, `adaptive_T32`) plus the analytic
  AR row (1 forward/token by construction). Token-forwards, not wall-clock,
  is the robust proxy (robust to sampler internals).
- **Quality anchor**: `inference/evaluate.py:heldout_x0_nll` — mean chunked
  x0-CE on held-out shard windows at sampled `t`, in nats/token, directly
  comparable to an AR baseline's CE on the same shard. Acceptance: within +5%
  (`scripts/loss_parity_eval.py --ar-nll <AR CE>`).
- **Honest gaps** (pending the A100 run): wall-clock tokens/s vs the
  LLaMA-3-Lite AR baseline, MFU ≥ 33%, peak VRAM vs the §4.0 budget. The
  scripts print the gap when inputs are absent — quoting an unmeasured
  number would violate the workspace's honesty contract.

---

## 18. The recipe deltas (why this differs from the upstream literature)

Full table in `DIFFUSION.md` §8. The three deltas worth internalizing:

| choice | here | common upstream | why |
|---|---|---|---|
| init | from scratch, diffusion objective only | SFT/RL on a pretrained AR teacher | from-scratch control (§4.4) |
| adaptive compute | closed entropy rule | learned/RL halting policy | no extra training phase; the posterior's own entropy is the signal |
| time granularity | per-canvas `t` | per-sequence `t` | 16 corruption levels per micro-batch row |

---

## 19. Training dynamics (advanced): determinism and the NaN guard

The foundations-level view is §9; the deep version lives in
[training.md](../training.md) (Phase 3). The two pieces every reader of this
chapter should already internalize:

1. **Resume determinism is a property of *where randomness lives*, not of
   luck.** All train-time draws flow from one per-micro-step generator
   (`training/pretrain.py:Pretrainer._step_rng`, seeded `seed·100_003 +
   micro_count`); the dataloader's permutation is a pure function of
   (seed, n_windows) replayed from an offset
   (`data/dataset.py:ShuffledRangeSampler`). Introduce one global-RNG draw
   into the step and bitwise resume dies
   (`tests/test_training.py::test_checkpoint_resume_determinism`).
2. **The NaN guard trades availability for strict determinism.** Skipped
   NaN steps advance the micro counter without an optimizer step, so resume
   bit-equality holds for NaN-free runs only (SDD ledger Minor 3) — a
   documented caveat, not a bug.

---

---

## 20. References

- D3PM — Austin et al. 2021, *Structured Denoising Diffusion Models in
  Discrete State-Spaces* (uniform-state transition matrices).
- MDLM / SEDD lineage — absorbing-state alternatives (the §3.2 contrast).
- LLaDA; Block-Diffusion (arXiv 2024/2025) — block-causal discrete diffusion
  LMs (upstream practice; §4.4 recipe delta).
- Chinchilla — Hoffmann et al. 2022 (the 20× token heuristic; §7.1).
- RoPE — Su et al. 2021; GQA — Ainslie et al. 2023.
- In-repo: [`DIFFUSION.md`](../../DIFFUSION.md) (authoritative) ·
  [diffusion-core](diffusion-core.md) · [block-causal-attention](block-causal-attention.md) ·
  [self-conditioning](self-conditioning.md) · [sampler](sampler.md) ·
  [memory-engineering](memory-engineering.md) · [data-pipeline](data-pipeline.md)
