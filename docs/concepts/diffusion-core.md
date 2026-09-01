# Concept: the diffusion core (uniform-state, x0-prediction, chunked loss)

> Full context: [`DIFFUSION.md`](../../DIFFUSION.md) §1 and §5. This page is
> the concept-level walkthrough; DIFFUSION.md stays authoritative.

## The forward process

Each 256-token canvas is corrupted independently at its own timestep
`t ~ U{1..T}` (train `T=16`) under the D3PM-style uniform-state process:

```
ᾱ(t) = cos²(π/2 · t/T)            # models/diffusion.py:alpha_bar
xt = x0      with prob ᾱ(t)
xt ~ U(vocab) with prob 1 − ᾱ(t)  # models/diffusion.py:q_sample
```

The forward-process row — `ᾱ` on the clean token, `(1−ᾱ)/V` elsewhere — is
`models/diffusion.py:corruption_probs`. Because the noise state is a valid
token (not `[MASK]`, not Gaussian), *any* sequence is a legal state at any
corruption level: that is what lets the sampler commit partial answers
(`DIFFUSION.md` §4.2).

Per-canvas `t` (not per-sequence) is deliberate: one micro-batch row of 16
canvases covers 16 corruption levels, reducing gradient variance across the
`t`-marginal (`models/diffusion.py:sample_canvas_t`).

## x0-parameterization

The model predicts the **clean tokens** directly; the loss is CE against `x0`
at the corrupted input `xt`:

- eager reference: `models/diffusion.py:x0_ce_loss` (test-only),
- production: `training/losses.py:chunked_x0_ce` — never materializes the
  full `(B, T, V)` logits tensor.

The softmax over the head **is** the x̂0 posterior the sampler needs to commit
and re-noise (`DIFFUSION.md` §1.2). The plan's `xt` wording is a recorded typo.

## The chunked-CE memory story (§4.0)

At micro_bs=8, seq=4096, V=50,257 the full fp32 logits tensor is ~6.6 GB.
`training/losses.py:chunked_x0_ce` computes `hidden @ E[chunk].T` for one
8192-token vocab chunk at a time; each chunk's bf16 logits are retained for
backward by the `training/losses.py:_ChunkTerms` autograd Function (the
previous checkpoint scheme re-ran the head GEMM on every backward);
per-chunk fp32 logsumexp combines into a global lse, then the target-logit
gather is masked to in-chunk targets. `Pretrainer` scales the chunk
inversely with micro-batch so retained bytes stay at this budget. Equivalence
to eager is pinned at `atol=1e-6` for loss and gradients (`tests/test_loss.py`).

The self-cond pre-pass needs `p @ E` without gradients —
`training/losses.py:chunked_p_embed` does the same chunked dance under
`no_grad` (an eager full-vocab softmax there would alone blow the budget).

Memory bounds are encoded by `utils/memory.py:estimate_model_memory_gb`
(§4.0 table; ~33 GB of retained activations with the grad-checkpointing-off
80 GB layout) and enforced pre-flight by
`utils/memory.py:assert_fits_in_available_gpu`.

## Time conditioning

`models/time_embed.py:CanvasTimeEmbedding` embeds the normalized time `t/T`
per canvas and adds it to the token embeddings. Train-time forwards use the
model's train `T`; eval forwards pass
`time_steps=SamplerConfig.n_diffusion_steps` so `t/T ∈ (0,1]` at any eval
schedule (`DIFFUSION.md` §4.3). `t=0` (prompt, finalized canvases) is
T-independent.
