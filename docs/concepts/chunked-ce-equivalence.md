# Concept: chunked-CE ≡ eager CE — the equivalence proof sketch

> **Audience: expert.** Why `training/losses.py:chunked_x0_ce` computes the
> same cross-entropy as the eager reference, stated as a proof sketch with
> the fp32 argument, the backward identity, and the tests that pin it. The
> memory *motivation* is covered in [diffusion-core](diffusion-core.md) §8 and
> [memory-engineering](memory-engineering.md) §2; this page proves the
> numerics. `AGENTS.md` §3 turns one line of it into a hard rule: keep the
> fp32 logsumexp boundaries, never simplify to BF16 lse.

**Depends on:** [block-diffusion](block-diffusion.md) §4 ·
**Read next:** [memory-engineering](memory-engineering.md)

---

## 1. Claim

For any chunking `vocab_chunk`, `training/losses.py:chunked_x0_ce(hidden, E,
targets, vocab_chunk)` equals the eager reference
`models/diffusion.py:x0_ce_loss(hidden @ E.t(), targets)` — mean
cross-entropy of x0-prediction logits against the clean tokens — up to
floating-point reduction order. Not bit-exact; pinned at `atol=1e-6` by
`tests/test_loss.py::test_chunked_equals_eager`.

## 2. The forward identity

Per position, CE decomposes into a denominator and a numerator:

```
# verified — the exact decomposition chunked_x0_ce implements
CE = log Σ_v e^{z_v}  −  z_target        # lse over the full vocab, minus target logit
  = log Σ_c e^{lse_c} −  z_target        # grouping the same sum by vocab chunk
```

1. **Per chunk**, `training/losses.py:_ChunkTerms.forward` computes the chunk
   logits `hidden @ E[c0:c1].t()` (bf16 under autocast), casts them
   `.float()`, and returns `lse_c = logsumexp` over the chunk plus the
   gathered target logit `z_target − c0` for positions whose target falls in
   this chunk.
2. **Combine**: the global denominator is
   `logsumexp(stack(lse_parts), dim=-1)` — exact in ℝ because
   `e^{lse_c} = Σ_{v ∈ c} e^{z_v}`, so the outer logsumexp re-adds exactly the
   same exponentials, regrouped. In fp32 this regrouping (plus kernel tiling
   differences between a full-GEMM and chunked GEMMs over `d_model`) is the
   entire deviation from the eager path — reduction order only, no
   approximation term. That is why the pins are `atol=1e-6`/`1e-5` rather
   than bit equality, and why the boundary stays in fp32: computing `lse_c`
   on bf16 inputs would inject a real (avoidable) error, hence the
   `AGENTS.md` §3 rule.
3. **Target logit, exactly once**: every target lands in exactly one chunk
   (`c0 ≤ target < c1` by construction of the `range(0, V, step)` loop). The
   caller zeroes the out-of-chunk gathers
   (`torch.where(in_chunk, tgt, zeros)`) and sums the parts, so
   `target_logit` is the true target logit selected once. The partial last
   chunk (`c1 = min(c0 + step, V)`) has no off-by-one —
   `tests/test_loss.py::test_partial_last_chunk` pins `V = 100`, chunk 64,
   36-token tail, plus the degenerate `vocab_chunk=None` (one chunk).
4. `(lse − target_logit).mean()` is the same mean-reduction
   `F.cross_entropy` applies in `models/diffusion.py:x0_ce_loss`.

## 3. The backward identity

`_ChunkTerms` saves each chunk's **bf16 logits** and derives the softmax from
them in backward: `g_logits = p · grad_lse` with
`scatter_add_(target, grad_tgt)` layered on top — the standard CE gradient
`∂CE/∂z_v = p_v − 𝟙[v = target]` factorized so the `grad_lse` part is shared
and the target part is per-position. `grad_tgt` arrives zeroed for
out-of-chunk targets (the same `in_chunk` mask as forward), so the clamped
scatter only ever adds zeros there. Gradients to `hidden` and `E` are then
the transposed chunk GEMMs. This is the exact CE gradient —
`tests/test_loss.py::test_chunked_matches_eager_grad_direction` pins both
gradients against the eager path at `atol=1e-5`, and
`test_chunked_loss_differentiable` pins finiteness. Retaining the bf16
logits (~3.9 GB at §4.0 scale, per the `training/losses.py:chunked_x0_ce`
docstring) is the price of reusing them instead of recomputing the head GEMM
in backward.

## 4. The companion equivalence

The self-cond pre-pass needs `p̂ @ E` without full-vocab logits;
`training/losses.py:chunked_p_embed` computes softmax weights via per-chunk
fp32 logsumexps combined into one global `lse`, then accumulates
`exp(logits − lse) @ E[c]` over chunks — the same regrouping argument as §2,
applied to a matrix product instead of a scalar. It is pinned to the eager
`models/selfcond.py:SelfConditioning.embed` at `atol=1e-5` by
`tests/test_loss.py::test_chunked_p_embed_matches_eager`, and runs under
`torch.no_grad()` in `training/pretrain.py:Pretrainer.diffusion_loss`
(see [self-conditioning-mechanism](self-conditioning-mechanism.md) §3).

## 5. What breaks if you weaken this

- bf16 chunk logsumexp → a real numerator/denominator error, silent drift vs
  the eager oracle; the parity harness (`inference/evaluate.py:heldout_x0_nll`)
  and `tests/test_loss.py` both stop being valid equivalence checks.
- Recomputing (checkpointing) instead of retaining logits → identical math
  but a full extra head-GEMM forward per step — the trade
  `DIFFUSION.md` §5 records as deliberately settled.
- Naive full logits `(8, 4096, 50257)` fp32 → ~6.6 GB of chain memory
  (`AGENTS.md` §2 rule 2); the chunked contract exists precisely to forbid
  that materialization.
