"""Training-pipeline memory optimizations: chunked x0-CE and the self-cond pre-pass.

Both follow the DESIGN §4.0 contract — the full ``(B, T, V)`` logits tensor is
never materialized. Vocab-chunked logits live one slice at a time, with fp32
normalization. Each chunk's bf16 logits are retained for backward (a custom
autograd Function derives the softmax from them), so the head GEMM runs once
instead of being checkpoint-recomputed — the eager references live in
``models/diffusion.py:x0_ce_loss`` and ``models/selfcond.py:SelfConditioning.embed``.
"""
import torch

from models.diffusion import x0_ce_loss  # noqa: F401  (eager reference two-path pair)


class _ChunkTerms(torch.autograd.Function):
    """One vocab chunk's fp32 logsumexp and target logit, saving bf16 logits for backward.

    The chunk GEMM runs once; backward computes the softmax from the saved bf16
    logits — bit-identical to what the previous ``checkpoint`` path re-derived by
    re-running the same bf16 GEMM. The trade: ~2 bytes/elem/every-chunk retained
    (~3.9 GB at §4.0 scale) buys back a full head-GEMM forward per step."""

    @staticmethod
    def forward(ctx, hidden, weight, local_targets):
        logits = hidden @ weight.to(hidden.dtype).t()      # bf16 under autocast
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        local = local_targets.clamp(0, weight.size(0) - 1).unsqueeze(-1)
        tgt = torch.gather(logits_f, -1, local).squeeze(-1)
        ctx.save_for_backward(hidden, weight, logits, local)
        return lse, tgt

    @staticmethod
    def backward(ctx, grad_lse, grad_tgt):
        hidden, weight, logits, local = ctx.saved_tensors
        p = logits.float().softmax(dim=-1)
        g_logits = p * grad_lse.unsqueeze(-1)
        # d(target_logit)/dlogits = onehot(target); grad_tgt is already signed
        # (dLoss/dtgt) and zeroed for out-of-chunk targets by the caller's
        # in-chunk mask, so the clamped scatter only ever adds zeros there.
        g_logits.scatter_add_(-1, local, grad_tgt.unsqueeze(-1))
        h_dtype = hidden.dtype
        g_hidden = g_logits.to(h_dtype) @ weight.to(h_dtype)
        B, T, C = g_logits.shape
        g_weight = (g_logits.reshape(-1, C).t() @ hidden.reshape(-1, hidden.size(-1))).to(weight.dtype)
        return g_hidden, g_weight, None


def chunked_x0_ce(hidden, embed_weight, targets, vocab_chunk: int = 8192):
    """x0 cross-entropy one vocab chunk at a time; never materializes (B, T, V).

    Chunk logsumexps combine into the global denominator. Each chunk's bf16
    logits stay alive for backward (~3.9 GB total at §4.0 scale, micro_bs 8 /
    seq 4096) in exchange for skipping the checkpointed head-GEMM recompute —
    the eager reference is ``x0_ce_loss``."""
    V = embed_weight.size(0)
    step = V if vocab_chunk is None else min(int(vocab_chunk), V)

    lse_parts, tgt_parts = [], []
    for c0 in range(0, V, step):
        c1 = min(c0 + step, V)
        lse, tgt = _ChunkTerms.apply(hidden, embed_weight[c0:c1], targets - c0)
        in_chunk = (targets >= c0) & (targets < c1)
        lse_parts.append(lse)
        tgt_parts.append(torch.where(in_chunk, tgt, torch.zeros_like(tgt)))
    lse = torch.logsumexp(torch.stack(lse_parts, dim=-1), dim=-1)
    target_logit = sum(tgt_parts)
    return (lse - target_logit).mean()


def chunked_p_embed(h_norm, embed_weight, vocab_chunk: int = 8192):
    """p @ E without full-vocab logits: softmax weights via per-chunk logsumexp.

    Training pre-pass runs this under ``torch.no_grad()`` (detached sc_input);
    under grad mode it would retain every chunk's logits — do not use as a
    differentiable path (the eager reference is ``selfcond.SelfConditioning.embed``).
    """
    V = embed_weight.size(0)
    step = V if vocab_chunk is None else min(vocab_chunk, V)
    lses = [torch.logsumexp((h_norm @ embed_weight[c0:c0 + step].t()).float(), dim=-1)
            for c0 in range(0, V, step)]
    lse = torch.logsumexp(torch.stack(lses, dim=-1), dim=-1).unsqueeze(-1)
    p_embed = torch.zeros(h_norm.shape[0], h_norm.shape[1], embed_weight.size(1),
                          device=h_norm.device, dtype=embed_weight.dtype)
    for c0 in range(0, V, step):
        logits = (h_norm @ embed_weight[c0:c0 + step].t()).float() - lse
        p_embed = p_embed + logits.exp() @ embed_weight[c0:c0 + step].float()
    return p_embed