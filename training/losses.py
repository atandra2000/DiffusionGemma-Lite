"""Training-pipeline memory optimizations: chunked x0-CE and the self-cond pre-pass.

Both follow the DESIGN §4.0 contract — the full ``(B, T, V)`` logits tensor is
never materialized. Vocab-chunked logits live one slice at a time (each chunk
recomputed under ``torch.utils.checkpoint`` so backward re-materializes instead
of retaining), with fp32 normalization. The eager references live in
``models/diffusion.py:x0_ce_loss`` and ``models/selfcond.py:SelfConditioning.embed``.
"""
import torch
from torch.utils.checkpoint import checkpoint

from models.diffusion import x0_ce_loss  # noqa: F401  (eager reference two-path pair)


def _chunk_terms(hidden, weight, local_targets):
    """One chunk's fp32 logsumexp and target logits (out-of-chunk targets gather-clamped)."""
    logits = (hidden @ weight.t()).float()
    local = local_targets.clamp(0, weight.size(0) - 1).unsqueeze(-1)
    return torch.logsumexp(logits, dim=-1), torch.gather(logits, -1, local).squeeze(-1)


def chunked_x0_ce(hidden, embed_weight, targets, vocab_chunk: int = 8192):
    """x0 cross-entropy one vocab chunk at a time; never materializes (B, T, V).

    Chunk logsumexps combine into the global denominator, and each chunk runs
    inside ``checkpoint`` so only one chunk's fp32 chain is ever alive — the
    ~1.1 GB budget of DESIGN §4.0 (the eager reference is ``x0_ce_loss``)."""
    V = embed_weight.size(0)
    step = V if vocab_chunk is None else min(int(vocab_chunk), V)

    lse_parts, tgt_parts = [], []
    for c0 in range(0, V, step):
        c1 = min(c0 + step, V)
        lse, tgt = checkpoint(_chunk_terms,
                              hidden, embed_weight[c0:c1], targets - c0, use_reentrant=False)
        lse_parts.append(lse)
        in_chunk = (targets >= c0) & (targets < c1)
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