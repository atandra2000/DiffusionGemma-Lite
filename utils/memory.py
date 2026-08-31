"""Conservative peak-VRAM estimates for DiffusionGemma-Lite pre-training.

The estimator encodes the DESIGN §4.0 A100 budget table: fp32 AdamW master
state, boundary activations under every-3 check-pointing, and the CE term that
motivates ``training/losses.py:chunked_x0_ce`` — a naive full-vocab fp32 CE at
micro_bs=8/seq=4096 costs ~6.6 GB, chunked ~1.1 GB. The self-conditioning
pre-pass runs under ``no_grad`` and is therefore free of retained memory.
"""
import torch
import torch.nn as nn


def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    vocab_chunk: int | None = 8192,
    overhead_gb: float | None = None,
) -> float:
    """Estimate peak GB from parameters, optimizer state, boundary activations, and CE."""
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return 0.0

    # AdamW keeps two moments plus an FP32 master copy.
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optim_bytes = sum(p.numel() for p in model.parameters()) * 12  # 4 (m) + 4 (v) + 4 (master)

    n_layers = len(model.blocks)
    dtype_bytes = 2  # BF16 activations

    if grad_checkpoint:
        # DESIGN §4.0: with every-3rd-layer check-pointing the retained state is the
        # block boundaries (~1.6 GB at ms=8/seq=4096) — SwiGLU intermediates recompute.
        act_bytes = n_layers * seq_len * batch_size * cfg.d_model * dtype_bytes
    else:
        act_bytes = n_layers * seq_len * batch_size * cfg.d_model * dtype_bytes
        if cfg.ffn_dim > 0:
            act_bytes += n_layers * 3 * seq_len * batch_size * cfg.ffn_dim * dtype_bytes

    # CE chain is fp32: full-vocab logits (naive) or one vocab chunk (chunked path).
    vocab = cfg.vocab_size if vocab_chunk is None else min(vocab_chunk, cfg.vocab_size)
    ce_bytes = batch_size * seq_len * vocab * 4

    # Reserve allocator and framework overhead not represented above.
    if overhead_gb is None:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            overhead_gb = min(13.7, max(2.0, total_gb * 0.17))
        else:
            overhead_gb = 2.0

    return (param_bytes + optim_bytes + act_bytes + ce_bytes) / 1024**3 + overhead_gb


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None:
    """Raise when the estimate leaves less than ``safety_margin_gb`` available."""
    if not torch.cuda.is_available():
        return
    try:
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return
    if estimate_gb > available - safety_margin_gb:
        raise RuntimeError(
            f"Estimated peak VRAM ({estimate_gb:.1f} GB) exceeds available GPU memory "
            f"({available:.1f} GB, {safety_margin_gb:.1f} GB margin)."
        )
    print(f"[memory] DiffusionGemma-Lite estimated peak VRAM: {estimate_gb:.1f} GB / {available:.1f} GB — OK.")