"""CPU-only tiny forward/backward smoke — the production two-pass path, no fixtures."""
import torch

from models.diffusion import q_sample, sample_canvas_t
from models.transformer import DiffusionGemma, DiffusionGemmaConfig
from training.losses import chunked_p_embed, chunked_x0_ce


def test_tiny_forward_backward():
    """Corrupt -> no-grad pre-pass -> denoise -> chunked x0-CE: finite loss, full grads."""
    cfg = DiffusionGemmaConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16,
        ffn_dim=128, max_seq_len=64, canvas_len=32, n_diffusion_steps=4, time_embed_dim=32,
        self_conditioning=True)
    model = DiffusionGemma(cfg)
    x0 = torch.randint(0, cfg.vocab_size, (2, 64))
    t = sample_canvas_t(batch=2, n_canvases=2, T=cfg.n_diffusion_steps, device="cpu")
    xt, _ = q_sample(x0, t, canvas_len=cfg.canvas_len,
                     n_diffusion_steps=cfg.n_diffusion_steps, vocab_size=cfg.vocab_size)

    with torch.no_grad():  # self-cond pre-pass, detached
        sc_input = chunked_p_embed(model.backbone(xt, t), model.embed.weight, vocab_chunk=128)
    loss = chunked_x0_ce(model.final_hidden(xt, t, sc_input), model.embed.weight, x0,
                         vocab_chunk=128)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
