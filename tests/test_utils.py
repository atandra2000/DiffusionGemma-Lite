"""House utils: checkpoint round-trip/resume and the peak-VRAM estimator."""
import torch

from models.transformer import DiffusionGemma, DiffusionGemmaConfig
from utils.checkpoint import CheckpointManager
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb


def test_checkpoint_roundtrip_resume_equivalence(tiny_model, tmp_ckpt_dir, device):
    """save -> latest_step() -> load: weights, optimizer state, and meta are restored bit-exact."""
    torch.manual_seed(3)
    twin = DiffusionGemma(tiny_model.cfg).to(device)
    assert not torch.equal(next(tiny_model.parameters()), next(twin.parameters()))

    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    (tiny_model.embed.weight * 0 + 1).sum().backward()  # populate optimizer state
    opt.step()

    manager = CheckpointManager(tmp_ckpt_dir)
    manager.save(tiny_model, opt, step=7, extra_meta={"tag": "roundtrip"})
    assert manager.latest_step() == 7

    twin_opt = torch.optim.AdamW(twin.parameters(), lr=1e-3)
    meta = manager.load(twin, step=7, device=device, optimizer=twin_opt)
    assert meta["step"] == 7 and meta["tag"] == "roundtrip"
    for (name, p), (_, q) in zip(tiny_model.named_parameters(), twin.named_parameters()):
        assert torch.equal(p, q), f"weight mismatch at {name}"
    for a, b in zip(opt.state.values(), twin_opt.state.values()):
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(a[key], b[key])


def test_latest_step_skips_incomplete_checkpoints(tiny_model, tmp_ckpt_dir):
    """A step missing any of its three files is not resumable."""
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_ckpt_dir)
    manager.save(tiny_model, opt, step=3)
    manager.save(tiny_model, opt, step=9)
    (tmp_ckpt_dir / "meta_step_9.json").unlink()
    assert manager.latest_step() == 3


def test_memory_estimator_monotone_in_batch(tiny_model):
    est1 = estimate_model_memory_gb(tiny_model, seq_len=128, batch_size=1)
    est2 = estimate_model_memory_gb(tiny_model, seq_len=128, batch_size=2)
    assert est1 > 0 and est2 > est1


def test_chunked_ce_term_bounds_memory_estimate():
    """Full-vocab fp32 CE vs 8192-chunked: the estimator encodes DESIGN §4.0's table."""
    with torch.device("meta"):
        m380 = DiffusionGemma(DiffusionGemmaConfig())
    naive = estimate_model_memory_gb(m380, seq_len=4096, batch_size=8, vocab_chunk=None)
    chunked = estimate_model_memory_gb(m380, seq_len=4096, batch_size=8, vocab_chunk=8192)
    assert naive > chunked
    # naive 6.6 GB fp32 vs chunked ~4.4 GB (all bf16 chunk logits retained for
    # backward + one transient fp32 chunk): ~2.2 GB of CE chain avoided by chunking.
    assert naive - chunked > 2.0

def test_gpu_guard_noops_without_cuda():
    if torch.cuda.is_available():
        return
    assert assert_fits_in_available_gpu(0.5) is None
