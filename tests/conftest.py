import pytest
import torch

from models.transformer import DiffusionGemmaConfig, DiffusionGemma


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def tiny_cfg():
    return DiffusionGemmaConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=16, ffn_dim=128, weight_tying=True, rms_norm_eps=1e-5,
        init_std=0.02, rope_theta=10000.0, max_seq_len=128, attn_impl="sdpa",
        canvas_len=32, n_diffusion_steps=16, eval_diffusion_steps=32,
        corruption="uniform", alpha_schedule="cosine", self_conditioning=True,
        self_cond_p=0.5, self_cond_detach=True, time_embed_dim=32)


@pytest.fixture
def tiny_model(tiny_cfg, device):
    torch.manual_seed(0)
    return DiffusionGemma(tiny_cfg).to(device)
