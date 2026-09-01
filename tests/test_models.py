from dataclasses import replace

import pytest
import torch

from models.diffusion import x0_ce_loss
from models.transformer import DiffusionGemma, DiffusionGemmaConfig


def test_forward_shapes(tiny_model, device):
    x = torch.randint(0, 256, (2, 64), device=device)
    t = torch.randint(1, 5, (2, 2), device=device)
    assert tiny_model(x, t).shape == (2, 64, 256)


def test_param_count():
    # full 380M config constructed on CPU meta device; band below from the mandated
    # config's true count (GQA attention + tied head), not the plan's 355-405M which
    # assumed untied/square QKVO -- see task-7-report.md deviation 1.
    full = DiffusionGemmaConfig()
    with torch.device("meta"):
        model = DiffusionGemma(full)
    n = sum(p.numel() for p in model.parameters())
    assert 3.35e8 < n < 3.55e8, f"param count {n} outside 335-355M guard"


def test_weight_tying_shared(tiny_model):
    assert tiny_model.head.weight is tiny_model.embed.weight


def test_two_step_overfit(tiny_model, device):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 64), device=device)
    t = torch.randint(1, 5, (2, 2), device=device)
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    logits = tiny_model(x, t)
    loss0 = x0_ce_loss(logits, x)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    loss1 = x0_ce_loss(tiny_model(x, t), x)
    assert loss1 < loss0, f"loss did not decrease: {loss0:.6f} -> {loss1:.6f}"


def test_grad_flow_all_params(tiny_model, device):
    # self-cond path on: first pass detached -> sc re-embedded -> second pass backward
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 64), device=device)
    t = torch.randint(1, 5, (2, 2), device=device)
    with torch.no_grad():
        h0 = tiny_model.backbone(x, t)
        sc_input = tiny_model.selfcond.embed(h0, tiny_model.embed.weight)
    assert sc_input.grad_fn is None  # detached first pass
    loss = x0_ce_loss(tiny_model(x, t, sc_input=sc_input), x)
    loss.backward()
    missing = [n for n, p in tiny_model.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing}"


@pytest.mark.numeric
def test_eager_attn_impl_matches_sdpa(tiny_cfg, device):
    # eager branch is a ruled requirement; identical weights must give identical logits
    eager_cfg = replace(tiny_cfg, attn_impl="eager")
    sdpa = DiffusionGemma(tiny_cfg).to(device)
    eager = DiffusionGemma(eager_cfg).to(device)
    eager.load_state_dict(sdpa.state_dict())
    torch.manual_seed(1)
    x = torch.randint(0, 256, (2, 64), device=device)
    t = torch.randint(1, 5, (2, 2), device=device)
    with torch.no_grad():
        max_diff = (sdpa(x, t) - eager(x, t)).abs().max().item()
    assert max_diff < 1e-5, f"eager diverges from sdpa: max |diff| = {max_diff:.3e}"


@pytest.mark.numeric
def test_flex_attn_impl_matches_sdpa(tiny_cfg, device):
    # flex branch: BlockMask block-causal kernel must match the bool-mask sdpa path
    try:
        flex_cfg = replace(tiny_cfg, attn_impl="flex")
        flex = DiffusionGemma(flex_cfg).to(device)
    except Exception as e:  # CPU/CUDA builds without flex support
        pytest.skip(f"flex_attention unavailable: {e}")
    sdpa = DiffusionGemma(tiny_cfg).to(device)
    flex.load_state_dict(sdpa.state_dict())
    torch.manual_seed(1)
    x = torch.randint(0, 256, (2, 64), device=device)
    t = torch.randint(1, 5, (2, 2), device=device)
    with torch.no_grad():
        max_diff = (sdpa(x, t) - flex(x, t)).abs().max().item()
    assert max_diff < 1e-5, f"flex diverges from sdpa: max |diff| = {max_diff:.3e}"


def test_yaml_config_roundtrip():
    cfg = DiffusionGemmaConfig.from_yaml("configs/pretrain_a100_380m.yaml")
    assert cfg.d_model == 1024 and cfg.canvas_len == 256
    assert cfg.n_diffusion_steps == 16 and cfg.eval_diffusion_steps == 32
