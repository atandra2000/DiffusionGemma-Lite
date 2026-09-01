"""Headline evaluation harness: schedule rows, FLOP counter, held-out NLL."""
import numpy as np
import torch

from inference.evaluate import SpeedupEvaluator, heldout_x0_nll, parse_baseline
from inference.generate import SamplerConfig


def test_parse_baseline():
    fixed = parse_baseline("fixed_T16")
    assert isinstance(fixed, SamplerConfig)
    assert fixed.n_diffusion_steps == 16 and fixed.adaptive is False
    adaptive = parse_baseline("adaptive_T32")
    assert adaptive.n_diffusion_steps == 32 and adaptive.adaptive is True


def test_evaluator_produces_three_rows(tiny_model, device):
    """DESIGN §4.2: fixed_T16 / fixed_T32 / adaptive_T32 rows + analytic AR row."""
    res = SpeedupEvaluator(tiny_model).evaluate(
        n_samples=1, prompt_tokens=32, gen_tokens=64,
        baselines=("fixed_T16", "fixed_T32", "adaptive_T32"))
    rows = res["rows"]
    assert set(rows) == {"fixed_T16", "fixed_T32", "adaptive_T32", "ar_kv_analytic"}
    # fixed schedules spend exactly 1 prefill + canvases*(T denoise steps + 1 encode)
    for name, T in (("fixed_T16", 16), ("fixed_T32", 32)):
        assert rows[name]["forwards"] == 1 + 2 * (T + 1)
        assert rows[name]["tokens_per_forward"] == 64 / (1 + 2 * (T + 1))
        # speedup entry mirrors the row (AR analytic row = 1 token/forward)
        assert res["speedup_vs_ar_tokens_per_forward"][name] == rows[name]["tokens_per_forward"]
    ar = rows["ar_kv_analytic"]
    assert ar["forwards"] == 64 and ar["token_forwards_per_token"] == 1.0
    assert ar["tokens_per_sec"] is None  # no wall-clock without an external AR run


def test_flop_counter_adaptive_le_fixed(tiny_model):
    """Adaptive stopping never spends more forwards than the fixed schedule."""
    res = SpeedupEvaluator(tiny_model).evaluate(
        n_samples=1, prompt_tokens=32, gen_tokens=64,
        baselines=("fixed_T32", "adaptive_T32"), wall_clock=False)
    a, f = res["rows"]["adaptive_T32"], res["rows"]["fixed_T32"]
    assert a["forwards"] <= f["forwards"]
    assert a["tokens_per_forward"] >= f["tokens_per_forward"]


def test_heldout_x0_nll_finite(tiny_model, tmp_path):
    """Quality anchor produces a positive finite NLL over synthetic shard windows."""
    rng = np.random.default_rng(0)
    shard = tmp_path / "shard_90000.bin"
    rng.integers(0, 256, size=2 * 128, dtype=np.uint32).tofile(shard)
    nll = heldout_x0_nll(tiny_model, shard, n_windows=2, seed=42)
    assert np.isfinite(nll) and nll > 0


def test_heldout_x0_nll_short_shard_raises(tiny_model, tmp_path):
    shard = tmp_path / "tiny.bin"
    np.zeros(10, dtype=np.uint32).tofile(shard)
    try:
        heldout_x0_nll(tiny_model, shard)
        raised = False
    except ValueError:
        raised = True
    assert raised
