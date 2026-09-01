"""Data layer: shard-window dataset contract and the data_config house pins."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data.dataset import ShardWindows, build_dataloader

_SEQ = 64


def _synthetic_shards(data_dir, n_shards=2, tokens_per_shard=320, seed=42):
    """Two uint32 shards of GPT-2-range tokens: 2 x 320 tokens -> 5 windows each."""
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        toks = rng.integers(0, 50257, size=tokens_per_shard, dtype=np.uint32)
        toks.tofile(data_dir / f"shard_{i:05d}.bin")
    return data_dir


def _expected_window(data_dir, window):
    flat = np.concatenate([np.fromfile(p, dtype=np.uint32)
                           for p in sorted(data_dir.glob("shard_*.bin"))])
    return torch.from_numpy(flat[window * _SEQ:(window + 1) * _SEQ].copy()).long()


def test_shard_windows_shape_dtype_and_range(tmp_data_dir):
    _synthetic_shards(tmp_data_dir)
    ds = ShardWindows(tmp_data_dir, seq_len=_SEQ)
    assert len(ds) == 10
    sample = ds[3]
    assert sample.shape == (_SEQ,) and sample.dtype == torch.int64
    assert int(sample.max()) < 50257


def test_shard_windows_content_matches_memmap(tmp_data_dir):
    _synthetic_shards(tmp_data_dir)
    ds = ShardWindows(tmp_data_dir, seq_len=_SEQ)
    assert torch.equal(ds[7], _expected_window(tmp_data_dir, 7))


def test_partial_tail_window_dropped(tmp_data_dir):
    _synthetic_shards(tmp_data_dir, tokens_per_shard=321)  # one token dead in each shard
    ds = ShardWindows(tmp_data_dir, seq_len=_SEQ)
    assert len(ds) == 2 * (321 // _SEQ)


def test_missing_shards_raises(tmp_ckpt_dir):
    with pytest.raises(FileNotFoundError):
        ShardWindows(tmp_ckpt_dir, seq_len=_SEQ)


def test_loader_batch_shapes(tmp_data_dir):
    _synthetic_shards(tmp_data_dir)
    loader = build_dataloader(tmp_data_dir, seq_len=_SEQ, batch_size=2, seed=42)
    batch = next(iter(loader))
    assert batch.shape == (2, _SEQ) and batch.dtype == torch.long


def test_loader_seed42_repeatability(tmp_data_dir):
    """Same seed -> identical batch order across independent constructions."""
    _synthetic_shards(tmp_data_dir)
    first = [b.clone() for b in build_dataloader(tmp_data_dir, _SEQ, 2, seed=42)]
    second = [b.clone() for b in build_dataloader(tmp_data_dir, _SEQ, 2, seed=42)]
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_loader_resumable_offset(tmp_data_dir):
    """offset skips leading windows deterministically — the resume primitive."""
    _synthetic_shards(tmp_data_dir, tokens_per_shard=1280)
    batch5 = next(iter(build_dataloader(tmp_data_dir, _SEQ, 2, seed=42, offset_batches=5)))
    it = iter(build_dataloader(tmp_data_dir, _SEQ, 2, seed=42))
    for _ in range(5):
        next(it)  # skip batches the resumed run would have consumed
    expected = next(it)
    assert torch.equal(batch5, expected)


def test_producer_consumer_shard_path_wiring(tmp_data_dir):
    """The contract the A100 launch depends on: prepare_data writes shards where
    the training config reads them (review Important-1 — pack subprocess honors
    only $LLM_DATA_ROOT, and packs to <DATA_ROOT>/shards/)."""
    from data.prepare_data import DEFAULT_DATA_ROOT

    assert DEFAULT_DATA_ROOT == Path(__file__).resolve().parents[1] / "data" / "pretrain_chinchilla"
    # consumer side: config + TrainingConfig default point at <root>/shards
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "pretrain_a100_380m.yaml").read_text())
    assert cfg["data"]["train_data_path"] == "data/pretrain_chinchilla/shards"
    # the produced layout is consumable: pack writes <data_root>/shards/shard_*.bin
    _synthetic_shards(tmp_data_dir / "root" / "shards")
    assert len(ShardWindows(tmp_data_dir / "root" / "shards", seq_len=_SEQ)) > 0


def test_data_config_contract():
    """data/data_config.yaml pins the house shard contract (DESIGN §3 data block)."""
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "data" / "data_config.yaml").read_text())
    tok = cfg["pipeline"]["tokenizer"]
    assert tok["name"] == "gpt2" and tok["vocab_size"] == 50257
    sharding = cfg["pipeline"]["sharding"]
    assert cfg["pipeline"]["pack"]["cross_document_boundary_ok"] is False
    assert cfg["pipeline"]["seed"] == 42
    assert sharding["shard_size_tokens"] == 50_000_000 and sharding["dtype"] == "uint32"
    assert sharding["target_total_tokens"] == 8_000_000_000
