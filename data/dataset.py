"""Training dataset over packed uint32 shards: flat x0 windows, no next-token shift.

Diffusion training reconstructs x0 at every position, so windows are flat
``seq_len``-token slices (no +1 AR shift). Shards are the ``shard_*.bin``
uint32 memmaps written by ``shared_data`` (via ``data/prepare_data.py``) and
are windowed independently, honoring the no-cross-boundary pack contract.
"""
import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class ShardWindows(Dataset):
    """Windows of exactly ``seq_len`` tokens from ``shard_*.bin`` uint32 memmaps."""

    def __init__(self, data_dir, seq_len: int):
        data_dir = Path(data_dir)
        paths = sorted(data_dir.glob("shard_*.bin"))
        if not paths:
            raise FileNotFoundError(
                f"No shard_*.bin under {data_dir} — run `python data/prepare_data.py` first")
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        self.seq_len = seq_len
        self.shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        self.shard_windows = [len(mm) // seq_len for mm in self.shards]
        self._starts = [0]
        for n in self.shard_windows[:-1]:
            self._starts.append(self._starts[-1] + n)
        self.n_windows = sum(self.shard_windows)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        w = int(idx) % self.n_windows
        shard = bisect.bisect_right(self._starts, w) - 1
        local = w - self._starts[shard]
        start = local * self.seq_len
        chunk = np.array(self.shards[shard][start:start + self.seq_len], copy=True)
        return torch.from_numpy(chunk).long()


class ShuffledRangeSampler(Sampler):
    """Deterministic, resumable window shuffler (house ``shared_data.loader`` contract).

    The permutation is fixed by (seed, n_windows); ``offset`` restarts mid-order
    after a checkpoint resume without regenerating any draws. Offsets wrap
    modulo n_windows, so long runs cycle the permutation deterministically.
    """

    def __init__(self, n_windows: int, seed: int = 42, offset: int = 0):
        if n_windows <= 0:
            raise ValueError(f"no complete windows available (n_windows={n_windows})")
        self.n_windows = int(n_windows)
        self.offset = int(offset) % self.n_windows
        self.indices = np.random.default_rng(seed).permutation(n_windows)

    def __iter__(self):
        for i in range(self.offset, len(self.indices)):
            yield int(self.indices[i])

    def __len__(self):
        return len(self.indices) - self.offset


def build_dataloader(data_dir, seq_len, batch_size, seed=42, offset_batches=0):
    """Deterministic shuffled loader over shard windows; offset resumes the order."""
    ds = ShardWindows(data_dir, seq_len)
    sampler = ShuffledRangeSampler(len(ds), seed=seed, offset=offset_batches * batch_size)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True)


__all__ = ["ShardWindows", "ShuffledRangeSampler", "build_dataloader"]