# Concept: the data pipeline — shards, windows, and the resumable shuffler

> **Canonical** for the shard/window contract, the no-+1-shift rationale,
> and deterministic-resume data order. `DIFFUSION.md` §6 stays authoritative
> for rulings; this page derives the pipeline from the bytes up.

**Depends on:** [foundations](foundations.md) §17 ·
[diffusion-core](diffusion-core.md) §3 ·
**Read next:** [training](../training.md)

---

## Table of Contents

1. [The contract: shards → windows → batches](#1-the-contract)
2. [Why there is no +1 shift](#2-why-there-is-no-1-shift)
3. [The resumable shuffler](#3-the-resumable-shuffler)
4. [Worked example: indexing a window at toy scale](#4-worked-example-indexing-a-window-at-toy-scale)
5. [What breaks if you change this](#5-what-breaks-if-you-change-this)
6. [Glossary](#6-glossary)
7. [Interview Q&A](#7-interview-qa)

---

## 1. The contract

Three objects move bytes to the trainer
(`data/dataset.py:ShardWindows`, `data/dataset.py:ShuffledRangeSampler`,
`data/dataset.py:build_dataloader`):

```
shared_data pipeline ──► shard_*.bin (uint32 memmap, 50M tokens each)
        │                                  │
        └ data/prepare_data.py main()      └ ShardWindows (flat seq_len windows)
                (tokenizer: gpt2, V=50,257,          │
                 EOS=PAD=50,256)                     ShuffledRangeSampler (seed 42)
                                                     │
                                            build_dataloader → DataLoader
                                                     │  (drop_last=True)
                                              batches of B windows  ──► trainer
```

`data/dataset.py:ShardWindows` memmaps every `shard_*.bin` under
`train_data_path` (`configs/pretrain_a100_380m.yaml` data block), slices each
into `len(mm) // seq_len` windows of exactly `seq_len = 4096`, and indexes
across shards via a cumulative-start table (`bisect`). At production scale:
**160 shards × 200 MB, 12,207 windows per shard, 1,953,120 windows total,
128 tokens per shard dropped as a partial window (0.000%)** — the pack
contract (`cross_document_boundary_ok: false`) means windows never straddle
shard boundaries; a 50M-token shard simply loses its sub-4,096-token tail.

Windows cross **document** boundaries freely (documents are concatenated
inside a shard before windowing); they never cross **shard** boundaries.
Those are two different boundaries, and only one is forbidden.

`data/prepare_data.py:main` is the producer: it materializes a project-local
`data_config.yaml` pinning the GPT-2 tokenizer contract
(`vocab 50,257, EOS = PAD = 50,256`) and delegates download → clean →
tokenize → pack to the workspace `shared_data` pipeline
(`data/prepare_data.py:_require_shared_data` raises a vendor-me error when
the package is absent). The `LLM_DATA_ROOT` env var is pinned before the
pack subprocess because the subprocess re-resolves the root from the
environment (`data/prepare_data.py:main`).

## 2. Why there is no +1 shift

An AR dataset returns `(x[i:i+seq], x[i+1:i+seq+1])` — inputs and targets
offset by one, because AR predicts token `i+1` from tokens `≤ i`.

Diffusion training reconstructs **x0 at every position simultaneously**: the
corruption (`q_sample`) and the target are the *same window*. So
`data/dataset.py:ShardWindows.__getitem__` returns a flat
`seq_len`-token slice — no shift, no `+1` offset, no second tensor:

```
window w:  tokens[start : start+seq_len]      # input = target = x0
```

The +1 would waste a token per window and, worse, imply an off-by-one
coupling between positions that does not exist in the objective — the model
never uses position i's *input* to predict position i+1; every position is
independently corrupted and independently reconstructed
([diffusion-core §3](diffusion-core.md)). Pinned:
`tests/test_data.py::test_producer_consumer_shard_path_wiring`.

## 3. The resumable shuffler

`data/dataset.py:ShuffledRangeSampler` — the determinism story in one class:

```
indices = np.random.default_rng(seed).permutation(n_windows)   # fixed by (seed, n)
__iter__:  yields indices[offset:]          # resume mid-permutation
__len__:   len(indices) - offset            # windows left in this pass
offset wraps modulo n_windows              # long runs cycle deterministically
```

- **The permutation is a pure function of `(seed, n_windows)`** — seed 42
  yields the *same* order on every machine, every restart. Pinned:
  `tests/test_data.py::test_loader_seed42_repeatability`.
- **Resume = offset.** The trainer passes `offset_batches · batch_size`
  (`data/dataset.py:build_dataloader`): after a checkpoint at batch `n`, the
  loader restarts at window `offset` *without regenerating any draws* —
  the same windows in the same order as an uninterrupted run. Pinned:
  `tests/test_data.py::test_loader_resumable_offset`.
- **No cross-process state**: the sampler holds `indices` and an integer;
  two ranks with the same seed walk identical orders (data-parallel
  sharding would be a later concern; this repo trains single-process).

Combined with the training side — per-step generators seeded from the step
count ([training.md](../training.md) §resume) — the *entire* data+corruption
pipeline is a pure function of `(seed, step)`, which is what makes
`tests/test_training.py::test_checkpoint_resume_determinism` a bit-equality
test rather than a statistical one.

## 4. Worked example: indexing a window at toy scale

Toy: 2 shards of 10 tokens, `seq_len = 4` → windows per shard:
`10 // 4 = 2`, 2 tokens of each shard dropped.

```
shard0:  [t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9]   shard1: [u0 ... u9]
windows:  shard0[0:4]=w0, shard0[4:8]=w1,  shard1[0:4]=w2, shard1[4:8]=w3
_starts  = [0, 2]          (cumulative window counts)
n_windows = 4
```

`__getitem__(3)`:

```
w = 3 % 4 = 3
shard = bisect_right([0, 2], 3) - 1 = 2 - 1 = 1     # window 3 lives in shard 1
local = 3 - _starts[1] = 3 - 2 = 1                  # second window of shard 1
start = 1 · 4 = 4  → tokens[4:8] of shard 1 = [u4 u5 u6 u7]
```

Take-away: global window id → shard via `bisect` on cumulative starts, local
id → byte offset via one multiply. O(log shards) per lookup, zero copies at
open time (memmap), one `np.array(..., copy=True)` per fetch to detach from
the mmap (`data/dataset.py:ShardWindows.__getitem__`). The modulo in
`__getitem__` (`idx % n_windows`) makes out-of-range indices cycle instead of
raising — the sampler never triggers it, but a user's hand-rolled index can.

## 5. What breaks if you change this

| change | immediate effect | the test that catches it |
|---|---|---|
| add the AR +1 shift | windows misaligned by one vs the x0 target; loss measures a hybrid objective | `tests/test_data.py::test_producer_consumer_shard_path_wiring` |
| regenerate the permutation on resume | data order after resume ≠ order before; resume no longer bit-deterministic | `tests/test_data.py::test_loader_resumable_offset` |
| permute per epoch (`random.shuffle` per pass) | resume across an epoch boundary replays a different order | `tests/test_data.py::test_loader_seed42_repeatability` |
| window across shard boundaries | corrupts the no-cross-boundary pack contract | `data/dataset.py:ShardWindows` per-shard windowing |
| `drop_last=False` in `build_dataloader` | ragged final batch breaks the (B, S) shape contract | `data/dataset.py:build_dataloader` |
| pass `pin_memory=False` on CUDA | `non_blocking=True` H2D copies become synchronous | `data/dataset.py:build_dataloader` docstring |
| `offset` not taken modulo n_windows | negative/out-of-range start after wrap | `data/dataset.py:ShuffledRangeSampler` (offset % n_windows) |

## 6. Glossary

| symbol | meaning | code |
|---|---|---|
| shard | 50M-token uint32 memmap (`shard_*.bin`) | `data/prepare_data.py:main` output |
| window | flat `seq_len`-token x0 slice, no shift | `data/dataset.py:ShardWindows` |
| `_starts` | cumulative window offsets across shards | `data/dataset.py:ShardWindows.__init__` |
| `offset` | resume point: windows already consumed | `data/dataset.py:ShuffledRangeSampler` |
| `seed=42` | the fixed permutation seed | `data/dataset.py:build_dataloader` |
| `LLM_DATA_ROOT` | env var the pack subprocess reads | `data/prepare_data.py:main` |

## 7. Interview Q&A

**Q: Why no +1 shift in the dataset?**
A: AR predicts token i+1 from prefix ≤ i; uniform-state diffusion
reconstructs every position of x0 independently — input and target are the
same window (`data/dataset.py:ShardWindows`). A +1 shift would waste a token
and imply a coupling the objective doesn't have
([diffusion-core §2](diffusion-core.md)).

**Q: How does resume-with-identical-data-order work?**
A: The permutation is `default_rng(seed).permutation(n_windows)` — fixed by
(seed, n_windows). Resume passes `offset_batches · batch_size`, and the
sampler yields `indices[offset:]` — no draws regenerated
(`data/dataset.py:ShuffledRangeSampler`). Combined with per-step corruption
generators, resume is bit-exact
(`tests/test_training.py::test_checkpoint_resume_determinism`).

**Q: What does a shard contain and how big is the corpus?**
A: 50M-token uint32 memmaps (200 MB each) packed by the shared pipeline;
the 8.0B-token config → 160 shards → 1,953,120 windows of 4,096, 128 tokens
per shard dropped as tail (0.000%)
(`data/dataset.py:ShardWindows.__init__`).

**Q: Why do windows cross document boundaries but not shard boundaries?**
A: The pack contract is per-shard (`cross_document_boundary_ok: false`
governs documents *inside* a shard's text stream, not window boundaries);
windowing is per-shard by construction so a window never spans two memmaps
(`data/dataset.py:ShardWindows.__getitem__`).

**Q: What breaks first if the shuffler is made stateful?**
A: Resume determinism: a shuffle regenerated from a fresh RNG would replay a
different order after restart, and the fp64 resume-determinism test — which
compares full training trajectories — fails
(`tests/test_data.py::test_loader_resumable_offset`,
`tests/test_training.py::test_checkpoint_resume_determinism`).

**Q: Why `pin_memory=True` on CUDA?**
A: The trainer's `non_blocking=True` H2D copy is only actually asynchronous
with pinned host memory; without the flag it silently degrades to a
synchronous copy (`data/dataset.py:build_dataloader` docstring).