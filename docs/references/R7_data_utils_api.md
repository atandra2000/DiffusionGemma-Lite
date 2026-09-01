# R7 — Data, checkpoint, memory & logging API

> Public symbols of `data/dataset.py`, `data/prepare_data.py`,
> `utils/`. Pipeline semantics: [data-pipeline](../concepts/data-pipeline.md),
> [training §5](../training.md).

## `data/dataset.py`

| symbol | signature | contract |
|---|---|---|
| `ShardWindows.__init__` | `(data_dir, seq_len: int)` | memmaps every `shard_*.bin`; flat `seq_len` windows — **no +1 shift** ([data-pipeline §2](../concepts/data-pipeline.md)) |
| `ShuffledRangeSampler.__init__` | `(n_windows: int, seed: int = 42, offset: int = 0)` | permutation fixed by `(seed, n_windows)`; `offset` resumes mid-order, wraps mod n_windows — `tests/test_data.py::test_loader_seed42_repeatability`, `test_loader_resumable_offset` |
| `build_dataloader` | `(data_dir, seq_len, batch_size, seed=42, offset_batches=0, pin_memory=False)` | DataLoader with `drop_last=True`; `offset = offset_batches · batch_size`; `pin_memory` makes the trainer's `non_blocking` H2D copy actually async |

Windowing math: `len(mm) // seq_len` windows per shard (50M tokens / 4096 =
12,207; 128-token tail dropped, 0.000%); global id → shard via `bisect` on
cumulative starts. Pinned by `tests/test_data.py::test_producer_consumer_shard_path_wiring`.

## `data/prepare_data.py`

| symbol | signature | contract |
|---|---|---|
| `main` | `() -> int` | CLI wrapper over the workspace `shared_data` pipeline: materializes `data/data_config.yaml` (GPT-2 tokenizer, V=50,257, EOS=PAD=50,256), pins `LLM_DATA_ROOT` for the pack subprocess — `tests/test_data.py::test_producer_consumer_shard_path_wiring` |

Requires `shared_data/` vendored (or sibling `LLM/shared_data/`);
`data/prepare_data.py:_require_shared_data` raises a vendor-me error
otherwise.

## `utils/checkpoint.py`

| symbol | signature | contract |
|---|---|---|
| `CheckpointManager.__init__` | `(save_dir: str)` | checkpoint dir owner |
| `.save` | `(model, optimizer, step, extra_meta=None, state_dict=None) -> None` | atomic write: weights + optimizer + step + `extra_meta` (scheduler, opt_steps, tag) |
| `.load` | `(model, step, device='cuda', optimizer=None, strict=True) -> dict` | restores weights (+ optimizer when passed); returns meta dict |
| `.latest_step` | `() -> Optional[int]` | highest **complete** checkpoint; skips incomplete writes — `tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints` |

4.81 GB per checkpoint at production dims (0.69 bf16 weights + 4.12 AdamW
fp32). Ops: [checkpoint-ops](../guides/checkpoint-ops.md).

## `utils/memory.py`

| symbol | signature | contract |
|---|---|---|
| `estimate_model_memory_gb` | `(model, seq_len, batch_size, grad_checkpoint=True, vocab_chunk: int \| None = 8192, overhead_gb: float \| None = None) -> float` | peak-GB estimate: params + AdamW(12 B/param) + activations + CE; overhead `min(13.7, max(2.0, 0.17·total))` — `tests/test_utils.py::test_memory_estimator_monotone_in_batch`, `test_chunked_ce_term_bounds_memory_estimate` |
| `assert_fits_in_available_gpu` | `(estimate_gb: float, safety_margin_gb: float = 2.0) -> None` | pre-launch guard; raises when estimate exceeds available − margin; no-op on CPU |

## `utils/logging.py`

| symbol | signature | contract |
|---|---|---|
| `TrainingLogger.__init__` | `(log_every: int = 10, seq_len: int = 1024, batch_size: int = 1)` | derived tok/s denominators |
| `.log` | `(step: int, loss: float, lr: float = 0.0) -> None` | one line per `log_interval` (config: 50) |