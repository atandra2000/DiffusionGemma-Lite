# Guide: checkpoint operations

Anatomy, layout, and safe handling of checkpoints. Loop context:
[training §5](../training.md); API: [R7](../references/R7_data_utils_api.md).

## Layout

`utils/checkpoint.py:CheckpointManager` writes one directory per step under
`save_dir` (config: `checkpoints/pretrain_a100`), every
`save_interval = 4000` steps plus a `final` tag — 16 checkpoints per full
run (steps 4000…60000 + final at 61,000), **4.81 GB each** (0.69 GB bf16
weights + 4.12 GB AdamW fp32 m/v/master):

```
checkpoints/pretrain_a100/
  model_step_4000.safetensors     # weights (safetensors)
  optim_step_4000.pt              # AdamW state (m, v, fp32 master)
  meta_step_4000.json             # step, scheduler state, opt_steps, tag
```

`load` restores model + optimizer + scheduler (`extra_meta`) and returns the
meta dict; `latest_step` returns the highest **complete** checkpoint and
skips incomplete writes
(`tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints`).

## Recipes

```bash
# resume latest (auto):
python -m training.pretrain --config configs/pretrain_a100_380m.yaml
# explicit step:
python -m training.pretrain --config ... --resume 4000
# wiring check without touching checkpoints:
python -m training.pretrain --config ... --dry-run
```

Resume determinism contract: weights + optimizer + scheduler + step counters
restore; the data loader restarts at
`offset = opt_steps × grad_accum` windows; every micro-step re-seeds its own
generator (`seed·100003 + step`) — pinned fp64 by
`tests/test_training.py::test_checkpoint_resume_determinism`. Caveat: the
NaN-guard's skipped micro-steps advance the count without an optimizer step,
so bit-equality holds for **NaN-free runs** (SDD ledger Minor 3).

## Disk hygiene

- 16 checkpoints × 4.81 GB ≈ 77 GB per full run — keep the last N interval
  checkpoints + `final`, prune older ones *only after* `heldout_x0_nll`
  confirms the successor.
- Never edit a checkpoint in place; re-save via
  `training/pretrain.py:Pretrainer.save_checkpoint` (atomic write + meta).
- The NaN guard restores `latest_step()` — an incomplete write silently
  changes what "latest" means; never kill a run mid-write
  (`tests/test_utils.py::test_latest_step_skips_incomplete_checkpoints`).