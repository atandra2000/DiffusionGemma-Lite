# Guide: quickstart

End-to-end: data → training → sampling → headline evaluation. Assumes the
repo root (`LLM/DiffusionGemma-Lite/`) and Python ≥ 3.10 with
`requirements.txt` installed.

## 0. Environment check

```bash
uv run --python 3.13 python -m pytest -m "not gpu and not slow"    # 77 tests, CPU-only
```

Expected: `77 passed` (plus 2 warnings). On this box bare `python3` is 3.9
and fails collection (PEP-604) — use the `uv run --python 3.13` form (see
`SKILLS.md`). Training needs CUDA (A100 80GB for the production config);
everything else runs on CPU.

A doc-integrity pass is part of the same gate:

```bash
python3 scripts/check_docs.py --coverage --links   # every public symbol cited, links resolve
python3 scripts/build_docs_html.py                 # rebuild docs_html/ portal
```

Expected: `[doc-refs] coverage: 38/38` (or a named symbol to fix), and the
portal path printed.

## 1. Data

`data/prepare_data.py:main` delegates to the workspace `shared_data`
pipeline and pins `LLM_DATA_ROOT` to `data/pretrain_chinchilla` — the pack
stage runs as a subprocess that honors only that env var
(`DIFFUSION.md` §6, SDD Ruling 26):

```bash
python data/prepare_data.py                     # full pipeline (download → tokenize → pack)
python data/prepare_data.py --skip-download     # if the corpus is already on disk
python data/prepare_data.py --skip-download --skip-clean --skip-tokenize --skip-pack
                                                # config-materialization smoke (no I/O heavy stages)
```

Output: `data/pretrain_chinchilla/shards/shard_*.bin` (50M-token uint32
shards, GPT-2 BPE, no cross-document boundaries, seed 42). Consumers read
that exact path — `configs/pretrain_a100_380m.yaml` `train_data_path` and
`training/pretrain.py:TrainingConfig.data_path`; the producer/consumer
agreement is pinned by `tests/test_data.py::test_producer_consumer_shard_path_wiring`.

**Expected output**: 160 shards × 200 MB for the 8.0B-token config
(1,953,120 windows of 4,096; 128 tokens per shard dropped as the sub-window
tail). Each stage prints a `[data/diffusiongemma] ...` line; the pack stage
reports shards written. The pipeline internals:
[data-pipeline](../concepts/data-pipeline.md).

Windows are flat `seq_len` slices — **no** +1 AR shift, diffusion
reconstructs x0 everywhere (`data/dataset.py:ShardWindows`); the resumable
shuffler is `data/dataset.py:ShuffledRangeSampler` via
`data/dataset.py:build_dataloader`.

## 2. Training

```bash
python -m training.pretrain --config configs/pretrain_a100_380m.yaml
python -m training.pretrain --config ... --dry-run        # 2-step wiring check
python -m training.pretrain --config ... --resume 4000    # explicit step
```

`training/pretrain.py:main` → `training/pretrain.py:Pretrainer`. Key
behaviors (details in `DIFFUSION.md` §6): optimizer-step counting, BF16 +
TF32, per-micro-step RNG for resume determinism, NaN guard with rollback
(`utils/checkpoint.py:CheckpointManager`), 3-file checkpoints every 4,000
steps, pre-flight VRAM estimate (`utils/memory.py:estimate_model_memory_gb`).

**Expected output** (dry run): startup lines — parameter count
`343,516,160 total`, the peak-VRAM estimate (`[memory] ... estimated peak
VRAM: 55.2 GB / 80.0 GB — OK` on A100; a warning + sdpa fallback on CPU) —
then two micro-steps of loss logs and a `final` checkpoint. Expected log
cadence: one `TrainingLogger` line every 50 optimizer steps
(`log_interval`). Full loop anatomy: [training](../training.md).

Resuming: `--resume <step>` (default: latest complete). Post-resume
bit-equality holds for NaN-free runs — see the caveat in `AGENTS.md` §5.

## 3. Generation

```python
import torch
from models.transformer import DiffusionGemma, DiffusionGemmaConfig
from inference.generate import BlockDiffusionSampler, SamplerConfig

model = DiffusionGemma(DiffusionGemmaConfig.from_yaml("configs/pretrain_a100_380m.yaml"))
prompt = torch.randint(0, 50257, (1, 64))
ids = model.generate(prompt, max_new_tokens=1024)          # one-call entry

# explicit sampler control (adaptive stop, temperature, seed):
s = BlockDiffusionSampler(model, SamplerConfig(n_diffusion_steps=32, seed=0))
ids = s.generate(prompt, max_new_tokens=1024)
```

Prompt lengths need not be canvas multiples (the prefill builds a
partial-canvas mask inline); generation is emitted in whole canvases and
trimmed.

## 4. Evaluation

```bash
python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
python scripts/speedup_eval.py --flops-only
python scripts/loss_parity_eval.py --shard data/pretrain_chinchilla/shards/shard_90000.bin \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors --ar-nll 3.21
```

Rows and their meaning are in `DIFFUSION.md` §7 (`inference/evaluate.py:SpeedupEvaluator`,
`inference/evaluate.py:heldout_x0_nll`). Without `--checkpoint` the harness
runs random-init weights (forward counts are architecture-bound; adaptive
staying inert is expected). Without `--ar-nll` the parity delta prints as an
honest gap.

**Expected output**: five rows — `fixed_T16` (≈15.06 tokens/forward),
`fixed_T32` (≈7.76×), `adaptive_T32` (≤ 33 forwards/canvas, ≥ 7.76×),
`ar_kv_analytic` (1.0, `seconds=None`), and the speedup map. With
`--ar-nll 3.21` the parity delta prints; without it the AR-NLL column is
marked as the disclosed gap. Reading rows:
[inference §5](../inference.md). Tune the sampler knobs:
[sampler-tuning](sampler-tuning.md). Full workflow discipline:
[benchmarking](benchmarking.md).
