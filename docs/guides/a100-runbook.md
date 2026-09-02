# Guide: A100 pod runbook

The operational order for the remaining pod work — data → boundary checks →
the 40–50 h pretrain → the headline evals. Commands live in the
[quickstart](quickstart.md) and the [eval-scripts reference](../references/eval-scripts.md);
this page is about execution sequence, resume semantics, and what to monitor.
The CPU test suite (`python3 -m pytest -m "not gpu and not slow"`) must pass
before anything is uploaded.

## 0. What ships to the pod

- The repo (`LLM/DiffusionGemma-Lite/`) with `requirements.txt` installed.
- The packed binary shards: `data/pretrain_chinchilla/shards/` (uint32, 50M tokens
  each, 8.0B tokens total) — either built on the pod with `python3 data/prepare_data.py`
  or copied from an existing storage volume.
- The LLaMA-3-Lite baseline run details (for the loss parity anchor).

Nothing else is needed: the pretraining loop is single-device, single-process;
no distributed overhead, no external services.

## 1. Boundary checks, in this order

Run these pre-flight gates before launching the 40–50 h pretraining run:

```bash
# 1. Peak VRAM boundary: must fit comfortably in 80 GB HBM
python3 scripts/microbench_a100.py                  # Gate: peak VRAM < 55 GB
python3 scripts/microbench_a100.py --checkpointed   # Gate: peak VRAM < 15 GB

# 2. Step time & MFU: ensure kernel efficiency
python3 scripts/step_time_a100.py --compile         # Gate: MFU ≥ 33% (target 35-40%)

# 3. End-to-end loop & recovery smoke
python3 scripts/e2e_gpu_smoke.py --steps 20         # 20 steps, loss descent, ckpt/resume, gen demo
```

**Why this order:**
- VRAM fails fast (seconds) and is the cheapest thing to verify — if memory explodes,
  do not proceed to timing.
- MFU is the throughput gate: at $<33\%$, the 40–50 h completion window is invalidated.
  Profile the [`training/losses.py:chunked_x0_ce`](../../training/losses.py) backward and
  [`models/mask.py:flex_block_causal_attention`](../../models/mask.py) compilation before tuning hyperparameters.
- The e2e smoke is last because it runs the real loop with checkpointing, resume, and sampling.

## 2. Launching and monitoring the run

Use the provided launch script:

```bash
nohup bash scripts/launch_a100.sh > pretrain.log 2>&1 &
tail -f pretrain.log
```

Or run inside a persistent `tmux` session. The launcher runs [`training/pretrain.py:Pretrainer`](../../training/pretrain.py)
on [`configs/pretrain_a100_380m.yaml`](../../configs/pretrain_a100_380m.yaml) with automatic resume.

### Resume semantics

When a spot pod preempts or restarts:
- [`utils/checkpoint.py:CheckpointManager`](../../utils/checkpoint.py) writes three files per step:
  - `model_step_XXXX.safetensors` (weights)
  - `optimizer_step_XXXX.pt` (optimizer state & moments)
  - `meta_step_XXXX.json` (metadata, scheduler, step counter)
- A step is resumable only when **all three files exist**. If a crash happens mid-save,
  the incomplete step is automatically ignored and the previous safe checkpoint is loaded.
- Re-running `bash scripts/launch_a100.sh` automatically resumes from `latest_step()` without losing progress.

### What to watch in `pretrain.log`

- **Loss descent:** Expected loss should trend steadily downward from ~10.8 (uniform random noise over 50,257 vocab)
  toward ~3.0–3.5.
- **Throughput:** Steady ~15,000–25,000 tokens/second on an A100 SXM 80GB.
- **NaN guard:** Look for `[nan-guard]` in logs. An isolated NaN will skip an accumulation step safely.
  5 consecutive NaNs trigger an automatic rollback to the last valid checkpoint.

## 3. Post-run headline evaluation

Once training completes at step 61,000:

```bash
# 1. Headline speedup evaluation vs AR baseline
python3 scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors

# 2. Loss parity evaluation on held-out test shard
python3 scripts/loss_parity_eval.py --shard data/pretrain_chinchilla/shards/shard_00000.bin \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors \
    --ar-nll <AR_BASELINE_NLL>
```
