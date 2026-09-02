# Reference: eval + boundary scripts

Every headline number and boundary check comes from one of the scripts below,
built on the same harness ([`inference/evaluate.py:SpeedupEvaluator`](../../inference/evaluate.py))
and the same honesty contract ([`AGENTS.md`](../../AGENTS.md)): a measured number is printed with an
explicit `[PASS …]` or `[DISCLOSED …]` marker — never a bare claim; a missing
checkpoint, baseline, tokenizer, or shard is disclosed with the reason, never
silently dropped; a measured miss is disclosed next to its gate. **The scripts
exit 0 in every reported case** — a disclosed miss is a result, not a crash.

Every script has a CPU self-check form (`--tiny` or CPU fallback) and an A100 headline form;
run the self-check before spending pod time on the headline.

## The gates at a glance

| gate | script | measures | criterion | gate evaluated at |
|---|---|---|---|---|
| B1 | `scripts/speedup_eval.py` | decode throughput vs AR baseline | ≥ 8× and ≥ 14 tok/fwd | production canvas L=256 |
| B2 | `scripts/speedup_eval.py` | adaptive FLOP ratio | ≤ 0.6× forwards of fixed T=32 | `--flops-only` |
| B3 | `scripts/loss_parity_eval.py` | held-out ΔNLL vs AR baseline | ≤ +5%, else disclose | `--shard` (held-out) |
| B4 | `scripts/microbench_a100.py` | peak VRAM per layout | prod < 55 GB, ckpt < 15 GB | production shapes |
| B5 | `scripts/step_time_a100.py` | optimizer-step MFU | ≥ 33% (A100 80GB BF16) | production step shape |
| B6 | `scripts/e2e_gpu_smoke.py` | multi-step loop + resume + gen | loss descent, quiet guards | 20 steps + resume |

## `scripts/speedup_eval.py` — B1 + B2

Measures forward passes and tokens generated per forward pass across fixed schedules
(`fixed_T16`, `fixed_T32`) and the entropy-bounded adaptive schedule (`adaptive_T32`).
Compares against the analytic AR baseline (`ar_kv_analytic`, 1 token/forward by construction).

| flag | default | meaning |
|---|---|---|
| `--config` | `configs/pretrain_a100_380m.yaml` | model & sampler configuration |
| `--checkpoint` | none | trained weights (`.safetensors`); omitted ⇒ uses random init |
| `--flops-only` | off | counts forward passes without timing wall-clock |
| `--n-samples` | 8 | number of generation samples to average over |
| `--gen-tokens` | 1024 | total generation tokens (4 canvases of 256) |

```bash
python scripts/speedup_eval.py --flops-only                                # analytic FLOP check
python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors   # A100 wall-clock
```

## `scripts/loss_parity_eval.py` — B3

Computes held-out $x_0$ negative log-likelihood ([`inference/evaluate.py:heldout_x0_nll`](../../inference/evaluate.py))
over held-out shard windows. Directly comparable to an AR baseline's cross-entropy on the
same tokens.

| flag | default | meaning |
|---|---|---|
| `--shard` | required | path to binary token shard (e.g. `shard_90000.bin`) |
| `--config` | `configs/pretrain_a100_380m.yaml` | model config |
| `--checkpoint` | none | checkpoint file |
| `--ar-nll` | none | AR baseline NLL in nats/token; omitted ⇒ prints disclosed gap |

```bash
python scripts/loss_parity_eval.py --shard data/pretrain_chinchilla/shards/shard_00000.bin \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors --ar-nll 3.25
```

## `scripts/microbench_a100.py` — B4

Runs one steady-state optimizer step at the production shape (`micro_bs=16, seq=4096, accum=2`)
with fused AdamW and BF16 autocast. A warmup step runs first to materialize AdamW moments and
fp32 master weights before resetting peak CUDA allocation stats.

| flag | default | meaning |
|---|---|---|
| `--config` | `configs/pretrain_a100_380m.yaml` | configuration path |
| `--checkpointed` | off | tests 8×4 + grad-checkpoint layout (gate < 15.0 GB) |
| `--tiny` | off | runs on CPU with 2-layer config for quick local self-check |

```bash
python scripts/microbench_a100.py --tiny         # CPU self-check
python scripts/microbench_a100.py                # A100 production layout (< 55 GB)
python scripts/microbench_a100.py --checkpointed # A100 checkpointed layout (< 15 GB)
```

## `scripts/step_time_a100.py` — B5

Times `grad_accum` micro-batches + gradient clipping + fused AdamW update.
Reports seconds per step, tokens per second, and Model FLOPs Utilization (MFU)
against the 312 TFLOPS A100 dense-BF16 peak.

| flag | default | meaning |
|---|---|---|
| `--config` | `configs/pretrain_a100_380m.yaml` | configuration path |
| `--steps` | 20 | timed optimizer steps |
| `--warmup` | 5 | untimed warmup steps |
| `--compile` | off | compiles residual blocks with `torch.compile` matching production |
| `--tiny` | off | CPU proxy self-check |

```bash
python scripts/step_time_a100.py --tiny          # CPU proxy check
python scripts/step_time_a100.py --compile       # A100 timed MFU run
```

## `scripts/e2e_gpu_smoke.py` — B6

End-to-end smoke test driving [`training/pretrain.py:Pretrainer`](../../training/pretrain.py):
1. Synthesizes binary shards in a temporary directory.
2. Runs optimizer steps, asserting loss decrease (`losses[-1] < losses[0]`).
3. Verifies checkpoint writing via [`utils/checkpoint.py:CheckpointManager`](../../utils/checkpoint.py).
4. Tests resume integrity, continuing seamlessly from the checkpoint.
5. Executes a 1-canvas test generation via [`inference/generate.py:BlockDiffusionSampler`](../../inference/generate.py).
6. Verifies stability guards remain quiet.

| flag | default | meaning |
|---|---|---|
| `--config` | `configs/pretrain_a100_380m.yaml` | configuration path |
| `--steps` | 20 | number of smoke optimizer steps |
| `--tiny` | off | runs in seconds on CPU with 2-layer model |
| `--workdir` | temp dir | custom directory for shards and checkpoints |

```bash
python scripts/e2e_gpu_smoke.py --tiny           # CPU self-check
python scripts/e2e_gpu_smoke.py                  # A100 full-config smoke
```
