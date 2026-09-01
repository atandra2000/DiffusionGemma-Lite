# Guide: benchmarking with honest numbers

How to produce speedup/VRAM/MFU numbers that survive scrutiny — this
portfolio's discipline: measured vs analytic, disclosed gaps, no invented
baselines. Metric definitions: [inference §2](../inference.md).

## The three claim classes

| claim type | how to produce it | where it lives |
|---|---|---|
| analytic (checkpoint-free) | token-forwards/forward counts — exact integers | `inference/evaluate.py:SpeedupEvaluator` rows |
| measured wall-clock | this sampler's `tokens_per_sec` on named hardware | row `seconds`/`tokens_per_sec` |
| analytic AR baseline | 1 forward/token by construction | `ar_kv_analytic` row, `seconds=None` — the disclosed gap |

Rule: never print a wall-clock AR number without an external checkpoint
run; the harness encodes this as `seconds=None`
(`inference/evaluate.py:SpeedupEvaluator.evaluate`).

## Speedup protocol

```bash
python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
```

Expected at production dims: `fixed_T16` ≈ 15.06 tokens/forward, `fixed_T32`
≈ 7.76×, `adaptive_T32` ≥ 7.76× (entropy bond cuts forwards), AR row = 1.0
analytic. The forwards are counted by instrumenting `model.backbone`
(`inference/evaluate.py:SpeedupEvaluator._count_forwards`) — exact integers,
not timers. Report protocol:

1. Name the GPU, precision, and `attn_impl` (flex vs sdpa changes kernels).
2. Report `tokens_per_forward` (architecture-bound) **and** wall-clock
   separately — never merge them into one "speedup".
3. `token_forwards_per_token` is the honest cost column (17.0 at T=16) —
   block-AR re-forwards committed positions; state both numbers.

## VRAM numbers

Compute, don't guess: `utils/memory.py:estimate_model_memory_gb` and the
printed `[memory] ... estimated peak VRAM` line at startup; the estimator's
accounting is [memory-engineering](../concepts/memory-engineering.md).
Quote both layouts (8×4+ckpt 23.6 GB / 16×2 55.2 GB) with the layout named.

## MFU claims

FLOPs are 6·N·D + ~12% self-cond = 1.8457e19 for the full run
([training §6](../training.md)); MFU = measured throughput × 6·N·D per step
vs 312 TFLOPS dense-BF16 A100 peak. Tag any utilization figure `[INFERENCE]`
unless a profiler run produced it. Never quote a wall-clock AR comparison
that wasn't run.

## Loss parity

`python scripts/loss_parity_eval.py --shard <heldout> --checkpoint <ckpt>
--ar-nll <x>` — the x0-NLL vs AR-CE delta is the quality anchor
(`inference/evaluate.py:heldout_x0_nll`); without `--ar-nll` it prints as
the disclosed gap. Never train on the parity shard.