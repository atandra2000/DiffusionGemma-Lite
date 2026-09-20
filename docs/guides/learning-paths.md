# Learning Paths — How to Read the DiffusionGemma-Lite Docs

> Audience: all levels. This guide is the navigational meta layer for the
> documentation tree — it teaches no model topic itself; it tells you which
> doc to read next depending on what you already know and what you want
> from the codebase. [`DIFFUSION.md`](../../DIFFUSION.md) remains the
> authoritative engineer's contract for every ruling these paths reference.

The docs are organized into three tracks: **concepts**
([concepts/](../concepts/foundations.md), theory from first principles),
**references** ([references/](../references/R1_config.md), symbol-anchored
API walkthroughs), and **guides** (this folder — operations and how-to),
plus the two pipeline docs [training.md](../training.md) and
[inference.md](../inference.md). Every citation to code uses symbol anchors
(`models/mask.py:build_block_causal_mask` style), verified by
`tests/test_doc_refs.py` and `scripts/check_docs.py --coverage`.

---

## Beginner path — What this model is and how it works

Who this is for: you know basic PyTorch and what autoregressive next-token
prediction is, and want to understand block diffusion from the ground up.
No prior discrete-diffusion background required.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [quickstart.md](quickstart.md) | How the repo is laid out, how to run data → train → eval commands on CPU, and what the canonical 343.5M numbers are. |
| 2 | [foundations.md](../concepts/foundations.md) (lineage + task) | The AR → diffusion → D3PM lineage, why uniform-state corruption pairs with x0 prediction, and the worked forward pass. |
| 3 | [foundations.md](../concepts/foundations.md) (param accounting) | Why the model pins exactly 343,516,160 parameters and where the early "~380M" number went wrong. |
| 4 | [diffusion-core.md](../concepts/diffusion-core.md) | The cosine schedule (`models/diffusion.py:alpha_bar`), per-canvas timestep sampling (`models/diffusion.py:sample_canvas_t`), and corruption (`models/diffusion.py:q_sample`). |
| 5 | [block-causal-attention.md](../concepts/block-causal-attention.md) | The causal-across-canvases / bidirectional-within mask (`models/mask.py:build_block_causal_mask`), GQA 16Q/4KV, and the three attention paths (eager / SDPA / FlexAttention). |
| 6 | [self-conditioning.md](../concepts/self-conditioning.md) | Zero-init equivalence (bit-exact at init), the detached pre-pass, and why the add happens exactly once per path (`models/selfcond.py:SelfConditioning`). |
| 7 | [R2 — Transformer API](../references/R2_transformer_api.md) + [R3 — Mask/Attention API](../references/R3_mask_attention_api.md) | The code tour: `models/transformer.py:DiffusionGemma`, `models/block.py:DenoiseBlock`, `models/attention.py:DenoiseAttention` with shape contracts. |

## Intermediate path — Train it and understand the numerics

Who this is for: you have read the beginner path and want to run, tune, or
resume training — including the chunked-CE memory argument and the data
pipeline.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [R1 — Config Schema](../references/R1_config.md) | Every model/runtime/sampler knob, its default, and its reader symbol. |
| 2 | [training.md](../training.md) | The applied pretrain loop: AdamW (3e-4, warmup 2000 → cosine to 0.05), grad checkpointing every 3 blocks, NaN guard, checkpointing. |
| 3 | [diffusion-core.md](../concepts/diffusion-core.md) (loss half) | Why the loss is cross-entropy against *clean* tokens (x0 parameterization — Ruling 19) and how `training/losses.py:chunked_x0_ce` never materializes full-vocab logits. |
| 4 | [memory-engineering.md](../concepts/memory-engineering.md) | The byte budget, the chunked-CE fp32-logsumexp argument, and the VRAM-for-MFU trade. |
| 5 | [data-pipeline.md](../concepts/data-pipeline.md) | The shard/window contract (`data/dataset.py:ShardWindows`), the no-+1-shift rationale, and the resumable shuffler (`data/dataset.py:ShuffledRangeSampler`). |
| 6 | [R5 — Loss/Training API](../references/R5_loss_training_api.md) + [R7 — Data/Checkpoint/Memory API](../references/R7_data_utils_api.md) | `training/pretrain.py:Pretrainer` internals and `utils/checkpoint.py:CheckpointManager` resume recipes. |
| 7 | [checkpoint-ops.md](checkpoint-ops.md) | Checkpoint layout, resume procedures, disk hygiene. |

## Expert path — Sample, evaluate, optimize

Who this is for: you want the sampler invariants, the eval harness, the
measurement discipline, and the A100 operational path.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [sampler.md](../concepts/sampler.md) | The decode loop (`inference/generate.py:BlockDiffusionSampler.denoise_canvas`), commit rule, temperature annealing (0.8 → 0.4), entropy bond, FLOP accounting. |
| 2 | [DIFFUSION.md](../../DIFFUSION.md) (Rulings 15–17) | Why finalized canvases re-encode under the ALL-ONES `models/mask.py:build_canvas_decode_mask`, why committed positions are never overwritten, and why eval passes `time_steps=32` while training uses 16. |
| 3 | [R6 — Sampler/Eval API](../references/R6_sampler_eval_api.md) + [R4 — Diffusion API](../references/R4_diffusion_api.md) | `inference/generate.py:BlockDiffusionSampler` (prefill → denoise → encode chaining) and `inference/evaluate.py:SpeedupEvaluator`. |
| 4 | [sampler-tuning.md](sampler-tuning.md) | Which `inference/generate.py:SamplerConfig` knobs to touch, which to leave alone. |
| 5 | [inference.md](../inference.md) + [eval-scripts.md](../references/eval-scripts.md) | The eval harness, headline-metric discipline, B1–B6 gates, PASS/DISCLOSED semantics. |
| 6 | [benchmarking.md](benchmarking.md) | How to measure speedup / VRAM / MFU honestly, without claiming unmeasured numbers. |
| 7 | [debugging-playbook.md](debugging-playbook.md) | NaN rollback, resume divergence, shape errors, anchor failures. |
| 8 | [a100-runbook.md](a100-runbook.md) | The A100 pod operational sequence, boundary checks, and resume — the path to the pending 8.0B-token run. |

---

## What each track is for

- **Concepts** build the mental model; they are self-contained and can be
  read without the code open.
- **References** are the code-keyed counterparts; keep the cited file open
  beside them.
- **Guides** assume the concepts and give procedures; [contributing.md](contributing.md)
  documents the doc contract and both gates
  (`tests/test_doc_refs.py`, `scripts/check_docs.py --coverage`).
