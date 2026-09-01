# Reference: public API

The module-level public surface. The coverage gate
(`scripts/check_docs.py --coverage`) requires every symbol below to stay
cited somewhere in the doc corpus.

## Models

- `models/mask.py:build_block_causal_mask` — the block-causal mask
  (causal across canvases, bidirectional within).
- `models/mask.py:build_canvas_decode_mask` — all-ones decode view for the
  in-flight canvas (SDD Ruling 15).
- `models/mask.py:block_causal_sdpa_attention` — production SDPA path.
- `models/mask.py:eager_block_causal_attention` — O(T²) ground-truth twin.
- `models/diffusion.py:alpha_bar` — cosine schedule `ᾱ(t) = cos²(π/2·t/T)`.
- `models/diffusion.py:corruption_probs` — one forward-process row.
- `models/diffusion.py:sample_canvas_t` — per-canvas `t ~ U{1..T}`.
- `models/diffusion.py:q_sample` — single-step corruption `xt | x0, t`.
- `models/diffusion.py:x0_ce_loss` — eager CE reference (test oracle).
- `models/attention.py:apply_rope` — canonical rotate-half RoPE.
- `models/attention.py:DenoiseAttention` — GQA 16Q/4KV attention over the
  block mask, optional KV return for chaining.
- `models/selfcond.py:SelfConditioning` — zero-init W_sc conditioning module.
- `models/time_embed.py:CanvasTimeEmbedding` — per-canvas t/T embedding.
- `models/block.py:RMSNorm`, `models/block.py:DenoiseBlock` — pre-norm
  residual block (attention + SwiGLU).
- `models/transformer.py:DiffusionGemma` — the model: `backbone`,
  `head_forward` (loss head path), `final_hidden` (chunked-CE input),
  `forward`, `generate` (sampler delegation).
- `models/transformer.py:DiffusionGemmaConfig` — dataclass config;
  `DiffusionGemmaConfig.from_yaml` loads `configs/*.yaml`.

## Training

- `training/losses.py:chunked_x0_ce` — production loss (§4.0 memory story;
  targets x0).
- `training/losses.py:chunked_p_embed` — no-grad `p @ E` for the self-cond
  pre-pass.
- `training/pretrain.py:TrainingConfig` — runtime settings dataclass.
- `training/pretrain.py:Pretrainer` — setup, diffusion step, checkpointing,
  NaN rollback; `Pretrainer._step_rng` is the resume-determinism anchor.
- `training/pretrain.py:main` — CLI entry (`--config`, `--resume`, ...).
- `training/pretrain.py:count_parameters` — total/trainable parameter counter (startup log).

## Data

- `data/prepare_data.py:main` — shared_data delegation shim; pins
  `LLM_DATA_ROOT` (`data/prepare_data.py:DEFAULT_DATA_ROOT`).
- `data/dataset.py:ShardWindows` — flat (seq_len,) uint32 windows, no +1 AR
  shift.
- `data/dataset.py:ShuffledRangeSampler` — deterministic, resumable shuffle.
- `data/dataset.py:build_dataloader` — loader over shard windows.

## Inference

- `inference/generate.py:SamplerConfig` — schedule/temperature/entropy knobs.
- `inference/generate.py:BlockDiffusionSampler` — prefill /
  `BlockDiffusionSampler.denoise_canvas` / `BlockDiffusionSampler.encode_canvas`
  / `BlockDiffusionSampler.generate`; `BlockDiffusionSampler._denoise_step`
  implements the commit rule.
- `inference/evaluate.py:SpeedupEvaluator` — headline harness;
  `SpeedupEvaluator.evaluate` returns the schedule rows + analytic AR row.
- `inference/evaluate.py:parse_baseline` — `"fixed_T16"` → `SamplerConfig`.
- `inference/evaluate.py:heldout_x0_nll` — held-out chunked x0-CE (nats/token).

## Utils

- `utils/checkpoint.py:CheckpointManager` — 3-file checkpoints;
  `CheckpointManager.latest_step` only returns complete steps.
- `utils/logging.py:TrainingLogger` — step/loss/ppl/lr/tps lines.
- `utils/memory.py:estimate_model_memory_gb` — DESIGN §4.0 estimator.
- `utils/memory.py:assert_fits_in_available_gpu` — pre-flight VRAM gate.
