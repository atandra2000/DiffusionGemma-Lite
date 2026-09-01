# Reference: config

Every key of `configs/pretrain_a100_380m.yaml`, loaded by
`models/transformer.py:DiffusionGemmaConfig.from_yaml` (the `model:` block)
and `training/pretrain.py:main` (`training:` + `data:` blocks). Runtime
defaults live in `training/pretrain.py:TrainingConfig`.

## `model:` — architecture (`models/transformer.py:DiffusionGemmaConfig`)

| key | value | meaning |
|---|---|---|
| `vocab_size` | 50257 | GPT-2 BPE (shared_data default, sibling-Lite parity) |
| `d_model` | 1024 | model width |
| `n_layers` | 24 | dense Denoise Blocks (`models/block.py:DenoiseBlock`) |
| `n_heads` / `n_kv_heads` | 16 / 4 | GQA query/KV heads (`models/attention.py:DenoiseAttention`) |
| `head_dim` | 64 | = d_model / n_heads |
| `ffn_dim` | 3072 | SwiGLU intermediate (3× d_model) |
| `weight_tying` | true | embed ↔ head share storage (part of the 343,516,160 count) |
| `rms_norm_eps` | 1.0e-5 | `models/block.py:RMSNorm` |
| `init_std` | 0.02 | N(0, σ) init; `selfcond.proj` re-zeroed after |
| `rope_theta` | 500000 | RoPE base (`models/attention.py:apply_rope`) |
| `max_seq_len` | 4096 | = 16 canvases |
| `attn_impl` | "sdpa" | production path; "eager" is the ground-truth twin |
| `canvas_len` | 256 | the block-AR block size |
| `n_diffusion_steps` | 16 | train-time T (per-canvas `t ~ U{1..T}`) |
| `eval_diffusion_steps` | 32 | eval-time max T (`eval T ≤ 32`) |
| `corruption` | "uniform" | D3PM-style uniform-state (the distinctive choice) |
| `alpha_schedule` | "cosine" | `ᾱ(t) = cos²(π/2 · t/T)` (`models/diffusion.py:alpha_bar`) |
| `self_conditioning` | true | `models/selfcond.py:SelfConditioning` |
| `self_cond_p` | 0.5 | fraction of training steps with sc input |
| `self_cond_detach` | true | pass 1 under no_grad, detached into pass 2 |
| `time_embed_dim` | 256 | `models/time_embed.py:CanvasTimeEmbedding` width |

## `training:` (consumed by `training/pretrain.py:Pretrainer`)

| key | value | meaning |
|---|---|---|
| `micro_batch_size` × `gradient_accumulation_steps` | 8 × 4 | effective batch 32 × 4096 |
| `total_steps` | 61000 | **optimizer** steps ≈ 8.0B tokens |
| `warmup_steps` | 2000 | linear LR warmup |
| `lr` / `min_lr_ratio` | 3.0e-4 / 0.05 | cosine decay to 5% of lr |
| `weight_decay`, `beta1`, `beta2` | 0.1, 0.9, 0.95 | AdamW |
| `grad_clip` | 1.0 | global-norm clip |
| `grad_checkpoint` / `grad_checkpoint_every` | true / 3 | boundary-only activations (DESIGN §4.0) |
| `compile` / `compile_mode` | true / max-autotune | per-block in-place, CUDA only |
| `save_interval` / `log_interval` | 4000 / 50 | optimizer-step cadence (`utils/logging.py:TrainingLogger`) |
| `nan_guard` / `nan_guard_max_consecutive` | true / 5 | rollback threshold |
| `save_dir` | checkpoints/pretrain_a100 | `utils/checkpoint.py:CheckpointManager` root |
| `vocab_chunk` | 8192 | pipeline-internal constant (not a yaml key) |

## `data:`

| key | value | meaning |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla/shards | must equal the producer's `<LLM_DATA_ROOT>/shards` (pinned by test) |
| `tokenizer` | gpt2 | 50,257 vocab |
| `shard_size_tokens` | 50,000,000 | uint32 shards |
| `max_tokens` | 8,000,000,000 | Chinchilla-optimal for ~343.5M params |
| `data_mix` | diffusiongemma-default | **currently unused** — shared_data has one universal mixture (SDD ledger Minor 2; doc note) |

CLI overrides: `training/pretrain.py:main` accepts `--config`, `--data-path`,
`--checkpoint-dir`, `--resume`, `--no-checkpoint`, `--no-compile`, `--dry-run`.