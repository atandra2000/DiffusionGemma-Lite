# R1 — Config reference

> Every model and runtime knob: field, type, default, production value, and
> the doc that owns its semantics. Fields verified against
> `models/transformer.py:DiffusionGemmaConfig` and
> `training/pretrain.py:TrainingConfig`.

**Reading order:** [concepts index](../README.md) → this reference.

## DiffusionGemmaConfig — `models/transformer.py:DiffusionGemmaConfig`

| field | type = default | production (`configs/pretrain_a100_380m.yaml`) | semantics |
|---|---|---|---|
| `vocab_size` | int = 50257 | 50,257 GPT-2 BPE | [foundations §7](../concepts/foundations.md) |
| `d_model` | int = 1024 | 1024 | foundations §5 |
| `n_layers` | int = 24 | 24 | foundations §5.2 |
| `n_heads` | int = 16 | 16 query heads | [block-causal-attention §2](../concepts/block-causal-attention.md) |
| `n_kv_heads` | int = 4 | 4 KV heads (GQA) | block-causal-attention §2 |
| `head_dim` | int = 64 | 64 | foundations §5.3 |
| `ffn_dim` | int = 3072 | 3072 (SwiGLU) | foundations §5.6 |
| `weight_tying` | bool = True | tied embed/head | foundations §7.1 |
| `rms_norm_eps` | float = 1e-5 | 1e-5 | foundations §5.1 |
| `init_std` | float = 0.02 | 0.02 | foundations §5 |
| `rope_theta` | float = 500000.0 | 500,000 | block-causal-attention §3 |
| `max_seq_len` | int = 4096 | 16 canvases × 256 | foundations §7 |
| `attn_impl` | str = "sdpa" | "flex" on A100 | block-causal-attention §4 |
| `canvas_len` | int = 256 | 256 | foundations §4 |
| `n_diffusion_steps` | int = 16 | 16 = train T | [diffusion-core §2](../concepts/diffusion-core.md) |
| `eval_diffusion_steps` | int = 32 | 32 (upstream uses 48) | [sampler §4](../concepts/sampler.md) |
| `corruption` | str = "uniform" | Uniform State Diffusion | diffusion-core §1 |
| `alpha_schedule` | str = "cosine" | ᾱ = cos²(π/2·t/T) | diffusion-core §2 |
| `self_conditioning` | bool = True | on | [self-conditioning §1](../concepts/self-conditioning.md) |
| `self_cond_p` | float = 0.5 | 0.5 | self-conditioning §2 |
| `self_cond_detach` | bool = True | no-grad pre-pass | self-conditioning §6 |
| `time_embed_dim` | int = 256 | 256 | foundations §5.5 |

Loader: `DiffusionGemmaConfig.from_yaml(path)` reads the `model:` block of a
config yaml (`configs/pretrain_a100_380m.yaml`); the production config's
`attn_impl: "flex"` is overridden to `"sdpa"` on CPU smoke runs
(`training/pretrain.py:Pretrainer.__init__`).

## TrainingConfig — `training/pretrain.py:TrainingConfig`

| field | type = default | shipped value | semantics |
|---|---|---|---|
| `model_config` | DiffusionGemmaConfig | from yaml | R1 (this file) |
| `data_path` | str = "data/pretrain_chinchilla/shards" | same | [data-pipeline §1](../concepts/data-pipeline.md) |
| `checkpoint_dir` | str = "checkpoints/pretrain_a100" | yaml `save_dir` | [checkpoint-ops](../guides/checkpoint-ops.md) |
| `micro_batch_size` | int = 8 | 16 | [memory-engineering §4](../concepts/memory-engineering.md) |
| `gradient_accumulation_steps` | int = 4 | 2 | training §1 |
| `total_steps` | int = 61000 | 61000 | training §6 |
| `warmup_steps` | int = 2000 | 2000 | training §3 |
| `lr` | float = 3e-4 | 3e-4 | training §3 |
| `min_lr_ratio` | float = 0.05 | 0.05 | training §3 |
| `weight_decay` | float = 0.1 | 0.1 | training §3 |
| `beta1` / `beta2` | 0.9 / 0.95 | same | training §3 |
| `grad_clip` | float = 1.0 | 1.0 | training §3 |
| `grad_checkpoint` | bool = True | **False** (shipped) | memory-engineering §4 |
| `grad_checkpoint_every` | int = 3 | 3 (inert when off) | memory-engineering §4 |
| `compile_model` | bool = True | True, `max-autotune` | training §1 |
| `save_interval` | int = 4000 | 4000 | [checkpoint-ops](../guides/checkpoint-ops.md) |
| `log_interval` | int = 50 | 50 | `utils/logging.py:TrainingLogger` |
| `nan_guard` | bool = True | True | training §4 |
| `nan_guard_max_consecutive` | int = 5 | 5 | training §4 |
| `vocab_chunk` | int = 8192 | 8192·8/micro_bs | [memory-engineering §3](../concepts/memory-engineering.md) |
| `seed` | int = 42 | 42 (house seed) | training §2 |

Note: `vocab_chunk` is pipeline-internal (not a yaml key); the yaml
equivalents live in the `training:` block (`configs/pretrain_a100_380m.yaml`).

## SamplerConfig — `inference/generate.py:SamplerConfig`

| field | default | semantics |
|---|---|---|
| `n_diffusion_steps` | 32 | eval T (≤ 32; training uses 16) — [sampler §5](../concepts/sampler.md) |
| `adaptive` | True | entropy bond on/off — sampler §4 |
| `entropy_threshold` | 1.0 | stop when mean entropy < 1.0 nat for `stability_steps` |
| `stability_steps` | 2 | consecutive low-entropy steps required |
| `temp_start` / `temp_end` | 0.8 / 0.4 | linear temperature anneal — sampler §3 |
| `seed` | None | optional generator seed for reproducible decode |

Pinned by `tests/test_sampler.py` (adaptive thresholds, Gumbel draw) and
`tests/test_inference.py::test_parse_baseline` (name → config).