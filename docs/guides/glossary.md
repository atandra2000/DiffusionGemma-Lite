# Glossary — DiffusionGemma-Lite

> Notation, component terms, and the config keys that appear throughout the
> documentation. Config semantics live in [R1 — Config Schema](../references/R1_config.md);
> this page is the quick lookup. Every code citation uses symbol anchors
> verified by `tests/test_doc_refs.py`.

---

## Notation

| Symbol | Meaning | Canonical value |
|--------|---------|-----------------|
| `d_model` | Residual-stream width | 1024 |
| `n_layers` | Denoiser blocks | 24 |
| `n_heads` / `n_kv_heads` | Query / KV heads (GQA) | 16 / 4 |
| `head_dim` | Per-head width | 64 |
| `ffn_dim` | FFN hidden width | 3072 |
| `vocab_size` | Tokenizer rows (GPT-2 BPE) | 50 257 |
| `rope_theta` | RoPE base frequency | 500 000.0 |
| `max_seq_len` | Training context window | 4096 |
| `canvas_len` (`L`) | Tokens per diffusion canvas | 256 |
| `n_diffusion_steps` (`T`) | Train-time corruption timesteps | 16 |
| `eval_diffusion_steps` | Eval/decode schedule length | 32 |
| `time_embed_dim` | Canvas-time embedding width | 256 |
| `rms_norm_eps` | RMSNorm epsilon | 1e-5 |
| `init_std` | Init standard deviation | 0.02 |
| `[B, T, d]` | Batch × sequence × hidden tensor shape | — |
| `N` | Parameter count | 343,516,160 (exactly) |

## Diffusion core

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Block diffusion** | Sequence generation in 256-token canvases denoised in parallel — causal across canvases, bidirectional within a canvas. | `models/mask.py:build_block_causal_mask` |
| **Uniform-state corruption** | Each position independently resamples from a uniform distribution over the vocab with probability `1 − α`; the clean token survives with probability `α`. | `models/diffusion.py:corruption_probs`, `models/diffusion.py:q_sample` |
| **Cosine schedule** | `ᾱ(t) = cos²(π/2 · ((t/T + s)/(1+s)))`-style cosine interpolation of the corruption schedule. | `models/diffusion.py:alpha_bar` |
| **Per-canvas timestep** | Each canvas draws its own `t ∈ {1..T}` per training step, not one global `t`. | `models/diffusion.py:sample_canvas_t` |
| **x0 prediction** | The network predicts the *clean* token distribution; the loss is CE against `x0`. The plan's `xt` wording was a recorded typo (Ruling 19). | `models/diffusion.py:x0_ce_loss` |
| **D3PM** | Discrete Denoising Diffusion Probabilistic Models — the uniform-state transition-matrix family this corruption scheme belongs to. | [foundations.md](../concepts/foundations.md) |

## Block-causal attention

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Block-causal mask** | Bool mask: a position attends to all earlier canvases causally and to its own canvas bidirectionally. Cached — pure function of `seq_len`/`canvas_len`. | `models/mask.py:build_block_causal_mask` |
| **GQA** | Grouped-Query Attention: 16 query heads share 4 KV heads (groups of 4). | `models/attention.py:DenoiseAttention` |
| **`attn_impl`** | Attention backend selector: `"sdpa"` (default), `"eager"`, or `"flex"`. | `models/transformer.py:DiffusionGemmaConfig` |
| **Eager ground-truth twin** | The manual masked-softmax attention path — a deliberate regression oracle kept alongside SDPA and FlexAttention; never consolidate it away. | `models/mask.py:eager_block_causal_attention` |
| **FlexAttention path** | The fused `attn_impl: "flex"` route through a compiled block mask. | `models/mask.py:flex_block_causal_attention` |
| **RoPE rotate-half** | Rotary embedding applied by rotating half the head dim; cos/sin tables precomputed in fp32 and cached. | `models/attention.py:apply_rope` |
| **RMSNorm** | Root-mean-square normalization (pre-norm, no mean subtraction). | `models/block.py:RMSNorm` |

## Self-conditioning

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Self-conditioning** | The model receives its own previous x̂0 prediction (embedded via the tied embedding table) as an extra input. | `models/selfcond.py:SelfConditioning` |
| **Zero-init equivalence** | The self-cond projection is zero-initialized, so with self-conditioning enabled the model is bit-exact (`atol=0, rtol=0`) to the model without it at init. | pinned by bitwise tests; theory in [self-conditioning.md](../concepts/self-conditioning.md) |
| **Detached pre-pass** | During training, the first x̂0 prediction is computed with `self_cond_p` = 0.5 probability and fed back detached — no second gradient path. | `models/selfcond.py:SelfConditioning.forward` |
| **Cross-step eval conditioning** | At decode time the previous step's prediction conditions the next step within a canvas. | [self-conditioning.md](../concepts/self-conditioning.md) |

## Training and precision

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Chunked x0 CE** | Production loss: computes cross-entropy against clean tokens one vocab chunk (8192) at a time with fp32 logsumexp, never materializing the ~6.6 GB full-vocab logits tensor. | `training/losses.py:chunked_x0_ce` |
| **Global-lse combination** | Per-chunk losses combine through a global logsumexp so the result is numerically identical to the unchunked loss. | `training/losses.py:_ChunkTerms` |
| **`chunked_p_embed`** | The chunked p(x0)-embedding variant used for the self-conditioning pre-pass. | `training/losses.py:chunked_p_embed` |
| **Warmup→cosine schedule** | 2000-step linear ramp to `lr` = 3e-4, cosine decay to `min_lr_ratio` = 0.05 over 61 000 steps. | `training/pretrain.py:TrainingConfig` |
| **Gradient checkpointing** | Every 3rd block re-computed in backward (`grad_checkpoint_every` = 3). | `training/pretrain.py:Pretrainer` |
| **NaN guard** | 5 consecutive NaN/Inf optimizer steps → rollback to the last complete checkpoint. | `training/pretrain.py:Pretrainer` |
| **Per-micro-step RNG** | All train-time randomness (diffusion `t` draws, corruption) comes from a dedicated generator so resume is bitwise-deterministic. | `training/pretrain.py:Pretrainer._step_rng` |
| **Atomic checkpoint** | Weights + optimizer + meta written together; a step is resumable only when all three exist. | `utils/checkpoint.py:CheckpointManager` |

## Sampler and inference

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Prefill** | Encode the prompt (possibly a partial final canvas) into the KV cache before any denoising. | `inference/generate.py:BlockDiffusionSampler.prefill` |
| **Denoise step** | One timestep of parallel within-canvas denoising: predict x̂0, sample, apply the commit rule. | `inference/generate.py:BlockDiffusionSampler._denoise_step` |
| **Commit rule** | A position whose sampled token is unchanged-or-stronger (confidence rose) becomes committed; committed positions are never overwritten — monotone. | `inference/generate.py:BlockDiffusionSampler.denoise_canvas` |
| **KV chaining** | A finalized canvas re-encodes under the ALL-ONES decode mask and its KV is appended; the next canvas attends causally to it (Rulings 15–17). | `models/mask.py:build_canvas_decode_mask`, `inference/generate.py:BlockDiffusionSampler.encode_canvas` |
| **Sampler ↔ single-shot equivalence** | The KV-chained decode must equal a fresh single-shot block-causal forward, fp64 bit-exact — pinned by `tests/test_sampler.py`. | [sampler.md](../concepts/sampler.md) |
| **Temperature annealing** | Sampling temperature ramps `temp_start` = 0.8 → `temp_end` = 0.4 across the schedule. | `inference/generate.py:SamplerConfig` |
| **Entropy bond** | Adaptive stopping: stop the schedule early when mean entropy stays below `entropy_threshold` = 1.0 for `stability_steps` = 2 consecutive steps. | `inference/generate.py:SamplerConfig` |
| **Speedup evaluator** | Counts model forwards (parallel denoise vs AR baseline `parse_baseline`) and measures wall-clock speedup. | `inference/evaluate.py:SpeedupEvaluator` |
| **Held-out NLL** | Teacher-forced x0 cross-entropy on held-out shards — the loss-parity gate. | `inference/evaluate.py:heldout_x0_nll` |

## Data pipeline

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **8.0B-token universal corpus** | Prepared once in workspace `shared_data/` (7-source mixture); 61 000 steps × 8 × 4 × 4096 tokens — Chinchilla-optimal for 343.5M. | [data-pipeline.md](../concepts/data-pipeline.md) |
| **Shim** | `data/prepare_data.py` — per-project entry that pins `LLM_DATA_ROOT` to `data/pretrain_chinchilla` and runs the shared pack stage as a subprocess. | `data/prepare_data.py` |
| **Window dataset** | mmap shard reader yielding fixed `seq_len` = 4096 windows; no +1 shift — the diffusion target is the window itself, not a shifted next token. | `data/dataset.py:ShardWindows` |
| **Resumable shuffler** | Deterministic window permutation driven by seed + offset batches, so a resumed run continues the data order exactly. | `data/dataset.py:ShuffledRangeSampler` |

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| AR | Autoregressive |
| BPE | Byte-Pair Encoding (GPT-2 tokenizer here) |
| CE | Cross-Entropy |
| D3PM | Discrete Denoising Diffusion Probabilistic Models |
| GQA | Grouped-Query Attention |
| KV cache | Key/Value cache |
| MFU | Model FLOPs Utilization |
| NLL | Negative Log-Likelihood |
| RMSNorm | Root Mean Square Normalization |
| RoPE | Rotary Position Embedding |
| SDPA | Scaled-Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`) |
| SDD | Spec-Driven Development (rulings ledger for this repo) |
| x0 | The clean (uncorrupted) token sequence |
