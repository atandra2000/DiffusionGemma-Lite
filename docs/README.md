# Docs — DiffusionGemma-Lite

Every code citation in this tree uses `file.py:Symbol` anchors
(e.g. `models/mask.py:build_block_causal_mask`); `scripts/check_docs.py`
fails on any anchor that stops resolving, and `--coverage` fails if a public
symbol in `models/`, `training/`, `data/`, `inference/`, `utils/` is never
cited. Line-number citations are rejected — they rot. The measured state of
this tree — verification runs, findings, acceptance criteria — is recorded in
[`AUDIT.md`](AUDIT.md).

## Visual systems atlas

Explore the [Interactive Visual Systems Guide](diagrams/diffusiongemma_visual_guide.html): five verified Archify showcase maps ([System Overview](diagrams/diffusiongemma-system.html), [Forward Pass & Block Attention](diagrams/diffusiongemma-forward-pass.html), [Data Pipeline](diagrams/diffusiongemma-data-pipeline.html), [Discrete Sampler](diagrams/diffusiongemma-sampler.html), [Training Pipeline](diagrams/diffusiongemma-training.html)), interactive D3PM cosine noise simulator, block-causal mask inspector, and [verification receipts](diagrams/RECEIPTS.md).

## Corpus size

Measured 2026-09-21 (`wc -w`, md files only; excludes `diagrams/` and
`docs_html/`):

| track | files | words |
|---|---:|---:|
| concepts/ | 11 | 41,366 |
| references/ | 10 | 3,983 |
| guides/ | 9 | 5,214 |
| training.md + inference.md | 2 | 7,477 |
| README.md + AUDIT.md | 2 | 2,226 |
| `../DIFFUSION.md` | 1 | 3,237 |
| **total** | **35** | **63,503** |

## Reading order

| # | doc | lines | read when |
|---|---|---|---|
| 1 | [`../DIFFUSION.md`](../DIFFUSION.md) | 464 | **start here** — the authoritative engineer's contract (all math + semantics + rulings) |
| 2 | [`concepts/block-diffusion.md`](concepts/block-diffusion.md) | 99 | beginner — the uniform-state formulation in one page: corruption process, what "uniform state" means, why canvases, x0-parameterization |
| 3 | [`concepts/foundations.md`](concepts/foundations.md) | 1,074 | the from-scratch textbook chapter: AR → diffusion → D3PM → every primitive derived, worked forward pass, param accounting |
| 4 | [`concepts/diffusion-core.md`](concepts/diffusion-core.md) | 719 | forward process, cosine schedule, x0-parameterization, chunked-CE |
| 5 | [`concepts/block-causal-attention.md`](concepts/block-causal-attention.md) | 720 | mask derivation, GQA + RoPE at production dims, three attention paths, KV chaining |
| 6 | [`concepts/canvas-denoising.md`](concepts/canvas-denoising.md) | 92 | how one 256-token canvas is generated in parallel: per-step semantics, commit rule, decode mask, re-encode |
| 7 | [`concepts/self-conditioning-mechanism.md`](concepts/self-conditioning-mechanism.md) | 98 | the self-conditioning mechanism in one page: zero-init, when the input is fed, train/test symmetry |
| 8 | [`concepts/self-conditioning.md`](concepts/self-conditioning.md) | 525 | zero-init equivalence, detached pre-pass, cross-step eval conditioning |
| 9 | [`concepts/sampler.md`](concepts/sampler.md) | 711 | decode loop, commit rule, temperature annealing, entropy bond, FLOP accounting |
| 10 | [`concepts/memory-engineering.md`](concepts/memory-engineering.md) | 644 | the byte budget, chunked-CE memory argument, VRAM-for-MFU trade |
| 11 | [`concepts/chunked-ce-equivalence.md`](concepts/chunked-ce-equivalence.md) | 100 | expert — the proof that `training/losses.py:chunked_x0_ce` equals the eager CE, and the tests that pin it |
| 12 | [`concepts/data-pipeline.md`](concepts/data-pipeline.md) | 482 | shard/window contract, no-+1-shift rationale, resumable shuffler |
| 13 | [`training.md`](training.md) | 523 | the training loop, determinism recipe, NaN guard, checkpointing, A100 run |
| 14 | [`inference.md`](inference.md) | 469 | eval harness, headline metric, honest gaps |
| 15 | [`references/R1_config.md`](references/R1_config.md) | 82 | every model/runtime/sampler knob |
| 16 | [`references/R2_transformer_api.md`](references/R2_transformer_api.md) | 54 | `models/transformer.py` API |
| 17 | [`references/R3_mask_attention_api.md`](references/R3_mask_attention_api.md) | 51 | mask + attention + block + time-embed APIs |
| 18 | [`references/R4_diffusion_api.md`](references/R4_diffusion_api.md) | 39 | diffusion + self-conditioning APIs |
| 19 | [`references/R5_loss_training_api.md`](references/R5_loss_training_api.md) | 35 | chunked loss + trainer API |
| 20 | [`references/R6_sampler_eval_api.md`](references/R6_sampler_eval_api.md) | 41 | sampler + eval APIs |
| 21 | [`references/R7_data_utils_api.md`](references/R7_data_utils_api.md) | 52 | data + checkpoint + memory + logging APIs |
| 22 | [`guides/quickstart.md`](guides/quickstart.md) | 126 | data → train → eval commands with expected outputs |
| 23 | [`guides/learning-paths.md`](guides/learning-paths.md) | 82 | three audience-routed step tables (beginner / intermediate / expert) through the whole corpus |
| 24 | [`guides/glossary.md`](guides/glossary.md) | 115 | notation table + per-component term tables — the single cross-cutting lookup |
| 25 | [`guides/debugging-playbook.md`](guides/debugging-playbook.md) | 106 | NaN rollback, resume divergence, shape errors, anchor failures |
| 26 | [`guides/sampler-tuning.md`](guides/sampler-tuning.md) | 46 | SamplerConfig knobs: what to touch, what not to |
| 27 | [`guides/benchmarking.md`](guides/benchmarking.md) | 57 | honest speedup/VRAM/MFU discipline |
| 28 | [`guides/checkpoint-ops.md`](guides/checkpoint-ops.md) | 53 | checkpoint layout, resume recipes, disk hygiene |
| 29 | [`guides/contributing.md`](guides/contributing.md) | 58 | the doc contract + gates for new code |
| 30 | [`guides/a100-runbook.md`](guides/a100-runbook.md) | 89 | A100 pod operational sequence, boundary checks, resume |
| 31 | [`references/eval-scripts.md`](references/eval-scripts.md) | 120 | B1–B6 gates, flags, CPU vs A100 forms, PASS/DISCLOSED semantics |
| 32 | [`AUDIT.md`](AUDIT.md) | 112 | measured verification runs, findings, condensed codebase map, acceptance criteria |

## Concepts track

| doc | audience | core topics |
|---|---|---|
| [`block-diffusion.md`](concepts/block-diffusion.md) | beginner | corruption process, uniform state, why canvases, x0-parameterization |
| [`foundations.md`](concepts/foundations.md) | beginner | AR → D3PM lineage, transformer primitives from scratch, param accounting, worked forward pass |
| [`diffusion-core.md`](concepts/diffusion-core.md) | intermediate | forward process, cosine schedule, per-canvas t, x0 vs xt, chunked-CE memory story, time conditioning |
| [`block-causal-attention.md`](concepts/block-causal-attention.md) | intermediate | mask derivation, GQA + RoPE, three attention paths, KV chaining |
| [`canvas-denoising.md`](concepts/canvas-denoising.md) | intermediate | per-step semantics of one canvas, commit rule, all-ones decode view, re-encode |
| [`self-conditioning-mechanism.md`](concepts/self-conditioning-mechanism.md) | intermediate | zero-init, feeding schedule, train/test symmetry, one-add routing |
| [`self-conditioning.md`](concepts/self-conditioning.md) | expert | zero-init proof, detached pre-pass, cross-step conditioning, worked examples |
| [`sampler.md`](concepts/sampler.md) | expert | decode loop, commit derivation, Gumbel draw + temperature, entropy bond, FLOPs |
| [`chunked-ce-equivalence.md`](concepts/chunked-ce-equivalence.md) | expert | chunked-CE ≡ eager CE proof sketch, fp32 logsumexp boundaries, backward identity, test pins |
| [`memory-engineering.md`](concepts/memory-engineering.md) | expert | byte budgets, chunk-scaling law, VRAM-for-MFU trade |
| [`data-pipeline.md`](concepts/data-pipeline.md) | intermediate | shard/window contract, no-+1-shift, resumable shuffler |

## File→doc map

| module | where documented |
|---|---|
| `models/transformer.py` | [references/R2](references/R2_transformer_api.md), [foundations](concepts/foundations.md) §5 |
| `models/mask.py`, `models/attention.py`, `models/block.py` | [block-causal-attention](concepts/block-causal-attention.md), [references/R3](references/R3_mask_attention_api.md) |
| `models/diffusion.py` | [block-diffusion](concepts/block-diffusion.md), [diffusion-core](concepts/diffusion-core.md), [references/R4](references/R4_diffusion_api.md) |
| `models/selfcond.py`, `models/time_embed.py` | [self-conditioning](concepts/self-conditioning.md) (+ [mechanism](concepts/self-conditioning-mechanism.md)), [references/R4](references/R4_diffusion_api.md) |
| `training/losses.py` | [chunked-ce-equivalence](concepts/chunked-ce-equivalence.md), [memory-engineering](concepts/memory-engineering.md), [references/R5](references/R5_loss_training_api.md) |
| `training/pretrain.py` | [training.md](training.md), [references/R5](references/R5_loss_training_api.md) |
| `data/prepare_data.py`, `data/dataset.py` | [data-pipeline](concepts/data-pipeline.md), [references/R7](references/R7_data_utils_api.md) |
| `inference/generate.py` | [canvas-denoising](concepts/canvas-denoising.md), [sampler](concepts/sampler.md), [references/R6](references/R6_sampler_eval_api.md) |
| `inference/evaluate.py` | [inference.md](inference.md), [references/R6](references/R6_sampler_eval_api.md) + [eval-scripts](references/eval-scripts.md) |
| `utils/checkpoint.py`, `utils/logging.py`, `utils/memory.py` | [references/R7](references/R7_data_utils_api.md), [checkpoint-ops](guides/checkpoint-ops.md) |

Every concept doc carries its own glossary, "what breaks if you change this"
table, and embedded interview Q&A. First read for the foundations:
`concepts/foundations.md`; for rulings and deltas, `../DIFFUSION.md` stays
authoritative. If you don't know where to start, [`guides/learning-paths.md`](guides/learning-paths.md)
routes you by background and goal; for a term you don't recognize,
[`guides/glossary.md`](guides/glossary.md).

## House rules

- **Docs ship with code; stale docs fail CI** (`AGENTS.md` §6). Rename/move a
  public symbol → update every citing doc in the same commit.
- **Anchor style:** `file.py:Symbol`, `file.py:Class`, or
  `file.py:Class.method`. Never line numbers.
- **Gates** (run after any doc or code change):

```bash
uv run --python 3.13 --with torch --with numpy --with pyyaml \
  --with safetensors --with pytest python -m pytest -m "not gpu and not slow"
uv run --python 3.13 --with torch --with numpy --with pyyaml \
  --with safetensors --with pytest python scripts/check_docs.py --coverage --links
python3 scripts/build_docs_html.py
```

- The SDD decision ledger (rulings 1–26) lives in
  `.superpowers/sdd/EXECUTION-PLAN-diffusiongemma-lite/progress.md`; the
  DESIGN/EXECUTION-PLAN specs live in `../../llm-research/` outside the repo.