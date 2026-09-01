# Docs — DiffusionGemma-Lite

Every code citation in this tree uses `file.py:Symbol` anchors
(e.g. `models/mask.py:build_block_causal_mask`); `scripts/check_docs.py`
fails on any anchor that stops resolving, and `--coverage` fails if a public
symbol in `models/`, `training/`, `data/`, `inference/`, `utils/` is never
cited. Line-number citations are rejected — they rot.

## Reading order

| # | doc | lines | read when |
|---|---|---|---|
| 1 | [`../DIFFUSION.md`](../DIFFUSION.md) | 464 | **start here** — the authoritative engineer's contract (all math + semantics + rulings) |
| 2 | [`concepts/foundations.md`](concepts/foundations.md) | 1,074 | the from-scratch textbook chapter: AR → diffusion → D3PM → every primitive derived, worked forward pass, param accounting |
| 3 | [`concepts/diffusion-core.md`](concepts/diffusion-core.md) | 324 | forward process, cosine schedule, x0-parameterization, chunked-CE |
| 4 | [`concepts/block-causal-attention.md`](concepts/block-causal-attention.md) | 302 | mask derivation, GQA + RoPE at production dims, three attention paths, KV chaining |
| 5 | [`concepts/self-conditioning.md`](concepts/self-conditioning.md) | 211 | zero-init equivalence, detached pre-pass, cross-step eval conditioning |
| 6 | [`concepts/sampler.md`](concepts/sampler.md) | 297 | decode loop, commit rule, temperature annealing, entropy bond, FLOP accounting |
| 7 | [`concepts/memory-engineering.md`](concepts/memory-engineering.md) | 219 | the byte budget, chunked-CE memory argument, VRAM-for-MFU trade |
| 8 | [`concepts/data-pipeline.md`](concepts/data-pipeline.md) | 204 | shard/window contract, no-+1-shift rationale, resumable shuffler |
| 9 | [`training.md`](training.md) | 234 | the training loop, determinism recipe, NaN guard, checkpointing, A100 run |
| 10 | [`inference.md`](inference.md) | 183 | eval harness, headline metric, honest gaps |
| 11 | [`references/R1_config.md`](references/R1_config.md) | 120 | every model/runtime/sampler knob |
| 12 | [`references/R2_transformer_api.md`](references/R2_transformer_api.md) | 80 | `models/transformer.py` API |
| 13 | [`references/R3_mask_attention_api.md`](references/R3_mask_attention_api.md) | 85 | mask + attention + block + time-embed APIs |
| 14 | [`references/R4_diffusion_api.md`](references/R4_diffusion_api.md) | 65 | diffusion + self-conditioning APIs |
| 15 | [`references/R5_loss_training_api.md`](references/R5_loss_training_api.md) | 60 | chunked loss + trainer API |
| 16 | [`references/R6_sampler_eval_api.md`](references/R6_sampler_eval_api.md) | 75 | sampler + eval APIs |
| 17 | [`references/R7_data_utils_api.md`](references/R7_data_utils_api.md) | 80 | data + checkpoint + memory + logging APIs |
| 18 | [`guides/quickstart.md`](guides/quickstart.md) | 126 | data → train → eval commands with expected outputs |
| 19 | [`guides/debugging-playbook.md`](guides/debugging-playbook.md) | 106 | NaN rollback, resume divergence, shape errors, anchor failures |
| 20 | [`guides/sampler-tuning.md`](guides/sampler-tuning.md) | 46 | SamplerConfig knobs: what to touch, what not to |
| 21 | [`guides/benchmarking.md`](guides/benchmarking.md) | 57 | honest speedup/VRAM/MFU discipline |
| 22 | [`guides/checkpoint-ops.md`](guides/checkpoint-ops.md) | 53 | checkpoint layout, resume recipes, disk hygiene |
| 23 | [`guides/contributing.md`](guides/contributing.md) | 57 | the doc contract + gates for new code |

Every concept doc carries its own glossary, "what breaks if you change this"
table, and embedded interview Q&A. First read for the foundations:
`concepts/foundations.md`; for rulings and deltas, `../DIFFUSION.md` stays
authoritative.

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