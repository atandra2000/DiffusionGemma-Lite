# AGENTS.md — DiffusionGemma-Lite

> Read the workspace `LLM/AGENTS.md` and the parent `CoreProjects/AGENTS.md`
> first. Higher-level rules are authoritative; this file adds project-specific
> rules only (and wins on DiffusionGemma-Lite-specific conflicts).

## Quick checks (run before claiming work is done)

```bash
cd LLM/DiffusionGemma-Lite
python3 -m pytest -m "not gpu and not slow"   # CPU-friendly suite
python3 scripts/check_docs.py --coverage      # doc↔code symbol anchors
# GPU gates (CUDA required, A100 pod):
#   python scripts/e2e_gpu_smoke.py ; python scripts/step_time_a100.py
```

> **Project:** `LLM/DiffusionGemma-Lite/` · **Type:** block-AR discrete
> diffusion LM (uniform-state) · **Scale:** 343,516,160 params (~343.5M — NOT
> the "~380M" of early planning docs; their table counted full-MHA K/V) ·
> 8.0B Chinchilla-optimal tokens · 14–18 h on A100 80GB (pending).
> **Stack:** PyTorch only — no diffusion library, no custom CUDA, no Triton.
> **Hardware:** A100 80GB (no offloading).
> **Architecture detail:** `DIFFUSION.md` is authoritative; `docs/README.md`
> is the doc map; SDD rulings live in
> `.superpowers/sdd/EXECUTION-PLAN-diffusiongemma-lite/progress.md`.

## 1. What this model is (know cold)

- **Block-AR discrete diffusion.** 256-token canvases denoised in parallel
  from uniform noise (D3PM-style, cosine schedule), committed under the
  unchanged-or-stronger rule, re-encoded into the KV cache; causal across
  canvases, bidirectional within (`models/mask.py:build_block_causal_mask`).
- **x0 prediction.** The loss is cross-entropy against the *clean* tokens —
  the plan's `xt` wording is a recorded typo (Ruling 19). The sampler's
  logits are the x̂0 posterior.
- **Self-conditioning** with zero-init equivalence (bit-exact at init,
  `atol=0, rtol=0`): `models/selfcond.py:SelfConditioning`; the add happens
  exactly once per path; the training pre-pass is detached (no second
  gradient path).
- **Sampler invariants** (SDD Rulings 15–17): finalized canvases re-encode
  under the ALL-ONES `models/mask.py:build_canvas_decode_mask` (not the
  superseded §2.5 "zero-mask" wording); the sampler redraws pure uniform
  (`q_sample` is train-time only); committed positions are never overwritten;
  eval passes `time_steps=SamplerConfig.n_diffusion_steps` (not the model's
  train-time 16).
- GQA 16Q/4KV head_dim 64 · rope rotate-half · SDPA production path with an
  eager ground-truth twin (`models/mask.py:eager_block_causal_attention`) —
  do not delete the "duplicate".

## 2. Hard rules

1. **Pure PyTorch.** No HuggingFace Trainer/Lightning, no diffusion
   libraries, no custom CUDA/Triton kernels. The two eager-vs-SDPA attention
   branches and the eager CE reference (`models/diffusion.py:x0_ce_loss`)
   are deliberate regression oracles — never consolidate them away.
2. **Never materialize full-vocab training logits.** The production loss is
   `training/losses.py:chunked_x0_ce` (and `chunked_p_embed` for the
   self-cond pre-pass); `x0_ce_loss` is the eager test reference only. At
   micro_bs 8 / seq 4096 / V 50,257 the full fp32 logits tensor is ~6.6 GB.
3. **Commit rule is monotone.** A committed canvas position is never
   overwritten; uncommitted positions redraw pure uniform. Do not "fix" the
   sampler to overwrite confident-but-changed tokens without re-deriving the
   DESIGN §2.5 posterior argument and updating the pinning tests.
4. **Sampler ↔ single-shot equivalence is sacred.** The KV-chained decode
   must equal a fresh single-shot block-causal forward (fp64, bit-exact) —
   `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`.
   Any change to attention/KV plumbing re-runs this test.
5. **Resume determinism.** All train-time draws come from the per-micro-step
   generator (`training/pretrain.py:Pretrainer._step_rng`); data order from
   `data/dataset.py:ShuffledRangeSampler` offsets. Never introduce global RNG
   into the training step without updating the bitwise-resume test.
6. **Shard-path contract:** the data shim pins `LLM_DATA_ROOT` to
   `data/pretrain_chinchilla` (the `shared_data` pack stage runs as a
   subprocess honoring only that env var) and consumers read
   `data/pretrain_chinchilla/shards/`. Pinned by
   `tests/test_data.py::test_producer_consumer_shard_path_wiring`.
7. **Doc anchors.** Docs cite code as `file.py:Symbol` — never line numbers.
   `scripts/check_docs.py --coverage` must stay clean; stale docs fail CI
   (see §4).
8. **Concise comments only.** Docstrings justify non-obvious code; no
   comment-per-line narration; section banners ≤ 3 per file (house rule, as
   in sibling Lites).

## 3. Numerical-stability rules

- NaN guard: 5 consecutive NaN/Inf optimizer steps → rollback to the last
  complete checkpoint (`utils/checkpoint.py:CheckpointManager` — a step is
  resumable only when weights + optimizer + meta all exist).
- Chunked-CE does fp32 logsumexp per chunk; loss combines via global lse.
  Keep fp32 boundaries inside chunks; don't "simplify" to BF16 lse.
- Bitwise-equality tests pin zero-init self-cond equivalence (fp64, atol=0)
  and rope norm preservation — treat any relaxation as a regression.
- `utils/memory.py:assert_fits_in_available_gpu` raises before an
  oversized run starts; a probe failure is logged, never silent.

## 4. Files

- `models/` — `mask.py` (masks + eager/SDPA paths), `diffusion.py`
  (schedule, q_sample, eager loss), `attention.py`, `selfcond.py`,
  `time_embed.py`, `block.py`, `transformer.py` (model + `from_yaml`).
- `training/` — `losses.py` (chunked-CE), `pretrain.py` (loop).
- `data/` — `prepare_data.py` (shared_data shim), `dataset.py` (ShardWindows).
- `inference/` — `generate.py` (sampler), `evaluate.py` (headline harness).
- `utils/` — `checkpoint.py`, `logging.py`, `memory.py`.
- `scripts/` — eval CLIs + docs tooling; `docs/` — concepts/guides/references
  (`docs/README.md` is the map; `tests/test_doc_refs.py` the checker).
- `DIFFUSION.md`, `README.md`, `SKILLS.md`, `pytest.ini`, `LICENSE`.

## 5. Known caveats

- **No trained checkpoint yet.** The 8.0B-token A100 run has not started;
  GPU gates (MFU ≥ 33%, peak VRAM < 15 GB, wall-clock speedup vs AR, loss
  parity) are pending the pod session (Task 15).
- **~343.5M, not ~380M.** Config filename and early docs say 380M; the
  architecture pins 343,516,160. Don't "correct" one side without the other.
- **Tiny-model eval numbers are misleading.** `tokens/forward` of fixed_T is
  `L/(T+1)`: at canvas 32, fixed_T32 < 1. Quote production-canvas numbers.
- **Adaptive stopping is inert on untrained weights** (entropy never drops
  below threshold); only meaningful post-training.
- **Eval `T ≤ 32`** stays within the time-embedding normalization range;
  raising `eval_diffusion_steps` needs the §4.3 normalization re-checked.
- Phase-1–4 review minors deferred to the final whole-branch review are
  tracked in the SDD ledger; consult it before "cleaning up" flagged spots.

## 6. Docs rule

**Docs ship with code; stale docs fail CI.** Any change that adds, renames,
or removes a public symbol must update the docs citing it.
`scripts/check_docs.py` parses every `file.py:Symbol` anchor in `docs/`,
`DIFFUSION.md`, `README.md`, `AGENTS.md`, `SKILLS.md` and fails on unknown
files or symbols; `--coverage` additionally requires every public symbol in
`models/`, `training/`, `data/`, `inference/`, `utils/` to be cited at least
once. `tests/test_doc_refs.py` runs the same gate under pytest.