# DiffusionGemma-Lite — Documentation & Codebase Audit

> Scope: the whole repo — `docs/` (28 md files), `DIFFUSION.md`, `README.md`,
> `AGENTS.md`, `SKILLS.md` — against the code in `models/`, `training/`,
> `data/`, `inference/`, `utils/`. Performed 2026-07-20 as the final piece of
> the portfolio documentation upgrade (four-track taxonomy completion):
> this repo already had concepts/references/training/inference tracks plus
> `check_docs.py --coverage` (the portfolio template); what was missing were
> the cross-cutting learning path, the top-level glossary, and this audit.

## 1. Verification runs (all measured, this audit)

| Run | Command | Result |
|-----|---------|--------|
| Symbol anchors | `python3 tests/test_doc_refs.py --strict-coverage` | PASS — scanned 35 docs, 732 anchors; resolution PASS, coverage PASS, line anchors PASS |
| Doc↔code gate | `python3 scripts/check_docs.py --coverage --links` | PASS — same gate under the standalone checker, links PASS |
| Test suite | `python3 -m pytest tests/ -m "not gpu and not slow" --tb=no` | **77 passed, 0 skipped, 2 warnings in ~10 s** (macOS CPU) |
| Citation density | `` grep -rEo '`[a-z_/]+\.py:[A-Za-z_]+`' `` | 678 `*.py:Symbol` anchors pre-upgrade → 732 after (+54 from the two new guides) |
| Stale-count sweep | grep for retired counts ("71 tests") and decontextualized "~380M" | Clean — every "~380M" mention is the deliberate correction context (Ruling 13/19); no retired test counts remain |

The 2 warnings are benign and known: flex_attention invoked uncompiled in
`tests/test_models.py::test_flex_attn_impl_matches_sdpa` (the test's purpose
is semantic equivalence, not kernel fusion), and an lr_scheduler ordering
warning in `tests/test_training.py::test_lr_schedule_shape` (shape probe).

Wave-1 closeout addendum, re-measured 2026-09-21:

| Run | Command | Result |
|-----|---------|--------|
| Doc↔code gate | `python3 scripts/check_docs.py --coverage --links` | PASS — scanned 39 docs, 776 anchors; resolution PASS, coverage PASS, line anchors PASS, links PASS |
| Test suite | `python3 -m pytest -m "not gpu and not slow"` | **77 passed, 2 warnings in ~12.6 s** (macOS CPU; same benign warnings as above) |

Addendum: the Wave-1 "Special work" concepts cluster landed as four compact,
audience-tagged docs under `docs/concepts/` —
[block-diffusion.md](concepts/block-diffusion.md),
[canvas-denoising.md](concepts/canvas-denoising.md),
[self-conditioning-mechanism.md](concepts/self-conditioning-mechanism.md),
[chunked-ce-equivalence.md](concepts/chunked-ce-equivalence.md) — each
cross-linked to the canonical deep dive it complements (no existing doc was
rewritten). `docs/README.md` gained the measured corpus-size table (dated),
a concepts-track table, and a file→doc map; the reading-order line counts
were re-measured (the R1–R7 rows had drifted from 120/80/85/65/60/75/80 down
to 82/54/51/39/35/41/52 — same failure mode as finding A1);
`guides/learning-paths.md` step tables route the new docs.

## 2. Findings

| ID | Severity | Evidence | Status |
|----|----------|----------|--------|
| A1 | minor | `docs/README.md` reading-order line counts drifted on two guides: `contributing.md` listed 57 (measured 58), `guides/a100-runbook.md` listed 75 (measured 89 — the pod runbook grew in commit `86d8a1c` without a table refresh). | Fixed in this upgrade's nav commit. |
| A2 | gap | No cross-cutting learning path and no top-level glossary: the corpus was deep but entry order was implicit, and every concept doc carried its own inline glossary with no single lookup page. | Fixed — `guides/learning-paths.md` (three tier tables) and `guides/glossary.md` (notation + term tables) added; wired into `docs/README.md`. |
| A3 | accepted | The nav map's reading-order table (line counts per doc) plays the "corpus size" role other Lites implement as a separate word-count table. Kept as-is — one maintained table beats two near-duplicates. | Accepted (no change). |
| A4 | target, no run yet | Headline GPU numbers (~40–50 h A100 80GB at 35–40% MFU; wall-clock speedup vs AR; MFU ≥ 33%, peak VRAM < 15 GB gates) are unmeasured — the 8.0B-token run has not started. Docs correctly label them as pending; nothing to fix until the pod session (Task 15). | Accepted (honest pending state). |

## 3. From-scratch codebase explanation (condensed map)

**What the model is.** A 343,516,160-parameter (exactly — pinned by
`tests/test_models.py::test_param_count`; not the "~380M" of early planning
docs, whose table counted full-MHA K/V) dense transformer that generates in
256-token canvases: each canvas starts as uniform noise over the 50,257-token
GPT-2 BPE vocab and is denoised in parallel over a schedule of diffusion
steps, then re-encoded into the KV cache so the next canvas attends to it
causally. Training context 4096 = 16 canvases × 256.

**Module map.**

| Module | Role | Key symbols |
|--------|------|-------------|
| `models/transformer.py` | Config + top-level wiring | `DiffusionGemmaConfig`, `DiffusionGemma` |
| `models/mask.py` | The block-causal geometry | `build_block_causal_mask`, `build_canvas_decode_mask`, `build_block_causal_block_mask`, `eager_block_causal_attention`, `block_causal_sdpa_attention`, `flex_block_causal_attention` |
| `models/diffusion.py` | Corruption process | `alpha_bar`, `corruption_probs`, `sample_canvas_t`, `q_sample`, `x0_ce_loss` (eager reference) |
| `models/attention.py`, `models/block.py` | Per-block compute | `apply_rope`, `DenoiseAttention`, `DenoiseBlock`, `RMSNorm` |
| `models/selfcond.py`, `models/time_embed.py` | Conditioning inputs | `SelfConditioning`, `CanvasTimeEmbedding` |
| `training/losses.py` | Memory-bounded loss | `chunked_x0_ce`, `chunked_p_embed`, `_ChunkTerms` |
| `training/pretrain.py` | Training loop | `TrainingConfig`, `Pretrainer` (per-micro-step `Pretrainer._step_rng`) |
| `data/dataset.py` | Shard reader | `ShardWindows`, `ShuffledRangeSampler`, `build_dataloader` |
| `inference/generate.py` | Decode sampler | `SamplerConfig`, `BlockDiffusionSampler` (`prefill` → `denoise_canvas` → `encode_canvas`) |
| `inference/evaluate.py` | Eval harness | `SpeedupEvaluator`, `parse_baseline`, `heldout_x0_nll` |
| `utils/` | Checkpointing, memory, logging | `CheckpointManager`, `estimate_model_memory_gb`, `assert_fits_in_available_gpu` |

**Invariants that hold the design together** (full rulings in
[`DIFFUSION.md`](../DIFFUSION.md); SDD ledger rulings 1–26):
x0-parameterized loss (Ruling 19), zero-init self-conditioning equivalence
(bit-exact), monotone commit rule, ALL-ONES re-encode mask (Rulings 15–17),
sampler ↔ single-shot fp64 bit-exact equivalence, bitwise-deterministic
resume, and the never-materialize-full-vocab-logits chunked-CE contract.

## 4. Modification plan (priority order)

1. ~~Add learning path + glossary + audit; wire nav map~~ (this upgrade).
2. ~~Refresh drifted reading-order line counts~~ (A1, this upgrade).
3. A100 pod session (Task 15): run the 8.0B-token pretrain, then fill the
   pending GPU gates (MFU, VRAM, wall-clock speedup, loss parity) into
   [guides/benchmarking.md](guides/benchmarking.md) and
   [inference.md](inference.md) — measured numbers only.
4. Post-training: recalibrate `SamplerConfig` (`entropy_threshold`,
   temperature ramp) against the trained model and record the tuned values
   in [guides/sampler-tuning.md](guides/sampler-tuning.md).
5. Keep `scripts/check_docs.py` as the portfolio `--coverage` template: any
   workspace port must re-derive `DOC_PATHS`/`COVERAGE_MODULES` per repo.

## 5. Acceptance criteria for "audit complete"

- [x] Both doc gates green: `check_docs.py --coverage --links` and
      `tests/test_doc_refs.py --strict-coverage` (0 stale anchors, coverage 0-gap).
- [x] CPU test suite green: 77 passed, 0 skipped.
- [x] Learning path exists with three tier tables citing real docs.
- [x] Top-level glossary exists with notation table + per-component term tables.
- [x] Nav map (`docs/README.md`) routes to the new guides and this audit.
- [x] Findings table resolved or explicitly accepted (A3, A4).
- [x] Every headline number in docs is measured (343,516,160 params,
      77 tests) or explicitly labeled pending (A100 run).
