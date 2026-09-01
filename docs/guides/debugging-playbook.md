# Guide: debugging playbook

Symptom-first recipes for the failures this codebase actually produces.
House rule from the sibling Lites: **read the training error before
deciding anything** — never diagnose a NaN as overfitting.

## NaN / Inf loss → rollback

The guard rolls back after `nan_guard_max_consecutive=5` consecutive bad
optimizer steps (`training/pretrain.py:Pretrainer`, message
`[nan-guard] ... restoring checkpoint step ...`). Check in order:

1. **LR too high for the step count** — warmup is 2,000 steps; a crash in
   the first 100 steps is usually LR, not architecture.
2. **fp32 boundaries** — chunked-CE does fp32 logsumexp per chunk
   (`training/losses.py:chunked_x0_ce`); keep it that way.
3. **No checkpoint to restore from** → the guard raises
   `NaN/Inf with no checkpoint to restore from`; lower `save_interval` for
   unstable early runs.

## Resume doesn't reproduce bitwise

`tests/test_training.py::test_checkpoint_resume_determinism` pins
bit-equality. If a manual resume diverges:

1. All draws come from the per-micro-step generator
   (`training/pretrain.py:Pretrainer._step_rng`, seed
   `seed·100_003 + micro_count`) — any new RNG use in the training step
   (global `torch.rand`, `random`, dataloader shuffling) breaks it.
2. Data order replays via `data/dataset.py:ShuffledRangeSampler` offsets;
   `_micro_count` restores as `opt_steps × grad_accum`.
3. NaN-guard skips advance the micro count without an optimizer step —
   bit-equality holds only for NaN-free runs (SDD ledger Minor 3).

## `FileNotFoundError` at training start (no shards)

The training config reads `data/pretrain_chinchilla/shards/`. If shards
land elsewhere, the env pin was bypassed — run
`python data/prepare_data.py` (it pins `LLM_DATA_ROOT` itself) rather than
invoking `shared_data` stages by hand. The producer/consumer contract is
pinned by `tests/test_data.py::test_producer_consumer_shard_path_wiring`.

## Sampler outputs differ from a single-shot forward

The KV-chained decode must equal a fresh block-causal forward bit-exactly
in fp64 (`tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`).
If it drifts: a new attention/KV path must (a) tile KV heads before SDPA,
(b) append the finalized canvas's KV under the all-ones
`models/mask.py:build_canvas_decode_mask`, (c) not re-derive positions
off-by-one (`inference/generate.py:BlockDiffusionSampler.encode_canvas`).

## Zero-init self-conditioning broke

If `test_zero_init_equivalence` fails: something re-initialized
`models/selfcond.py:SelfConditioning`'s proj to non-zero — check that
`models/transformer.py:DiffusionGemma`'s `_init_weights` still re-zeros it,
and that the loss path adds conditioning exactly once
(`models/transformer.py:DiffusionGemma.final_hidden`, not composed with `head_forward`).

## Doc-anchor / coverage failures

`python scripts/check_docs.py --coverage` names the offending anchor:

- `FAIL ... not found` → a cited symbol was renamed/removed; update the
  citing doc in the same commit as the code change.
- `UNCOVERED ...` → a public symbol in `models/`, `training/`, `data/`,
  `inference/`, `utils/` is never cited; add one anchor to DIFFUSION.md,
  `docs/references/api.md`, or the relevant concept page.
- `LINE-ANCHOR` → a doc cites a line number instead of a symbol; cite symbols only.
- `BROKEN-LINK` → a markdown link target doesn't exist (check both
  doc-relative and repo-root-relative resolution).

## Chunked-CE disagrees with the eager loss

Equivalence (loss + grads, atol 1e-6) is pinned by
`tests/test_loss.py::test_chunked_equals_eager` and
`test_chunked_matches_eager_grad_direction`. If they drift:

1. The fp32 logsumexp per chunk is load-bearing — a bf16 logsumexp drifts
   across 50,257-vocab chunks (`training/losses.py:chunked_x0_ce`).
2. The custom autograd Function
   (`training/losses.py:_ChunkTerms`) must save the **bf16** logits and
   derive softmax from them in backward — saving fp32 doubles retained
   memory, recomputing the GEMM defeats the point.
3. Partial last chunk: `V = 50,257` is not a multiple of 8192 — the last
   chunk is 1,153 wide; `tests/test_loss.py::test_partial_last_chunk` pins
   it. A hand-rolled `range(0, V, chunk)` without the clamp hits it.

## Empty loader / wrong window counts

`No complete {seq_len}-token windows` (`training/pretrain.py:Pretrainer.train`)
means shard windows < 1: shard shorter than `seq_len`, or
`micro_batch_size × seq` exceeding available windows. Verify window counts
directly (`data/dataset.py:ShardWindows.n_windows`, 12,207 per 50M-token
shard at seq 4,096) before blaming the trainer. Data pipeline internals:
[data-pipeline](../concepts/data-pipeline.md).

## Shape errors cheat-sheet

| error | likely cause |
|---|---|
| `seq_len must be a multiple of canvas_len` | raw `models/mask.py:build_block_causal_mask` on a non-multiple; use the sampler's partial mask |
| SDPA head-count mismatch | KV heads not tiled to 16 before `models/mask.py:block_causal_sdpa_attention` |
| `(B, T, V)` OOM | someone bypassed `training/losses.py:chunked_x0_ce` for the eager loss |
| mask shape `(1,1,L,prefix+L)` mismatch | `models/mask.py:build_canvas_decode_mask` called with the wrong `prefix_len` |
| empty loader (`No complete ... windows`) | shard shorter than `seq_len`, or `micro_batch_size` × seq larger than the data — check `data/dataset.py:ShardWindows` window counts |
