# R6 — Sampler & evaluation API

> Public symbols of `inference/generate.py` and `inference/evaluate.py`.
> Loop semantics: [sampler](../concepts/sampler.md); harness:
> [inference](../inference.md).

## `inference/generate.py`

| symbol | signature | contract |
|---|---|---|
| `SamplerConfig` | dataclass — fields in [R1](R1_config.md) | eval sampler knobs |
| `BlockDiffusionSampler.__init__` | `(model, cfg: SamplerConfig)` | picks flex vs mask path from `model.cfg.attn_impl` |
| `BlockDiffusionSampler.prefill` | `(prompt_ids)` → `(kv, prefix_len)` | ONE forward over the prompt under the block-causal (partial-first-canvas) mask; KV static afterwards |
| `BlockDiffusionSampler.denoise_canvas` | `(kv, prefix_len, cfg=None)` | T_eval (+early stop) forwards from pure noise; returns `(canvas_ids, steps_used, entropy_trace)`; commit-and-renoise per [sampler §2](../concepts/sampler.md) |
| `BlockDiffusionSampler.encode_canvas` | `(kv, prefix_len, canvas_ids)` | 1 forward over the finalized canvas under the all-ones decode view; per-layer `cat(past_kv, new_kv)`; asserts whole-canvas input |
| `BlockDiffusionSampler.generate` | `(prompt_ids, max_new_tokens)` | prefill → (denoise → encode)×⌈new/L⌉ → preallocated output buffer |

Internal but load-bearing (cited by tests):

| symbol | role |
|---|---|
| `inference/generate.py:_prefix_mask` | partial-first-canvas prefill mask, same row semantics as `build_block_causal_mask` |
| `_decode_mask` | all-ones decode view — `None` under flex (mask-free full attention) |
| `_denoise_step` | one uniform-state step; returns `(x, committed, (am, conf), entropy, sc_next, x0)`; computes `sc_next = p @ E` for cross-step conditioning |

Pinned: `tests/test_sampler.py::test_block_ar_chaining_matches_single_shot`
(fp64 bit-exact), `test_commit_rule_monotone`,
`test_adaptive_off_ignores_entropy`, `test_gumbel_temperature_draw`.

## `inference/evaluate.py`

| symbol | signature | contract |
|---|---|---|
| `parse_baseline` | `(name: str) -> SamplerConfig` | `"fixed_T16"` / `"adaptive_T32"` → config — `tests/test_inference.py::test_parse_baseline` |
| `SpeedupEvaluator.__init__` | `(model, tokenizer=None)` | tokenizer accepted for future decode previews; counting is tokenizer-free |
| `SpeedupEvaluator.evaluate` | `(n_samples=100, prompt_tokens=64, gen_tokens=1024, baselines=("fixed_T16", "fixed_T32", "adaptive_T32"), wall_clock=True)` | `{"rows": {...}, "speedup_vs_ar_tokens_per_forward": {...}}`; adds `ar_kv_analytic` (1.0 by construction, `seconds=None`) — `test_evaluator_produces_three_rows` |
| `heldout_x0_nll` | `(model, shard_path, n_windows=64, seq_len=None, seed=42)` | mean chunked x0-CE over held-out windows, **no sc input**, nats/token — `tests/test_inference.py::test_heldout_x0_nll_finite` |

Row metrics (from `_measure`): `forwards`, `token_forwards`,
`token_forwards_per_token`, `tokens_per_forward`, `seconds`,
`tokens_per_sec` — the AR row is analytic with `seconds=None` (the honest
gap, [inference §4](../inference.md)).