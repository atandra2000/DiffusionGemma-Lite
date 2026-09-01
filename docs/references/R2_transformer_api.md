# R2 — `models/transformer.py` API reference

> Signatures extracted from source; every entry carries shapes, defaults,
> callers, and the pinning test. Concept depth lives in
> [foundations](../concepts/foundations.md) §5.

## `class DiffusionGemmaConfig`

Dataclass — all 22 fields with defaults in [R1](R1_config.md);
`DiffusionGemmaConfig.from_yaml(path: str) -> DiffusionGemmaConfig` reads the
`model:` block. Pinned by `tests/test_models.py::test_param_count` (the
343,516,160 invariant at production dims).

## `class DiffusionGemma(self, cfg: DiffusionGemmaConfig)`

The full denoiser: embedding (weight-tied with the LM head), 24
`models/block.py:DenoiseBlock`s, final RMSNorm, zero-init
`models/selfcond.py:SelfConditioning`, CanvasTimeEmbedding.

| method | signature | returns | contract |
|---|---|---|---|
| `backbone` | `(input_ids, t, past_kv=None, positions=None, mask=None, time_steps=None, return_kv=False)` | `(h, kv)` or `h` | the transformer trunk; `t` is (B, n_canvases); mask semantics per [R3](R3_mask_attention_api.md) |
| `final_hidden` | `(input_ids, t, sc_input=None)` | (B, S, d_model) | backbone + **the** single W_sc add — loss path ([self-conditioning §5](../concepts/self-conditioning.md)) |
| `head_forward` | `(h_norm, sc_input=None)` | (B, S, V) logits | same single add, then `h @ Eᵀ`; never compose with `final_hidden` |
| `forward` | `(input_ids, t, sc_input=None)` | logits (B, S, V) | `final_hidden` → `head_forward` (training convenience) |
| `generate` | `(prompt_ids, max_new_tokens, n_diffusion_steps=None, adaptive=None)` | (B, P+new) | convenience wrapper over `inference/generate.py:BlockDiffusionSampler` |

Shapes at production dims: `input_ids (B, 4096)`, `t (B, 16)`, `h (B, 4096,
1024)`, logits `(B, 4096, 50257)` — but the loss path never materializes
those logits ([memory-engineering §2](../concepts/memory-engineering.md)).

Key invariants (each pinned):

- parameter count **343,516,160** — `tests/test_models.py::test_param_count`
- weight tying shares storage — `tests/test_models.py::test_weight_tying_shared`
- exactly one W_sc add per path — `tests/test_models.py::test_two_step_overfit`
- `past_kv` forwards require an explicit mask on non-flex paths — asserted in
  `models/transformer.py:DiffusionGemma.backbone`

`models/transformer.py:DiffusionGemma._init_weights` applies `init_std 0.02`
to every Linear then **re-zeros `selfcond.proj`** — the only tensor the
generic pass must undo ([self-conditioning §3](../concepts/self-conditioning.md)).

## Module map

| import | provides | concept doc |
|---|---|---|
| `models.transformer` | `DiffusionGemmaConfig`, `DiffusionGemma` | [foundations §5](../concepts/foundations.md) |
| — | `.backbone` KV contract | [R3](R3_mask_attention_api.md), [block-causal-attention §6](../concepts/block-causal-attention.md) |
| — | `.final_hidden` / `.head_forward` routing | [self-conditioning §5](../concepts/self-conditioning.md) |
| — | `.generate` | [sampler](../concepts/sampler.md) §1 |

Callers: `training/pretrain.py:Pretrainer` (backbone/final_hidden),
`training/losses.py:chunked_x0_ce` (consumes `final_hidden` output),
`inference/generate.py:BlockDiffusionSampler` (backbone + head_forward).