# SKILLS.md — DiffusionGemma-Lite

> Companion to `AGENTS.md` (architecture + rules). This file holds day-to-day
> developer workflows.

## Skill 1: Run the CPU-friendly test suite

```bash
cd LLM/DiffusionGemma-Lite
python3 -m pytest -m "not gpu and not slow"
```

Covers mask semantics, the diffusion process, rope/self-conditioning
invariants (zero-init bit-exact), model wiring, sampler invariants (commit
monotonicity, KV-chaining ≡ single-shot in fp64), chunked-CE ≡ eager CE,
resume bit-equality, and the doc-anchor gate. Must pass before any change.

## Skill 2: Verify the chunked-CE loss equals eager CE

The load-bearing optimization must not drift. To re-verify at production
vocab:

```python
import torch
from training.losses import chunked_x0_ce
from models.diffusion import x0_ce_loss

torch.manual_seed(0)
hidden = torch.randn(1, 128, 256)
E = torch.randn(50257, 256)
x0 = torch.randint(0, 50257, (1, 128))
eager = x0_ce_loss(hidden @ E.t(), x0)
chunked = chunked_x0_ce(hidden, E, x0, vocab_chunk=8192)
print(float(eager), float(chunked), float((eager - chunked).abs()))
```

Max abs diff should be ~0 (review-measured 0.0; the test pins 1e-6). If it
drifts, suspect the fp32 logsumexp boundaries or the chunk weighting.

## Skill 3: Smoke the sampler invariants

```python
import torch
from models.transformer import DiffusionGemma, DiffusionGemmaConfig
from inference.generate import BlockDiffusionSampler, SamplerConfig

m = DiffusionGemma(DiffusionGemmaConfig.from_yaml('configs/pretrain_a100_380m.yaml'))
s = BlockDiffusionSampler(m, SamplerConfig(n_diffusion_steps=8, seed=0))
ids = s.generate(torch.randint(0, 50257, (1, 256)), max_new_tokens=512)
```

Invariants to check after touching the sampler: commit monotonicity
(`tests/test_sampler.py::test_commit_rule_monotone`), adaptive-off fallback ≡
fixed (`test_adaptive_off_ignores_entropy`), and the fp64
chaining-vs-single-shot equivalence (`test_block_ar_chaining_matches_single_shot`).

## Skill 4: Smoke-run training before an A100 session

```bash
python -m training.pretrain --config configs/pretrain_a100_380m.yaml --dry-run
```

2 steps to verify wiring (data → corruption → pre-pass → chunked-CE →
backward → checkpoint). On CPU this runs tiny-config scale only; the full
config needs the A100 pod (`scripts/launch_a100.sh`, Task 15).

## Skill 5: Headline evaluation

```bash
python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
python scripts/speedup_eval.py --flops-only     # forwards/token-forwards only
python scripts/loss_parity_eval.py --shard <heldout.bin> --ar-nll <AR CE>
```

Rows: `fixed_T16`, `fixed_T32`, `adaptive_T32`, analytic AR (1 token/forward).
On random-init weights adaptive never fires (entropy never settles) — the
adaptive row only diverges post-training. Do not quote tiny-canvas numbers.

## Skill 6: Docs gates

```bash
python scripts/check_docs.py --coverage --links   # anchors + coverage + links
python scripts/build_docs_html.py                 # HTML portal → docs_html/
python3 -m pytest tests/test_doc_refs.py tests/test_build_docs_html.py
```

Every doc citation is `file.py:Symbol` — line-number anchors fail CI. If you
add/rename a public symbol in `models/`, `training/`, `data/`, `inference/`,
or `utils/`, cite it in a doc (DIFFUSION.md or docs/references/) or the
coverage gate fails.

## Pitfalls

- **`xt` vs x0:** the plan's Task-13 loss line says `xt`; the correct target
  is x0 (sampler semantics depend on it). Don't "fix" the code to match the
  plan text.
- **Self-cond double-add:** `final_hidden` already includes the W_sc add;
  never compose it with `head_forward`'s add.
- **Eval time normalization:** sampler forwards pass
  `time_steps=SamplerConfig.n_diffusion_steps`; the model default (train-time
  16) is for training only.
- **`LLM_DATA_ROOT`:** `prepare_data.py` pins it — the pack subprocess ignores
  in-process data-root overrides.
- **Canvas divisibility:** `build_block_causal_mask` asserts
  `seq_len % canvas_len == 0`; non-multiple prompts go through the sampler's
  inline partial mask (`inference/generate.py:_prefix_mask`).
- **NaN guard:** `nan_guard_max_consecutive=5` — after 5 consecutive NaN steps
  the run rolls back to the last complete checkpoint. Post-resume bit-equality
  holds only for NaN-free runs (SDD ledger Minor 3).