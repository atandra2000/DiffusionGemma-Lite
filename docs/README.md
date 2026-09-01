# Docs — DiffusionGemma-Lite

Every code citation in this tree uses `file.py:Symbol` anchors
(e.g. `models/mask.py:build_block_causal_mask`); `scripts/check_docs.py`
fails on any anchor that stops resolving, and `--coverage` fails if a public
symbol in `models/`, `training/`, `data/`, `inference/`, `utils/` is never
cited. Line-number citations are rejected — they rot.

## Map

| doc | read when |
|---|---|
| [`../DIFFUSION.md`](../DIFFUSION.md) | **start here** — the authoritative technical doc (all math + semantics + recipe deltas) |
| [`../README.md`](../README.md) | headline metric, quick start, honest gaps |
| [`concepts/diffusion-core.md`](concepts/diffusion-core.md) | the forward process, schedule, x0-parameterization, chunked-CE |
| [`concepts/block-causal-attention.md`](concepts/block-causal-attention.md) | mask derivation, GQA+RoPE, SDPA vs eager, KV chaining |
| [`concepts/self-conditioning.md`](concepts/self-conditioning.md) | zero-init equivalence, detached pre-pass, cross-step eval conditioning |
| [`guides/quickstart.md`](guides/quickstart.md) | data → train → eval commands end to end |
| [`guides/debugging-playbook.md`](guides/debugging-playbook.md) | NaN rollback, resume divergence, anchor failures, common shape errors |
| [`references/config.md`](references/config.md) | every key of `configs/pretrain_a100_380m.yaml` |
| [`references/api.md`](references/api.md) | public API surface, symbol-anchored |

## House rules

- **Docs ship with code; stale docs fail CI** (`AGENTS.md` §6). Rename/move a
  public symbol → update every citing doc in the same commit.
- **Anchor style:** `file.py:Symbol`, `file.py:Class`, or
  `file.py:Class.method`. Never line numbers.
- The SDD decision ledger (rulings 1–26) lives in
  `.superpowers/sdd/EXECUTION-PLAN-diffusiongemma-lite/progress.md`; the
  DESIGN/EXECUTION-PLAN specs live in `../../llm-research/` outside the repo.
