# Guide: contributing — the contracts this repo enforces

Read `AGENTS.md` first (workspace), then this page (repo). The theme: this
repo is a **spec-first, test-pinned** implementation — every change keeps a
documented ruling true and a test green.

## The non-negotiables

1. **Deliberate duplication is load-bearing.** The eager attention twin
   (`models/mask.py:eager_block_causal_attention`) and the eager loss
   (`models/diffusion.py:x0_ce_loss`) are the implementation-independent
   oracles the equivalence tests compare against — never "simplify" them
   (AGENTS.md §2).
2. **Never materialize (B, T, V) logits** in the training path —
   `training/losses.py:chunked_x0_ce` / `chunked_p_embed` only (DESIGN §4.0).
3. **x0 is the target, never xt** (Ruling 13); parameter count is
   343,516,160 — never "380M" (Ruling 19/343.5M ruling).
4. **Per-canvas t**, one draw per canvas from the step generator — not
   per-token, not global RNG ([training §2](../training.md)).
5. **All three attention paths stay** (sdpa / flex / eager) with their
   weight-transplant equivalence tests
   (`tests/test_models.py::test_eager_attn_impl_matches_sdpa`,
   `test_flex_attn_impl_matches_sdpa`).

## Changing code → the doc contract

Any symbol rename/removal must update every citing doc in the same change:

```bash
python3 scripts/check_docs.py --coverage --links
```

- `FAIL ... not found` → a cited symbol changed; fix the citing doc.
- `UNCOVERED` → a new public symbol needs ≥1 `file.py:Symbol` anchor
  (DIFFUSION.md, the R-file, or its concept page).
- Anchors cite a file and symbol (as `models/mask.py` + the symbol name) —
  never line numbers (LINE-ANCHOR
  gate).

New concept docs follow the house format (see any
`docs/concepts/*.md`): reading-order header (Depends on / Read next), TOC,
shape annotations under every equation, tiny-dim worked example with an
explicit take-away, "what breaks if you change this" table, glossary
(symbols used ≥ 2×), embedded interview Q&A, and every load-bearing claim
citing its pinning test. TOC changes re-run the slug-collapse + fence-balance
checks (`SKILLS.md` documents the recipes).

## Tests and gates

```bash
uv run --python 3.13 python -m pytest -m "not gpu and not slow"   # 77 tests
python3 scripts/check_docs.py --coverage --links
python3 scripts/build_docs_html.py
```

New invariants get a pinning test in the same change; new public symbols get
an R-file row ([R2](../references/R2_transformer_api.md)–[R7](../references/R7_data_utils_api.md)).
`uv run --python 3.13` is mandatory on this box (python3.9 collection
failure — `SKILLS.md`).