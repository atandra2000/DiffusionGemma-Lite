# Guide: sampler tuning

How to choose `SamplerConfig` values and what each knob trades. Semantics:
[sampler](../concepts/sampler.md); API: [R6](../references/R6_sampler_eval_api.md).

## The knobs

| knob | default | range | effect |
|---|---|---|---|
| `n_diffusion_steps` | 32 | 16–48 | quality/speed dial: tokens/forward = 256/(T+1) — 15.06× at T=16, 7.76× at 32 ([sampler §5](../concepts/sampler.md)) |
| `adaptive` | True | — | entropy bond on/off; off = fixed schedule (`test_adaptive_off_ignores_entropy`) |
| `entropy_threshold` | 1.0 nat | 0.5–2.0 | lower = stricter stop, more steps; 1.0 ≈ 2.7 effective tokens of uncertainty |
| `stability_steps` | 2 | 1–4 | consecutive low-entropy steps required — higher = later stop |
| `temp_start` / `temp_end` | 0.8 / 0.4 | 0.3–1.0 | linear anneal within a canvas; lower = more argmax-like draws ([sampler §3](../concepts/sampler.md)) |
| `seed` | None | int | reproducible decode (generator seeded once per sampler) |

## Recipes

**Fastest acceptable decode** — `n_diffusion_steps=16, adaptive=True`:
the T=16 fixed schedule already yields 15.06×; the entropy bond only helps
at higher T (it can only cut forwards, never add). Start here.

**Quality-first** — `n_diffusion_steps=32, adaptive=True,
entropy_threshold=0.7, stability_steps=3`: stricter bond, more refinement
steps, still bounded by 33 forwards/canvas.

**Reproducible decode** — `seed=0` on the `SamplerConfig`:
corruption + Gumbel draws replay exactly
(`inference/generate.py:BlockDiffusionSampler._generator`).

**Ablation parity** — `adaptive=False, n_diffusion_steps=16` must produce
the fixed-schedule row exactly (`parse_baseline("fixed_T16")`,
`tests/test_inference.py::test_parse_baseline`); use it as the control arm
when abimating the adaptive path.

## What NOT to tune

- `temp_start=0` (kill exploration entirely) — the first step's argmax
  freeze cascades; quality collapses to greedy-with-renoise.
- `entropy_threshold` below ~0.5 — the bond almost never fires; you paid
  for adaptive and got fixed.
- `stability_steps > 4` — the schedule usually ends before the streak
  accumulates; the bond stops firing.

All changes re-run `tests/test_sampler.py` (commit-rule monotonicity,
Gumbel equivalence, adaptive bounds) and
`tests/test_inference.py::test_flop_counter_adaptive_le_fixed`.