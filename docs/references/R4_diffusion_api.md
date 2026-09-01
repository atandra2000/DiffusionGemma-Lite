# R4 — Diffusion core & self-conditioning API

> Public symbols of `models/diffusion.py`, `models/selfcond.py` — the
> corruption process and the conditioning path. Derivations:
> [diffusion-core](../concepts/diffusion-core.md),
> [self-conditioning](../concepts/self-conditioning.md).

## `models/diffusion.py`

| symbol | signature | returns | contract |
|---|---|---|---|
| `alpha_bar` | `(t: torch.Tensor, T: int) -> torch.Tensor` | ᾱ(t) = cos²(π/2·t/T), shape of `t` | monotone 1→0; ᾱ(0)=1, ᾱ(T)≈0 — `tests/test_diffusion.py::test_alpha_bar_monotone_and_bounded` |
| `corruption_probs` | `(alpha_bar_t: float, vocab_size: int, clean_token: int) -> torch.Tensor` | (V,) row of q(xt\|x0) | mass ᾱ on clean token + (1−ᾱ)/V uniform elsewhere — `test_forward_process_marginals` |
| `sample_canvas_t` | `(batch, n_canvases, T, device, generator=None)` | (B, n_canvases) ints | t ~ U{1..T} **per canvas** — the per-canvas-t ruling |
| `q_sample` | `(x0, t, canvas_len, n_diffusion_steps, vocab_size, generator=None)` | `(xt, alpha (B, seq, 1))` | keep x0 w.p. ᾱ(t) else uniform draw; t=0 identity, t=T pure noise — `test_q_sample_*` |
| `x0_ce_loss` | `(logits, x0)` | scalar | **eager reference** CE against x0 (not xt); the test-only twin of `training/losses.py:chunked_x0_ce` — never consolidate (AGENTS.md §2) |

ᾱ(t) table for T=16 (computed): t=1 → 0.9936, 2 → 0.9745, 3 → 0.9455,
4 → 0.9045, 5 → 0.8536, 6 → 0.7500, 7 → 0.6402, 8 → 0.5000, 9 → 0.3536,
10 → 0.2500, 11 → 0.1464, 12 → 0.1464, 13 → 0.0955, 14 → 0.0545, 15 → 0.0254,
16 → 0.0063 — full table with derivation in
[diffusion-core §2](../concepts/diffusion-core.md).

## `models/selfcond.py`

| symbol | signature | contract |
|---|---|---|
| `SelfConditioning.__init__` | `(d_model: int)` | Linear `d_model → d_model`, weight **and** bias zero-init |
| `SelfConditioning.embed` | `(h_norm, embed_weight)` | `softmax(h @ Eᵀ) @ E` — posterior re-embedding (B, S, d_model); `test_embed_is_softmax_weighted_mean` |
| `SelfConditioning.forward` | `(h_norm, sc_input, embed_weight)` | conditioning add: `h + proj(sc)`; zero-init ⇒ bit-exact no-op at init — `test_zero_init_equivalence` (fp64, atol=0) |

Routing contract: exactly one `proj` add per path —
`models/transformer.py:DiffusionGemma.final_hidden` (loss path) and
`.head_forward` (head path) each apply it once via
`_conditioned_hidden`; composing the two double-conditions
(`tests/test_models.py::test_two_step_overfit`).

The training pre-pass uses the chunked twin
(`training/losses.py:chunked_p_embed`) under `no_grad` —
[self-conditioning §6](../concepts/self-conditioning.md).