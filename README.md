# DiffusionGemma-Lite

A from-scratch PyTorch implementation of **uniform-state block-diffusion LM**:
a Gemma-class dense transformer that generates text **256 tokens at a time**,
denoising each canvas in parallel from uniform noise over a small number of
diffusion steps.

**~343.5M params (exactly 343,516,160) · 8.0B Chinchilla-optimal tokens · target 14–18 h on a single A100 80GB · canvas 256 · GPT-2 BPE**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch%202.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://www.pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3DDC84?logo=apache&logoColor=white)](LICENSE)
[![GPU: A100 80GB](https://img.shields.io/badge/GPU-A100%2080GB-76B900?logo=nvidia&logoColor=white)](#-status-measured-vs-honest-gaps)

[**Architecture**](#-architecture) · [**Headline metric**](#-headline-metric) · [**Quick start**](#-quick-start) · [**Docs**](#-documentation)

---

## 📖 Overview

**DiffusionGemma-Lite** is a **block-autoregressive (block-AR) discrete diffusion**
language model, built end-to-end from scratch — no diffusion library, no
`transformers`, no custom CUDA. Three components carry the design:

1. **Uniform-state diffusion over the vocabulary.** Each 256-token canvas is
   corrupted by D3PM-style uniform noise with a cosine schedule
   (`models/diffusion.py:q_sample`); the transformer predicts the clean tokens
   **x0** directly (`training/losses.py:chunked_x0_ce`). The "noise" is random
   valid tokens, so partial answers can be committed while the rest re-noises.
2. **Block-causal attention.** One dense GQA transformer does two jobs under
   one mask (`models/mask.py:build_block_causal_mask`): bidirectional within a
   canvas (the parallel denoising win), causal across canvases (a valid
   left-to-right LM factorization).
3. **Entropy-bounded adaptive stopping.** A closed rule — stop refining a
   canvas when its posterior entropy settles — cuts eval forwards without any
   learned halting policy (`inference/generate.py:denoise_canvas`).

Plus self-conditioning with zero-init equivalence
(`models/selfcond.py:SelfConditioning`): the model sees its previous step's
posterior re-embedded at the input, and provably starts as if self-conditioning
didn't exist.

### How it compares to the rest of the portfolio

| Project | Generation | KV cache growth | Parallel decode |
|---|---|---|---|
| [LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite) | token AR | 1/token | ❌ |
| [DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) | token AR (MLA+MoE) | 1/token | ❌ |
| [Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite) | token AR (SSM state) | 1/token | ❌ |
| **DiffusionGemma-Lite** | **block-AR diffusion** | **1 / finalized canvas (256 tok)** | **✅ 256 tokens/step** |

---

## 🏆 Headline metric

> **Decode throughput at acceptable quality:** one forward pass refines a
> 256-token canvas; `fixed_T32` spends `T+1` forwards per canvas (≈7.8
> tokens/forward at production settings) and entropy-bounded adaptive stopping
> targets a further ~2× eval-FLOP cut — vs. exactly 1 token/forward for a
> KV-cached AR decoder.

**Measured vs honest gaps.** The forward-count arithmetic above is verified by
`tests/test_inference.py` (FLOP counter: adaptive ≤ fixed, closed form pinned).
Not yet measured (A100 pod session pending): wall-clock tokens/s vs the
LLaMA-3-Lite AR baseline, peak VRAM vs the §4.0 budget, MFU ≥ 33%, and the
loss-parity verdict on a trained checkpoint (`scripts/loss_parity_eval.py
--ar-nll <AR CE>`, acceptance within +5%). These print as explicit honest gaps
until the trained run exists — see the SDD ledger for phase gating.

---

## 🏗 Architecture

```
GPT-2 BPE tokens (vocab 50,257)
    │
    ▼
Embedding (d_model=1024)  ← weight-tied with LM head
    │  + per-canvas time embedding (t/T, sinusoidal → 256)
    ▼
24 × Denoise Blocks:
    ┌─────────────────────────────────────────────────────────┐
    │ RMSNorm → GQA attention (16Q/4KV, head_dim 64, RoPE)    │
    │           mask: causal across canvases,                 │
    │                 bidirectional within a canvas           │
    │ RMSNorm → SwiGLU FFN (ffn_dim 3072)                     │
    └─────────────────────────────────────────────────────────┘
    │  (+ zero-init self-conditioning add on the loss path)
    ▼
chunked x0-CE loss (vocab chunks of 8192 — never materializes (B,T,V) logits)
```

- **Canvas:** `canvas_len=256`, `max_seq_len=4096` (16 canvases).
- **Diffusion:** train `T=16` per-canvas timesteps, eval `T≤32`; cosine
  `ᾱ(t) = cos²(π/2·t/T)`; uniform-token corruption; direct x0 prediction.
- **Params:** exactly 343,516,160 under the mandated GQA + weight tying.
  (Early planning docs said "~380M" — their param table counted full-MHA K/V;
  the honest number is 343.5M, derived digit-for-digit in the SDD ledger.)

---

## 🚀 Quick start

```bash
# 1. Data (workspace shared_data pipeline; writes data/pretrain_chinchilla/shards/)
python data/prepare_data.py --skip-download      # 2-shard synthetic check

# 2. Training (single A100; CPU is smoke-test only)
python -m training.pretrain --config configs/pretrain_a100_380m.yaml

# 3. Headline evaluation (schedule rows + FLOP counter)
python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
python scripts/speedup_eval.py --flops-only

# 4. Quality anchor (AR CE from the baseline run on the same shard)
python scripts/loss_parity_eval.py --shard data/pretrain_chinchilla/shards/shard_90000.bin \
    --checkpoint ... --ar-nll 3.21

# 5. Generate
python -c "
from models.transformer import DiffusionGemma, DiffusionGemmaConfig
import torch
m = DiffusionGemma(DiffusionGemmaConfig.from_yaml('configs/pretrain_a100_380m.yaml'))
print(m.generate(torch.randint(0, 50257, (1, 64)), max_new_tokens=1024).shape)
"
```

```bash
python3 -m pytest -m "not gpu and not slow"   # CPU-friendly suite (63 tests)
python scripts/check_docs.py --coverage        # doc↔code anchor gate
```

---

## 📚 Documentation

| doc | contents |
|---|---|
| [`DIFFUSION.md`](DIFFUSION.md) | **the authoritative technical doc** — diffusion math, mask derivation, self-conditioning, sampler semantics, training recipe, recipe deltas vs upstream |
| [`docs/README.md`](docs/README.md) | doc map |
| `docs/concepts/` | diffusion core, block-causal attention, self-conditioning |
| `docs/guides/` | quickstart, debugging playbook |
| `docs/references/` | config reference, API reference |
| [`AGENTS.md`](AGENTS.md) | coding-agent contract (rules, caveats) |
| [`SKILLS.md`](SKILLS.md) | day-to-day developer workflows |

---

## 📊 Status: measured vs honest gaps

| item | status |
|---|---|
| Diffusion core, sampler, training loop, eval harness | ✅ implemented, 63 CPU tests green |
| Chunked-CE ≡ eager CE at production vocab | ✅ measured: max abs diff **0.0**, grads ≤ 1.7e-7 |
| 100-step smoke descent + resume bit-equality | ✅ measured on CPU |
| **8.0B-token A100 training run** | ❌ not started (Task 15 scripts ready for the pod) |
| **MFU ≥ 33%, peak VRAM < 15 GB, wall-clock speedup vs AR** | ❌ GPU gates pending |
| **Loss parity vs AR baseline (± disclosure)** | ❌ needs trained checkpoint + baseline run |

The AR-baseline side of the headline is analytic (1 token/forward by
construction); wall-clock speedup and the +5% quality verdict require the
external LLaMA-3-Lite baseline run on the same shard — the scripts expose
those inputs and print the gap when absent.

## 📄 License

Apache-2.0 — see [`LICENSE`](LICENSE).