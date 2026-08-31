# DiffusionGemma-Lite (scaffold — build in progress)

From-scratch PyTorch reimplementation of **DiffusionGemma** uniform-state block
diffusion, scaled to ~380M dense for a single A100 80GB (Chinchilla-optimal,
8.0B tokens). Not trained yet — see the design and execution plan docs under
`CoreProjects/llm-research/`.

**Status:** Phase 0 (pre-flight) + Phase 1 scaffold only. No model code exists
yet; the build follows `EXECUTION-PLAN-diffusiongemma-lite.md` strictly.