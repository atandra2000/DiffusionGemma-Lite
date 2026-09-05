# Archify delivery evidence — DiffusionGemma-Lite

The [interactive visual guide](diffusiongemma_visual_guide.html) links five standalone showcase architecture, dataflow, and workflow diagrams.

All five: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light/dark. All required viewport measurements passed horizontal/vertical containment, minimum projected text size and viewer-control clearance.

## Artifact bindings

### System Overview

- Diagram type: `architecture`
- Output: [diffusiongemma-system.html](diffusiongemma-system.html)
- Specification: `docs/diagrams/diffusiongemma-system.architecture.json`
- Artifact SHA-256: `c9ab4f78e54f387522395195f543ce7ac5a9ba9c9845fca42420f6fadb328301` (715,789 bytes)
- [Browser receipt](diffusiongemma-system.visual-check.json) · [Screenshot contact sheet](diffusiongemma-system.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Forward Pass & Block Attention

- Diagram type: `architecture`
- Output: [diffusiongemma-forward-pass.html](diffusiongemma-forward-pass.html)
- Specification: `docs/diagrams/diffusiongemma-forward-pass.architecture.json`
- Artifact SHA-256: `990af736500434091ecd96836c9488270aed4822c8b3676b2f3369cb6c619e11` (714,975 bytes)
- [Browser receipt](diffusiongemma-forward-pass.visual-check.json) · [Screenshot contact sheet](diffusiongemma-forward-pass.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Data Pipeline & Block Packing

- Diagram type: `dataflow`
- Output: [diffusiongemma-data-pipeline.html](diffusiongemma-data-pipeline.html)
- Specification: `docs/diagrams/diffusiongemma-data-pipeline.dataflow.json`
- Artifact SHA-256: `2578268bbcc6b6c9cb33bf9458d06cd5f3f778e7c95df1d3cb61b3aaca634f10` (711,840 bytes)
- [Browser receipt](diffusiongemma-data-pipeline.visual-check.json) · [Screenshot contact sheet](diffusiongemma-data-pipeline.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Discrete Diffusion Sampler

- Diagram type: `workflow`
- Output: [diffusiongemma-sampler.html](diffusiongemma-sampler.html)
- Specification: `docs/diagrams/diffusiongemma-sampler.workflow.json`
- Artifact SHA-256: `929b8f9b3345c06417ec8f1bb466c9af83df881ffbf47c56471860d667b6b5fe` (718,972 bytes)
- [Browser receipt](diffusiongemma-sampler.visual-check.json) · [Screenshot contact sheet](diffusiongemma-sampler.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Training Pipeline & Loss Rollout

- Diagram type: `workflow`
- Output: [diffusiongemma-training.html](diffusiongemma-training.html)
- Specification: `docs/diagrams/diffusiongemma-training.workflow.json`
- Artifact SHA-256: `1270350232e9d70d56aba57a0255f86ea58ac5cf9f2caf3bfda7e39f5d08618a` (717,737 bytes)
- [Browser receipt](diffusiongemma-training.visual-check.json) · [Screenshot contact sheet](diffusiongemma-training.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

## Verification limits

Parameter count: exactly 343,516,160 parameters (~343.5M). Trained on 8.0B tokens with uniform-state discrete diffusion (D3PM), block-causal attention across 256-token canvases, and self-conditioning. Single A100 80GB baseline budget estimated at ~40–50 hours.
