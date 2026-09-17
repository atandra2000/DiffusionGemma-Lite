# DiffusionGemma diagram evidence

Source revision: `83acb7cc6e3a597800cb6ae3be8ad673b100e6f3`. Configuration: `configs/pretrain_a100_380m.yaml`.

All five corrected specs: 9/9 showcase checks, zero composition errors/warnings. Fresh automated browser evidence passed at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light theme, with light/dark endpoint captures. Exact HTML hashes match their browser receipts.

Perceptual visual review: **skipped, image input unavailable**. Proposed 12px essential / 11px secondary target: **not met**. No interaction/export or guide mobile acceptance claim. No GPU, corpus-inventory or trained-quality measurements.

## diffusiongemma-data-pipeline

- Spec: `diffusiongemma-data-pipeline.dataflow.json`
- Spec SHA-256: `64b27ff99f4e05a574ad192f5a0b31cf9d8e16b8168aabc641fc9bbb92942fa4`
- HTML SHA-256: `2663a1e534d5c43fa79367f010830ddcdd6c78c0afe7436305b4832703adc120`
- Browser receipt: `diffusiongemma-data-pipeline.visual-check.json`
- Minimum recorded node text: 7 px.

## diffusiongemma-forward-pass

- Spec: `diffusiongemma-forward-pass.architecture.json`
- Spec SHA-256: `0a1053157f3cf7de0e8ab289153f4de5f689bbc229d41e143ca4779b5d878890`
- HTML SHA-256: `9f7a4afd02749919237268d5ecc85c862247f7fc4dc49f0867ccf268e50aed9d`
- Browser receipt: `diffusiongemma-forward-pass.visual-check.json`
- Minimum recorded node text: 6.665255474452555 px.

## diffusiongemma-sampler

- Spec: `diffusiongemma-sampler.workflow.json`
- Spec SHA-256: `a6b3fafb88ee04c909fe7a72ce0339bab8afa725d58060fc5ae03ca2f47961b1`
- HTML SHA-256: `20cda9de4cdd32ab7b7cfb1097784d4a89e3230d928fc98effd15896e17a115e`
- Browser receipt: `diffusiongemma-sampler.visual-check.json`
- Minimum recorded node text: 8 px.

## diffusiongemma-system

- Spec: `diffusiongemma-system.architecture.json`
- Spec SHA-256: `dd87554ec79584c2c0f113ca26018140bda854dded6a9d58f5bfb7d9c5c7683a`
- HTML SHA-256: `e3033a5246af95a82dde7f49d7647c38bcfa3321164a2166f6ff957aed932825`
- Browser receipt: `diffusiongemma-system.visual-check.json`
- Minimum recorded node text: 6.071739130434782 px.

## diffusiongemma-training

- Spec: `diffusiongemma-training.workflow.json`
- Spec SHA-256: `3eb514d5ed91d24fa454bff0f94e5c621384e475d254d43fe0de761195fbd4b0`
- HTML SHA-256: `f7090417a87dadf31bc18a192ca75c618d89b026e680039448044b7e3b2076a3`
- Browser receipt: `diffusiongemma-training.visual-check.json`
- Minimum recorded node text: 8 px.

## Source corrections

Time conditioning enters backbone embeddings. Training CE consumes hidden states/weights/clean targets, separately from sampling posterior. Copied int64 windows and dtype uncertainty are disclosed. Finite-loss and accumulation gates, rollback/abort and continuation are drawn. Sampler denoise/next-canvas loops, immutable commitment, temperature and strict consecutive entropy stopping match source. Checkpoint multi-file writes and resume limits are explicit. The obsolete 1000-step guide simulator and stale symbols were removed.
