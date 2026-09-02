#!/usr/bin/env bash
# Launch DiffusionGemma-Lite pre-training on an A100 80GB pod (plan §4.1, SKILLS Skill 2).
#
# Model & Schedule: 343.5M params, 61,000 steps @ seq 4096 (8.0B tokens) —
# configs/pretrain_a100_380m.yaml.
# Resumes automatically from the latest complete checkpoint in
# checkpoints/pretrain_a100 (utils/checkpoint.py:CheckpointManager).
#
# Before launching: ensure shards are packed (python3 data/prepare_data.py)
# and boundary checks pass:
#   python3 scripts/microbench_a100.py
#   python3 scripts/step_time_a100.py
#   python3 scripts/e2e_gpu_smoke.py
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Pre-flight: checking GPU availability ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required on A100 pod'"

echo "=== Pre-flight: running boundary checks ==="
python3 scripts/microbench_a100.py
python3 scripts/step_time_a100.py --compile
python3 scripts/e2e_gpu_smoke.py --steps 20

echo "=== Launching 8.0B-token pretraining ==="
exec python3 -m training.pretrain --config configs/pretrain_a100_380m.yaml
