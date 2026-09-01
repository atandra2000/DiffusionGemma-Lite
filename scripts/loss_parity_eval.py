"""Quality anchor (DESIGN §4.2(3)): held-out x0-NLL on a shard, Δ vs AR baseline.

Examples:
  python scripts/loss_parity_eval.py --shard data/pretrain_chinchilla/shards/shard_90000.bin \
      --config configs/pretrain_a100_380m.yaml --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
  python scripts/loss_parity_eval.py --shard <heldout.bin> --ar-nll 3.21

The AR side is a number (AR baseline's CE on the same shard, nats/token), not a
cross-repo model load — pass --ar-nll from the baseline run; without it the
script prints the diffusion NLL and marks the delta an honest gap. Acceptance
(DESIGN §4.2): within +5% of the AR number, disclosed either way.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference.evaluate import heldout_x0_nll  # noqa: E402
from models.transformer import DiffusionGemma, DiffusionGemmaConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", required=True, help="held-out uint32 shard file")
    parser.add_argument("--config", default="configs/pretrain_a100_380m.yaml")
    parser.add_argument("--checkpoint", default=None, help="safetensors weights path")
    parser.add_argument("--ar-nll", type=float, default=None,
                        help="AR baseline CE (nats/token) on the same shard")
    parser.add_argument("--n-windows", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="window length (default: model max_seq_len)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionGemma(DiffusionGemmaConfig.from_yaml(args.config)).to(device).eval()
    if args.checkpoint:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=False)
        print(f"[parity] weights: {args.checkpoint}")

    nll = heldout_x0_nll(model, args.shard, n_windows=args.n_windows,
                         seq_len=args.seq_len, seed=args.seed)
    print(f"\nheld-out x0-NLL ({args.n_windows} windows, seed {args.seed}): "
          f"{nll:.4f} nats/token")
    if args.ar_nll is None:
        print("delta vs AR: not computed (pass --ar-nll from the AR baseline run)")
        return 0
    delta = nll - args.ar_nll
    rel = delta / args.ar_nll
    print(f"AR baseline CE: {args.ar_nll:.4f}  ->  ΔNLL: {delta:+.4f} ({rel:+.2%})")
    verdict = "WITHIN +5% acceptance" if rel <= 0.05 else "EXCEEDS +5% acceptance"
    print(f"quality anchor ({verdict}): {nll:.4f} vs {args.ar_nll:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
