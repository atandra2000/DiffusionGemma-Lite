"""Headline speedup eval (DESIGN §4.2): AR baselines, ours, tokens/forward, speedup.

Examples:
  python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml \
      --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors
  python scripts/speedup_eval.py --config configs/pretrain_a100_380m.yaml --flops-only
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference.evaluate import SpeedupEvaluator  # noqa: E402
from models.transformer import DiffusionGemma, DiffusionGemmaConfig  # noqa: E402

DEFAULT_BASELINES = "fixed_T16,fixed_T32,adaptive_T32"


def load_model(args, device):
    model = DiffusionGemma(DiffusionGemmaConfig.from_yaml(args.config)).to(device).eval()
    if args.checkpoint:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=False)
        print(f"[speedup] weights: {args.checkpoint}")
    else:
        print("[speedup] no --checkpoint: random init (FLOP/forward counts are "
              "architecture-bound; treat tokens_per_sec as machine noise only)")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/pretrain_a100_380m.yaml")
    parser.add_argument("--checkpoint", default=None, help="safetensors weights path")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=1024)
    parser.add_argument("--baselines", default=DEFAULT_BASELINES,
                        help=f"comma-separated schedule rows (default: {DEFAULT_BASELINES})")
    parser.add_argument("--flops-only", action="store_true",
                        help="count forwards/token-forwards only; skip wall-clock")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = load_model(args, device)
    results = SpeedupEvaluator(model).evaluate(
        n_samples=args.n_samples, prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens, baselines=tuple(args.baselines.split(",")),
        wall_clock=not args.flops_only)

    cols = ["forwards", "token_forwards", "token_forwards_per_token",
            "tokens_per_forward", "tokens_per_sec"]
    header = f"{'row':<18}" + "".join(f"{c:>26}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, row in results["rows"].items():
        cells = "".join(
            f"{('—' if row[c] is None else f'{row[c]:.2f}'):>26}" for c in cols)
        print(f"{name:<18}{cells}")
    print("\nspeedup vs AR (tokens/forward):")
    for name, x in results["speedup_vs_ar_tokens_per_forward"].items():
        print(f"  {name:<14} {x:>8.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
