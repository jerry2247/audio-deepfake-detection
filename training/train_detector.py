from __future__ import annotations

import argparse
import sys

from .engine import TrainConfig, fit_detector, metrics_to_json
from .modeling import ModelAccessError
from .specs import METHODS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one cached-feature audio deepfake detector head.",
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--cache-root", default="training")
    parser.add_argument("--output-root", default="final_models")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainConfig(
        cache_root=args.cache_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        use_amp=not args.no_amp,
    )
    try:
        metrics = fit_detector(args.method, config)
    except ModelAccessError as exc:
        print(f"MODEL ACCESS ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"TRAINING ERROR: {exc}", file=sys.stderr)
        return 1

    print(metrics_to_json(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
