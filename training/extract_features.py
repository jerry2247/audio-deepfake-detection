from __future__ import annotations

import argparse
import sys

from .engine import TrainConfig, extract_feature_cache, metrics_to_json
from .modeling import ModelAccessError
from .specs import METHODS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact frozen-backbone features for one detector method.",
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--cache-root", default="training")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = TrainConfig(
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        print_progress=True,
    )
    try:
        summary = extract_feature_cache(
            args.method,
            config,
            splits=tuple(args.splits),
            overwrite=args.overwrite,
        )
    except ModelAccessError as exc:
        print(f"MODEL ACCESS ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"FEATURE EXTRACTION ERROR: {exc}", file=sys.stderr)
        return 1

    print(metrics_to_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

