# -*- coding: utf-8 -*-
"""Fit the XGBoost control and cache it under ``results_harness/control/_models/``.

    python -m src.control.train_xgboost
    python -m src.control.train_xgboost --limit 5000    # a quick model, for a smoke test

    python harness/run.py --strategy xgboost --split test --workers 8   # then score it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.control import ml_data, xgboost_baseline  # noqa: E402


def build_parser():
    """Command-line interface of the trainer."""
    parser = argparse.ArgumentParser(description="Train the XGBoost control on the train split.")
    parser.add_argument("--cases", type=int, default=None, help="use only the first N train cases")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of fit windows")
    parser.add_argument(
        "--eval-limit", type=int, default=None, help="cap the number of windows scored at the end"
    )
    parser.add_argument("--model-path", default=None, help="where to write the model")
    parser.add_argument("--verbose", action="store_true", help="show the boosting progress")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    metrics = xgboost_baseline.train(
        cases=args.cases,
        limit=args.limit,
        eval_limit=args.eval_limit,
        model_path=args.model_path,
        verbose=args.verbose,
    )
    print(
        f"features {metrics['n_features']} | fit {metrics['n_fit']} windows "
        f"({metrics['accuracy_fit']:.4f}) | validation {metrics['n_valid']} "
        f"({metrics['accuracy_valid']:.4f}) | test {metrics['n_test']} "
        f"({metrics['accuracy_test']:.4f})"
    )
    print(f"best iteration: {metrics['best_iteration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
