# -*- coding: utf-8 -*-
"""Is the XGBoost accuracy clinical signal, or is it labelling bookkeeping?

The model reaches ~0.81 top-1 on the test split, twice the best prompt architecture. Before
believing that number, this audit refits the *same* configuration on feature subsets defined
by the dataset's own data dictionary (``dataset/reports/data_dictionary.csv``, column
``type``) instead of by a hand-picked list:

- ``all`` -- every numeric feature (the published model);
- ``no_metadata`` -- without the ``metadata`` variables: the session clock, the staleness
  counters and the window bookkeeping, i.e. the ones that could fingerprint a case rather
  than describe its state;
- ``signals_only`` -- only ``signal`` and ``trend``: the window's measured aggregates and
  the recent trend of the vitals and the drug signals, i.e. the state at the bedside. No
  labs, no patient history, no ``metadata`` bookkeeping.

If the accuracy survives losing the bookkeeping -- and survives keeping only the measured
state -- the model is reading physiology, not the labelling machinery.

    python -m src.control.xgboost_audit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.control import ml_data, xgboost_baseline  # noqa: E402

#: The dataset's own dictionary: the taxonomy comes from there, not from me.
DEFAULT_DICTIONARY = Path(config.PROJECT_ROOT) / "dataset" / "reports" / "data_dictionary.csv"

#: Where the ablation models and their table go (the leading "_" keeps runs discoverable).
DEFAULT_MODEL_DIR = Path(config.RESULTS_DIR) / "control" / "_models" / "ablations"
DEFAULT_OUT_DIR = Path(config.RESULTS_DIR) / "control" / "_analysis"

#: One entry per refit.
VARIANTS = ("all", "no_metadata", "signals_only")

#: ``type`` values kept by each variant (``None``: keep everything).
#: ``trend`` carries the lowercase families (``hr_current``, ``map_slope_short``,
#: ``propofol_ce_current``...): the recent behaviour of the signals, not bookkeeping.
KEEP_TYPES = {"signals_only": ("signal", "trend")}

#: ``type`` values dropped by each variant.
DROP_TYPES = {"no_metadata": ("metadata",)}

COLUMNS = [
    "variant",
    "n_features",
    "accuracy_fit",
    "accuracy_valid",
    "accuracy_test",
    "best_iteration",
]


def variable_types(path=None):
    """``{variable: type}`` from the data dictionary."""
    path = Path(path) if path is not None else DEFAULT_DICTIONARY
    if not path.is_file():
        raise FileNotFoundError(f"the data dictionary is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["variable"]: row["type"]
            for row in csv.DictReader(handle)
            if row.get("variable") and row.get("type")
        }


def feature_subset(variant, names=None, types=None):
    """The feature names a variant keeps, in the canonical index order."""
    if variant != "all" and variant not in KEEP_TYPES and variant not in DROP_TYPES:
        raise ValueError(f"unknown variant: {variant!r} (expected one of {VARIANTS})")
    names = list(names if names is not None else ml_data.feature_names())
    if variant == "all":
        return names
    types = types if types is not None else variable_types()
    keep = KEEP_TYPES.get(variant)
    drop = DROP_TYPES.get(variant, ())
    selected = []
    for name in names:
        kind = types.get(name)
        if keep is not None and kind not in keep:
            continue
        if kind in drop:
            continue
        selected.append(name)
    return selected


def run_audit(variants=VARIANTS, dictionary=None, model_dir=None, out_dir=None, limit=None,
              cases=None, eval_limit=None, params=None, verbose=False):
    """Refit the model once per variant and write the comparison table; returns the rows."""
    model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    names = ml_data.feature_names()
    types = variable_types(dictionary)

    rows = []
    for variant in variants:
        subset = feature_subset(variant, names=names, types=types)
        metrics = xgboost_baseline.train(
            names=subset,
            model_path=model_dir / f"{variant}.json",
            meta_path=model_dir / f"{variant}.meta.json",
            limit=limit,
            cases=cases,
            eval_limit=eval_limit,
            params=params,
            verbose=verbose,
        )
        rows.append(
            {
                "variant": variant,
                "n_features": metrics["n_features"],
                "accuracy_fit": round(metrics["accuracy_fit"], 6),
                "accuracy_valid": (
                    round(metrics["accuracy_valid"], 6)
                    if metrics["accuracy_valid"] is not None
                    else ""
                ),
                "accuracy_test": round(metrics["accuracy_test"], 6),
                "best_iteration": metrics["best_iteration"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "xgboost_ablation.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, table


def build_parser():
    """Command-line interface of the audit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variants", default=",".join(VARIANTS), help=f"which refits to run ({', '.join(VARIANTS)})"
    )
    parser.add_argument("--cases", type=int, default=None, help="use only the first N train cases")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of fit windows")
    parser.add_argument("--eval-limit", type=int, default=None, help="cap the windows scored")
    parser.add_argument("--model-dir", default=None, help="where to write the ablation models")
    parser.add_argument("--out-dir", default=None, help="where to write the table")
    parser.add_argument("--verbose", action="store_true", help="show the boosting progress")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows, table = run_audit(
        variants=[name.strip() for name in args.variants.split(",") if name.strip()],
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        cases=args.cases,
        eval_limit=args.eval_limit,
        verbose=args.verbose,
    )
    print(f"{'variant':16s} {'features':>8s} {'fit':>7s} {'valid':>7s} {'test':>7s}")
    for row in rows:
        valid = f"{row['accuracy_valid']:.4f}" if row["accuracy_valid"] != "" else "-"
        print(
            f"{row['variant']:16s} {row['n_features']:8d} {row['accuracy_fit']:.4f} "
            f"{valid:>7s} {row['accuracy_test']:.4f}"
        )
    print(f"\ntable: {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
