# -*- coding: utf-8 -*-
"""How collinear are the dataset's own labels with the window features?

The ``result_*`` variables never enter a model: they live in ``window.output`` while every
control -- the gradient boosting one included -- reads ``window.input``. The VIF report covers
the 234 input features only, so it cannot answer "which feature is most collinear with the
labels?". This module answers it on the train split:

- ``label_collinearity.csv`` -- |Pearson r| of every feature with every label target;
- ``label_vif.csv`` -- the VIF the label would have if it were added to the design matrix,
  i.e. ``1 / (1 - R^2)`` of the label regressed on all the features, plus the feature that
  correlates with it the most.

The targets are the five variables the data dictionary declares as ``output``: the two ratios
(continuous), and the three action labels turned into one-vs-rest indicators (``no_action``,
``reduce_opioid``...), so a categorical label becomes readable as a number.

Two caveats, both inherited from the VIF report:

- missing values are imputed with the column mean (what the few-shot index does too), so a
  feature that is mostly missing is mostly a constant;
- the VIF is a **linear** view. The label is a decision rule with thresholds, so a moderate
  ``R^2`` here is consistent with a model that predicts 0.81 of it: that is the non-linear part.

    python -m src.control.label_collinearity
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, dataset  # noqa: E402
from src.control import ml_data, xgboost_diagnostics  # noqa: E402

#: Where the reports go (the leading "_" keeps them out of the run scan).
DEFAULT_OUT_DIR = Path(config.RESULTS_DIR) / "control" / "_analysis"

#: The labels, as the data dictionary lists them.
RATIO_LABELS = ("result_1", "result_aux")
ACTION_LABELS = ("result_1", "result_aux", "result_real")

#: A correlation needs at least this many complete pairs to be reported.
MIN_OBSERVATIONS = 30

COLLINEARITY_COLUMNS = ["target", "feature", "abs_corr", "rank"]

VIF_COLUMNS = ["target", "n_windows", "n_features", "r2", "vif", "top_feature", "top_abs_corr"]


def action_of(window, key):
    """The action a label holds: ``result_real`` is a string, the others a dict."""
    if key == "result_real":
        return window.output.get("result_real")
    value = window.output.get(key)
    return value.get("action") if isinstance(value, dict) else None


def label_targets(windows, actions=None):
    """``{target: values}``: the dataset's labels as numeric targets, one value per window.

    Missing labels are ``NaN`` (a window without ``result_aux`` gets no indicator for it).
    """
    actions = list(actions if actions is not None else dataset.ACTIONS)
    targets = {}
    for key in RATIO_LABELS:
        targets[f"{key}_ratio"] = np.asarray(
            [dataset.get_ratio(window.output.get(key)) for window in windows], dtype=float
        )
    for key in ACTION_LABELS:
        labels = np.asarray([action_of(window, key) for window in windows], dtype=object)
        for action in actions:
            targets[f"{key}__{action}"] = np.asarray(
                [np.nan if label is None else float(label == action) for label in labels],
                dtype=float,
            )
    return targets


def correlation(column, values, mask):
    """Pearson ``r`` of one feature with one target over their complete pairs, or ``NaN``."""
    if mask.sum() < MIN_OBSERVATIONS:
        return np.nan
    x = np.asarray(column, dtype=float)[mask]
    y = values[mask]
    complete = ~np.isnan(x)
    if complete.sum() < MIN_OBSERVATIONS:
        return np.nan
    x, y = x[complete], y[complete]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def explained_variance(X, values, mask):
    """``R^2`` of the target regressed on the whole feature matrix (least squares)."""
    Z, kept = xgboost_diagnostics.design_matrix(X[mask])
    y = values[mask]
    if not len(y) or np.isnan(y).any() or y.std() == 0:
        return np.nan, len(kept)
    design = np.hstack([np.ones((len(y), 1)), Z])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ solution
    total = y - y.mean()
    return float(1.0 - (residual @ residual) / (total @ total)), len(kept)


def load_windows(limit=None, cases=None):
    """The train windows the report reads, optionally capped by cases and by count."""
    if cases is None:
        return dataset.load_split(ml_data.FIT_SPLIT, limit=limit)
    windows = [
        window
        for path in dataset.case_files(ml_data.FIT_SPLIT)[:cases]
        for window in dataset.load_case(path)
    ]
    return windows[:limit] if limit is not None else windows


def collinearity(X=None, names=None, targets=None, limit=None, cases=None):
    """``(rows, vif_rows)``: the two reports of the label/feature collinearity."""
    names = list(names if names is not None else ml_data.feature_names())
    if X is None or targets is None:
        windows = load_windows(limit=limit, cases=cases)
        X = ml_data.matrix(windows, names)
        targets = label_targets(windows)

    rows = []
    summary = []
    for target_name, values in targets.items():
        mask = ~np.isnan(values)
        correlations = [correlation(X[:, index], values, mask) for index in range(len(names))]
        order = np.argsort([np.abs(value) if not np.isnan(value) else -1.0
                            for value in correlations])[::-1]
        for rank, index in enumerate(order[: len(names)], start=1):
            if np.isnan(correlations[index]):
                break
            rows.append(
                {
                    "target": target_name,
                    "feature": names[index],
                    "abs_corr": abs(round(correlations[index], 6)),
                    "rank": rank,
                }
            )
        r2, n_features = explained_variance(X, values, mask)
        best = order[0] if len(order) and not np.isnan(correlations[order[0]]) else None
        explained = not np.isnan(r2)
        summary.append(
            {
                "target": target_name,
                "n_windows": int(mask.sum()),
                "n_features": n_features,
                "r2": round(r2, 4) if explained else "",
                # a perfect fit has no finite VIF: report nothing rather than "inf"
                "vif": (
                    round(1.0 / (1.0 - r2), 2) if explained and r2 < 1.0 - 1e-9 else ""
                ),
                "top_feature": names[best] if best is not None else "",
                "top_abs_corr": round(abs(correlations[best]), 4) if best is not None else "",
            }
        )
    return rows, summary


def run(out_dir=None, limit=None, cases=None):
    """Write both reports and return their rows."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = collinearity(limit=limit, cases=cases)
    with (out_dir / "label_collinearity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLLINEARITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "label_vif.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIF_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)
    return out_dir, rows, summary


def build_parser():
    """Command-line interface of the label/feature collinearity report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=None, help="use only the first N train cases")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of windows")
    parser.add_argument("--out-dir", default=None, help="where to write the reports")
    parser.add_argument("--top", type=int, default=5, help="features to print per target")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir, rows, summary = run(out_dir=args.out_dir, limit=args.limit, cases=args.cases)
    for entry in summary:
        best = [row for row in rows if row["target"] == entry["target"]][: args.top]
        print(
            f"{entry['target']:34s} n={entry['n_windows']:6d}  R2={str(entry['r2']):>7s} "
            f"VIF={str(entry['vif']):>9s}"
        )
        print(
            "    "
            + ", ".join(f"{row['feature']} {row['abs_corr']:.3f}" for row in best)
        )
    print(f"\nreports: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
