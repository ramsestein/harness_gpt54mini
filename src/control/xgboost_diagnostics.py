# -*- coding: utf-8 -*-
"""Diagnostics of the XGBoost control: what carries the decision, and how collinear it is.

Two reports, both written under ``results_harness/control/_analysis/``:

- ``shap_importance.csv`` -- mean |SHAP| per feature (and per predicted class) over a sample
  of test windows. It answers "which variables move this model, and how much", in the model's
  own units.
- ``vif.csv`` -- the variance inflation factor of every feature, from the inverse correlation
  matrix of the train matrix, plus a per-family summary. The dataset derives many features
  from the same signals (``X_current``/``X_mean``/``X_min``/``X_max``/``X_slope``), so
  severe collinearity is expected *by construction*; the table measures it instead of
  assuming it.

    python -m src.control.xgboost_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.control import ml_data, xgboost_baseline  # noqa: E402

#: Where the reports go (the leading "_" keeps them out of the run scan).
DEFAULT_OUT_DIR = Path(config.RESULTS_DIR) / "control" / "_analysis"

#: Test windows explained by SHAP (tree traversal is linear in this number).
DEFAULT_SAMPLE = 1500

#: Correlations above this are treated as duplicated columns.
DUPLICATE_R = 0.999

SHAP_COLUMNS = ["feature", "mean_abs_shap", "rank"]

VIF_COLUMNS = ["feature", "family", "vif", "max_abs_corr_with", "max_abs_corr", "duplicate_of"]

FAMILY_COLUMNS = ["family", "n_features", "median_vif", "max_vif", "n_above_10", "n_above_100"]


def family_of(name):
    """The feature family: the prefix before the first ``_`` (``bis_slope_short`` -> ``bis``).

    ``lab_k`` -> ``lab``, ``propofol_ce_current`` -> ``propofol``, ``pt_age`` -> ``pt``.
    """
    return str(name).split("_")[0].lower()


def design_matrix(X, min_observed=2):
    """``(Z, kept)``: ``NaN`` imputed with the column mean, then standardised.

    Imputing with the mean is what the few-shot index does too, so the diagnostics describe
    the representation the controls actually see. Caveat worth remembering when reading the
    table: a column that is missing for most of its rows becomes nearly constant after
    imputation, so its VIF partly describes the imputation. Columns with no variance (or
    almost no data) are dropped: they carry nothing and make the correlation matrix singular.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("the feature matrix is empty")
    counts = np.sum(~np.isnan(X), axis=0)
    means = np.nanmean(np.where(counts > 0, X, 0.0), axis=0)
    means = np.nan_to_num(means, nan=0.0)
    Z = np.where(np.isnan(X), means, X)
    stds = Z.std(axis=0)
    kept = np.flatnonzero((stds > 0) & (counts >= min_observed))
    Z = (Z[:, kept] - Z[:, kept].mean(axis=0)) / stds[kept]
    return Z, kept


def vif_table(X=None, names=None, limit=None, cases=None, split=None):
    """Rows of the VIF report, one per feature (ordered by descending VIF)."""
    names = list(names if names is not None else ml_data.feature_names())
    if X is None:
        X, _, _ = ml_data.load_split(
            split or ml_data.FIT_SPLIT, names=names, cases=cases, limit=limit
        )
    Z, kept = design_matrix(X)

    if len(kept) == 1:  # nothing to be collinear with
        variance = np.ones(1)
        correlation = np.ones((1, 1))
    else:
        correlation = np.corrcoef(Z, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        variance = np.diag(np.linalg.pinv(correlation))

    abs_corr = np.abs(correlation)
    np.fill_diagonal(abs_corr, 0.0)
    partner = abs_corr.argmax(axis=1)

    rows = []
    for column, position in enumerate(kept):
        name = names[position]
        twin = names[kept[partner[column]]]
        best = float(abs_corr[column, partner[column]])
        rows.append(
            {
                "feature": name,
                "family": family_of(name),
                "vif": float(variance[column]),
                "max_abs_corr_with": twin,
                "max_abs_corr": round(best, 6),
                "duplicate_of": twin if best >= DUPLICATE_R else "",
            }
        )
    rows.sort(key=lambda row: row["vif"], reverse=True)
    return rows


def family_table(vif_rows):
    """Per-family summary of the VIF report."""
    families = {}
    for row in vif_rows:
        families.setdefault(row["family"], []).append(row["vif"])
    rows = []
    for family, values in sorted(families.items()):
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "family": family,
                "n_features": len(values),
                "median_vif": float(np.median(array)),
                "max_vif": float(array.max()),
                "n_above_10": int((array > 10).sum()),
                "n_above_100": int((array > 100).sum()),
            }
        )
    rows.sort(key=lambda row: row["median_vif"], reverse=True)
    return rows


def contributions(model, X, sample=None, seed=0):
    """``(matrix, X_used)``: TreeSHAP contributions, laid out ``(n_windows, outputs, features+1)``.

    xgboost's own TreeSHAP (``predict(..., pred_contribs=True)``) rather than the ``shap``
    package, whose release available here (0.48) cannot parse the per-class ``base_score``
    that xgboost 3.x writes: it tries ``float()`` on a vector. The trailing column is the bias
    term, and the contributions are in margin space, so the softmax of a row's sum is the
    predicted probability of that class.
    """
    import xgboost as xgb

    X = np.asarray(X)
    if sample is not None and sample < len(X):
        generator = np.random.default_rng(seed)
        rows = np.sort(generator.choice(len(X), size=sample, replace=False))
        X = X[rows]
    values = np.asarray(model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True))
    if values.ndim == 2:  # a single output: (n_windows, features + 1)
        values = values[:, None, :]
    return values, X


def shap_importance(model=None, names=None, sample=DEFAULT_SAMPLE, seed=0, model_path=None,
                    meta_path=None):
    """``(rows, per_class)``: mean |SHAP| per feature over a sample of test windows.

    The contributions use :func:`contributions`; ``per_class`` is ``(n_features, n_classes)``.
    """
    if model is None:
        model, names = xgboost_baseline.load_model(model_path, meta_path)
    names = list(names if names is not None else ml_data.feature_names())
    X, y, _ = ml_data.load_split(ml_data.EVAL_SPLIT, names=names)

    contributions_matrix, _ = contributions(model, X, sample=sample, seed=seed)
    per_window = contributions_matrix[:, :, : len(names)]
    per_class = np.abs(per_window).mean(axis=0).T  # (n_features, n_classes)
    overall = per_class.mean(axis=1)

    order = np.argsort(overall)[::-1]
    importance = [
        {
            "feature": names[index],
            "mean_abs_shap": float(overall[index]),
            "rank": rank + 1,
        }
        for rank, index in enumerate(order)
    ]
    return importance, per_class


def run_diagnostics(out_dir=None, sample=DEFAULT_SAMPLE, limit=None, cases=None, seed=0,
                    model_path=None, meta_path=None):
    """Write both reports; returns ``(out_dir, vif_rows, importance_rows)``."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ml_data.feature_names()

    importance, _ = shap_importance(
        names=names, sample=sample, seed=seed, model_path=model_path, meta_path=meta_path
    )
    with (out_dir / "shap_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHAP_COLUMNS)
        writer.writeheader()
        writer.writerows(importance)

    vif_rows = vif_table(names=names, limit=limit, cases=cases)
    with (out_dir / "vif.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIF_COLUMNS)
        writer.writeheader()
        writer.writerows(vif_rows)
    with (out_dir / "vif_by_family.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAMILY_COLUMNS)
        writer.writeheader()
        writer.writerows(family_table(vif_rows))
    return out_dir, vif_rows, importance


def build_parser():
    """Command-line interface of the diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="test windows for SHAP")
    parser.add_argument("--cases", type=int, default=None, help="cap the fit cases for the VIF")
    parser.add_argument("--limit", type=int, default=None, help="cap the fit windows for the VIF")
    parser.add_argument("--out-dir", default=None, help="where to write the reports")
    parser.add_argument("--model-path", default=None, help="the model to explain")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir, vif_rows, importance = run_diagnostics(
        out_dir=args.out_dir,
        sample=args.sample,
        limit=args.limit,
        cases=args.cases,
        model_path=args.model_path,
    )
    above = lambda cut: sum(1 for row in vif_rows if row["vif"] > cut)  # noqa: E731
    print(f"VIF: {len(vif_rows)} features | >10: {above(10)} | >100: {above(100)}")
    print(f"     worst: " + ", ".join(f"{r['feature']} {r['vif']:.0f}" for r in vif_rows[:5]))
    print("SHAP top-10: " + ", ".join(r["feature"] for r in importance[:10]))
    print(f"\nreports: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
