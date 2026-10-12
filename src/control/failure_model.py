# -*- coding: utf-8 -*-
"""Can we tell in advance when the XGBoost control will be wrong?

The control answers one of the five actions, and it is right about four times in five. The
remaining windows are the interesting ones: if the *input alone* announced them -- before the
clinician's action is known -- a system could flag them for a human instead of trusting the
model. This module trains a binary classifier on exactly that question, using as features
**only the window data the prompt architectures are also given** (never the answer, never
``result_*``).

Two design choices make the labels honest:

- the failure labels come from **out-of-fold** predictions on the train split: every window is
  labelled by a model that never saw that patient, so the label is not contaminated by the
  model being explained;
- the folds hold out **whole cases**: two windows of one patient are not independent.

Then the predictor is evaluated on the test split, where the label is the failure of the
production model, and the report is not just AUC: what the user can act on is the
**selective-accuracy curve**, i.e. the accuracy of the control on the windows the predictor is
most comfortable with. The curve is also computed from the control's own confidence, so it is
visible how much the input data adds on top of "the model looked unsure".

    python -m src.control.failure_model
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, dataset, metrics, view  # noqa: E402
from src.control import ml_data, xgboost_baseline, xgboost_diagnostics  # noqa: E402

#: Folder under ``results_harness/control/`` (nothing is launched through the harness here).
NAME = "failure_predictor"

#: Where the reports go (the leading "_" keeps them out of the run scan).
DEFAULT_OUT_DIR = Path(config.RESULTS_DIR) / "control" / "_analysis"

#: The fitted failure classifier, cached beside the control's model so a strategy can reuse it.
DEFAULT_PREDICTOR_PATH = (
    Path(config.RESULTS_DIR) / "control" / "_models" / "failure_predictor.json"
)
DEFAULT_PREDICTOR_META_PATH = DEFAULT_PREDICTOR_PATH.with_suffix(".meta.json")

#: Cases per out-of-fold round. Five folds: ~6 min of fitting for the full corpus.
DEFAULT_FOLDS = 5

#: The failure classifier: shallow, because the signal (if any) is broad, not intricate.
PREDICTOR_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 0,
    "eval_metric": "logloss",
}

#: Coverages of the selective-accuracy curve (the kept windows are the most confident ones).
COVERAGES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)

#: Bootstrap replicates for the confidence interval of every point of the curve.
BOOTSTRAP_REPLICATES = 1000

METRIC_COLUMNS = ["metric", "value", "n"]

SELECTIVE_COLUMNS = ["ranking", "coverage", "n_kept", "accuracy", "ci95_low", "ci95_high"]

FEATURE_COLUMNS = ["feature", "mean_abs_shap", "rank"]


def production_trees(meta_path=None):
    """The tree count of the fitted control, read from its sidecar.

    The out-of-fold folds must be the same size as the model whose failures they label.
    """
    path = Path(meta_path) if meta_path is not None else xgboost_baseline.DEFAULT_META_PATH
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("n_trees"):
            return int(recorded["n_trees"])
    return int(xgboost_baseline.PARAMS["n_estimators"])


def case_folds(window_ids, folds=DEFAULT_FOLDS, seed=0):
    """``(fold_of_each_window, fold_cases)``: whole cases shuffled into ``folds`` groups."""
    cases = ml_data.case_ids(window_ids)
    unique = sorted(set(cases.tolist()))
    generator = np.random.default_rng(seed)
    generator.shuffle(unique)
    groups = [sorted(group.tolist()) for group in np.array_split(unique, folds)]
    positions = {case: index for index, group in enumerate(groups) for case in group}
    return np.asarray([positions[case] for case in cases]), groups


def oof_probabilities(names=None, folds=DEFAULT_FOLDS, n_trees=None, seed=0, limit=None, cases=None,
                      params=None, progress=False):
    """``(X, probs, y, window_ids)``: the control's out-of-fold answer on every train window.

    Each window is predicted by a model fitted on the other folds only, so the failure label
    it produces is honest. The fold models are fitted without early stopping, at the tree
    count of the production model, so all folds are comparable.
    """
    names = list(names if names is not None else ml_data.feature_names())
    X, y, window_ids = ml_data.load_split(ml_data.FIT_SPLIT, names=names, cases=cases, limit=limit)
    fold_of, groups = case_folds(window_ids, folds=folds, seed=seed)
    settings = {**xgboost_baseline.PARAMS, "n_estimators": int(n_trees or production_trees())}
    settings.update(params or {})

    probs = np.zeros((len(y), len(dataset.ACTIONS)), dtype=np.float32)
    for fold in range(len(groups)):
        test_rows = np.flatnonzero(fold_of == fold)
        fit_rows = np.flatnonzero(fold_of != fold)
        if not len(fit_rows) or not len(test_rows):
            raise ValueError(
                f"fold {fold + 1} of {len(groups)} left {len(fit_rows)} windows to fit on: "
                "use fewer folds or more cases"
            )
        model = xgboost_baseline._boosters()(**settings)
        model.fit(X[fit_rows], y[fit_rows])
        probs[test_rows] = model.predict_proba(X[test_rows])
        if progress:
            print(
                f"  fold {fold + 1}/{len(groups)}: fit {len(fit_rows)} windows, "
                f"predicted {len(test_rows)}"
            )
    return X, probs, y, window_ids


def failure_labels(probs, y):
    """``(failed, predicted)``: whether the control was wrong, and what it answered."""
    predicted = np.asarray(probs).argmax(axis=1)
    return predicted != np.asarray(y), predicted


def confidence(probs):
    """The control's own certainty: the largest probability it returned."""
    return np.asarray(probs).max(axis=1)


def train_predictor(X, failures, params=None):
    """Fit the failure classifier on the window features."""
    settings = {**PREDICTOR_PARAMS}
    settings.update(params or {})
    model = xgboost_baseline._boosters()(**settings)
    model.fit(np.asarray(X), np.asarray(failures, dtype=np.int8))
    return model


def auc_score(labels, scores):
    """ROC-AUC, or ``None`` when only one class is present."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def average_precision(labels, scores):
    """Average precision (the honest summary for an imbalanced target), or ``None``."""
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    return float(average_precision_score(labels, scores))


def selective_curve(correct, score, cases, coverages=COVERAGES, replicates=BOOTSTRAP_REPLICATES,
                    seed=0):
    """Accuracy on the kept windows, most-confident first (``score``: low = keep first).

    ``cases`` clusters the bootstrap: the interval resamples whole patients, because the
    windows of one patient are not independent.
    """
    correct = np.asarray(correct, dtype=bool)
    cases = np.asarray(cases)
    order = np.argsort(np.asarray(score, dtype=float), kind="stable")  # most confident first
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(len(order) * coverage)))
        selected = order[:keep]
        hits = correct[selected].astype(float)
        low, high, _ = metrics.cluster_bootstrap_ci(
            cases[selected], hits, replicates=replicates, seed=seed
        )
        rows.append(
            {
                "coverage": f"{coverage:.2f}",
                "n_kept": int(keep),
                "accuracy": float(hits.mean()),
                "ci95_low": float(low) if low is not None else "",
                "ci95_high": float(high) if high is not None else "",
            }
        )
    return rows


def shap_of_predictor(model, X, names, sample=None, seed=0):
    """``(rows, per_feature)``: mean |SHAP| per feature of the failure classifier."""
    values, _ = xgboost_diagnostics.contributions(model, X, sample=sample, seed=seed)
    per_feature = np.abs(values[:, 0, : len(names)]).mean(axis=0)
    order = np.argsort(per_feature)[::-1]
    return (
        [
            {"feature": names[index], "mean_abs_shap": float(per_feature[index]), "rank": rank + 1}
            for rank, index in enumerate(order)
        ],
        per_feature,
    )


def load_predictor(predictor_path=None, meta_path=None):
    """``(classifier, names)`` from the cache written by :func:`run`."""
    predictor_path = (
        Path(predictor_path) if predictor_path is not None else DEFAULT_PREDICTOR_PATH
    )
    meta_path = (
        Path(meta_path) if meta_path is not None else DEFAULT_PREDICTOR_META_PATH
    )
    if not predictor_path.is_file():
        raise FileNotFoundError(
            f"no fitted failure predictor at {predictor_path}: run "
            "'python -m src.control.failure_model' first"
        )
    model = xgboost_baseline._boosters()()
    model.load_model(predictor_path)
    names = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xgboost_baseline.check_view(meta, predictor_path)
        names = meta.get("names")
    return model, names


def run(out_dir=None, names=None, folds=DEFAULT_FOLDS, n_trees=None, limit=None, cases=None,
        seed=0, sample=None, progress=False, model_path=None, meta_path=None,
        predictor_path=None, predictor_meta_path=None):
    """Fit the failure classifier, evaluate it on test and write the reports."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(names if names is not None else ml_data.feature_names())

    if progress:
        print(f"out-of-fold failures on {ml_data.FIT_SPLIT} ({folds} folds of whole cases)")
    X_train, train_probs, y_train, train_ids = oof_probabilities(
        names=names, folds=folds, n_trees=n_trees, seed=seed, limit=limit, cases=cases,
        progress=progress,
    )
    train_failed, _ = failure_labels(train_probs, y_train)

    predictor = train_predictor(X_train, train_failed)

    # The label on test is the failure of the production model, which is the one on the
    # leaderboard (fitted on the whole train split and cached beside these reports).
    control, _ = xgboost_baseline.load_model(model_path, meta_path)
    X_test, y_test, test_ids = ml_data.load_split(ml_data.EVAL_SPLIT, names=names)
    test_probs = control.predict_proba(X_test)
    test_failed, _ = failure_labels(test_probs, y_test)
    test_cases = ml_data.case_ids(test_ids)

    risk = np.asarray(predictor.predict_proba(X_test))[:, 1]
    confident = 1.0 - confidence(test_probs)  # low = the control is sure
    random_order = np.random.default_rng(seed).random(len(X_test))  # a fair baseline ordering

    rows = [
        {"metric": "n_train_windows", "value": int(len(y_train)), "n": int(len(y_train))},
        {"metric": "n_test_windows", "value": int(len(y_test)), "n": int(len(y_test))},
        {"metric": "failure_rate_train_oof", "value": float(train_failed.mean()), "n": len(y_train)},
        {"metric": "failure_rate_test", "value": float(test_failed.mean()), "n": len(y_test)},
        {"metric": "n_train_cases", "value": len(set(ml_data.case_ids(train_ids))), "n": ""},
        {"metric": "n_test_cases", "value": len(set(test_cases)), "n": ""},
        {
            "metric": "auc_from_features",
            "value": auc_score(test_failed, risk),
            "n": int(len(y_test)),
        },
        {
            "metric": "average_precision_from_features",
            "value": average_precision(test_failed, risk),
            "n": int(len(y_test)),
        },
        {
            "metric": "auc_from_control_confidence",
            "value": auc_score(test_failed, confident),
            "n": int(len(y_test)),
        },
        {
            "metric": "average_precision_from_control_confidence",
            "value": average_precision(test_failed, confident),
            "n": int(len(y_test)),
        },
        {"metric": "accuracy_of_control_on_test", "value": float((~test_failed).mean()), "n": ""},
        {
            "metric": "correlation_features_vs_confidence",
            "value": float(np.corrcoef(risk, confident)[0, 1]),
            "n": "",
        },
        {"metric": "folds", "value": folds, "n": ""},
        {"metric": "trees_per_fold_model", "value": int(n_trees or production_trees()), "n": ""},
    ]

    curve_rows = []
    for ranking, score in (
        ("failure_predictor", risk),
        ("control_confidence", confident),
        ("random", random_order),
    ):
        for row in selective_curve(~test_failed, score, test_cases, seed=seed):
            curve_rows.append({"ranking": ranking, **row})

    importance, _ = shap_of_predictor(predictor, X_test, names, sample=sample, seed=seed)

    # Cache the classifier: ``support`` reuses it to decide which windows to hand over,
    # and the labels it was trained on (out-of-fold failures) are expensive to rebuild.
    predictor_path = (
        Path(predictor_path) if predictor_path is not None else DEFAULT_PREDICTOR_PATH
    )
    predictor_meta_path = (
        Path(predictor_meta_path) if predictor_meta_path is not None
        else DEFAULT_PREDICTOR_META_PATH
    )
    predictor_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(predictor_path)
    predictor_meta_path.write_text(
        json.dumps(
            {
                "names": names,
                "view_version": view.VIEW_VERSION,
                "split": ml_data.FIT_SPLIT,
                "folds": folds,
                "trees_per_fold_model": int(n_trees or production_trees()),
                "failure_rate_train_oof": float(train_failed.mean()),
                "params": dict(PREDICTOR_PARAMS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with (out_dir / "failure_predictor.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "failure_selective.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTIVE_COLUMNS)
        writer.writeheader()
        writer.writerows(curve_rows)
    with (out_dir / "failure_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(importance)

    return {
        "out_dir": out_dir,
        "metrics": {row["metric"]: row["value"] for row in rows},
        "curve": curve_rows,
        "importance": importance,
        "predictor": predictor,
        "predictor_path": predictor_path,
        "test": {
            "risk": risk,
            "confident": confident,
            "failed": test_failed,
            "y": y_test,
            "cases": test_cases,
        },
    }


def build_parser():
    """Command-line interface of the failure study."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="out-of-fold rounds")
    parser.add_argument("--n-trees", type=int, default=None, help="trees per fold model")
    parser.add_argument("--cases", type=int, default=None, help="cap the train cases")
    parser.add_argument("--limit", type=int, default=None, help="cap the train windows")
    parser.add_argument("--sample", type=int, default=None, help="windows explained by SHAP")
    parser.add_argument("--out-dir", default=None, help="where to write the reports")
    parser.add_argument("--progress", action="store_true", help="show the folds as they fit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outcome = run(
        out_dir=args.out_dir,
        folds=args.folds,
        n_trees=args.n_trees,
        limit=args.limit,
        cases=args.cases,
        sample=args.sample,
        progress=args.progress,
    )
    numbers = outcome["metrics"]
    print(
        f"train {numbers['n_train_windows']} windows / {numbers['n_train_cases']} cases, "
        f"failure rate {numbers['failure_rate_train_oof']:.4f}\n"
        f"test  {numbers['n_test_windows']} windows / {numbers['n_test_cases']} cases, "
        f"failure rate {numbers['failure_rate_test']:.4f}"
    )
    print(
        f"AUC  from features {numbers['auc_from_features']:.4f} | "
        f"from confidence {numbers['auc_from_control_confidence']:.4f}\n"
        f"AP   from features {numbers['average_precision_from_features']:.4f} | "
        f"from confidence {numbers['average_precision_from_control_confidence']:.4f}"
    )
    print("\nselective accuracy (keep the windows the ranking trusts most):")
    print(f"{'ranking':20s} {'coverage':>8s} {'kept':>6s} {'accuracy':>9s}   95% CI")
    for row in outcome["curve"]:
        interval = (
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            if row["ci95_low"] != ""
            else "-"
        )
        print(
            f"{row['ranking']:20s} {row['coverage']:>8s} {row['n_kept']:6d} "
            f"{row['accuracy']:9.4f}   {interval}"
        )
    print("\ntop failure indicators: " + ", ".join(r["feature"] for r in outcome["importance"][:10]))
    print(f"\nreports: {outcome['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
