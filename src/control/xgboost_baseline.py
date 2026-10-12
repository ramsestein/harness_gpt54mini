# -*- coding: utf-8 -*-
"""The gradient-boosting control: XGBoost on the window features.

The dataset's own reference model is a LightGBM trained to predict the clinician's action
(top-1 accuracy 55.8 % on the whole corpus, per ``dataset/docs/evaluation_guide.md``), so a
gradient-boosting model is the baseline the prompt architectures have to be measured
against. This one is trained on the ``train`` split over the same 234 numeric features the
few-shot index uses, and answers with its most probable action.

The model is fitted once by ``python -m src.control.train_xgboost`` and cached under
``results_harness/control/_models/``; the strategy only loads it, so it stays a control that
costs nothing per window (``uses_model = False``: no language model is called).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from src import config, dataset, view
from src.control import ml_data
from src.control.baselines import ControlStrategy
from src.harness_base import Prediction

__all__ = ["NAME", "PARAMS", "DEFAULT_MODEL_PATH", "DEFAULT_META_PATH", "train", "XGBoostStrategy"]

#: Folder name under ``results_harness/control/``.
NAME = "xgboost"

#: Where the fitted model and its sidecar live (``_`` keeps them out of the run scan).
DEFAULT_MODEL_PATH = Path(config.RESULTS_DIR) / "control" / "_models" / "xgboost.json"
DEFAULT_META_PATH = Path(config.RESULTS_DIR) / "control" / "_models" / "xgboost.meta.json"

#: One sensible configuration, not a tuned one: the point is a reference, not a leaderboard win.
PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(dataset.ACTIONS),
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 0,
    "eval_metric": "mlogloss",
}

#: The validation split groups by case: two windows of one patient are not independent.
VALID_FRACTION = 0.1
EARLY_STOPPING_ROUNDS = 50


def _boosters():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - dependency present
        raise ImportError(
            "the xgboost control needs 'xgboost': python -m pip install xgboost"
        ) from exc
    return XGBClassifier


def case_split(window_ids, fraction=VALID_FRACTION):
    """``(fit_positions, valid_positions)`` splitting whole cases, never windows.

    With a single case in hand there is nothing to group, so the tail of the windows is
    held out instead (only reachable through a deliberately tiny ``--limit``).
    """
    cases = ml_data.case_ids(window_ids)
    unique = sorted(set(cases.tolist()))
    if len(unique) < 2:
        cut = max(1, int(round(len(cases) * (1.0 - fraction))))
        return np.arange(cut), np.arange(cut, len(cases))
    cut = max(1, int(round(len(unique) * (1.0 - fraction))))
    cut = min(cut, len(unique) - 1)  # at least one whole case is always held out
    fit_cases = set(unique[:cut])
    fit = np.asarray([index for index, case in enumerate(cases) if case in fit_cases])
    valid = np.asarray([index for index, case in enumerate(cases) if case not in fit_cases])
    return fit, valid


def train(names=None, model_path=None, meta_path=None, limit=None, cases=None, params=None,
          valid_fraction=VALID_FRACTION, eval_limit=None, verbose=False):
    """Fit the model on the train split and cache it; returns the metrics."""
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else DEFAULT_META_PATH
    names = list(names if names is not None else ml_data.feature_names())

    X, y, window_ids = ml_data.load_split(ml_data.FIT_SPLIT, names=names, cases=cases, limit=limit)
    fit_rows, valid_rows = case_split(window_ids, fraction=valid_fraction)
    # ``hist`` + early stopping on a case-grouped validation slice of train.
    # xgboost 3.x wants the stopping rule on the constructor, not on ``fit``.
    fit_kwargs = {}
    settings = dict(params or PARAMS)
    if len(valid_rows):
        settings["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        fit_kwargs = {"eval_set": [(X[valid_rows], y[valid_rows])]}
    classifier = _boosters()(**settings)
    classifier.fit(X[fit_rows], y[fit_rows], verbose=verbose, **fit_kwargs)

    # Refit on the fit rows with the number of trees the early stop chose. Without this the
    # cached model would have a ``best_iteration``, and xgboost would then predict from a
    # different forest depending on the entry point: ``predict_proba`` uses ``best_iteration``
    # while the booster's own ``predict`` (and therefore the SHAP contributions) uses every
    # tree. One model, one definition of its own prediction.
    chosen = int(getattr(classifier, "best_iteration", 0) or 0)
    if len(valid_rows) and chosen > 0:
        settings.pop("early_stopping_rounds", None)
        settings["n_estimators"] = chosen + 1
        classifier = _boosters()(**settings)
        classifier.fit(X[fit_rows], y[fit_rows], verbose=verbose)

    X_test, y_test, test_ids = ml_data.load_split(
        ml_data.EVAL_SPLIT, names=names, limit=eval_limit
    )
    predictions = classifier.predict(X_test)
    metrics = {
        "n_fit": int(len(fit_rows)),
        "n_valid": int(len(valid_rows)),
        "n_test": int(len(y_test)),
        "n_features": len(names),
        "best_iteration": chosen,
        "n_trees": int(settings.get("n_estimators", PARAMS["n_estimators"])),
        "accuracy_fit": float((classifier.predict(X[fit_rows]) == y[fit_rows]).mean()),
        "accuracy_valid": (
            float((classifier.predict(X[valid_rows]) == y[valid_rows]).mean())
            if len(valid_rows)
            else None
        ),
        "accuracy_test": float((predictions == y_test).mean()),
        "params": dict(settings),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_path)
    meta_path.write_text(
        json.dumps({**metrics, "names": names, "split": ml_data.FIT_SPLIT,
                    "view_version": view.VIEW_VERSION}, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def load_model(model_path=None, meta_path=None):
    """``(classifier, names)`` from the cache."""
    model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    meta_path = Path(meta_path) if meta_path is not None else DEFAULT_META_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_path}: run 'python -m src.control.train_xgboost' first"
        )
    classifier = _boosters()()
    classifier.load_model(model_path)
    names = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        check_view(meta, model_path)
        names = meta.get("names")
    return classifier, names


def check_view(meta, path):
    """Refuse a model fitted on another view of the window (stale features)."""
    names = meta.get("names") or []
    hidden = view.hidden_in(names)
    if meta.get("view_version") != view.VIEW_VERSION or hidden:
        raise ValueError(
            f"{path} was fitted on view {meta.get('view_version')!r}, not the current shared "
            f"view {view.VIEW_VERSION}"
            + (f" (it reads hidden fields: {', '.join(hidden[:5])}...)" if hidden else "")
            + ": retrain it ('python -m src.control.train_xgboost')"
        )


class XGBoostStrategy(ControlStrategy):
    """Answer with the most probable action of the gradient-boosting model."""

    name = NAME
    description = (
        "XGBoost trained on the train split over the numeric window features, answering "
        "with its most probable action (no language model)."
    )

    def __init__(self, model=None, names=None, model_path=None, meta_path=None, **options):
        super().__init__(**options)
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.meta_path = Path(meta_path) if meta_path is not None else DEFAULT_META_PATH
        self._model = model
        self._names = list(names) if names is not None else None
        self._lock = threading.Lock()

    @property
    def loaded(self):
        """``(model, names)``, loaded once (thread-safe: windows run in a pool)."""
        if self._model is None or self._names is None:
            with self._lock:
                if self._model is None or self._names is None:
                    model, names = load_model(self.model_path, self.meta_path)
                    self._model = model
                    self._names = list(names or ml_data.feature_names())
        return self._model, self._names

    def probabilities(self, window):
        """The five action probabilities of one window, in ``dataset.ACTIONS`` order."""
        model, names = self.loaded
        row = ml_data.matrix([window], names)
        return np.asarray(model.predict_proba(row))[0]

    def predict(self, window):
        model, names = self.loaded
        probabilities = self.probabilities(window)
        best = int(np.argmax(probabilities))
        return Prediction(
            window_id=window.window_id,
            action=dataset.ACTIONS[best],
            raw_output=dataset.ACTIONS[best],
            meta={
                "baseline": NAME,
                "action": dataset.ACTIONS[best],
                "probability": round(float(probabilities[best]), 6),
                "probabilities": {
                    action: round(float(probabilities[index]), 6)
                    for index, action in enumerate(dataset.ACTIONS)
                },
                "n_features": len(names),
            },
        )
