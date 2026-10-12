# -*- coding: utf-8 -*-
"""Support strategy: trust the tree model where it is sure, hand the rest to the model.

The gradient-boosting control answers about four windows in five and sits far above every
prompt architecture, but it is not equally reliable everywhere. This strategy runs that
control over the split, keeps its answer only where it is **sure enough**, and for the rest
reuses what the language model already answered in an earlier ``zero_shot`` run: nothing is
called, so a run costs no tokens and the fallback is exactly the answers already on record.

Knobs (constructor options; the numeric one comes from the launcher's ``--samples``):

``effort``
    which ``zero_shot`` run supplies the fallback answers. The effort is **the run's**:
    ``--effort none`` defers to the ``zero_shot`` run launched with no reasoning effort,
    ``--effort high`` to the high one, and so on, so the hybrid is measured against the
    language model it actually leans on. It is recorded in every prediction, which is how
    the leaderboard tells the three variants apart.
``threshold``
    how sure is sure enough, on the ``[0, 1]`` scale of the score below. Given as a
    percentage through ``--samples`` (``--samples 60`` means "trust the control when its score
    is at least 0.60"), so the range can be swept run by run. ``0`` keeps everything and
    reproduces the control's own accuracy, which is the reference to beat.
``signal``
    what the score means: the control's own probability (``confidence``, the default), the
    failure predictor under ``src/control/`` (``failure``, which needs
    ``python -m src.control.failure_model`` to have run once), or the two averaged
    (``combined``).
"""

import json
import threading
from pathlib import Path

import numpy as np

from src import config, dataset, results, view
from src.harness_base import Prediction, Strategy

from src.control import failure_model, ml_data, xgboost_baseline

#: The fallback answers come from a stored run of this strategy.
FALLBACK_STRATEGY = "zero_shot"

#: What the score can mean.
SIGNALS = ("confidence", "failure", "combined")

#: Trust the control from this score upwards when nothing else is given.
DEFAULT_THRESHOLD = 0.60

#: Sentinel: "do not filter the candidate runs by effort".
ANY_EFFORT = object()


def _read_json(path):
    """A JSON file as a dict, or ``{}`` when it is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_finished_run(run_dir):
    """Whether a folder holds a scored, non-dry run (the rule the leaderboard uses too)."""
    info = _read_json(Path(run_dir) / "summary.json")
    return bool(info) and not info.get("dry_run")


def first_prediction_meta(run_dir):
    """The metadata of a run's first stored prediction (only that line is read)."""
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    return json.loads(line).get("meta") or {}
    except (OSError, ValueError):
        return {}
    return {}


def run_effort(run_dir):
    """The reasoning effort a stored run was launched with; ``None`` when it was omitted.

    ``run.json`` is the intended source (``"effort" in it`` distinguishes "omitted" from
    "not recorded"); a run too old for the field falls back to its predictions' metadata.
    """
    settings = _read_json(Path(run_dir) / "run.json")
    if "effort" in settings:
        return settings["effort"]
    return first_prediction_meta(run_dir).get("reasoning_effort")


def run_model(run_dir):
    """The model a stored run was launched with; ``None`` for runs that predate the column.

    A ``None`` here means the project's own model: that is what those runs used.
    """
    return _read_json(Path(run_dir) / "run.json").get("model") or None


def latest_run(strategy=FALLBACK_STRATEGY, split=None, results_root=None, effort=ANY_EFFORT,
               model=None):
    """The newest finished run of ``strategy``, optionally of one split, effort and model.

    ``model`` keeps the answers of one model only: with two models in the same results tree
    the newest run is not necessarily the one this run is meant to lean on. The model is
    read from ``run.json``; a run stored before the model was written there counts as the
    project's own model.
    """
    base = Path(results_root) if results_root is not None else Path(config.RESULTS_DIR)
    root = base / strategy
    if not root.is_dir():
        return None
    candidates = [
        child
        for child in sorted(root.iterdir(), reverse=True)
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (split is None or child.name.endswith(f"_{split}"))
        and is_finished_run(child)
        and run_view(child) == view.VIEW_VERSION
        and (effort is ANY_EFFORT or run_effort(child) == effort)
        and (model is None or (run_model(child) or config.DEFAULT_MODEL) == model)
    ]
    return candidates[0] if candidates else None


def run_view(run_dir):
    """The shared view a stored run was made on (``None`` for runs older than the view)."""
    return _read_json(Path(run_dir) / "run.json").get("view_version")


def load_fallback(run_dir):
    """``({window_id: action}, run_dir)`` of a stored run's answers.

    The run must have seen the current shared view: answers given on another window are not
    the language model this hybrid is meant to lean on.
    """
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no predictions in {run_dir}")
    if run_view(run_dir) != view.VIEW_VERSION:
        raise ValueError(
            f"{run_dir} was made on view {run_view(run_dir)!r}, not the current shared view "
            f"{view.VIEW_VERSION}: rerun {FALLBACK_STRATEGY} first"
        )
    actions = {
        row["window_id"]: row["action"]
        for row in results.read_predictions(path)
        if row.get("window_id") and row.get("action")
    }
    return actions, Path(run_dir)


class SupportStrategy(Strategy):
    """The tree model where it is sure, the stored ``zero_shot`` answers everywhere else."""

    name = "support"
    description = (
        "Hybrid: the XGBoost control answers the windows it is sure enough about (score at "
        "least the threshold, by its own confidence or by the failure predictor) and the "
        "zero_shot answers of the run matching this run's --effort answer the rest. It calls "
        "no model, so a run is free."
    )
    #: The run itself makes no model call: the fallback answers were produced earlier.
    uses_model = False

    def __init__(self, threshold=None, signal="confidence", fallback_run=None,
                 fallback_split=None, fallback_effort=None, fallback_root=None,
                 fallback_model=None, model_path=None,
                 meta_path=None, predictor_path=None, predictor_meta_path=None, n_samples=None,
                 k=None, **options):
        super().__init__(**options)
        # ``n_samples``/``k`` are accepted as aliases so the launcher's --samples works:
        # an integer percentage of the [0, 1] score.
        for alias in (n_samples, k):
            if alias is not None:
                threshold = float(alias) / 100.0
        self.threshold = float(DEFAULT_THRESHOLD if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold!r}")
        self.signal = str(signal).strip().lower()
        if self.signal not in SIGNALS:
            raise ValueError(f"unknown signal: {signal!r} (expected one of {SIGNALS})")
        self.fallback_run = Path(fallback_run) if fallback_run else None
        self.fallback_split = fallback_split
        self.fallback_root = Path(fallback_root) if fallback_root else None
        # The effort of this run is the effort of the zero_shot answers it leans on:
        # ``--effort high`` must defer to the high-effort run, not to whichever is newest.
        self.fallback_effort = self.reasoning_effort if fallback_effort is None else fallback_effort
        # Same for the model: ``--model deepseek-flash`` leans on the deepseek answers, not on
        # the newest ones. It defaults to the model this run was launched with.
        self.fallback_model = self.model if fallback_model is None else fallback_model
        #: What ``run.json`` records. An empty ``model`` cell already means the project's own
        #: model, so recording it would split one experiment into two table rows.
        self.inherited_model = (
            None if self.fallback_model == config.DEFAULT_MODEL else self.fallback_model
        )
        self.model_path = Path(model_path) if model_path else xgboost_baseline.DEFAULT_MODEL_PATH
        self.meta_path = Path(meta_path) if meta_path else xgboost_baseline.DEFAULT_META_PATH
        self.predictor_path = (
            Path(predictor_path) if predictor_path else failure_model.DEFAULT_PREDICTOR_PATH
        )
        self.predictor_meta_path = (
            Path(predictor_meta_path) if predictor_meta_path
            else failure_model.DEFAULT_PREDICTOR_META_PATH
        )
        self._control = None
        self._control_names = None
        self._predictor = None
        self._predictor_names = None
        self._fallback = None
        self._fallback_dir = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Loading (lazy: a strategy must construct without touching the disk)
    # ------------------------------------------------------------------ #

    def control(self):
        """``(model, names)`` of the tree control."""
        if self._control is None:
            with self._lock:
                if self._control is None:
                    model, names = xgboost_baseline.load_model(self.model_path, self.meta_path)
                    self._control_names = list(names or ml_data.feature_names())
                    self._control = model
        return self._control, self._control_names

    def predictor(self):
        """``(model, names)`` of the failure predictor."""
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    model, names = failure_model.load_predictor(
                        self.predictor_path, self.predictor_meta_path
                    )
                    self._predictor_names = list(names or ml_data.feature_names())
                    self._predictor = model
        return self._predictor, self._predictor_names

    def fallback(self):
        """``({window_id: action}, run_dir)`` of the stored answers this run defers to."""
        if self._fallback is None:
            with self._lock:
                if self._fallback is None:
                    run_dir = self.fallback_run or latest_run(
                        FALLBACK_STRATEGY, split=self.fallback_split, effort=self.fallback_effort,
                        results_root=self.fallback_root, model=self.fallback_model,
                    )
                    if run_dir is None:
                        raise FileNotFoundError(
                            f"no finished {FALLBACK_STRATEGY} run found for effort "
                            f"{self.fallback_effort!r} and model {self.fallback_model!r}: run that "
                            "strategy with that model first, or pass fallback_run=..."
                        )
                    self._fallback, self._fallback_dir = load_fallback(run_dir)
        return self._fallback, self._fallback_dir

    # ------------------------------------------------------------------ #
    # Scoring and answering
    # ------------------------------------------------------------------ #

    def probabilities(self, window):
        """The control's five action probabilities, in ``dataset.ACTIONS`` order."""
        model, names = self.control()
        return np.asarray(model.predict_proba(ml_data.matrix([window], names)))[0]

    def score(self, window, probabilities):
        """``(score, detail)``: how much the control is trusted here (low = defer)."""
        sure = float(probabilities.max())
        detail = {"confidence": round(sure, 6)}
        if self.signal == "confidence":
            return sure, detail
        model, names = self.predictor()
        risk = float(np.asarray(model.predict_proba(ml_data.matrix([window], names)))[0][1])
        detail["failure_risk"] = round(risk, 6)
        if self.signal == "failure":
            return 1.0 - risk, detail
        return (sure + (1.0 - risk)) / 2.0, detail

    def predict(self, window) -> Prediction:
        """The control's answer when it is trusted, the stored ``zero_shot`` one otherwise."""
        probabilities = self.probabilities(window)
        best = int(np.argmax(probabilities))
        action = dataset.ACTIONS[best]
        score, detail = self.score(window, probabilities)
        fallback, fallback_dir = self.fallback()

        meta = {
            "baseline": "support",
            "signal": self.signal,
            "threshold": self.threshold,
            "k": int(round(self.threshold * 100)),  # the leaderboard's knob column
            "score": round(score, 6),
            "xgb_action": action,
            "fallback_run": fallback_dir.name,
            # the model of the answers this run leans on: the control answers the rest, so the
            # model is the run's own (it is what the leaderboard's model column must show)
            "fallback_model": self.fallback_model,
            # the effort of the run this one leans on: the leaderboard reads it from here
            # (a strategy that calls no model records no effort of its own in run.json)
            "reasoning_effort": self.fallback_effort,
            "effort": self.fallback_effort,
            **detail,
        }

        if score < self.threshold:
            stored = fallback.get(window.window_id)
            if stored:
                meta["source"] = "fallback"
                meta["fallback_action"] = stored
                return Prediction(
                    window_id=window.window_id, action=stored, raw_output=stored, meta=meta
                )
            # No stored answer for this window: the control is the only thing left.
            meta["source"] = "control"
            meta["fallback_missing"] = True
            return Prediction(
                window_id=window.window_id, action=action, raw_output=action, meta=meta
            )

        meta["source"] = "control"
        return Prediction(window_id=window.window_id, action=action, raw_output=action, meta=meta)
