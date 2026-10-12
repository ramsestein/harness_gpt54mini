# -*- coding: utf-8 -*-
"""The feature matrix and the labels the machine-learning controls share.

Every model in this folder sees **exactly the numeric features the few-shot index uses** --
the numeric fields of the shared view (``src.view``), which is also what ``render_case`` shows
the language models --
with the same definitions and the same order: the names come from that index's cache, so a
tree model here and the retrieval controls are comparable feature for feature.

Missing values stay ``NaN``: XGBoost handles them natively, and imputing them would invent
information the model does not have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harness.few_shot import retrieval
from src import dataset, view

__all__ = [
    "FIT_SPLIT",
    "EVAL_SPLIT",
    "ACTION_INDEX",
    "ACTION_LABEL",
    "feature_names",
    "matrix",
    "labels",
    "load_split",
]

#: The models are fitted here and evaluated there.
FIT_SPLIT = retrieval.SOURCE_SPLIT
EVAL_SPLIT = "test"

#: Label encoding: the five actions, in ``dataset.ACTIONS`` order.
ACTION_INDEX = {action: index for index, action in enumerate(dataset.ACTIONS)}
ACTION_LABEL = {index: action for action, index in ACTION_INDEX.items()}


def feature_names(index_path=None):
    """The canonical feature names, in index order.

    The few-shot index cache is the single source of truth, so the models and the
    retrieval always agree on the feature set; without it, the names are read from a
    sample of the training corpus (slower, same definition).
    """
    path = Path(index_path) if index_path is not None else retrieval.DEFAULT_INDEX_PATH
    names = None
    if path.is_file():
        try:
            names = list(retrieval.SimilarityIndex.load(path).names)
        except ValueError:  # an index built on another view: fall back, never reuse it
            names = None
    if names is None:
        names = retrieval.feature_names(list(retrieval.train_windows(cases=20)))
    return [name for name in names if not view.is_hidden(name)]


def matrix(windows, names):
    """The feature matrix of an iterable of windows: float32, ``NaN`` where nothing is there."""
    names = list(names)
    rows = []
    for window in windows:
        inp = window.input
        rows.append(
            [inp.get(name) if retrieval.is_feature(inp.get(name)) else np.nan for name in names]
        )
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), len(names))


def labels(windows):
    """The ground-truth action of each window, as an index into ``dataset.ACTIONS``."""
    return np.asarray([ACTION_INDEX[window.real_action] for window in windows], dtype=np.int8)


def load_split(split, names=None, cases=None, limit=None):
    """``(X, y, window_ids)`` of a dataset split, streaming the windows.

    ``limit`` caps the number of windows (useful for tests and for quick runs).
    """
    names = list(names if names is not None else feature_names())
    if cases is None:
        windows = dataset.iter_split(split)
    else:
        windows = (
            window
            for path in dataset.case_files(split)[:cases]
            for window in dataset.load_case(path)
        )
    if limit is not None:
        windows = (window for index, window in enumerate(windows) if index < limit)

    rows = []
    targets = []
    ids = []
    for window in windows:
        inp = window.input
        rows.append(
            [inp.get(name) if retrieval.is_feature(inp.get(name)) else np.nan for name in names]
        )
        targets.append(ACTION_INDEX[window.real_action])
        ids.append(window.window_id)
    return (
        np.asarray(rows, dtype=np.float32).reshape(len(rows), len(names)),
        np.asarray(targets, dtype=np.int8),
        ids,
    )


def case_ids(window_ids):
    """The case of every window, from ``case<id>_w<number>``."""
    return np.asarray([str(window_id).split("_")[0] for window_id in window_ids])
