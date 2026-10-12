# -*- coding: utf-8 -*-
"""The failure predictor: honest labels, a readable curve, the three reports."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src import dataset
from src.control import failure_model, ml_data, xgboost_baseline

pytest.importorskip("xgboost")

#: A corner of the corpus, small enough to fit five folds in a test.
SMALL = {"cases": 12, "limit": 600, "folds": 3, "n_trees": 40}


@pytest.fixture(scope="module")
def names():
    return ml_data.feature_names()


@pytest.fixture(scope="module")
def tiny_study(names, tmp_path_factory):
    """A miniature version of the whole study, with the production model as arbiter."""
    folder = tmp_path_factory.mktemp("failure-study")
    model_path = folder / "control.json"
    meta_path = folder / "control.meta.json"
    xgboost_baseline.train(
        cases=SMALL["cases"],
        limit=SMALL["limit"],
        eval_limit=120,
        params={**xgboost_baseline.PARAMS, "n_estimators": 40, "max_depth": 4},
        model_path=model_path,
        meta_path=meta_path,
    )
    outcome = failure_model.run(
        out_dir=folder / "analysis",
        names=names,
        folds=SMALL["folds"],
        n_trees=SMALL["n_trees"],
        cases=SMALL["cases"],
        limit=SMALL["limit"],
        sample=80,
        model_path=model_path,
        meta_path=meta_path,
        predictor_path=folder / "predictor.json",
        predictor_meta_path=folder / "predictor.meta.json",
    )
    return outcome


# --------------------------------------------------------------------------- #
# The out-of-fold labels
# --------------------------------------------------------------------------- #


def test_the_folds_hold_out_whole_cases(names):
    X, probs, y, ids = failure_model.oof_probabilities(
        names=names, folds=SMALL["folds"], n_trees=SMALL["n_trees"],
        cases=SMALL["cases"], limit=SMALL["limit"],
    )

    assert len(probs) == len(y) == len(ids) == len(X)
    # every window got a prediction from some fold model, and none is left at zero
    assert np.allclose(np.asarray(probs).sum(axis=1), 1.0, atol=1e-4)


def test_the_fold_assignment_is_by_case(names):
    _, _, ids = ml_data.load_split(ml_data.FIT_SPLIT, names=names, cases=SMALL["cases"])
    fold_of, groups = failure_model.case_folds(ids, folds=SMALL["folds"])

    assert len(groups) == SMALL["folds"]
    for case in set(ml_data.case_ids(ids)):
        windows = fold_of[ml_data.case_ids(ids) == case]
        assert len(set(windows.tolist())) == 1  # a case never straddles two folds
    assert sorted(case for group in groups for case in group) == sorted(set(ml_data.case_ids(ids)))


def test_every_out_of_fold_prediction_is_an_action(names):
    _, probs, _, _ = failure_model.oof_probabilities(
        names=names, folds=SMALL["folds"], n_trees=SMALL["n_trees"],
        cases=SMALL["cases"], limit=SMALL["limit"],
    )

    predicted = np.asarray(probs).argmax(axis=1)

    assert set(predicted.tolist()) <= set(range(len(dataset.ACTIONS)))


def test_the_failure_label_is_the_wrong_action():
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]])
    y = np.array([0, 0, 1])

    failed, predicted = failure_model.failure_labels(probs, y)

    assert list(predicted) == [0, 1, 0]
    assert list(failed) == [False, True, True]


def test_the_confidence_is_the_biggest_probability():
    from src.control.failure_model import confidence

    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])

    assert list(confidence(probs)) == [0.9, 0.4]


def test_the_folds_are_reproducible(names):
    first = failure_model.oof_probabilities(
        names=names, folds=SMALL["folds"], n_trees=SMALL["n_trees"], cases=SMALL["cases"], limit=300
    )[1]
    second = failure_model.oof_probabilities(
        names=names, folds=SMALL["folds"], n_trees=SMALL["n_trees"], cases=SMALL["cases"], limit=300
    )[1]

    assert np.allclose(first, second)


# --------------------------------------------------------------------------- #
# The failure classifier
# --------------------------------------------------------------------------- #


def test_the_tree_count_defaults_to_the_production_model(names):
    """The fold models must be the size of the model whose failures they label."""
    assert failure_model.production_trees() >= 1


def test_the_predictor_sees_only_window_features(tiny_study, names):
    """No answer, no reference label: the feature names are the window's own."""
    X, probs, y, ids = failure_model.oof_probabilities(
        names=names, folds=SMALL["folds"], n_trees=SMALL["n_trees"],
        cases=SMALL["cases"], limit=SMALL["limit"],
    )

    assert X.shape[1] == len(names)
    assert not any("result" in name for name in names)


# --------------------------------------------------------------------------- #
# The selective-accuracy curve
# --------------------------------------------------------------------------- #


def test_the_curve_keeps_the_most_confident_first():
    correct = np.array([False] * 10 + [True] * 90, dtype=bool)
    score = np.arange(100, dtype=float)  # low score = keep first = the wrong ones
    cases = np.array([f"case{i // 10}" for i in range(100)])

    rows = failure_model.selective_curve(correct, score, cases, coverages=(1.0, 0.5))

    assert rows[0]["n_kept"] == 100
    assert rows[0]["accuracy"] == pytest.approx(0.9)
    # the kept half is the ten failures plus the first forty successes
    assert rows[1]["n_kept"] == 50
    assert rows[1]["accuracy"] == pytest.approx(0.8)
    assert rows[1]["accuracy"] < rows[0]["accuracy"]  # trusting the wrong windows hurts


def test_the_curve_improves_when_the_ranking_is_informative():
    correct = np.array([True] * 90 + [False] * 10, dtype=bool)
    informative = np.arange(100, dtype=float)  # low = keep first = the correct ones
    blind = np.random.default_rng(0).random(100)  # a fair ordering would pick up some failures

    informed = failure_model.selective_curve(correct, informative, np.arange(100), (1.0, 0.5))
    unaware = failure_model.selective_curve(correct, blind, np.arange(100), (1.0, 0.5))

    assert informed[1]["accuracy"] == pytest.approx(1.0)
    assert unaware[1]["accuracy"] < informed[1]["accuracy"]


def test_the_curve_carries_case_clustered_intervals():
    correct = np.array([True, False] * 50)
    cases = np.array([f"case{i}" for i in range(100)])

    rows = failure_model.selective_curve(correct, np.arange(100, dtype=float), cases, (0.5,))

    assert rows[0]["ci95_low"] < rows[0]["accuracy"] < rows[0]["ci95_high"]


# --------------------------------------------------------------------------- #
# The reports
# --------------------------------------------------------------------------- #


def test_the_reports_are_written(tiny_study):
    out_dir = tiny_study["out_dir"]

    metrics = list(csv.DictReader((out_dir / "failure_predictor.csv").open(encoding="utf-8")))
    assert [row["metric"] for row in metrics][:2] == ["n_train_windows", "n_test_windows"]
    numbers = {row["metric"]: row["value"] for row in metrics}
    assert float(numbers["failure_rate_test"]) > 0
    assert float(numbers["accuracy_of_control_on_test"]) < 1.0

    curve = list(csv.DictReader((out_dir / "failure_selective.csv").open(encoding="utf-8")))
    assert {row["ranking"] for row in curve} == {
        "failure_predictor",
        "control_confidence",
        "random",
    }
    assert len(curve) == 3 * len(failure_model.COVERAGES)

    features = list(csv.DictReader((out_dir / "failure_features.csv").open(encoding="utf-8")))
    assert len(features) == len(ml_data.feature_names())
    assert int(features[0]["rank"]) == 1


def test_the_study_keeps_the_control_and_the_predictor_apart(tiny_study):
    """The predictor answers one thing (will it fail), the control another (which action)."""
    assert set(tiny_study["test"]) == {"risk", "confident", "failed", "y", "cases"}
    assert set(tiny_study["test"]["risk"].shape) == {len(tiny_study["test"]["y"])}
    assert tiny_study["predictor"].n_features_in_ == len(ml_data.feature_names())


def test_the_predictor_is_cached_for_the_strategies(tiny_study):
    """``support`` must be able to reuse it without refitting anything."""
    path = Path(tiny_study["predictor_path"])
    meta_path = path.parent / "predictor.meta.json"

    model, names = failure_model.load_predictor(path, meta_path)

    assert path.is_file() and meta_path.is_file()
    assert names == ml_data.feature_names()
    X, _, _ = ml_data.load_split(ml_data.EVAL_SPLIT, names=names, limit=5)
    risk = model.predict_proba(X)[:, 1]
    assert risk.shape == (5,)
    assert ((risk >= 0.0) & (risk <= 1.0)).all()
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    assert info["split"] == "train"
    assert 0.0 < info["failure_rate_train_oof"] < 1.0


def test_a_missing_predictor_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="failure_model"):
        failure_model.load_predictor(tmp_path / "nope.json", tmp_path / "nope.meta.json")


def test_the_curve_of_a_coverage_never_leaves_zero_and_one(tiny_study):
    for row in tiny_study["curve"]:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["n_kept"] >= 1


def test_the_auc_of_the_control_confidence_is_reported(tiny_study):
    numbers = tiny_study["metrics"]

    assert numbers["auc_from_features"] is None or 0.0 <= numbers["auc_from_features"] <= 1.0
    assert 0.0 <= numbers["auc_from_control_confidence"] <= 1.0
