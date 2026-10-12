# -*- coding: utf-8 -*-
"""The XGBoost control: fitting, caching and answering."""

import json
from pathlib import Path

import numpy as np
import pytest

from src import config, dataset
from src.control import ml_data
from src.harness_base import Strategy

pytest.importorskip("xgboost")

from src.control import xgboost_baseline  # noqa: E402
from src.control.baselines import ControlStrategy  # noqa: E402

#: Small enough to stay fast, big enough to have more than one case.
FIT_CASES = 3
FIT_LIMIT = 300
EVAL_LIMIT = 40


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A model trained on a corner of the train split."""
    folder = tmp_path_factory.mktemp("model")
    model_path = folder / "xgboost.json"
    meta_path = folder / "xgboost.meta.json"
    metrics = xgboost_baseline.train(
        cases=FIT_CASES, limit=FIT_LIMIT, eval_limit=EVAL_LIMIT,
        model_path=model_path, meta_path=meta_path,
    )
    return model_path, meta_path, metrics


# --------------------------------------------------------------------------- #
# It is a control: no language model, scored like every other strategy
# --------------------------------------------------------------------------- #


def test_it_is_a_strategy():
    strategy = xgboost_baseline.XGBoostStrategy()

    assert isinstance(strategy, Strategy)
    assert isinstance(strategy, ControlStrategy)
    assert strategy.name == xgboost_baseline.NAME == "xgboost"
    assert strategy.description.strip()


def test_it_does_not_claim_a_language_model():
    assert xgboost_baseline.XGBoostStrategy().uses_model is False


def test_the_model_lives_under_the_control_folder():
    assert Path(config.RESULTS_DIR) / "control" in xgboost_baseline.DEFAULT_MODEL_PATH.parents
    assert xgboost_baseline.DEFAULT_MODEL_PATH.parent.name.startswith("_")  # out of the scan


def test_the_five_actions_are_the_dataset_ones():
    assert xgboost_baseline.PARAMS["num_class"] == len(dataset.ACTIONS) == 5


# --------------------------------------------------------------------------- #
# Splitting and fitting
# --------------------------------------------------------------------------- #


def test_the_validation_split_groups_whole_cases():
    ids = [f"case{case}_w{window}" for case in range(1, 5) for window in range(3)]

    fit, valid = xgboost_baseline.case_split(ids, fraction=0.25)

    fit_cases = set(ml_data.case_ids([ids[i] for i in fit]))
    valid_cases = set(ml_data.case_ids([ids[i] for i in valid]))
    assert not fit_cases & valid_cases
    assert sorted(list(fit) + list(valid)) == list(range(len(ids)))
    assert len(valid) == 3


def test_a_single_case_still_leaves_something_to_validate_on():
    ids = [f"case7_w{window}" for window in range(10)]

    fit, valid = xgboost_baseline.case_split(ids, fraction=0.2)

    assert len(valid) == 2
    assert not set(fit) & set(valid)


def test_training_writes_the_model_and_its_sidecar(tiny_model):
    model_path, meta_path, metrics = tiny_model

    assert model_path.is_file()
    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    assert sidecar["names"] == ml_data.feature_names()
    assert sidecar["split"] == "train"
    assert sidecar["n_features"] == metrics["n_features"] == len(ml_data.feature_names())
    assert 0 < metrics["n_fit"] <= FIT_LIMIT
    assert 0 < metrics["n_valid"]  # a whole case is always held out
    assert metrics["n_fit"] + metrics["n_valid"] <= FIT_LIMIT
    assert metrics["accuracy_valid"] is not None
    assert metrics["accuracy_test"] is not None


def test_the_model_is_much_better_than_chance_even_on_this_corner(tiny_model):
    _, _, metrics = tiny_model

    # 300 windows of three cases: a real signal should already stand out from 1/5.
    assert metrics["accuracy_fit"] > 0.3
    assert metrics["best_iteration"] >= 0


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


def test_the_strategy_reads_the_cached_model(tiny_model):
    model_path, meta_path, _ = tiny_model
    strategy = xgboost_baseline.XGBoostStrategy(model_path=model_path, meta_path=meta_path)

    predictions = strategy.predict_many(dataset.load_split("dev", limit=5))

    assert len(predictions) == 5
    for prediction in predictions:
        assert prediction.action in dataset.ACTIONS
        assert prediction.meta["baseline"] == "xgboost"
        assert set(prediction.meta["probabilities"]) == set(dataset.ACTIONS)
        assert sum(prediction.meta["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
        assert prediction.meta["action"] == prediction.action


def test_the_answer_is_the_most_probable_action(tiny_model):
    model_path, meta_path, _ = tiny_model
    strategy = xgboost_baseline.XGBoostStrategy(model_path=model_path, meta_path=meta_path)

    for window in dataset.load_split("dev", limit=3):
        probabilities = strategy.probabilities(window)
        assert strategy.predict(window).action == dataset.ACTIONS[int(np.argmax(probabilities))]


def test_it_is_deterministic(tiny_model):
    model_path, meta_path, _ = tiny_model
    strategy = xgboost_baseline.XGBoostStrategy(model_path=model_path, meta_path=meta_path)
    windows = dataset.load_split("dev", limit=4)

    first = [p.action for p in strategy.predict_many(windows)]
    second = [p.action for p in strategy.predict_many(windows)]

    assert first == second


def test_a_missing_model_says_how_to_build_one(tmp_path):
    strategy = xgboost_baseline.XGBoostStrategy(model_path=tmp_path / "nope.json")

    with pytest.raises(FileNotFoundError, match="train_xgboost"):
        strategy.predict(dataset.load_split("dev", limit=1)[0])


def test_the_launcher_scores_it_like_any_strategy(tiny_model, tmp_path):
    from harness import run as harness_run

    model_path, meta_path, _ = tiny_model
    strategy = xgboost_baseline.XGBoostStrategy(model_path=model_path, meta_path=meta_path)

    outcome = harness_run.run(strategy, split="dev", limit=6, results_root=tmp_path, quiet=True)

    assert outcome["summary"]["errors"] == 0
    assert outcome["summary"]["n_windows"] == 6
    assert len(outcome["predictions"]) == 6
    assert outcome["summary"]["accuracy"] is not None
    # the run must say it did not call a language model
    run_info = json.loads((Path(outcome["out_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert run_info["model"] is None
    assert run_info["strategy"] == "xgboost"
