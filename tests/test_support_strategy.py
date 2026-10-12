# -*- coding: utf-8 -*-
"""The ``support`` strategy: the tree control where it is sure, stored zero_shot elsewhere."""

import json
from pathlib import Path

import numpy as np
import pytest

from src import config, dataset, results, view
from src.harness_base import Strategy

pytest.importorskip("xgboost")

from harness.support import strategy as support  # noqa: E402
from src.control import failure_model, xgboost_baseline  # noqa: E402

#: Small enough to fit in a test, big enough to have more than one case.
FIT_CASES = 3
FIT_LIMIT = 300


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """A control model and a failure predictor, both trained on a corner of the corpus."""
    folder = tmp_path_factory.mktemp("support-models")
    model_path = folder / "control.json"
    meta_path = folder / "control.meta.json"
    xgboost_baseline.train(
        cases=FIT_CASES, limit=FIT_LIMIT, eval_limit=60,
        params={**xgboost_baseline.PARAMS, "n_estimators": 40, "max_depth": 4},
        model_path=model_path, meta_path=meta_path,
    )
    failure_model.run(
        out_dir=folder / "analysis", folds=2, n_trees=40, cases=FIT_CASES, limit=FIT_LIMIT,
        sample=40, model_path=model_path, meta_path=meta_path,
        predictor_path=folder / "predictor.json", predictor_meta_path=folder / "predictor.meta.json",
    )
    return model_path, meta_path, folder / "predictor.json", folder / "predictor.meta.json"


@pytest.fixture(scope="module")
def fallback_run(tmp_path_factory):
    """A stored zero_shot-shaped run: the answers a support run defers to."""
    run_dir = tmp_path_factory.mktemp("fallback") / "20261009-120000_dev"
    run_dir.mkdir(parents=True)
    windows = dataset.load_split("dev", limit=40)
    rows = [
        {
            "window_id": window.window_id,
            "action": "no_action",
            "real_action": window.real_action,
            "result_1": "no_action",
            "result_aux": "no_action",
            "case_id": window.case_id,
            "meta": {"reasoning_effort": None},
        }
        for window in windows
    ]
    results.write_predictions(run_dir / "predictions.jsonl", rows)
    (run_dir / "summary.json").write_text(
        json.dumps({"split": "dev", "dry_run": False, "accuracy": 0.2}), encoding="utf-8"
    )
    _stamp(run_dir)
    return run_dir


def _stamp(run_dir, **settings):
    """Write ``run.json`` the way the launcher does: on the current shared view."""
    (Path(run_dir) / "run.json").write_text(
        json.dumps({"view_version": view.VIEW_VERSION, **settings}), encoding="utf-8"
    )


@pytest.fixture
def strategy(models, fallback_run):
    model_path, meta_path, predictor_path, predictor_meta_path = models
    return support.SupportStrategy(
        threshold=0.0,
        fallback_run=fallback_run,
        model_path=model_path,
        meta_path=meta_path,
        predictor_path=predictor_path,
        predictor_meta_path=predictor_meta_path,
    )


def _windows(limit=8):
    return dataset.load_split("dev", limit=limit)


# --------------------------------------------------------------------------- #
# It is a strategy
# --------------------------------------------------------------------------- #


def test_it_is_a_strategy():
    instance = support.SupportStrategy()

    assert isinstance(instance, Strategy)
    assert instance.name == support.SupportStrategy.name == "support"
    assert instance.description.strip()
    assert instance.uses_model is False  # the run itself calls no model


def test_it_constructs_without_touching_the_disk(tmp_path):
    """The structure suite loads every strategy with no arguments."""
    instance = support.SupportStrategy(fallback_run=tmp_path / "nowhere")

    assert instance.threshold == support.DEFAULT_THRESHOLD
    assert instance.signal == "confidence"


def test_the_launcher_samples_flag_is_the_threshold_in_percent():
    instance = support.SupportStrategy(n_samples=75)

    assert instance.threshold == pytest.approx(0.75)


def test_the_run_effort_is_the_effort_of_the_answers_it_leans_on():
    """``--effort high`` defers to the high-effort run, ``--effort none`` to the none one."""
    from src import config

    # with no flag at all it follows the shared default, like every other strategy
    assert support.SupportStrategy().fallback_effort == config.get_settings().reasoning_effort
    assert support.SupportStrategy(reasoning_effort="high").fallback_effort == "high"
    assert support.SupportStrategy(reasoning_effort="low").fallback_effort == "low"
    assert support.SupportStrategy(reasoning_effort=None).fallback_effort is None
    assert support.SupportStrategy(reasoning_effort="none").fallback_effort is None


def test_an_explicit_fallback_effort_wins():
    instance = support.SupportStrategy(reasoning_effort="high", fallback_effort="low")

    assert instance.fallback_effort == "low"


def test_a_zero_threshold_keeps_everything_and_a_high_one_defers_everything(strategy, fallback_run):
    """The two ends of the sweep: the control alone, and the stored answers alone."""
    windows = _windows(6)

    keep_all = support.SupportStrategy(
        threshold=0.0, fallback_run=fallback_run, model_path=strategy.model_path,
        meta_path=strategy.meta_path, predictor_path=strategy.predictor_path,
        predictor_meta_path=strategy.predictor_meta_path,
    )
    defer_all = support.SupportStrategy(
        threshold=1.0, fallback_run=fallback_run, model_path=strategy.model_path,
        meta_path=strategy.meta_path, predictor_path=strategy.predictor_path,
        predictor_meta_path=strategy.predictor_meta_path,
    )

    assert {p.meta["source"] for p in keep_all.predict_many(windows)} == {"control"}
    assert {p.action for p in defer_all.predict_many(windows)} == {"no_action"}


def test_an_unknown_signal_is_rejected():
    with pytest.raises(ValueError, match="unknown signal"):
        support.SupportStrategy(signal="vibes")


def test_a_threshold_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="threshold must be"):
        support.SupportStrategy(threshold=1.5)


# --------------------------------------------------------------------------- #
# The deferral
# --------------------------------------------------------------------------- #


def test_a_confident_window_keeps_the_control_answer(strategy, fallback_run):
    """Above the threshold the answer must be the control's, whatever the fallback says."""
    window = _windows(1)[0]
    instance = support.SupportStrategy(
        threshold=0.0, fallback_run=fallback_run, model_path=strategy.model_path,
        meta_path=strategy.meta_path, predictor_path=strategy.predictor_path,
        predictor_meta_path=strategy.predictor_meta_path,
    )

    prediction = instance.predict(window)

    assert prediction.meta["source"] == "control"
    assert prediction.action == prediction.meta["xgb_action"]
    assert prediction.action != "no_action" or prediction.meta["xgb_action"] == "no_action"


def test_a_doubtful_window_gets_the_stored_answer(strategy, fallback_run):
    """Below the threshold the answer must be the one already on record."""
    window = _windows(1)[0]
    instance = support.SupportStrategy(
        threshold=1.0, fallback_run=fallback_run, model_path=strategy.model_path,
        meta_path=strategy.meta_path, predictor_path=strategy.predictor_path,
        predictor_meta_path=strategy.predictor_meta_path,
    )

    prediction = instance.predict(window)

    assert prediction.meta["source"] == "fallback"
    assert prediction.action == "no_action"
    assert prediction.meta["fallback_action"] == "no_action"
    assert prediction.meta["fallback_run"] == fallback_run.name


def test_a_window_the_fallback_does_not_cover_keeps_the_control(strategy, fallback_run):
    """A stored run that did not cover this window is not an answer."""
    window = dataset.load_split("test", limit=1)[0]
    instance = support.SupportStrategy(
        threshold=1.0, fallback_run=fallback_run, model_path=strategy.model_path,
        meta_path=strategy.meta_path, predictor_path=strategy.predictor_path,
        predictor_meta_path=strategy.predictor_meta_path,
    )

    prediction = instance.predict(window)

    assert prediction.meta["source"] == "control"
    assert prediction.meta["fallback_missing"] is True
    assert prediction.action == prediction.meta["xgb_action"]


def test_every_answer_is_an_action(strategy):
    assert {p.action for p in strategy.predict_many(_windows(10))} <= set(dataset.ACTIONS)


def test_the_effort_travels_in_the_metadata(strategy, fallback_run):
    """The leaderboard reads the effort from the predictions of a run that calls no model."""
    instance = support.SupportStrategy(
        threshold=0.65, fallback_run=fallback_run, reasoning_effort="high",
        model_path=strategy.model_path, meta_path=strategy.meta_path,
        predictor_path=strategy.predictor_path, predictor_meta_path=strategy.predictor_meta_path,
    )

    prediction = instance.predict(_windows(1)[0])

    assert prediction.meta["reasoning_effort"] == "high"
    assert prediction.meta["effort"] == "high"
    assert prediction.meta["threshold"] == pytest.approx(0.65)
    assert prediction.meta["k"] == 65


# --------------------------------------------------------------------------- #
# The signals
# --------------------------------------------------------------------------- #


def test_the_confidence_signal_is_the_biggest_probability(strategy):
    window = _windows(1)[0]

    probabilities = strategy.probabilities(window)
    score, detail = strategy.score(window, probabilities)

    assert score == pytest.approx(float(probabilities.max()))
    assert detail["confidence"] == pytest.approx(score, abs=1e-6)
    assert "failure_risk" not in detail


def test_the_failure_signal_inverts_the_predicted_risk(strategy):
    window = _windows(1)[0]
    instance = support.SupportStrategy(
        signal="failure", threshold=0.5, fallback_run=strategy.fallback_run,
        model_path=strategy.model_path, meta_path=strategy.meta_path,
        predictor_path=strategy.predictor_path, predictor_meta_path=strategy.predictor_meta_path,
    )

    score, detail = instance.score(window, instance.probabilities(window))

    assert 0.0 <= detail["failure_risk"] <= 1.0
    assert score == pytest.approx(1.0 - detail["failure_risk"], abs=1e-6)


def test_the_combined_signal_averages_the_two(strategy):
    window = _windows(1)[0]
    instance = support.SupportStrategy(
        signal="combined", threshold=0.5, fallback_run=strategy.fallback_run,
        model_path=strategy.model_path, meta_path=strategy.meta_path,
        predictor_path=strategy.predictor_path, predictor_meta_path=strategy.predictor_meta_path,
    )

    score, detail = instance.score(window, instance.probabilities(window))

    expected = (detail["confidence"] + (1.0 - detail["failure_risk"])) / 2
    assert score == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# Finding the stored answers
# --------------------------------------------------------------------------- #


def test_a_dry_run_is_not_a_source_of_answers(tmp_path):
    run_dir = tmp_path / "zero_shot" / "20261009-120000_dev"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"dry_run": True}), encoding="utf-8")

    assert support.latest_run(results_root=tmp_path) is None


def test_the_newest_finished_run_of_the_right_split_wins(tmp_path, fallback_run):
    older = tmp_path / "zero_shot" / "20261008-100000_dev"
    older.mkdir(parents=True)
    (older / "summary.json").write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    _stamp(older)
    newer = tmp_path / "zero_shot" / "20261009-120000_test"
    newer.mkdir(parents=True)
    (newer / "summary.json").write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    _stamp(newer)

    assert support.latest_run(results_root=tmp_path, split="dev").name == "20261008-100000_dev"
    assert support.latest_run(results_root=tmp_path, split="test").name == "20261009-120000_test"
    assert support.latest_run(results_root=tmp_path).name == "20261009-120000_test"


def _write_effort_run(root, run_id, effort, split="test"):
    """A run folder as the launcher writes it, with the effort in run.json."""
    run_dir = Path(root) / "zero_shot" / f"{run_id}_{split}"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"split": split, "dry_run": False}), encoding="utf-8"
    )
    _stamp(run_dir, effort=effort)
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"window_id": "w1", "action": "no_action",
                    "meta": {"reasoning_effort": effort}}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_the_effort_is_read_from_the_run_settings(tmp_path):
    run_dir = _write_effort_run(tmp_path, "20261009-120000", "high")

    assert support.run_effort(run_dir) == "high"


def test_an_omitted_effort_reads_as_none_not_as_missing(tmp_path):
    """``--effort none`` is a real setting: it must match the run that omitted the parameter."""
    run_dir = _write_effort_run(tmp_path, "20261009-120000", None)

    assert support.run_effort(run_dir) is None


def test_the_effort_falls_back_to_the_predictions(tmp_path):
    """A run recorded before run.json carried the field keeps it in its metadata."""
    run_dir = Path(tmp_path) / "zero_shot" / "20261008-120000_test"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"dry_run": False}), encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"window_id": "w1", "action": "no_action",
                    "meta": {"reasoning_effort": "low"}}) + "\n",
        encoding="utf-8",
    )

    assert support.run_effort(run_dir) == "low"


def test_a_run_folder_without_metadata_is_reported_as_unknown(tmp_path):
    (Path(tmp_path) / "empty").mkdir()

    assert support.run_effort(tmp_path / "empty") is None


def test_the_fallback_is_selected_by_effort(tmp_path):
    """The whole point of the effort dimension: one fallback per effort, not one overall."""
    _write_effort_run(tmp_path, "20261008-100000", None)
    _write_effort_run(tmp_path, "20261008-110000", "low")
    _write_effort_run(tmp_path, "20261008-120000", "high")

    assert support.latest_run(results_root=tmp_path).name == "20261008-120000_test"
    assert support.latest_run(results_root=tmp_path, effort="low").name == "20261008-110000_test"
    assert support.latest_run(results_root=tmp_path, effort=None).name == "20261008-100000_test"
    assert support.latest_run(results_root=tmp_path, effort="medium") is None


def test_a_missing_fallback_for_that_effort_names_the_effort(tmp_path):
    _write_effort_run(tmp_path, "20261008-100000", None)
    instance = support.SupportStrategy(reasoning_effort="high", fallback_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="high"):
        instance.fallback()


def test_the_strategy_finds_the_fallback_of_its_own_effort(tmp_path, fallback_run):
    """End to end: ``--effort low`` must load the low run, not the newest one."""
    _write_effort_run(tmp_path, "20261008-100000", None)
    low = _write_effort_run(tmp_path, "20261008-110000", "low")
    Path(low / "predictions.jsonl").write_text(
        json.dumps({"window_id": "w1", "action": "reduce_opioid",
                    "meta": {"reasoning_effort": "low"}}) + "\n",
        encoding="utf-8",
    )
    _write_effort_run(tmp_path, "20261009-120000", "high")

    instance = support.SupportStrategy(reasoning_effort="low", fallback_root=tmp_path,
                                       threshold=0.0)
    actions, run_dir = instance.fallback()

    assert run_dir.name == "20261008-110000_test"
    assert actions == {"w1": "reduce_opioid"}


# --------------------------------------------------------------------------- #
# The model of the answers it leans on
# --------------------------------------------------------------------------- #


def _write_model_run(root, run_id, model, effort="high", split="test"):
    """A run folder as the launcher writes it, with the model in run.json."""
    run_dir = Path(root) / "zero_shot" / f"{run_id}_{split}"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"split": split, "dry_run": False}), encoding="utf-8"
    )
    _stamp(run_dir, effort=effort, model=model)
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"window_id": "w1", "action": "no_action",
                    "meta": {"reasoning_effort": effort}}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_the_fallback_is_selected_by_model(tmp_path):
    """Two models in the same tree: each support run must lean on the answers of its own."""
    _write_model_run(tmp_path, "20261008-100000", "gpt-5.4-mini")
    _write_model_run(tmp_path, "20261008-120000", "deepseek-flash")

    assert support.latest_run(results_root=tmp_path,
                              model="gpt-5.4-mini").name == "20261008-100000_test"
    assert support.latest_run(results_root=tmp_path,
                              model="deepseek-flash").name == "20261008-120000_test"
    assert support.latest_run(results_root=tmp_path, model="gemini-pro-latest") is None


def test_a_run_without_a_model_counts_as_the_projects_own(tmp_path):
    """Runs stored before the model was part of run.json are the project model's runs."""
    _write_effort_run(tmp_path, "20261008-100000", "high")

    assert support.latest_run(results_root=tmp_path,
                              model=config.DEFAULT_MODEL).name == "20261008-100000_test"


def test_the_strategy_finds_the_fallback_of_its_own_model(tmp_path, models):
    """End to end: ``--model deepseek-flash`` must not lean on the gpt-5.4-mini answers."""
    _write_model_run(tmp_path, "20261008-100000", "gpt-5.4-mini")
    mine = _write_model_run(tmp_path, "20261009-120000", "deepseek-flash")
    Path(mine / "predictions.jsonl").write_text(
        json.dumps({"window_id": "w1", "action": "reduce_opioid",
                    "meta": {"reasoning_effort": "high"}}) + "\n",
        encoding="utf-8",
    )
    model_path, meta_path, predictor_path, predictor_meta_path = models

    instance = support.SupportStrategy(
        model="deepseek-flash", fallback_root=tmp_path, threshold=0.0,
        model_path=model_path, meta_path=meta_path,
        predictor_path=predictor_path, predictor_meta_path=predictor_meta_path,
    )
    actions, run_dir = instance.fallback()

    assert run_dir.name == "20261009-120000_test"
    assert actions == {"w1": "reduce_opioid"}


def test_a_missing_fallback_for_that_model_names_the_model(tmp_path):
    _write_model_run(tmp_path, "20261008-100000", "gpt-5.4-mini")
    instance = support.SupportStrategy(model="deepseek-flash", fallback_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="deepseek-flash"):
        instance.fallback()


def test_the_model_it_leans_on_travels_in_the_metadata(strategy):
    prediction = strategy.predict(_windows(1)[0])

    assert prediction.meta["fallback_model"] == config.DEFAULT_MODEL


def test_the_inherited_model_is_recorded_unless_it_is_the_projects_own():
    """An empty ``model`` cell already means the project's own model."""
    assert support.SupportStrategy(model="deepseek-flash").inherited_model == "deepseek-flash"
    assert support.SupportStrategy().inherited_model is None


def test_the_answers_are_read_by_window(fallback_run):
    actions, directory = support.load_fallback(fallback_run)

    assert len(actions) == 40
    assert directory == fallback_run
    assert set(actions.values()) == {"no_action"}


def test_a_stored_run_without_predictions_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="no predictions"):
        support.load_fallback(tmp_path / "empty")


def test_missing_models_say_how_to_build_them(tmp_path, fallback_run):
    instance = support.SupportStrategy(
        fallback_run=fallback_run,
        model_path=tmp_path / "nope.json",
        meta_path=tmp_path / "nope.meta.json",
    )

    with pytest.raises(FileNotFoundError, match="train_xgboost"):
        instance.predict(_windows(1)[0])


def test_a_run_made_on_another_view_is_never_a_fallback(tmp_path):
    """Answers given on an older window are not the model this hybrid leans on."""
    old = _write_effort_run(tmp_path, "20261009-120000", "high")
    (old / "run.json").write_text(json.dumps({"effort": "high"}), encoding="utf-8")

    assert support.latest_run(results_root=tmp_path) is None
    with pytest.raises(ValueError, match="view"):
        support.load_fallback(old)
