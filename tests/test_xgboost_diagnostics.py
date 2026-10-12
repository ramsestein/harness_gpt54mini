# -*- coding: utf-8 -*-
"""The SHAP and VIF reports of the XGBoost control."""

import csv

import numpy as np
import pytest

from harness.few_shot import retrieval
from src import dataset
from src.control import ml_data, xgboost_baseline, xgboost_diagnostics

pytest.importorskip("xgboost")


# --------------------------------------------------------------------------- #
# VIF
# --------------------------------------------------------------------------- #


def test_a_pair_of_identical_features_is_flagged():
    base = np.random.default_rng(0).normal(size=(400, 1))
    X = np.hstack([base, base, np.random.default_rng(1).normal(size=(400, 1))])
    names = ["twin_a", "twin_b", "lonely"]

    rows = xgboost_diagnostics.vif_table(X=X, names=names)
    by_name = {row["feature"]: row for row in rows}

    assert by_name["twin_a"]["duplicate_of"] == "twin_b"
    assert by_name["twin_b"]["duplicate_of"] == "twin_a"
    assert by_name["lonely"]["duplicate_of"] == ""
    assert by_name["twin_a"]["max_abs_corr"] == pytest.approx(1.0, abs=1e-6)


def test_independent_features_have_a_vif_near_one():
    X = np.random.default_rng(2).normal(size=(2000, 8))

    rows = xgboost_diagnostics.vif_table(X=X, names=[f"independent_{i}" for i in range(8)])

    assert all(row["vif"] == pytest.approx(1.0, abs=0.05) for row in rows)
    assert all(row["vif"] > 0 for row in rows)


def test_a_collinear_copy_gets_a_high_vif():
    generator = np.random.default_rng(3)
    x = generator.normal(size=3000)
    X = np.column_stack([x, x + generator.normal(scale=0.01, size=3000)])

    rows = xgboost_diagnostics.vif_table(X=X, names=["signal", "signal_echo"])

    assert all(row["vif"] > 50 for row in rows)


def test_the_vif_is_sorted_and_covers_every_variable():
    rows = xgboost_diagnostics.vif_table(X=np.random.default_rng(4).normal(size=(200, 6)),
                                         names=list("abcdef"))

    assert len(rows) == 6
    assert {row["feature"] for row in rows} == set("abcdef")
    assert [row["vif"] for row in rows] == sorted((row["vif"] for row in rows), reverse=True)


def test_a_constant_column_is_left_out():
    X = np.column_stack([np.random.default_rng(5).normal(size=300), np.full(300, 7.0)])

    rows = xgboost_diagnostics.vif_table(X=X, names=["varying", "constant"])

    assert [row["feature"] for row in rows] == ["varying"]


def test_a_missing_value_does_not_break_the_table():
    X = np.random.default_rng(6).normal(size=(300, 3))
    X[:100, 1] = np.nan

    rows = xgboost_diagnostics.vif_table(X=X, names=["a", "b", "c"])

    assert len(rows) == 3
    assert all(np.isfinite(row["vif"]) for row in rows)


def test_the_family_table_groups_by_prefix():
    rows = [
        {"feature": "bis_current", "family": "bis", "vif": 300.0},
        {"feature": "bis_mean", "family": "bis", "vif": 200.0},
        {"feature": "hr_current", "family": "hr", "vif": 2.0},
    ]

    families = {row["family"]: row for row in xgboost_diagnostics.family_table(rows)}

    assert families["bis"]["n_features"] == 2
    assert families["bis"]["max_vif"] == 300.0
    assert families["bis"]["n_above_10"] == 2
    assert families["bis"]["n_above_100"] == 2
    assert families["hr"]["n_above_10"] == 0
    assert families["bis"]["median_vif"] == 250.0


def test_the_family_of_a_name_is_its_prefix():
    assert xgboost_diagnostics.family_of("propofol_ce_current") == "propofol"
    assert xgboost_diagnostics.family_of("lab_k") == "lab"
    assert xgboost_diagnostics.family_of("BIS_current") == "bis"


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_model(tmp_path_factory):
    """A cheap model that still splits, so the explainer has something to explain.

    The production hyper-parameters refuse to split a handful of windows
    (``min_child_weight=5``): a model of leaves explains nothing, which is correct but not
    what these tests are about. Same features, same data, shallower and fewer trees.
    """
    folder = tmp_path_factory.mktemp("shap-model")
    model_path = folder / "xgboost.json"
    meta_path = folder / "xgboost.meta.json"
    xgboost_baseline.train(
        cases=8,
        limit=2000,
        eval_limit=150,
        params={
            **xgboost_baseline.PARAMS,
            "n_estimators": 80,
            "max_depth": 4,
            "min_child_weight": 1,
        },
        model_path=model_path,
        meta_path=meta_path,
    )
    return model_path, meta_path


def test_shap_explains_every_feature(small_model):
    model_path, meta_path = small_model
    names = ml_data.feature_names()

    importance, per_class = xgboost_diagnostics.shap_importance(
        names=names, sample=60, model_path=model_path, meta_path=meta_path
    )

    assert len(importance) == len(names)
    assert per_class.shape == (len(names), len(dataset.ACTIONS))
    assert [row["rank"] for row in importance] == list(range(1, len(names) + 1))
    assert {row["feature"] for row in importance} == set(names)


def test_the_contributions_add_up_to_the_prediction(small_model):
    """TreeSHAP's defining property: softmax(sum of contributions) is the prediction."""
    import numpy as np
    import xgboost as xgb

    from src.control import xgboost_baseline

    model_path, meta_path = small_model
    names = ml_data.feature_names()
    model, _ = xgboost_baseline.load_model(model_path, meta_path)
    X, _, _ = ml_data.load_split(ml_data.EVAL_SPLIT, names=names, limit=25)

    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    rebuilt = np.exp(contributions.sum(axis=2))
    rebuilt = rebuilt / rebuilt.sum(axis=1, keepdims=True)

    assert np.allclose(rebuilt, model.predict_proba(X), atol=1e-4)


def test_shap_is_sorted_by_importance(small_model):
    model_path, meta_path = small_model

    importance, _ = xgboost_diagnostics.shap_importance(
        names=ml_data.feature_names(), sample=50, model_path=model_path, meta_path=meta_path
    )

    values = [row["mean_abs_shap"] for row in importance]
    assert values == sorted(values, reverse=True)
    assert all(value >= 0 for value in values)


def test_the_model_really_uses_something(small_model):
    model_path, meta_path = small_model

    importance, _ = xgboost_diagnostics.shap_importance(
        names=ml_data.feature_names(), sample=80, model_path=model_path, meta_path=meta_path
    )

    assert importance[0]["mean_abs_shap"] > 1e-3  # a decision, not a constant
    assert sum(row["mean_abs_shap"] for row in importance) > 0.01


def test_a_model_that_could_not_split_explains_nothing(tmp_path):
    """With the production parameters a handful of windows gives a forest of leaves.

    That model still answers (always the same class, at chance level), so the explainer
    correctly attributes everything to the bias and nothing to any feature.
    """
    model_path = tmp_path / "stumps.json"
    meta_path = tmp_path / "stumps.meta.json"
    metrics = xgboost_baseline.train(
        cases=2, limit=250, eval_limit=60, model_path=model_path, meta_path=meta_path
    )

    importance, _ = xgboost_diagnostics.shap_importance(
        names=ml_data.feature_names(), sample=40, model_path=model_path, meta_path=meta_path
    )

    assert metrics["n_fit"] < 100  # too little data to split
    assert max(row["mean_abs_shap"] for row in importance) == pytest.approx(0.0, abs=1e-12)


def test_shap_is_deterministic(small_model):
    model_path, meta_path = small_model
    names = ml_data.feature_names()

    first, _ = xgboost_diagnostics.shap_importance(
        names=names, sample=40, seed=7, model_path=model_path, meta_path=meta_path
    )
    second, _ = xgboost_diagnostics.shap_importance(
        names=names, sample=40, seed=7, model_path=model_path, meta_path=meta_path
    )

    assert first == second


# --------------------------------------------------------------------------- #
# The reports on disk
# --------------------------------------------------------------------------- #


def test_the_diagnostics_write_three_tables(small_model, tmp_path):
    model_path, meta_path = small_model
    names = ml_data.feature_names()

    out_dir, vif_rows, importance = xgboost_diagnostics.run_diagnostics(
        out_dir=tmp_path / "analysis", sample=50, limit=300, cases=2,
        model_path=model_path, meta_path=meta_path,
    )

    assert out_dir == tmp_path / "analysis"
    written = list(csv.DictReader((out_dir / "shap_importance.csv").open(encoding="utf-8")))
    assert len(written) == len(names)
    assert list(written[0]) == xgboost_diagnostics.SHAP_COLUMNS
    families = list(csv.DictReader((out_dir / "vif_by_family.csv").open(encoding="utf-8")))
    assert families and list(families[0]) == xgboost_diagnostics.FAMILY_COLUMNS
    # a two-case corpus leaves many columns empty; they are dropped, the rest is reported
    assert 0 < len(vif_rows) <= len(names)
    assert {row["feature"] for row in vif_rows} <= set(names)
    assert len(importance) == len(names)


def test_the_vif_uses_the_same_features_as_the_model():
    """The VIF must describe the matrix the model sees, not another one."""
    names = ml_data.feature_names()

    assert names == retrieval.SimilarityIndex.load(retrieval.DEFAULT_INDEX_PATH).names
