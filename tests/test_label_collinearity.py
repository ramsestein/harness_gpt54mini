# -*- coding: utf-8 -*-
"""How collinear the dataset's labels are with the window features.

The labels are not model inputs (they live in ``output``), so this report is about the
dataset, not about any strategy: it measures how much of the answer is written in the
features, linearly.
"""

import csv

import numpy as np
import pytest

from src import dataset
from src.control import label_collinearity as report
from src.control import ml_data


class FakeWindow:
    """Just enough of a window: the labels this report reads live in ``output``."""

    def __init__(self, output):
        self.output = output


def test_a_ratio_label_is_read_from_the_output():
    windows = [
        FakeWindow({"result_1": {"action": "no_action", "ratio": 0.75}}),
        FakeWindow({"result_1": {"action": "no_action", "ratio": 0.25}}),
    ]

    targets = report.label_targets(windows, actions=["no_action"])

    assert list(targets["result_1_ratio"]) == [0.75, 0.25]


def test_an_action_label_becomes_a_one_vs_rest_indicator():
    windows = [
        FakeWindow({"result_real": "no_action"}),
        FakeWindow({"result_real": "reduce_opioid"}),
    ]

    targets = report.label_targets(windows, actions=["no_action", "reduce_opioid"])

    assert list(targets["result_real__no_action"]) == [1.0, 0.0]
    assert list(targets["result_real__reduce_opioid"]) == [0.0, 1.0]


def test_a_missing_label_is_not_a_zero():
    """A window without an answer must not look like "none of the actions"."""
    windows = [FakeWindow({}), FakeWindow({"result_real": "no_action"})]

    targets = report.label_targets(windows, actions=["no_action"])

    assert np.isnan(targets["result_real__no_action"][0])
    assert targets["result_real__no_action"][1] == 1.0
    assert np.isnan(targets["result_1_ratio"][0])


def test_the_targets_cover_the_five_documented_output_variables():
    windows = dataset.load_split("dev", limit=3)

    targets = report.label_targets(windows)

    assert "result_1_ratio" in targets and "result_aux_ratio" in targets
    assert "result_real__no_action" in targets
    assert "result_1__increase_hypnotic" in targets
    assert "result_aux__reduce_opioid" in targets


def test_the_correlation_of_a_feature_with_itself_is_one():
    values = np.random.default_rng(0).normal(size=200)
    mask = np.ones(200, dtype=bool)

    assert report.correlation(values, values, mask) == pytest.approx(1.0)


def test_a_feature_that_is_the_label_gets_the_top_rank_and_a_huge_vif():
    generator = np.random.default_rng(1)
    X = generator.normal(size=(300, 4))
    target = {"a_free_label": np.asarray(X[:, 2], dtype=float)}

    rows, summary = report.collinearity(X=X, names=["a", "b", "c", "d"], targets=target)

    assert rows[0]["feature"] == "c"
    assert rows[0]["abs_corr"] == pytest.approx(1.0)
    assert float(summary[0]["top_abs_corr"]) == pytest.approx(1.0)
    assert float(summary[0]["r2"]) > 0.999
    # a perfect linear fit leaves no finite VIF, so the cell stays empty
    assert summary[0]["vif"] == "" or float(summary[0]["vif"]) > 1e6


def test_an_independent_label_has_no_collinear_feature():
    generator = np.random.default_rng(2)
    X = generator.normal(size=(500, 6))

    rows, summary = report.collinearity(
        X=X, names=list("abcdef"), targets={"noise": generator.normal(size=500)}
    )

    assert float(summary[0]["r2"]) < 0.05
    assert abs(float(summary[0]["top_abs_corr"])) < 0.2
    assert len(rows) == 6


def test_every_feature_is_ranked_once_per_target():
    generator = np.random.default_rng(3)
    X = generator.normal(size=(200, 5))
    targets = {"one": generator.normal(size=200), "two": generator.normal(size=200)}

    rows, summary = report.collinearity(X=X, names=list("abcde"), targets=targets)

    assert len(rows) == 10
    assert len(summary) == 2
    for name in ("one", "two"):
        ranked = [row for row in rows if row["target"] == name]
        assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))
        assert {row["feature"] for row in ranked} == set("abcde")


def test_a_constant_target_is_reported_as_unexplained():
    X = np.random.default_rng(4).normal(size=(200, 3))

    _, summary = report.collinearity(X=X, names=list("abc"), targets={"flat": np.ones(200)})

    assert summary[0]["r2"] == ""


def test_a_too_short_target_is_left_out():
    X = np.random.default_rng(5).normal(size=(200, 3))
    target = np.random.default_rng(6).normal(size=200)
    target[: 200 - report.MIN_OBSERVATIONS + 1] = np.nan

    rows, summary = report.collinearity(X=X, names=list("abc"), targets={"short": target})

    assert rows == []
    assert summary[0]["n_windows"] < report.MIN_OBSERVATIONS


def test_the_report_is_written(tmp_path):
    out_dir, rows, summary = report.run(out_dir=tmp_path / "analysis", cases=3)

    written = list(csv.DictReader((out_dir / "label_collinearity.csv").open(encoding="utf-8")))
    assert list(written[0]) == report.COLLINEARITY_COLUMNS
    assert len(written) == len(rows)
    vif = list(csv.DictReader((out_dir / "label_vif.csv").open(encoding="utf-8")))
    assert list(vif[0]) == report.VIF_COLUMNS
    assert len(vif) == len(summary)
    assert {row["target"] for row in vif} >= {"result_1_ratio", "result_real__no_action"}


def test_it_uses_the_same_feature_set_as_the_vif_report():
    """Same matrix as the VIF report, so the two are read the same way."""
    rows, _ = report.collinearity(cases=2)

    assert {row["feature"] for row in rows} <= set(ml_data.feature_names())
    assert len({row["feature"] for row in rows}) > 50
