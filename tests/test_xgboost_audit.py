# -*- coding: utf-8 -*-
"""The ablation that checks the XGBoost control is reading physiology, not bookkeeping."""

import csv
from pathlib import Path

import pytest

from src.control import ml_data, xgboost_audit

pytest.importorskip("xgboost")


@pytest.fixture(scope="module")
def names():
    return ml_data.feature_names()


def test_the_taxonomy_comes_from_the_dataset_dictionary():
    types = xgboost_audit.variable_types()

    assert types
    assert types["decision_time_abs"] == "metadata"
    assert types["BIS_current"] == "signal"
    assert set(types.values()) >= {"signal", "vital", "metadata", "lab", "patient"}


def test_a_missing_dictionary_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="data dictionary"):
        xgboost_audit.variable_types(tmp_path / "nope.csv")


def test_the_full_variant_keeps_everything(names):
    assert xgboost_audit.feature_subset("all", names=names) == names


def test_dropping_the_metadata_drops_the_clocks_and_counters(names):
    subset = xgboost_audit.feature_subset("no_metadata", names=names)

    assert "decision_time_abs" not in subset
    assert "n_actions_hist" not in subset
    assert "staleness_max" not in subset
    assert "BIS_current" in subset  # physiology stays
    assert "propofol_ce_current" in subset
    assert len(subset) < len(names)


def test_the_signals_only_variant_keeps_just_the_measured_state(names):
    subset = xgboost_audit.feature_subset("signals_only", names=names)
    types = xgboost_audit.variable_types()

    # the window aggregates (signal) and the recent trend families (trend)
    assert "BIS_current" in subset and "hr_current" in subset and "map_slope_short" in subset
    assert "propofol_ce_current" in subset
    assert "lab_k" not in subset  # a lab is not the bedside state
    assert "pt_age" not in subset
    assert "decision_time_abs" not in subset
    assert "map_below_65_sec" not in subset  # a band counter, not a signal
    assert all(types[name] in ("signal", "trend") for name in subset)


def test_the_variants_shrink_in_the_expected_order(names):
    sizes = [
        len(xgboost_audit.feature_subset(variant, names=names))
        for variant in ("all", "no_metadata", "signals_only")
    ]

    assert sizes[0] > sizes[1] > sizes[2] > 10


def test_an_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        xgboost_audit.feature_subset("vibes")

def test_the_audit_writes_one_row_per_variant(tmp_path):
    rows, table = xgboost_audit.run_audit(
        variants=("all", "signals_only"),
        model_dir=tmp_path / "models",
        out_dir=tmp_path / "analysis",
        cases=3,
        limit=300,
        eval_limit=40,
    )

    assert [row["variant"] for row in rows] == ["all", "signals_only"]
    assert table.is_file()
    written = list(csv.DictReader(table.open(encoding="utf-8")))
    assert [row["variant"] for row in written] == ["all", "signals_only"]
    assert int(written[0]["n_features"]) > int(written[1]["n_features"])
    assert (tmp_path / "models" / "all.json").is_file()
    assert (tmp_path / "models" / "signals_only.meta.json").is_file()
