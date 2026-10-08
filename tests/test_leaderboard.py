# -*- coding: utf-8 -*-
"""The cross-strategy table: one row per generated run of every strategy.

``results_harness/leaderboard.csv`` holds every run found under ``results_harness/``
(``<strategy>/<run_id>/`` and ``control/<baseline>/<run_id>/``) with its agreement
against the real action, against ``result_1`` and against ``result_1`` or ``result_aux``.
"""

import csv
import json
from pathlib import Path

import pytest

from src import config, dataset, leaderboard, results


def _summary(split="dev", n=4, real=0.5, first=0.75, plausible=None, dry_run=False):
    summary = {
        "strategy": "anything",
        "split": split,
        "dry_run": dry_run,
        "n_windows": n,
        "n_scored": n,
        "errors": 0,
        "accuracy": real,
        "agreement_result_1": first,
    }
    if plausible is not None:
        summary["agreement_result_plausible"] = plausible
    return summary


def _prediction(window_id, action, first, aux):
    """A stored prediction: the analysis needs the truth, the summary does not."""
    return {
        "window_id": window_id,
        "action": action,
        "real_action": first,
        "result_1": first,
        "result_aux": aux,
    }


def _scored(window_id, action, real, case_id=None):
    """A stored prediction with the truth the analysis needs."""
    return {
        "window_id": window_id,
        "action": action,
        "real_action": real,
        "result_1": action,
        "result_aux": action,
        "case_id": case_id or window_id.split("_")[0],
    }


def _write_run(root, group, strategy, run_id, summary, predictions=None, settings=None):
    """Build one run folder the way the launcher writes it."""
    base = Path(root) / (Path("control") / strategy if group == "control" else strategy)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if settings is not None:
        (run_dir / "run.json").write_text(json.dumps(settings), encoding="utf-8")
    if predictions is not None:
        results.write_predictions(run_dir / "predictions.jsonl", predictions)
    return run_dir


def _read_table(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# Finding the runs
# --------------------------------------------------------------------------- #


def test_iter_runs_finds_harness_and_control_runs(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary())
    _write_run(tmp_path, "control", "random", "20261008-121000_test", _summary(split="test"))

    found = [(group, strategy, run_dir.name) for group, strategy, run_dir in leaderboard.iter_runs(tmp_path)]

    assert found == [
        ("control", "random", "20261008-121000_test"),
        ("harness", "zero_shot", "20261008-120000_dev"),
    ]


def test_iter_runs_ignores_caches_and_folders_without_a_summary(tmp_path):
    _write_run(tmp_path, "harness", "few_shot", "20261008-120000_dev", _summary())
    (tmp_path / "few_shot" / "_index").mkdir(parents=True)  # cache, not a run
    (tmp_path / "cot").mkdir()  # no run yet
    (tmp_path / "cot" / "20261008-130000_dev").mkdir()  # incomplete run
    (tmp_path / "leaderboard.csv").write_text("", encoding="utf-8")  # the table itself

    assert [strategy for _, strategy, _ in leaderboard.iter_runs(tmp_path)] == ["few_shot"]


def test_a_dry_run_is_not_a_result(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary(dry_run=True))

    assert list(leaderboard.iter_runs(tmp_path)) == []


def test_a_missing_results_folder_is_empty_not_an_error(tmp_path):
    assert list(leaderboard.iter_runs(tmp_path / "nowhere")) == []


def test_a_root_inside_control_still_reports_the_control_group(tmp_path):
    """Pointing the scan at ``control/`` must not turn the controls into strategies."""
    _write_run(tmp_path, "control", "random", "20261008-121000_test", _summary(split="test"))

    found = [
        (group, strategy)
        for group, strategy, _ in leaderboard.iter_runs(tmp_path / "control")
    ]

    assert found == [("control", "random")]


# --------------------------------------------------------------------------- #
# One row
# --------------------------------------------------------------------------- #


def test_a_row_carries_the_three_agreements(tmp_path):
    run_dir = _write_run(
        tmp_path, "harness", "zero_shot", "20261008-120000_dev",
        _summary(n=10, real=0.3, first=0.5, plausible=0.8),
    )

    row = leaderboard.run_row("harness", "zero_shot", run_dir)

    assert row["group"] == "harness"
    assert row["strategy"] == "zero_shot"
    assert row["split"] == "dev"
    assert float(row["agreement_result_real"]) == 0.3
    assert float(row["agreement_result_1"]) == 0.5
    assert float(row["agreement_result_plausible"]) == 0.8
    assert row["run_dir"] == str(run_dir)


def test_the_table_leaves_the_run_bookkeeping_out():
    for dropped in ("run_id", "n_windows", "n_scored"):
        assert dropped not in leaderboard.COLUMNS
    for kept in ("effort", "k", "group", "strategy", "split", "errors", "run_dir"):
        assert kept in leaderboard.COLUMNS


def test_the_table_carries_the_uncertainty_and_prior_free_metrics():
    for column in (
        "ci95_low",
        "ci95_high",
        "balanced_accuracy",
        "macro_f1",
        "kappa",
        *[f"recall_{action}" for action in dataset.ACTIONS],
        *[f"predicted_{action}" for action in dataset.ACTIONS],
    ):
        assert column in leaderboard.COLUMNS, column


def test_the_analysis_is_computed_from_the_predictions(tmp_path):
    """A constant answer to a half-and-half truth: accuracy 0.5, no skill at all."""
    predictions = [
        _scored(
            f"case{case}_w{index}",
            "no_action",
            "no_action" if index < 5 else "reduce_opioid",
            f"case{case}",
        )
        for case in range(2)
        for index in range(10)
    ]
    run_dir = _write_run(
        tmp_path, "control", "always_no_action", "20261008-120000_dev",
        _summary(real=0.5), predictions,
    )

    row = leaderboard.run_row("control", "always_no_action", run_dir)

    assert float(row["agreement_result_real"]) == 0.5
    assert float(row["balanced_accuracy"]) == 0.2  # only one of the five labels is recalled
    assert float(row["kappa"]) == pytest.approx(0.0)  # and it carries no information
    assert float(row["recall_no_action"]) == 1.0
    assert float(row["recall_reduce_opioid"]) == 0.0
    assert float(row["predicted_no_action"]) == 1.0
    assert float(row["predicted_reduce_opioid"]) == 0.0
    assert float(row["ci95_low"]) <= 0.5 <= float(row["ci95_high"])


def test_a_run_without_predictions_leaves_the_analysis_empty(tmp_path):
    run_dir = _write_run(tmp_path, "harness", "cot", "20261008-120000_dev", _summary())

    row = leaderboard.run_row("harness", "cot", run_dir)

    assert row["ci95_low"] is None and row["ci95_high"] is None
    assert row["balanced_accuracy"] is None and row["macro_f1"] is None
    assert row["recall_no_action"] is None and row["predicted_no_action"] is None


def test_the_effort_and_the_k_come_from_the_run_settings(tmp_path):
    run_dir = _write_run(
        tmp_path, "harness", "few_shot", "20261008-120000_dev", _summary(),
        settings={"effort": "high", "k": 3},
    )

    row = leaderboard.run_row("harness", "few_shot", run_dir)

    assert row["effort"] == "high"
    assert row["k"] == 3


def test_the_effort_and_the_k_fall_back_to_the_predictions(tmp_path):
    """A run recorded before those fields existed keeps them in its prediction metadata."""
    predictions = [dict(_prediction("w1", "a", "a", "b"), meta={"reasoning_effort": "low", "k": 10})]
    run_dir = _write_run(
        tmp_path, "harness", "few_shot", "20261008-120000_dev", _summary(), predictions
    )

    row = leaderboard.run_row("harness", "few_shot", run_dir)

    assert row["effort"] == "low"
    assert row["k"] == 10


def test_the_run_settings_win_over_the_predictions(tmp_path):
    predictions = [dict(_prediction("w1", "a", "a", "b"), meta={"reasoning_effort": "low", "k": 10})]
    run_dir = _write_run(
        tmp_path, "harness", "few_shot", "20261008-120000_dev", _summary(), predictions,
        settings={"effort": "high", "k": 3},
    )

    row = leaderboard.run_row("harness", "few_shot", run_dir)

    assert row["effort"] == "high"
    assert row["k"] == 3


def test_a_strategy_that_uses_neither_leaves_the_cells_empty(tmp_path):
    run_dir = _write_run(
        tmp_path, "control", "always_no_action", "20261008-120000_dev", _summary(),
        predictions=[_prediction("w1", "a", "a", "b")],
        settings={"effort": None, "model": None},
    )

    row = leaderboard.run_row("control", "always_no_action", run_dir)

    assert row["effort"] is None
    assert row["k"] is None


def test_a_run_without_settings_still_produces_a_row(tmp_path):
    run_dir = _write_run(tmp_path, "harness", "cot", "20261008-120000_dev", _summary())

    row = leaderboard.run_row("harness", "cot", run_dir)

    assert row["strategy"] == "cot"
    assert row["effort"] is None and row["k"] is None


def test_the_plausible_agreement_is_read_from_the_summary(tmp_path):
    _write_run(
        tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary(plausible=0.61)
    )

    row = leaderboard.run_row("harness", "zero_shot", tmp_path / "zero_shot" / "20261008-120000_dev")

    assert float(row["agreement_result_plausible"]) == 0.61


def test_the_plausible_agreement_is_backfilled_from_the_predictions(tmp_path):
    """Runs older than the metric still get a value, straight from their predictions."""
    predictions = [
        _prediction("w1", "a", "a", "b"),  # equals result_1
        _prediction("w2", "b", "c", "b"),  # equals result_aux
        _prediction("w3", "d", "a", "b"),  # neither: not a plausible answer
    ]
    run_dir = _write_run(
        tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary(n=3), predictions
    )

    row = leaderboard.run_row("harness", "zero_shot", run_dir)

    assert float(row["agreement_result_plausible"]) == pytest.approx(2 / 3)


def test_the_backfill_ignores_rows_without_an_action(tmp_path):
    predictions = [_prediction("w1", "a", "a", "b"), _prediction("w2", None, "c", "d")]
    run_dir = _write_run(
        tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary(n=2), predictions
    )

    row = leaderboard.run_row("harness", "zero_shot", run_dir)

    assert float(row["agreement_result_plausible"]) == 1.0


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_the_table_has_one_row_per_run(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary())
    _write_run(tmp_path, "harness", "zero_shot", "20261008-130000_dev", _summary())
    _write_run(tmp_path, "control", "human", "20261008-121000_test", _summary(split="test"))

    rows = leaderboard.collect_rows(tmp_path)

    assert [(row["strategy"], row["split"]) for row in rows] == [
        ("human", "test"),
        ("zero_shot", "dev"),
        ("zero_shot", "dev"),
    ]


def test_the_table_is_ordered_by_group_strategy_and_run(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-130000_dev", _summary())
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary())
    _write_run(tmp_path, "harness", "cot", "20261008-140000_dev", _summary())

    names = [(row["group"], row["strategy"]) for row in leaderboard.collect_rows(tmp_path)]

    assert names == [
        ("harness", "cot"),
        ("harness", "zero_shot"),
        ("harness", "zero_shot"),
    ]


def test_the_older_run_of_a_strategy_comes_first(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-130000_dev", _summary(real=0.9))
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary(real=0.1))

    rows = leaderboard.collect_rows(tmp_path)

    assert [float(row["agreement_result_real"]) for row in rows] == [0.1, 0.9]


def test_write_leaderboard_writes_the_csv(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary())
    _write_run(tmp_path, "control", "human", "20261008-121000_test", _summary(split="test"))

    path = leaderboard.write_leaderboard(tmp_path)

    assert Path(path) == tmp_path / "leaderboard.csv"
    rows = _read_table(path)
    assert list(rows[0]) == leaderboard.COLUMNS
    assert [row["strategy"] for row in rows] == ["human", "zero_shot"]
    assert rows[0]["split"] == "test"


def test_write_leaderboard_accepts_another_output(tmp_path):
    _write_run(tmp_path, "harness", "zero_shot", "20261008-120000_dev", _summary())
    target = tmp_path / "elsewhere" / "table.csv"

    path = leaderboard.write_leaderboard(tmp_path, output=target)

    assert Path(path) == target and target.is_file()


# --------------------------------------------------------------------------- #
# The real folder
# --------------------------------------------------------------------------- #


def test_the_real_results_folder_produces_a_table(tmp_path, repo_root):
    path = leaderboard.write_leaderboard(config.RESULTS_DIR, output=tmp_path / "leaderboard.csv")

    rows = _read_table(path)

    assert rows, "the results folder holds runs"
    assert {row["group"] for row in rows} == {"harness", "control"}
    for row in rows:
        assert row["strategy"]
        assert row["split"] in dataset.SPLITS
        assert Path(row["run_dir"]).is_dir()
    # the language strategies ran with a reasoning effort; the retrieval ones with a k
    assert any(row["effort"] for row in rows)
    assert any(row["k"] for row in rows)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_cli_defaults_to_the_results_folder():
    args = leaderboard.build_parser().parse_args([])
    assert Path(args.results_root) == Path(config.RESULTS_DIR)
    assert args.output is None


def test_the_cli_accepts_an_explicit_root_and_output():
    args = leaderboard.build_parser().parse_args(["--results-root", "x", "--output", "y.csv"])
    assert (args.results_root, args.output) == ("x", "y.csv")
