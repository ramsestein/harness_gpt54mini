# -*- coding: utf-8 -*-
"""The ``shap_prompt`` strategy: zero_shot plus the SHAP analysis as reference."""

from pathlib import Path

import pytest

from src import dataset, prompting
from src.harness_base import Strategy

from harness import run as harness_run
from harness.shap_prompt import strategy as shap_prompt


@pytest.fixture(scope="module")
def ranking_file(tmp_path_factory):
    """A small SHAP report, shaped exactly like the one the diagnostics write."""
    path = tmp_path_factory.mktemp("shap-report") / "shap_importance.csv"
    rows = ["feature,mean_abs_shap,rank"]
    for rank, (name, value) in enumerate(
        (("remi_ce_slope_short", 0.9274), ("remi_ce_current", 0.5019), ("ce_remi", 0.4639),
         ("hr_current", 0.4198), ("bis_current", 0.2431)), start=1
    ):
        rows.append(f"{name},{value},{rank}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("dev", limit=1)[0]


@pytest.fixture
def strategy(ranking_file):
    return shap_prompt.ShapPromptStrategy(importance_path=ranking_file)


# --------------------------------------------------------------------------- #
# It is a strategy
# --------------------------------------------------------------------------- #


def test_it_is_a_strategy(strategy):
    assert isinstance(strategy, Strategy)
    assert strategy.name == shap_prompt.ShapPromptStrategy.name == "shap_prompt"
    assert strategy.description.strip()
    assert strategy.uses_model is True


def test_it_constructs_without_touching_the_disk(tmp_path):
    """The structure suite loads every strategy with no arguments."""
    strategy = shap_prompt.ShapPromptStrategy(importance_path=tmp_path / "nowhere.csv")

    assert strategy.rows is None  # everything by default


def test_the_report_is_the_one_the_diagnostics_write():
    assert shap_prompt.DEFAULT_IMPORTANCE_PATH.name == "shap_importance.csv"
    assert shap_prompt.DEFAULT_IMPORTANCE_PATH.parent.name == "_analysis"


def test_the_real_report_is_where_the_strategy_looks_for_it():
    """The default path must exist in this repository, not only in the tests."""
    assert shap_prompt.DEFAULT_IMPORTANCE_PATH.is_file()


# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #


def test_the_ranking_is_read_in_order(strategy):
    assert strategy.ranking()[0] == ("remi_ce_slope_short", pytest.approx(0.9274))
    assert [name for name, _ in strategy.ranking()] == [
        "remi_ce_slope_short", "remi_ce_current", "ce_remi", "hr_current", "bis_current"
    ]


def test_all_the_rows_are_shown_by_default(strategy):
    assert len(strategy.shown()) == 5


def test_the_launcher_samples_flag_never_cuts_the_ranking(ranking_file):
    """The whole SHAP analysis is always sent: --samples is accepted and ignored."""
    strategy = shap_prompt.ShapPromptStrategy(importance_path=ranking_file, n_samples=2)

    assert strategy.rows is None
    assert len(strategy.shown()) == len(strategy.ranking())
    assert strategy.shown()[0][0] == "remi_ce_slope_short"


def test_a_missing_report_says_how_to_build_one(tmp_path):
    strategy = shap_prompt.ShapPromptStrategy(importance_path=tmp_path / "nope.csv")

    with pytest.raises(FileNotFoundError, match="xgboost_diagnostics"):
        strategy.ranking()


def test_a_row_that_is_not_a_number_is_skipped(tmp_path):
    path = tmp_path / "shap_importance.csv"
    path.write_text(
        "feature,mean_abs_shap,rank\nbis_current,0.1,1\nbroken,,2\nhr_current,0.4,3\n",
        encoding="utf-8",
    )

    strategy = shap_prompt.ShapPromptStrategy(importance_path=path)

    assert [name for name, _ in strategy.ranking()] == ["bis_current", "hr_current"]


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def test_the_reference_explains_what_the_numbers_mean(strategy):
    reference = strategy.render_reference()

    assert "mean absolute SHAP value" in reference
    assert "most influential first" in reference


def test_the_reference_lists_the_variables_with_their_values(strategy):
    reference = strategy.render_reference()

    assert "1. remi_ce_slope_short: 0.9274" in reference
    assert "5. bis_current: 0.2431" in reference


def test_the_prompt_keeps_the_zero_shot_wording(strategy, window):
    prompt = strategy.build_prompt(window)

    for text in ("expert anesthesiologist", "Choose exactly one", "e) no_action"):
        assert text in prompt
    assert "{case}" not in prompt


def test_the_patient_state_is_there_too(strategy, window):
    prompt = strategy.build_prompt(window)

    assert prompting.render_case(window.input) in prompt
    assert shap_prompt.REFERENCE_HEADER in prompt


def test_the_ranking_comes_after_the_state_and_before_the_options(strategy, window):
    prompt = strategy.build_prompt(window)
    state = prompting.render_case(window.input)

    assert prompt.index(state[:40]) < prompt.index(shap_prompt.REFERENCE_HEADER)
    assert prompt.index(shap_prompt.REFERENCE_HEADER) < prompt.index("Choose exactly one")


def test_a_smaller_prompt_than_the_rag_one(strategy, window):
    """The analysis is a table, not a corpus: it must not double the prompt."""
    from harness.rag import strategy as rag

    assert len(strategy.build_prompt(window)) < len(rag.RagStrategy().build_prompt(window))


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #


def test_the_launcher_records_how_much_of_the_analysis_it_sent(strategy, tmp_path, monkeypatch):
    monkeypatch.setattr(
        strategy,
        "ask",
        lambda messages: type("R", (), {"text": "b", "model": "fake", "latency_s": 0.0,
                                       "dropped_params": []})(),
    )

    outcome = harness_run.run(strategy, split="dev", limit=2, results_root=tmp_path, quiet=True)

    assert outcome["summary"]["errors"] == 0
    meta = outcome["predictions"][0]["meta"]
    assert meta["rows_shown"] == 5
    assert meta["k"] == 5  # the leaderboard's knob column
    assert meta["prompt_chars"] > 0


@pytest.mark.live
def test_it_answers_like_any_other_strategy(ranking_file, window):
    strategy = shap_prompt.ShapPromptStrategy(importance_path=ranking_file)

    prediction = strategy.predict(window)

    assert prediction.action in dataset.ACTIONS


def test_a_report_of_a_model_fitted_on_another_view_is_refused(tmp_path):
    """A ranking that names hidden fields comes from a stale model: never advice."""
    path = tmp_path / "shap_importance.csv"
    path.write_text("feature,mean_abs_shap,rank\nsurgery_pct,0.9,1\nce_remi,0.5,2\n",
                    encoding="utf-8")
    strategy = shap_prompt.ShapPromptStrategy(importance_path=path)

    with pytest.raises(ValueError, match="shared view"):
        strategy.ranking()
