# -*- coding: utf-8 -*-
"""Evaluation metrics."""

import pytest

from src import metrics


def test_accuracy_perfect_and_partial():
    assert metrics.accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert metrics.accuracy(["a", "b"], ["a", "a"]) == 0.5
    assert metrics.accuracy([], []) == 0.0


def test_top2_recall():
    preds_top2 = [("a", "b"), ("c", "d"), ("e", "a")]
    refs = ["b", "c", "d"]  # "b" and "c" are covered, "d" is not
    assert metrics.top2_recall(preds_top2, refs) == 2 / 3


def test_confusion_matrix():
    cm = metrics.confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert cm[("a", "a")] == 1
    assert cm[("a", "b")] == 1
    assert cm[("b", "b")] == 1
    assert sum(cm.values()) == 3


def test_accuracy_per_difficulty():
    result = metrics.per_difficulty(["a", "b"], ["a", "a"], ["easy", "hard"])
    assert result["easy"]["n"] == 1
    assert result["easy"]["accuracy"] == 1.0
    assert result["hard"]["accuracy"] == 0.0


def test_plausible_agreement_counts_the_predictions_that_are_an_acceptable_answer():
    preds = ["a", "b", "c", "d"]
    plausible = [("a", "x"), ("y", "z"), ("c", "a"), ("d", "d")]
    assert metrics.plausible_agreement(preds, plausible) == 3 / 4


def test_plausible_agreement_edges():
    assert metrics.plausible_agreement([], []) == 0.0
    assert metrics.plausible_agreement(["a"], [("a", "b")]) == 1.0
    assert metrics.plausible_agreement(["b"], [("a", "b")]) == 1.0
    assert metrics.plausible_agreement(["a"], [("b", "c")]) == 0.0
    assert metrics.plausible_agreement([None], [("a", "b")]) == 0.0
    with pytest.raises(ValueError):
        metrics.plausible_agreement(["a", "b"], [("a", "b")])


def test_per_class_recall():
    preds = ["a", "a", "b", "b"]
    refs = ["a", "b", "b", "b"]

    # of the three windows whose truth is "b", two were answered with "b"
    assert metrics.per_class_recall(preds, refs, ["a", "b"]) == {"a": 1.0, "b": pytest.approx(2 / 3)}


def test_balanced_accuracy_does_not_reward_the_majority_class():
    """A constant answer to an imbalanced truth: high accuracy, useless strategy."""
    preds = ["b"] * 4
    refs = ["a", "b", "b", "b"]

    assert metrics.accuracy(preds, refs) == 0.75
    assert metrics.balanced_accuracy(preds, refs, ["a", "b"]) == 0.5


def test_a_label_with_no_instances_counts_as_zero():
    """Uniform across runs: every label of the vocabulary weighs the same."""
    assert metrics.balanced_accuracy(["a"], ["a"], ["a", "b"]) == 0.5
    assert metrics.per_class_recall(["a"], ["a"], ["a", "b"]) == {"a": 1.0, "b": 0.0}


def test_macro_f1():
    preds = ["a", "b", "b"]
    refs = ["a", "a", "b"]  # both classes: precision .5 or 1, recall .5 or 1 -> F1 2/3 each

    assert metrics.macro_f1(preds, refs, ["a", "b"]) == pytest.approx(2 / 3)


def test_cohen_kappa():
    assert metrics.cohen_kappa(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    # a constant answer against a balanced truth carries no information at all
    assert metrics.cohen_kappa(["a", "a"], ["a", "b"], ["a", "b"]) == pytest.approx(0.0)
    assert metrics.cohen_kappa([], [], ["a", "b"]) == 0.0


def test_cluster_bootstrap_brackets_the_mean_and_is_deterministic():
    cases = [f"case{index // 10}" for index in range(100)]
    hits = [1] * 50 + [0] * 50

    low, high, mean = metrics.cluster_bootstrap_ci(cases, hits)

    assert mean == 0.5
    assert low < 0.5 < high
    assert (low, high) == metrics.cluster_bootstrap_ci(cases, hits)[:2]


def test_cluster_bootstrap_of_a_perfect_and_of_an_empty_run():
    cases = [f"case{index // 10}" for index in range(100)]

    assert metrics.cluster_bootstrap_ci(cases, [1] * 100)[:2] == (1.0, 1.0)
    assert metrics.cluster_bootstrap_ci([], []) == (0.0, 0.0, 0.0)


def test_the_cluster_interval_is_wide_when_the_cases_are():
    """Twenty-five cases that hit everything and twenty-five that miss everything."""
    cases = [f"case{index // 10}" for index in range(500)]
    hits = [1] * 250 + [0] * 250

    low, high, _ = metrics.cluster_bootstrap_ci(cases, hits)

    # resampling cases gives a binomial spread; resampling windows would give ~0.04
    assert high - low > 0.2
