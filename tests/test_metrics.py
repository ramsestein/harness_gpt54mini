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
