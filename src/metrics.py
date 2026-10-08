# -*- coding: utf-8 -*-
"""Evaluation metrics against ``result_real``.

STATE: implemented. The contract is defined in
``tests/test_metrics.py``.

Contract
-----------------
``accuracy(preds, refs) -> float``
``top2_recall(preds_top2, refs) -> float``
    ``preds_top2`` is a list of ``(first, second)`` tuples.
``plausible_agreement(preds, plausible) -> float``
    ``plausible`` is a list of ``(result_1, result_aux)`` pairs; a prediction counts
    as a hit when it is one of the two.
``confusion_matrix(preds, refs) -> dict[(pred, ref)] -> int``
``per_difficulty(preds, refs, difficulties) -> dict[str, dict]``
    each group -> ``{"n": int, "accuracy": float}``
"""

from __future__ import annotations

__all__ = ["accuracy", "top2_recall", "plausible_agreement", "confusion_matrix", "per_difficulty"]


def accuracy(preds, refs):
    """Top-1 accuracy."""
    refs = list(refs)
    if not refs:
        return 0.0
    preds = list(preds)
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    hits = sum(1 for pred, ref in zip(preds, refs) if pred == ref)
    return hits / len(refs)


def top2_recall(preds_top2, refs):
    """Fraction of cases whose truth is among the two best actions."""
    refs = list(refs)
    if not refs:
        return 0.0
    preds_top2 = list(preds_top2)
    if len(preds_top2) != len(refs):
        raise ValueError("preds_top2 and refs must have the same length")
    hits = sum(1 for top2, ref in zip(preds_top2, refs) if ref in tuple(top2))
    return hits / len(refs)


def plausible_agreement(preds, plausible):
    """Fraction of predictions that are a plausible answer -- ``result_1`` or ``result_aux``.

    ``plausible`` holds the two actions the dataset considers acceptable for each case
    (the consensus primary action and its alternative), so a prediction is a hit when
    it is either of them. This is the metric behind
    ``agreement_result_plausible`` in the summaries: the answer is not the one the
    clinician took, but it is a reasonable one.
    """
    preds = list(preds)
    plausible = list(plausible)
    if len(preds) != len(plausible):
        raise ValueError("preds and plausible must have the same length")
    if not preds:
        return 0.0
    hits = sum(1 for pred, options in zip(preds, plausible) if pred in tuple(options))
    return hits / len(preds)


def confusion_matrix(preds, refs):
    """Confusion matrix as ``{(pred, ref): count}``."""
    matrix = {}
    for pred, ref in zip(preds, refs):
        matrix[(pred, ref)] = matrix.get((pred, ref), 0) + 1
    return matrix


def per_difficulty(preds, refs, difficulties):
    """Accuracy grouped by case difficulty: ``{difficulty: {"n": ..., "accuracy": ...}}``."""
    groups = {}
    for pred, ref, difficulty in zip(preds, refs, difficulties):
        group = groups.setdefault(difficulty, {"n": 0, "hits": 0})
        group["n"] += 1
        group["hits"] += int(pred == ref)
    return {
        difficulty: {
            "n": group["n"],
            "accuracy": (group["hits"] / group["n"]) if group["n"] else 0.0,
        }
        for difficulty, group in groups.items()
    }
