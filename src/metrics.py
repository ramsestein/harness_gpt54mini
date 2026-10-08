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
``per_class_recall(preds, refs, labels) -> dict[label, float]``
``balanced_accuracy(preds, refs, labels) -> float``
    Mean per-class recall: every class counts the same, whatever its prevalence. On a
    corpus where ``no_action`` and ``reduce_opioid`` are ~27 % each, accuracy alone hides
    a strategy that simply leans on the majority class.
``macro_f1(preds, refs, labels) -> float``
``cohen_kappa(preds, refs, labels) -> float``
    Agreement above what chance alone would give.
``cluster_bootstrap_ci(cases, hits, replicates, seed) -> (low, high)``
    Interval of the mean of ``hits`` resampling whole **cases**: the windows of one case
    are not independent observations, so resampling windows would give an interval that
    is far too narrow.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy",
    "top2_recall",
    "plausible_agreement",
    "confusion_matrix",
    "per_difficulty",
    "per_class_recall",
    "balanced_accuracy",
    "macro_f1",
    "cohen_kappa",
    "cluster_bootstrap_ci",
]

#: Replicates and seed of the bootstrap interval: fixed, so the table is reproducible.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 95.0


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


def per_class_recall(preds, refs, labels):
    """Recall of every label: of the windows whose truth is that label, how many were hit."""
    counts = {label: [0, 0] for label in labels}
    for pred, ref in zip(preds, refs):
        if ref not in counts:
            continue
        counts[ref][0] += 1
        counts[ref][1] += int(pred == ref)
    return {
        label: (hits / total) if total else 0.0 for label, (total, hits) in counts.items()
    }


def balanced_accuracy(preds, refs, labels):
    """Mean per-class recall."""
    recalls = list(per_class_recall(preds, refs, labels).values())
    return sum(recalls) / len(recalls) if recalls else 0.0


def macro_f1(preds, refs, labels):
    """Mean F1 over the labels: precision and recall of every class weigh the same."""
    scores = []
    for label in labels:
        hits = sum(1 for pred, ref in zip(preds, refs) if pred == label and ref == label)
        predicted = sum(1 for pred in preds if pred == label)
        actual = sum(1 for ref in refs if ref == label)
        precision = hits / predicted if predicted else 0.0
        recall = hits / actual if actual else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
    return sum(scores) / len(scores) if scores else 0.0


def cohen_kappa(preds, refs, labels):
    """Agreement corrected by the agreement expected from the marginals alone."""
    preds = list(preds)
    refs = list(refs)
    total = len(refs)
    if not total:
        return 0.0
    observed = accuracy(preds, refs)
    expected = sum(
        (sum(1 for pred in preds if pred == label) / total)
        * (sum(1 for ref in refs if ref == label) / total)
        for label in labels
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0


def cluster_bootstrap_ci(cases, hits, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                         confidence=BOOTSTRAP_CONFIDENCE):
    """Percentile interval of ``mean(hits)``, resampling whole cases with replacement.

    Returns ``(low, high, mean)``. The windows of one case are decisions of the same
    patient, so they are not independent; a bootstrap over cases is the honest interval.
    """
    cases = list(cases)
    hits = np.asarray(list(hits), dtype=float)
    if not len(cases):
        return 0.0, 0.0, 0.0
    grouped = {}
    for position, case in enumerate(cases):
        grouped.setdefault(case, []).append(position)
    clusters = [np.asarray(positions) for positions in grouped.values()]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    means = np.array(
        [hits[np.concatenate([clusters[index] for index in draw])].mean() for draw in draws]
    )
    tail = (100.0 - confidence) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high), float(hits.mean())
