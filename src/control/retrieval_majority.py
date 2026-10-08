# -*- coding: utf-8 -*-
"""The pure retrieval control (no model).

The non-model reading of the few-shot strategy, and nothing more. Per window:

1. **Retrieve** the ``k`` most similar training windows with the few-shot similarity
   tool (numeric features, standardised, Euclidean, same-case windows excluded). The
   case window goes in raw: no simulation, no model, nothing of the outcome.
2. **Vote** on the neighbourhood: every neighbour counts once, for the action its
   clinician took.
3. **Answer** with the most frequent action -- what anesthesiologists did in the
   situations that look most like this one. Ties go to ``dataset.ACTIONS`` order, like
   ``self_consistency``.

Only ``result_real`` takes part. The dataset's ``result_1``/``result_aux`` are the output
of a model trained on the clinician's actions, so voting with them would be circular --
this control measures the clinicians, not a consensus. An earlier version of this control
did score the consensus (it lives on as ``results_harness/control/retrieval_consensus/``);
it was dropped for exactly that reason.

The similarity index already stores the real action of every training window, so this
control needs no extra corpus or cache of its own.

Results go to ``results_harness/control/retrieval_majority/<run_id>/``.
"""

from __future__ import annotations

import threading
from pathlib import Path

from harness.few_shot import retrieval
from src import dataset
from src.control.baselines import ControlStrategy
from src.harness_base import Prediction

__all__ = [
    "NAME",
    "SOURCE_SPLIT",
    "DEFAULT_K",
    "RetrievalMajorityStrategy",
]

#: Folder name under ``results_harness/control/``.
NAME = "retrieval_majority"

#: The corpus the neighbours -- and therefore the vote -- come from.
SOURCE_SPLIT = retrieval.SOURCE_SPLIT

#: Neighbours retrieved per window.
DEFAULT_K = 10


class RetrievalMajorityStrategy(ControlStrategy):
    """Retrieve the k nearest training windows and answer with the real majority action."""

    name = NAME
    description = (
        "Retrieves the k most similar training windows of the case window with the "
        "few-shot similarity tool and answers with the action the clinicians took most "
        "often in that neighbourhood (no model, result_real only)."
    )

    def __init__(self, k=DEFAULT_K, exclude_same_case=True, index=None, index_path=None,
                 **options):
        super().__init__(**options)
        self.k = int(k)
        self.exclude_same_case = bool(exclude_same_case)
        self.index_path = (
            Path(index_path) if index_path is not None else retrieval.DEFAULT_INDEX_PATH
        )
        self._index = index
        self._lock = threading.Lock()

    @property
    def index(self):
        """The training index, loaded once (thread-safe: windows run in a pool)."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = retrieval.load_or_build(self.index_path)
        return self._index

    @staticmethod
    def _size(index):
        try:
            return len(index)
        except TypeError:
            return None

    # -- The vote ----------------------------------------------------------- #

    def score_neighbours(self, neighbours):
        """How many neighbours took each action."""
        scores = []
        for action in dataset.ACTIONS:
            count = sum(1 for neighbour in neighbours if neighbour.action == action)
            scores.append(
                {
                    "action": action,
                    "count": count,
                    "rate": round(count / len(neighbours), 4) if neighbours else 0.0,
                }
            )
        return scores

    # -- The decision ------------------------------------------------------- #

    def predict(self, window):
        neighbours = self.index.query(
            window,
            k=self.k,
            exclude_case=window.case_id if self.exclude_same_case else None,
        )
        scores = self.score_neighbours(neighbours)

        best = max(score["rate"] for score in scores)
        winners = [score["action"] for score in scores if score["rate"] == best]
        order = {action: position for position, action in enumerate(dataset.ACTIONS)}
        chosen = min(winners, key=lambda action: order[action])

        return Prediction(
            window_id=window.window_id,
            action=chosen,
            raw_output=chosen,
            meta={
                "baseline": NAME,
                "k": self.k,
                "exclude_same_case": self.exclude_same_case,
                "n_neighbours": len(neighbours),
                "neighbours": [
                    {
                        "window_id": neighbour.window_id,
                        "distance": round(neighbour.distance, 4),
                        "action": neighbour.action,
                    }
                    for neighbour in neighbours
                ],
                "scores": scores,
                "chosen": chosen,
                "rate": best,
                "tied": winners if len(winners) > 1 else [],
                "index_size": self._size(self.index),
            },
        )
