# -*- coding: utf-8 -*-
"""Prediction persistence (JSONL) and summaries.

STATE: implemented. The contract is defined in
``tests/test_results.py``.

Contract
-----------------
``write_predictions(path, predictions)``     writes JSONL (one line per prediction)
``read_predictions(path) -> list[dict]``
``summarize(predictions) -> dict``           ``{"n": int, "action_counts": {action: n}}``
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["write_predictions", "read_predictions", "summarize"]


def write_predictions(path, predictions):
    """Write the predictions as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return path


def read_predictions(path):
    """Read a predictions JSONL file."""
    predictions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def summarize(predictions):
    """Aggregate summary of a list of predictions."""
    counts = {}
    for prediction in predictions:
        action = prediction.get("action")
        counts[action] = counts.get(action, 0) + 1
    return {"n": len(predictions), "action_counts": counts}
