# -*- coding: utf-8 -*-
"""Prediction persistence."""

from src import results


def test_write_and_read_predictions(tmp_path):
    preds = [
        {"window_id": "case1_w1", "action": "no_action", "raw_output": "e"},
        {"window_id": "case1_w2", "action": "increase_opioid", "raw_output": "c"},
    ]
    path = tmp_path / "preds.jsonl"

    results.write_predictions(path, preds)

    assert path.exists()
    assert results.read_predictions(path) == preds


def test_summarize():
    preds = [
        {"window_id": "a", "action": "no_action"},
        {"window_id": "b", "action": "no_action"},
        {"window_id": "c", "action": "reduce_opioid"},
    ]
    summary = results.summarize(preds)
    assert summary["n"] == 3
    assert summary["action_counts"]["no_action"] == 2
    assert summary["action_counts"]["reduce_opioid"] == 1
