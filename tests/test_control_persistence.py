# -*- coding: utf-8 -*-
"""The persistence control: repeat the adjustment the clinician made most recently."""

import dataclasses
import json
from pathlib import Path

import pytest

from harness import run as harness_run
from src import dataset
from src.control import persistence, run_controls
from src.harness_base import Strategy

NAME = "persistence"


def _event(offset, drug, old, new, action_type="tci_change"):
    delta = new - old if isinstance(old, (int, float)) and isinstance(new, (int, float)) else None
    return {
        "t_offset_sec": offset,
        "action_type": action_type,
        "drug": drug,
        "old_target": old,
        "new_target": new,
        "delta": delta,
    }


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("test", limit=1)[0]


def _strategy():
    return persistence.PersistenceStrategy()


def _predict(window, events):
    return _strategy().predict(dataclasses.replace(window, input={"actions_history_json": events}))


# --------------------------------------------------------------------------- #
# Shape of the control
# --------------------------------------------------------------------------- #


def test_it_is_a_control_without_a_model():
    strategy = _strategy()

    assert isinstance(strategy, Strategy)
    assert strategy.name == NAME
    assert strategy.description.strip()
    assert strategy.uses_model is False


def test_it_runs_in_the_cheap_sweep():
    assert NAME in run_controls.default_names()
    assert NAME not in run_controls.OPT_IN_NAMES


# --------------------------------------------------------------------------- #
# Reading the history
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "drug,old,new,expected",
    [
        ("propofol", 3.0, 4.0, "increase_hypnotic"),
        ("propofol", 4.0, 3.0, "reduce_hypnotic"),
        ("remifentanil", 4.0, 6.0, "increase_opioid"),
        ("remifentanil", 6.0, 4.0, "reduce_opioid"),
        ("propofol", 4.0, 4.0, "no_action"),
        ("norepinephrine", 0.0, 0.05, "no_action"),
        ("propofol", None, None, "no_action"),
    ],
)
def test_the_last_target_change_is_translated_to_an_action(drug, old, new, expected):
    assert persistence.action_of(_event(-60, drug, old, new)) == expected


def test_the_most_recent_event_decides():
    events = [
        _event(-510, "remifentanil", 0.0, 0.37),
        _event(-390, "remifentanil", 11.76, 5.78),
        _event(-120, "propofol", 3.0, 4.5),
    ]

    assert persistence.action_of(persistence.last_event({"actions_history_json": events})) == (
        "increase_hypnotic"
    )


def test_the_delta_can_come_from_the_targets():
    event = {
        "t_offset_sec": -30,
        "drug": "remifentanil",
        "old_target": 5.0,
        "new_target": 7.5,
    }

    assert persistence.action_of(event) == "increase_opioid"


def test_a_json_string_history_is_parsed():
    encoded = json.dumps([_event(-60, "propofol", 3.0, 2.0)])

    assert persistence.action_of(persistence.last_event({"actions_history_json": encoded})) == (
        "reduce_hypnotic"
    )


@pytest.mark.parametrize(
    "value",
    [None, [], "not json", {"unexpected": "shape"}, [{"no": "offset"}], 42],
)
def test_a_usable_history_is_not_required(value):
    assert persistence.action_of(persistence.last_event({"actions_history_json": value})) == (
        "no_action"
    )


def test_an_event_without_an_offset_is_ignored():
    events = [_event(-60, "propofol", 3.0, 4.0), {"drug": "propofol", "delta": -1.0}]

    assert persistence.last_event({"actions_history_json": events}) == events[0]


# --------------------------------------------------------------------------- #
# Prediction and metadata
# --------------------------------------------------------------------------- #


def test_the_prediction_carries_the_event_that_decided(window):
    events = [_event(-510, "remifentanil", 0.0, 0.37)]

    prediction = _predict(window, events)

    assert prediction.action == "increase_opioid"
    assert prediction.meta["baseline"] == NAME
    assert prediction.meta["last_event"] == events[0]
    assert prediction.meta["n_events"] == 1


def test_a_window_without_history_answers_no_action(window):
    prediction = _predict(window, [])

    assert prediction.action == "no_action"
    assert prediction.meta["last_event"] is None


def test_it_predicts_a_real_window(window):
    prediction = _strategy().predict(window)

    assert prediction.action in dataset.ACTIONS
    assert prediction.window_id == window.window_id


def test_the_launcher_runs_it(tmp_path):
    outcome = harness_run.run(
        _strategy(), split="test", limit=20, results_root=tmp_path, quiet=True
    )

    run_dir = Path(outcome["out_dir"])
    assert run_dir.parent.name == NAME
    assert outcome["summary"]["errors"] == 0
    assert outcome["summary"]["n_windows"] == 20
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["model"] is None
