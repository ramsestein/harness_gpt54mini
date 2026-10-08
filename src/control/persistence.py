# -*- coding: utf-8 -*-
"""The persistence control (no model): repeat what the clinician just did.

The simplest thing a clinician can do is nothing new. This control reads the window's own
action history (``actions_history_json``: the TCI changes with their time offsets, the drug
and the target before and after) and answers with the **most recent** adjustment, mapped to
the five-action vocabulary:

- propofol target up / down  -> ``increase_hypnotic`` / ``reduce_hypnotic``
- remifentanil target up / down -> ``increase_opioid`` / ``reduce_opioid``
- a change of zero, another drug (a vasopressor) or no history at all -> ``no_action``

Nothing of the outcome is read: the history is part of the window, so there is no leakage,
and the previous decision of the same patient is legitimately known at decision time. With
consecutive windows of the same case, this is the "do what you did a moment ago" baseline,
which temporal autocorrelation makes worth measuring.

It needs no model, no simulator and no corpus, so it runs in the cheap sweep.

Results go to ``results_harness/control/persistence/<run_id>/``.
"""

from __future__ import annotations

import json

from src.control.baselines import ControlStrategy
from src.harness_base import Prediction

__all__ = ["NAME", "DRUG_ACTIONS", "history", "last_event", "action_of", "PersistenceStrategy"]

#: Folder name under ``results_harness/control/``.
NAME = "persistence"

#: The action each drug moves when its target goes up (index 0) or down (index 1).
DRUG_ACTIONS = {
    "propofol": ("increase_hypnotic", "reduce_hypnotic"),
    "remifentanil": ("increase_opioid", "reduce_opioid"),
}

#: Answer when the history says nothing actionable.
NO_ACTION = "no_action"


def history(inp):
    """The clinician's action history of a window, however the field is encoded."""
    raw = inp.get("actions_history_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    return raw if isinstance(raw, list) else []


def _delta(event):
    """The target change of one event, from ``delta`` or from the two targets."""
    delta = event.get("delta")
    if isinstance(delta, (int, float)) and not isinstance(delta, bool):
        return float(delta)
    old, new = event.get("old_target"), event.get("new_target")
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return float(new) - float(old)
    return None


def last_event(inp):
    """The most recent event of the history (the largest time offset), or ``None``."""
    events = [
        event
        for event in history(inp)
        if isinstance(event, dict)
        and isinstance(event.get("t_offset_sec"), (int, float))
        and not isinstance(event.get("t_offset_sec"), bool)
    ]
    return max(events, key=lambda event: event["t_offset_sec"]) if events else None


def action_of(event):
    """The five-action label of one history event."""
    if not isinstance(event, dict):
        return NO_ACTION
    pair = DRUG_ACTIONS.get(str(event.get("drug", "")).strip().lower())
    if pair is None:
        return NO_ACTION
    delta = _delta(event)
    if delta is None or delta == 0.0:
        return NO_ACTION
    return pair[0] if delta > 0 else pair[1]


class PersistenceStrategy(ControlStrategy):
    """Answer with the adjustment the clinician made most recently in this window."""

    name = NAME
    description = (
        "Repeats the adjustment the clinician made most recently in the window's own "
        "action history, mapped to the five-action vocabulary (no model)."
    )

    def predict(self, window):
        event = last_event(window.input)
        action = action_of(event)
        return Prediction(
            window_id=window.window_id,
            action=action,
            raw_output=action,
            meta={
                "baseline": NAME,
                "action": action,
                "last_event": event,
                "n_events": len(history(window.input)),
            },
        )
