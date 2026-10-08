# -*- coding: utf-8 -*-
"""The simulator + normality control (no model).

The idea: the best action is the one that leaves the patient closest to normality, so
let the simulator say where each action leads. Per window:

1. **Calibrate** a patient from the window and simulate the **five** actions five
   minutes ahead (one calibration for the five futures).
2. **Score** each future by its distance to the centre of the normal range, with every
   simulated signal as one dimension, normalised by that signal's tolerance so no
   dimension explodes:

   ``score = sqrt( sum_i ((x_i - target_i) / tolerance_i)^2 )``

3. **Answer** with the action of the smallest score (ties go to ``dataset.ACTIONS``
   order, like ``self_consistency``).

The signal table (centre of normality and the deviation that counts as one unit):

===========  ========  ===========  ===============================================
signal       target    tolerance    why
===========  ========  ===========  ===============================================
``bis``      50.0      10.0         surgical anaesthesia band 40-60
``map``      77.5      12.5         perfusion band 65-90 (no hypotension, no hypertension)
``co``       6.0       2.0          cardiac output band 4-8 L/min
``tol``      0.0       1.0          nociception index in [0, 1]; the target is 0
===========  ========  ===========  ===============================================

All four are measured, not guessed: the simulator's own spread and the windows it
reproduces were inspected first (see ``tests/test_sim_normality.py``).

Everything is deterministic: no seed, no model, no sampling. Results go to
``results_harness/control/sim_normality/<run_id>/``.
"""

from __future__ import annotations

import math
from pathlib import Path

from harness.critic_verifier_sim import simulator as simulator_module
from src import dataset
from src.control.baselines import ControlStrategy
from src.harness_base import Prediction

__all__ = [
    "NAME",
    "SIGNALS",
    "TARGETS",
    "TOLERANCES",
    "SimNormalityStrategy",
]

#: Folder name under ``results_harness/control/``.
NAME = "sim_normality"

#: The simulated signals that make up the state, in order.
SIGNALS = ("bis", "map", "co", "tol")

#: Centre of the normal range of each signal.
TARGETS = {
    "bis": 50.0,
    "map": 77.5,
    "co": 6.0,
    "tol": 0.0,
}

#: Deviation of each signal that counts as one unit of "far from the centre".
TOLERANCES = {
    "bis": 10.0,
    "map": 12.5,
    "co": 2.0,
    "tol": 1.0,
}


class SimNormalityStrategy(ControlStrategy):
    """Simulate the five actions and keep the future closest to the normal centre."""

    name = NAME
    description = (
        "Calibrates the patient from the window, simulates the five actions five "
        "minutes ahead, and answers with the one whose future sits closest to the "
        "centre of the normal range (each signal normalised by its tolerance, no model)."
    )

    def __init__(self, simulator=None, targets=None, tolerances=None, **options):
        super().__init__(**options)
        self.targets = {**TARGETS, **(targets or {})}
        self.tolerances = {**TOLERANCES, **(tolerances or {})}
        for signal in SIGNALS:
            if not self.tolerances.get(signal):
                raise ValueError(f"the tolerance of {signal!r} must be a positive number")
        self.simulator = (
            simulator if simulator is not None else simulator_module.AnesthesiaSimulator()
        )

    # -- The score ---------------------------------------------------------- #

    def deviation(self, signal, value):
        """How far ``value`` is from the normal centre, in units of its own tolerance."""
        return (float(value) - self.targets[signal]) / self.tolerances[signal]

    def candidate(self, outcome):
        """The normalised score of one simulated future, with its per-signal breakdown."""
        deviations = {signal: self.deviation(signal, getattr(outcome, signal)) for signal in SIGNALS}
        return {
            "action": outcome.action,
            "score": math.sqrt(sum(value * value for value in deviations.values())),
            "signals": {signal: round(float(getattr(outcome, signal)), 4) for signal in SIGNALS},
            "deviations": {signal: round(value, 4) for signal, value in deviations.items()},
        }

    # -- The decision ------------------------------------------------------- #

    def predict(self, window):
        calibration, outcomes = self.simulator.simulate(window.input, dataset.ACTIONS)
        candidates = [self.candidate(outcome) for outcome in outcomes]

        best = min(candidate["score"] for candidate in candidates)
        winners = [candidate["action"] for candidate in candidates if candidate["score"] == best]
        order = {action: position for position, action in enumerate(dataset.ACTIONS)}
        chosen = min(winners, key=lambda action: order[action])
        runner_up = sorted(candidate["score"] for candidate in candidates)[
            1 if len(candidates) > 1 else 0
        ]

        return Prediction(
            window_id=window.window_id,
            action=chosen,
            raw_output=chosen,
            meta={
                "baseline": NAME,
                "targets": dict(self.targets),
                "tolerances": dict(self.tolerances),
                "calibration": calibration.as_dict(),
                "candidates": candidates,
                "chosen": chosen,
                "score": best,
                "margin": runner_up - best,
                "tied": winners if len(winners) > 1 else [],
            },
        )
