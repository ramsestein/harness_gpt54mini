# -*- coding: utf-8 -*-
"""Control baselines: what the dataset answers without any model.

Three families, all built on the same ``Strategy`` contract so they run through the
same launcher as the harness strategies:

- ``RandomBaseline`` -- draws uniformly among the five actions.
- ``ConstantBaseline`` -- always answers the same action. The five of them together
  measure the label prevalence, and the best of them is the majority-class baseline.
- ``HumanBaseline`` -- replays the clinician's own action: the human value of the
  experiment. It is 1.0 against the real action by construction, so what it measures
  is the clinician's agreement with the dataset's plausible answers.

The random draw depends on the seed **and** the window id, never on the position of
the window in the batch: the result is independent of ``--workers`` and rerunning the
same seed reproduces the same run.

Results go to ``results_harness/control/`` (see ``run_controls.py``).
"""

from __future__ import annotations

import random
from pathlib import Path

from src import config, dataset
from src.harness_base import Prediction, Strategy

__all__ = [
    "CONTROL_DIR",
    "BASELINE_NAMES",
    "RANDOM_NAME",
    "HUMAN_NAME",
    "ControlStrategy",
    "RandomBaseline",
    "ConstantBaseline",
    "HumanBaseline",
    "build",
    "random_baseline",
    "constant_baselines",
    "human_baseline",
    "all_baselines",
]

#: Every control writes to ``results_harness/control/<baseline>/<run_id>/``.
CONTROL_DIR = Path(config.RESULTS_DIR) / "control"

#: Name of the random baseline (also its strategies folder name).
RANDOM_NAME = "random"

#: Name of the baseline that replays the clinician.
HUMAN_NAME = "human"

#: Name of the constant baseline of a given action.
CONSTANT_PREFIX = "always_"


class ControlStrategy(Strategy):
    """A strategy that answers without calling the model."""

    uses_model = False


class RandomBaseline(ControlStrategy):
    """Uniform random choice among the five actions."""

    name = RANDOM_NAME
    description = "Uniform random choice among the five actions (no model)."

    def __init__(self, seed=0, **options):
        super().__init__(**options)
        self.seed = int(seed)

    def _rng(self, window_id):
        """Per-window generator, so the draw only depends on the seed and the window."""
        return random.Random(f"{self.seed}:{window_id}")

    def predict(self, window):
        draw = self._rng(window.window_id).randrange(len(dataset.ACTIONS))
        action = dataset.ACTIONS[draw]
        return Prediction(
            window_id=window.window_id,
            action=action,
            raw_output=action,
            meta={"baseline": RANDOM_NAME, "seed": self.seed, "draw": draw},
        )


class ConstantBaseline(ControlStrategy):
    """Always the same action -- one per option the model is given."""

    description = "Always predicts the same action (no model)."

    def __init__(self, action, **options):
        if action not in dataset.ACTIONS:
            raise ValueError(
                f"unknown action: {action!r} (expected one of {dataset.ACTIONS})"
            )
        super().__init__(**options)
        self.action = action
        self.name = f"{CONSTANT_PREFIX}{action}"
        self.description = f"Always predicts {action} (no model)."

    def predict(self, window):
        return Prediction(
            window_id=window.window_id,
            action=self.action,
            raw_output=self.action,
            meta={"baseline": "always", "action": self.action},
        )


class HumanBaseline(ControlStrategy):
    """Replay the clinician's action: the human value of the experiment.

    It scores 1.0 against the real action by construction, so its interest is the other
    two agreements: how often the clinician's action equals the consensus ``result_1``
    and how often it is a plausible answer (``result_1`` or ``result_aux``).
    """

    name = HUMAN_NAME
    description = "Replays the real action taken by the clinician (no model)."

    def predict(self, window):
        return Prediction(
            window_id=window.window_id,
            action=window.real_action,
            raw_output=window.real_action,
            meta={"baseline": HUMAN_NAME},
        )


#: Every control, in the order they are run.
BASELINE_NAMES = [
    RANDOM_NAME,
    *[f"{CONSTANT_PREFIX}{a}" for a in dataset.ACTIONS],
    HUMAN_NAME,
]


def random_baseline(seed=0):
    """The uniform random baseline."""
    return RandomBaseline(seed=seed)


def constant_baselines():
    """One constant baseline per action, in ``dataset.ACTIONS`` order."""
    return [ConstantBaseline(action) for action in dataset.ACTIONS]


def human_baseline():
    """The clinician's own action -- the human value of the experiment."""
    return HumanBaseline()


def all_baselines(seed=0):
    """Every control: the random one, the five constants, then the human one."""
    return [random_baseline(seed=seed), *constant_baselines(), human_baseline()]


def build(name, seed=0, **options):
    """Build one of these controls by name (``src/control/run_controls.py`` uses this)."""
    if name == RANDOM_NAME:
        return RandomBaseline(seed=seed, **options)
    if name == HUMAN_NAME:
        return HumanBaseline(**options)
    if name.startswith(CONSTANT_PREFIX):
        return ConstantBaseline(name[len(CONSTANT_PREFIX):], **options)
    raise KeyError(f"unknown control: {name!r} (expected one of {BASELINE_NAMES})")
