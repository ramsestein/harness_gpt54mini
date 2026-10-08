# -*- coding: utf-8 -*-
"""Control baselines (no model) and their launcher.

- ``baselines.py``     the predictors: uniform random and the five constants.
- ``run_controls.py``  runs them over a split and writes the comparison table.

Results go to ``results_harness/control/``. See ``README.md``.
"""

from .baselines import (  # noqa: F401
    BASELINE_NAMES,
    CONSTANT_PREFIX,
    CONTROL_DIR,
    HUMAN_NAME,
    RANDOM_NAME,
    ConstantBaseline,
    HumanBaseline,
    RandomBaseline,
    all_baselines,
    constant_baselines,
    human_baseline,
    random_baseline,
)

__all__ = [
    "BASELINE_NAMES",
    "CONSTANT_PREFIX",
    "CONTROL_DIR",
    "HUMAN_NAME",
    "RANDOM_NAME",
    "ConstantBaseline",
    "HumanBaseline",
    "RandomBaseline",
    "all_baselines",
    "constant_baselines",
    "human_baseline",
    "random_baseline",
]
