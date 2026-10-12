# -*- coding: utf-8 -*-
"""Discovery and loading of the strategies under ``harness/``.

STATE: implemented. The contract is defined in
``tests/test_harness_structure.py``.

Contract
-----------------
``EXPECTED_STRATEGIES`` -> ordered list of the project strategies:
    ["zero_shot", "few_shot", "cot", "self_consistency", "structured_json",
     "critic_verifier"]

``strategy_dirs() -> list[str]``        subfolders of ``harness/`` holding ``strategy.py``
``load_strategy(name) -> Strategy``      instantiate ``harness/<name>/``
``available_strategies() -> list[str]``
``load_all() -> dict[str, Strategy]``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from . import config
from .harness_base import get_strategy_class

__all__ = [
    "EXPECTED_STRATEGIES",
    "strategy_dirs",
    "load_strategy",
    "available_strategies",
    "load_all",
]

#: Roadmap of the strategies this project compares. A strategy is *available*
#: only once ``harness/<name>/strategy.py`` exists.
EXPECTED_STRATEGIES = [
    "zero_shot",
    "few_shot",
    "cot",
    "self_consistency",
    "structured_json",
    "critic_verifier",
    "critic_verifier_sim",
    "support",
    "rag",
    "shap_prompt",
]


def strategy_dirs():
    """Names of the strategies implemented under ``harness/``."""
    root = Path(config.HARNESS_DIR)
    if not root.is_dir():
        return []
    return [
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (child / "strategy.py").is_file()
    ]


def _load_module(name, path):
    module_name = f"harness_{name}_strategy"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_strategy(name, **options):
    """Load and instantiate the strategy ``name`` (``options`` go to its constructor)."""
    path = Path(config.HARNESS_DIR) / name / "strategy.py"
    if not path.is_file():
        raise KeyError(f"no strategy named {name!r} under {config.HARNESS_DIR}")
    return get_strategy_class(_load_module(name, path))(**options)


def available_strategies():
    """Loadable strategies."""
    return strategy_dirs()


def load_all():
    """Load every available strategy."""
    return {name: load_strategy(name) for name in strategy_dirs()}
