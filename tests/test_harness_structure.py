# -*- coding: utf-8 -*-
"""Harness structure: one folder per strategy, loadable once implemented.

``EXPECTED_STRATEGIES`` is the roadmap; strategies are added one at a time, so
only the folders holding a ``strategy.py`` are required to load.
"""

import pytest

from src import config, dataset, harness_registry, llm
from src.harness_base import Strategy

EXPECTED = [
    "zero_shot",
    "few_shot",
    "cot",
    "self_consistency",
    "structured_json",
    "critic_verifier",
    "critic_verifier_sim",
]


def test_expected_strategies_is_the_roadmap():
    assert harness_registry.EXPECTED_STRATEGIES == EXPECTED


def test_implemented_strategies_are_a_subset_of_the_roadmap():
    assert set(harness_registry.strategy_dirs()) <= set(EXPECTED)


def test_zero_shot_and_structured_json_are_implemented(harness_dir):
    for name in ("zero_shot", "structured_json", "self_consistency"):
        folder = harness_dir / name
        assert folder.is_dir(), f"missing harness/{name}/"
        assert (folder / "strategy.py").is_file(), f"missing harness/{name}/strategy.py"


def test_strategy_dirs_detects_every_folder_with_a_strategy_file(harness_dir):
    detected = harness_registry.strategy_dirs()
    expected = sorted(
        child.name
        for child in harness_dir.iterdir()
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (child / "strategy.py").is_file()
    )
    assert detected == expected
    assert detected, "no strategy implemented yet"


def test_every_implemented_strategy_loads():
    names = harness_registry.strategy_dirs()
    assert names
    for name in names:
        strategy = harness_registry.load_strategy(name)
        assert isinstance(strategy, Strategy)
        assert strategy.name == name
        assert strategy.description.strip()


def test_loading_an_unknown_strategy_raises():
    with pytest.raises(KeyError):
        harness_registry.load_strategy("does_not_exist")


def test_load_all_matches_available_strategies():
    all_strategies = harness_registry.load_all()
    assert sorted(all_strategies) == sorted(harness_registry.available_strategies())
    assert "zero_shot" in all_strategies


def test_strategies_default_to_the_configured_model_and_effort():
    settings = config.get_settings()
    names = harness_registry.strategy_dirs()
    assert names
    for name in names:
        options = harness_registry.load_strategy(name).generation_options()
        assert set(options) == {"model", "temperature", "reasoning_effort"}
        assert options["model"] == settings.model
        assert options["reasoning_effort"] == settings.reasoning_effort


def test_single_call_strategies_use_the_configured_temperature():
    settings = config.get_settings()
    for name in ("zero_shot", "structured_json"):
        assert harness_registry.load_strategy(name).temperature == settings.temperature


def test_strategy_options_are_not_forwarded_to_the_model(monkeypatch):
    """A strategy-specific option must stay out of the model request."""
    sent = {}

    def fake_complete_messages(messages, **kwargs):
        sent.update(kwargs)
        return type("R", (), {"text": "a", "model": "m", "usage": {},
                              "latency_s": 0.0, "dropped_params": []})()

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    window = dataset.load_split("test", limit=1)[0]
    strategy = harness_registry.load_strategy("self_consistency", n_samples=1, custom_option=42)
    strategy.predict(window)

    assert "custom_option" not in sent
    assert "n_samples" not in sent
    assert sent["model"] == config.get_settings().model
