# -*- coding: utf-8 -*-
"""The model as a property of the run: per-run settings, and how a strategy uses them."""

import pytest

from src import config, llm
from src.harness_base import Strategy


class Dummy(Strategy):
    """The smallest strategy there is: it only needs ``ask``."""

    name = "dummy"
    description = "test double"

    def predict(self, window):  # pragma: no cover - not used here
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# settings_for / api_key
# --------------------------------------------------------------------------- #


def test_the_base_settings_are_untouched(monkeypatch):
    monkeypatch.setenv("ANESLLM_MODEL", "model-base")
    monkeypatch.setenv("OPENAI_API_KEY", "key-base")

    settings = config.get_settings()
    other = config.settings_for(model="model-other")

    assert other.model == "model-other"
    assert settings.model == "model-base"  # the globals never change
    assert other.api_key == settings.api_key
    assert other.reasoning_effort == settings.reasoning_effort
    assert other.timeout == settings.timeout


def test_another_key_comes_from_its_own_variable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-base")
    monkeypatch.setenv("CLAUDE_API_KEY", "key-claude")

    settings = config.settings_for(model="haiku", api_key_env="CLAUDE_API_KEY")

    assert settings.api_key == "key-claude"


def test_a_missing_key_variable_says_which_one(monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)

    with pytest.raises(KeyError, match="NOPE_API_KEY"):
        config.settings_for(model="x", api_key_env="NOPE_API_KEY")


def test_an_endpoint_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-base")

    settings = config.settings_for(model="x", base_url="https://example.invalid/v1")

    assert settings.base_url == "https://example.invalid/v1"


def test_nothing_given_returns_the_project_settings(monkeypatch):
    monkeypatch.setenv("ANESLLM_MODEL", "model-base")

    assert config.settings_for().model == "model-base"


# --------------------------------------------------------------------------- #
# The strategy side
# --------------------------------------------------------------------------- #


def test_a_strategy_without_settings_uses_the_project_model(monkeypatch):
    monkeypatch.setenv("ANESLLM_MODEL", "model-base")

    strategy = Dummy()

    assert strategy.settings is None
    assert strategy.model == "model-base"
    assert strategy.llm_client() is None  # the shared helper does the call


def test_a_strategy_with_settings_takes_the_model_from_them(monkeypatch):
    monkeypatch.setenv("ANESLLM_MODEL", "model-base")
    settings = config.settings_for(model="model-other")

    strategy = Dummy(settings=settings)

    assert strategy.model == "model-other"
    assert strategy.settings is settings


def test_an_explicit_model_argument_still_wins(monkeypatch):
    monkeypatch.setenv("ANESLLM_MODEL", "model-base")

    strategy = Dummy(settings=config.settings_for(model="model-other"), model="model-third")

    assert strategy.model == "model-third"


def test_the_per_run_client_is_built_once_and_reused(monkeypatch):
    settings = config.settings_for(model="model-other")
    built = []

    def fake_client(settings=None, client=None):
        built.append(settings)
        return type("C", (), {"settings": settings})()

    monkeypatch.setattr(llm, "LLMClient", fake_client)
    strategy = Dummy(settings=settings)

    first = strategy.llm_client()
    second = strategy.llm_client()

    assert first is second
    assert built == [settings]


def test_the_calls_go_through_the_per_run_client(monkeypatch):
    settings = config.settings_for(model="model-other")
    seen = {}

    def fake_complete(self, messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return type("R", (), {"text": "ok", "model": kwargs.get("model"),
                              "latency_s": 0.0, "dropped_params": []})()

    monkeypatch.setattr(llm, "complete_messages", lambda *a, **k: pytest.fail("wrong client"))
    monkeypatch.setattr(llm.LLMClient, "complete_messages", fake_complete)
    strategy = Dummy(settings=settings, reasoning_effort="high")

    response = strategy.ask([{"role": "user", "content": "hi"}])

    assert response.text == "ok"
    assert seen["kwargs"]["model"] == "model-other"
    assert seen["kwargs"]["reasoning_effort"] == "high"
