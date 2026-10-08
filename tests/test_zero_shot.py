# -*- coding: utf-8 -*-
"""The zero-shot strategy: prompt template, metadata and prediction."""

from pathlib import Path

from src import config, dataset, harness_registry, llm, prompting
from src.harness_base import Prediction

PROMPT_FILE = Path(config.HARNESS_DIR) / "zero_shot" / "prompt.txt"


class FakeResponse:
    """Stand-in for ``src.llm.LLMResponse``."""

    def __init__(self, text):
        self.text = text
        self.model = "gpt-5.4-mini"
        self.usage = {}
        self.latency_s = 0.0
        self.dropped_params = []


def _patch_llm(monkeypatch, text):
    """Replace the model call and capture the messages it receives."""
    captured = {}

    def fake_complete_messages(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return FakeResponse(text)

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    return captured


def test_prompt_file_exists():
    assert PROMPT_FILE.is_file()


def test_prompt_file_offers_exactly_five_options():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "{case}" in text
    for letter, action in prompting.LETTER_TO_ACTION.items():
        assert f"{letter})" in text
        assert action in text
    assert "f)" not in text
    assert "vasopressor" not in text


def test_strategy_metadata():
    strategy = harness_registry.load_strategy("zero_shot")
    assert strategy.name == "zero_shot"
    assert strategy.description.strip()


def test_predict_parses_the_model_answer(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    prediction = harness_registry.load_strategy("zero_shot").predict(window)

    assert isinstance(prediction, Prediction)
    assert prediction.window_id == window.window_id
    assert prediction.action == "reduce_hypnotic"
    assert prediction.raw_output == "b"
    assert prediction.meta["parsed"] is True


def test_predict_asks_for_temperature_0_and_high_reasoning(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    prediction = harness_registry.load_strategy("zero_shot").predict(window)

    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["reasoning_effort"] == "high"
    assert prediction.meta["temperature"] == 0.0
    assert prediction.meta["reasoning_effort"] == "high"
    assert prediction.meta["dropped_params"] == []


def test_reasoning_effort_can_be_overridden(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    strategy = harness_registry.load_strategy("zero_shot", reasoning_effort="medium")
    prediction = strategy.predict(window)

    assert captured["kwargs"]["reasoning_effort"] == "medium"
    assert prediction.meta["reasoning_effort"] == "medium"


def test_effort_none_omits_the_parameter(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    strategy = harness_registry.load_strategy("zero_shot", reasoning_effort="none")
    prediction = strategy.predict(window)

    assert captured["kwargs"]["reasoning_effort"] is None
    assert prediction.meta["reasoning_effort"] is None


def test_predict_sends_system_and_user_messages(monkeypatch):
    captured = _patch_llm(monkeypatch, "e")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("zero_shot").predict(window)

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"


def test_predict_renders_the_window_without_identifiers(monkeypatch):
    captured = _patch_llm(monkeypatch, "e")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("zero_shot").predict(window)

    content = captured["messages"][-1]["content"]
    assert "{case}" not in content
    assert "patient" in content.lower()
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in content


def test_unparseable_answer_falls_back_to_no_action(monkeypatch):
    _patch_llm(monkeypatch, "I cannot decide")
    window = dataset.load_split("test", limit=1)[0]

    prediction = harness_registry.load_strategy("zero_shot").predict(window)

    assert prediction.action == "no_action"
    assert prediction.meta["parsed"] is False
