# -*- coding: utf-8 -*-
"""The structured-JSON strategy: the whole prompt is a JSON document."""

import json
from pathlib import Path

from src import config, dataset, harness_registry, llm, prompting
from src.harness_base import Prediction

PROMPT_FILE = Path(config.HARNESS_DIR) / "structured_json" / "prompt.json"


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


def test_prompt_file_is_a_json_object():
    payload = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["task"]
    assert payload["output_format"]
    assert "patient_state" not in payload  # injected per window by the strategy


def test_prompt_offers_exactly_five_options():
    payload = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    assert payload["options"] == {
        "a": "increase_hypnotic",
        "b": "reduce_hypnotic",
        "c": "increase_opioid",
        "d": "reduce_opioid",
        "e": "no_action",
    }
    assert set(payload["options"]) == set("abcde")
    assert "vasopressor" not in PROMPT_FILE.read_text(encoding="utf-8")


def test_strategy_metadata():
    strategy = harness_registry.load_strategy("structured_json")
    assert strategy.name == "structured_json"
    assert strategy.description.strip()


def test_predict_parses_the_model_answer(monkeypatch):
    _patch_llm(monkeypatch, "d")
    window = dataset.load_split("test", limit=1)[0]

    prediction = harness_registry.load_strategy("structured_json").predict(window)

    assert isinstance(prediction, Prediction)
    assert prediction.window_id == window.window_id
    assert prediction.action == "reduce_opioid"
    assert prediction.meta["parsed"] is True


def test_the_whole_user_prompt_is_one_json_document(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("structured_json").predict(window)

    content = captured["messages"][-1]["content"].strip()
    assert content.startswith("{") and content.endswith("}")
    payload = json.loads(content)
    assert payload["patient_state"] == prompting.render_case(window.input)


def test_the_patient_state_is_the_shared_rendering(monkeypatch):
    """Only the wrapper is JSON: the window is the same text every strategy sends."""
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("structured_json").predict(window)

    payload = json.loads(captured["messages"][-1]["content"])
    assert payload["patient_state"] == prompting.render_case(window.input)
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in captured["messages"][-1]["content"]


def test_indent_pretty_prints_the_prompt(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("structured_json", indent=2).predict(window)

    content = captured["messages"][-1]["content"]
    assert "\n  " in content
    assert json.loads(content)["patient_state"] == prompting.render_case(window.input)


def test_asks_for_the_same_generation_settings_as_zero_shot(monkeypatch):
    captured = _patch_llm(monkeypatch, "b")
    window = dataset.load_split("test", limit=1)[0]

    harness_registry.load_strategy("structured_json").predict(window)

    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["reasoning_effort"] == "high"


def test_same_system_prompt_as_zero_shot(monkeypatch):
    """Both strategies share the system message; only the user message differs."""
    window = dataset.load_split("test", limit=1)[0]

    prose = _patch_llm(monkeypatch, "b")
    harness_registry.load_strategy("zero_shot").predict(window)
    prose_messages = prose["messages"]

    structured = _patch_llm(monkeypatch, "b")
    harness_registry.load_strategy("structured_json").predict(window)
    json_messages = structured["messages"]

    assert prose_messages[0] == json_messages[0]
    assert prompting.render_case(window.input) in prose_messages[-1]["content"]
    assert json.loads(json_messages[-1]["content"])["patient_state"] == prompting.render_case(
        window.input
    )
