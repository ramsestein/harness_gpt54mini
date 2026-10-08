# -*- coding: utf-8 -*-
"""The critic-verifier strategy: generate with zero_shot, then review."""

from pathlib import Path

from src import config, dataset, harness_registry, llm, prompting

ZERO_SHOT_PROMPT = Path(config.HARNESS_DIR) / "zero_shot" / "prompt.txt"
CRITIC_PROMPT = Path(config.HARNESS_DIR) / "critic_verifier" / "critic_prompt.txt"


class FakeResponse:
    """Stand-in for ``src.llm.LLMResponse``."""

    def __init__(self, text, latency=0.1, dropped=()):
        self.text = text
        self.model = "gpt-5.4-mini"
        self.usage = {}
        self.latency_s = latency
        self.dropped_params = list(dropped)


def _patch_llm_sequence(monkeypatch, texts):
    """Answer the i-th call with ``texts[i % len(texts)]`` and record the calls."""
    calls = {"messages": [], "kwargs": []}

    def fake_complete_messages(messages, **kwargs):
        index = len(calls["messages"])
        calls["messages"].append(messages)
        calls["kwargs"].append(kwargs)
        return FakeResponse(texts[index % len(texts)])

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    return calls


def _window():
    return dataset.load_split("test", limit=1)[0]


def test_strategy_metadata():
    strategy = harness_registry.load_strategy("critic_verifier")
    assert strategy.name == "critic_verifier"
    assert strategy.description.strip()


def test_generator_uses_the_zero_shot_prompt():
    strategy = harness_registry.load_strategy("critic_verifier")
    assert strategy.prompt_file == ZERO_SHOT_PROMPT
    assert strategy.critic_prompt_file == CRITIC_PROMPT


def test_first_call_is_identical_to_zero_shot(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["b", "b"])

    harness_registry.load_strategy("critic_verifier").predict(window)

    expected = harness_registry.load_strategy("zero_shot").build_messages(window)
    assert calls["messages"][0] == expected


def test_makes_exactly_two_calls(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["b", "b"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert len(calls["messages"]) == 2
    assert len(prediction.meta["latencies_s"]) == 2


def test_the_critic_receives_the_window_and_the_proposal(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["b", "e"])

    harness_registry.load_strategy("critic_verifier").predict(window)

    critic_prompt = calls["messages"][1][-1]["content"]
    assert "reduce_hypnotic" in critic_prompt  # the proposal, by name
    assert "{case}" not in critic_prompt
    assert "{proposal}" not in critic_prompt
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in critic_prompt
    assert prompting.render_case(window.input) in critic_prompt


def test_the_critic_can_confirm(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["b", "b"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.meta["proposal"] == "reduce_hypnotic"
    assert prediction.action == "reduce_hypnotic"
    assert prediction.meta["changed"] is False
    assert prediction.meta["critic_parsed"] is True
    assert prediction.meta["parsed"] is True


def test_the_critic_can_correct(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["b", "e"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.meta["proposal"] == "reduce_hypnotic"
    assert prediction.action == "no_action"
    assert prediction.meta["changed"] is True


def test_the_critic_review_is_the_final_answer(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["a", "d"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.action == "reduce_opioid"
    assert prediction.meta["final"] == "reduce_opioid"
    assert prediction.raw_output == "d"  # the critic's raw text


def test_unparseable_critic_keeps_the_proposal(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["b", "I am not sure"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.action == "reduce_hypnotic"
    assert prediction.meta["critic_parsed"] is False
    assert prediction.meta["changed"] is False
    assert prediction.meta["parsed"] is True


def test_unparseable_generator_is_flagged_in_the_critic_prompt(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["no idea", "e"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    critic_prompt = calls["messages"][1][-1]["content"]
    assert "none (the first answer could not be parsed)" in critic_prompt
    assert prediction.meta["proposal"] is None
    assert prediction.action == "no_action"  # the critic still decides
    assert prediction.meta["generator_parsed"] is False


def test_both_unparseable_falls_back_to_no_action(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["no idea", "still no idea"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.action == "no_action"
    assert prediction.meta["parsed"] is False
    assert prediction.meta["final"] is None


def test_records_both_raw_answers(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["b", "e"])

    prediction = harness_registry.load_strategy("critic_verifier").predict(window)

    assert prediction.meta["generator_raw"] == "b"
    assert prediction.meta["critic_raw"] == "e"


def test_asks_for_the_same_generation_settings_as_zero_shot(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["b", "b"])

    harness_registry.load_strategy("critic_verifier").predict(window)

    for kwargs in calls["kwargs"]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["reasoning_effort"] == "high"


def test_critic_prompt_file_offers_exactly_five_options():
    text = CRITIC_PROMPT.read_text(encoding="utf-8")
    assert "{case}" in text
    assert "{proposal}" in text
    for letter, action in prompting.LETTER_TO_ACTION.items():
        assert f"{letter})" in text
        assert action in text
    assert "f)" not in text
    assert "vasopressor" not in text
