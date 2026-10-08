# -*- coding: utf-8 -*-
"""The self-consistency strategy: zero_shot sampled N times + majority vote."""

from pathlib import Path

from src import config, dataset, harness_registry, llm

PROMPT_FILE = Path(config.HARNESS_DIR) / "zero_shot" / "prompt.txt"


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
    strategy = harness_registry.load_strategy("self_consistency")
    assert strategy.name == "self_consistency"
    assert strategy.description.strip()


def test_shares_the_prompt_file_with_zero_shot():
    strategy = harness_registry.load_strategy("self_consistency")
    assert strategy.prompt_file == PROMPT_FILE


def test_sends_the_very_same_messages_as_zero_shot(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["e"])

    harness_registry.load_strategy("self_consistency", n_samples=1).predict(window)

    expected = harness_registry.load_strategy("zero_shot").build_messages(window)
    assert calls["messages"][0] == expected


def test_samples_the_model_n_times(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["a", "b", "c"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=3).predict(window)

    assert len(calls["messages"]) == 3
    assert prediction.meta["n_samples"] == 3
    assert prediction.meta["samples"] == ["a", "b", "c"]


def test_the_majority_action_wins(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["a", "a", "a", "b"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=4).predict(window)

    assert prediction.action == "increase_hypnotic"
    assert prediction.meta["votes"] == {"increase_hypnotic": 3, "reduce_hypnotic": 1}
    assert prediction.meta["agreement"] == 0.75
    assert prediction.meta["parsed"] is True


def test_ties_are_broken_deterministically(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["a", "b"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=2).predict(window)

    # increase_hypnotic ("a") comes first in dataset.ACTIONS, so it wins the tie.
    assert prediction.action == "increase_hypnotic"


def test_unparseable_samples_do_not_vote(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["hmm", "b", "b", "???"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=4).predict(window)

    assert prediction.action == "reduce_hypnotic"
    assert prediction.meta["parsed_samples"] == 2
    assert prediction.meta["votes"] == {"reduce_hypnotic": 2}
    assert prediction.meta["agreement"] == 1.0


def test_all_unparseable_falls_back_to_no_action(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["hmm", "???"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=2).predict(window)

    assert prediction.action == "no_action"
    assert prediction.meta["parsed"] is False
    assert prediction.meta["parsed_samples"] == 0
    assert prediction.meta["votes"] == {}
    assert prediction.meta["agreement"] == 0.0


def test_raw_output_is_a_sample_that_voted_for_the_winner(monkeypatch):
    window = _window()
    _patch_llm_sequence(monkeypatch, ["b", "e", "e", "e"])

    prediction = harness_registry.load_strategy("self_consistency", n_samples=4).predict(window)

    assert prediction.action == "no_action"
    assert prediction.raw_output == "e"


def test_asks_for_a_sampling_temperature(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["a"])

    harness_registry.load_strategy("self_consistency", n_samples=1).predict(window)

    assert calls["kwargs"][0]["temperature"] == 1.0
    assert calls["kwargs"][0]["reasoning_effort"] == "high"


def test_temperature_can_be_overridden(monkeypatch):
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["a"])

    strategy = harness_registry.load_strategy("self_consistency", n_samples=1, temperature=0.5)
    strategy.predict(window)

    assert calls["kwargs"][0]["temperature"] == 0.5


def test_n_samples_is_never_forwarded_to_the_model(monkeypatch):
    """Strategy options must stay out of the model request."""
    window = _window()
    calls = _patch_llm_sequence(monkeypatch, ["a"])

    harness_registry.load_strategy("self_consistency", n_samples=1).predict(window)

    assert "n_samples" not in calls["kwargs"][0]
