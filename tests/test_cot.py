# -*- coding: utf-8 -*-
"""The chain-of-thought strategy: reason first, then answer with one letter."""

from pathlib import Path

from src import config, dataset, harness_registry, llm, prompting

PROMPT_FILE = Path(config.HARNESS_DIR) / "cot" / "prompt.txt"
ZERO_SHOT_PROMPT = Path(config.HARNESS_DIR) / "zero_shot" / "prompt.txt"

REACT_ANSWER = (
    "Thought: the depth of hypnosis is the axis I need first.\n"
    "Action: LOOKUP bis_current\n"
    "Observation: BIS is 74.9 and falling.\n"
    "Thought: option c is tempting, but with this BIS the hypnotic is the problem.\n"
    "Action: COMPARE increase_hypnotic vs increase_opioid\n"
    "Observation: the hypnotic effect-site concentration is only 3.33 while the opioid is 5.\n"
    "Thought: I now have enough information to decide.\n"
    "Answer: a"
)


class FakeResponse:
    """Stand-in for ``src.llm.LLMResponse``."""

    def __init__(self, text, latency=0.1, dropped=()):
        self.text = text
        self.model = "gpt-5.4-mini"
        self.usage = {}
        self.latency_s = latency
        self.dropped_params = list(dropped)


def _patch_llm(monkeypatch, text):
    """Replace the model call and capture the messages it receives."""
    captured = {}

    def fake_complete_messages(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return FakeResponse(text)

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    return captured


def _window():
    return dataset.load_split("test", limit=1)[0]


def _strategy(**kwargs):
    return harness_registry.load_strategy("cot", **kwargs)


def _option_lines(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("a)", "b)", "c)", "d)", "e)"))
    ]


def test_strategy_metadata():
    strategy = _strategy()
    assert strategy.name == "cot"
    assert strategy.description.strip()


def test_prompt_file_offers_exactly_the_same_five_options_as_zero_shot():
    assert _option_lines(PROMPT_FILE) == _option_lines(ZERO_SHOT_PROMPT)
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "vasopressor" not in text
    assert "f)" not in text


def test_asks_for_a_react_loop_before_the_closing_answer():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "{case}" in text
    assert "Answer: <letter>" in text
    for marker in ("Thought:", "Action:", "Observation:"):
        assert marker in text
    for action in ("LOOKUP", "TREND", "HISTORY", "COMPARE"):
        assert action in text
    assert "write nothing after them" in text
    assert "Never invent" in text


def test_prompt_keeps_the_three_clinical_axes_of_the_cot():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "haemodynamics" in text
    assert "depth of hypnosis" in text
    assert "analgesia" in text


def test_same_system_prompt_and_window_rendering_as_zero_shot(monkeypatch):
    window = _window()
    captured = _patch_llm(monkeypatch, REACT_ANSWER)

    _strategy().predict(window)

    zero_shot = harness_registry.load_strategy("zero_shot").build_messages(window)
    messages = captured["messages"]
    assert messages[0] == zero_shot[0]  # identical system prompt
    assert prompting.render_case(window.input) in messages[-1]["content"]


def test_makes_a_single_call_per_window(monkeypatch):
    calls = []

    def fake_complete_messages(messages, **kwargs):
        calls.append(messages)
        return FakeResponse(REACT_ANSWER)

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)

    _strategy().predict(_window())

    assert len(calls) == 1


def test_the_closing_answer_wins_over_options_named_in_the_trace(monkeypatch):
    _patch_llm(monkeypatch, REACT_ANSWER)

    prediction = _strategy().predict(_window())

    # The trace mentions option c, increase_hypnotic and increase_opioid; the
    # closing line is "Answer: a".
    assert prediction.action == "increase_hypnotic"
    assert prediction.meta["parsed"] is True


def test_the_trace_is_recorded_without_the_answer_line(monkeypatch):
    _patch_llm(monkeypatch, REACT_ANSWER)

    prediction = _strategy().predict(_window())

    reasoning = prediction.meta["reasoning"]
    assert reasoning.startswith("Thought: the depth of hypnosis")
    assert "Action: LOOKUP bis_current" in reasoning
    assert "Observation: BIS is 74.9" in reasoning
    assert "Answer:" not in reasoning
    assert prediction.raw_output == REACT_ANSWER  # the whole answer is kept


def test_the_react_trace_is_counted_in_the_metadata(monkeypatch):
    _patch_llm(monkeypatch, REACT_ANSWER)

    prediction = _strategy().predict(_window())

    assert prediction.meta["react"] == {"thoughts": 3, "actions": 2, "observations": 2}


def test_a_plain_letter_without_reasoning_still_parses(monkeypatch):
    _patch_llm(monkeypatch, "c")

    prediction = _strategy().predict(_window())

    assert prediction.action == "increase_opioid"
    assert prediction.meta["reasoning"] == ""


def test_unparseable_answer_falls_back_to_no_action(monkeypatch):
    _patch_llm(monkeypatch, "I would rather not say.")

    prediction = _strategy().predict(_window())

    assert prediction.action == "no_action"
    assert prediction.meta["parsed"] is False


def test_asks_for_the_same_generation_settings_as_zero_shot(monkeypatch):
    captured = _patch_llm(monkeypatch, REACT_ANSWER)

    _strategy().predict(_window())

    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["reasoning_effort"] == "high"


def test_the_prompt_does_not_leak_identifiers(monkeypatch):
    captured = _patch_llm(monkeypatch, REACT_ANSWER)
    window = _window()

    _strategy().predict(window)

    content = captured["messages"][-1]["content"]
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in content


# --------------------------------------------------------------------------- #
# extract_reasoning
# --------------------------------------------------------------------------- #


def test_extract_reasoning_keeps_everything_before_the_answer():
    assert _strategy().extract_reasoning("line one\nline two\nAnswer: b") == (
        "line one\nline two"
    )


def test_extract_reasoning_is_empty_without_an_answer_line():
    assert _strategy().extract_reasoning("just some text") == ""
    assert _strategy().extract_reasoning("") == ""
    assert _strategy().extract_reasoning(None) == ""


def test_extract_reasoning_uses_the_last_answer_line():
    assert _strategy().extract_reasoning("Answer: a\nmore doubts\nAnswer: b") == (
        "Answer: a\nmore doubts"
    )


def test_count_react_steps():
    steps = _strategy().count_react_steps(REACT_ANSWER)
    assert steps == {"thoughts": 3, "actions": 2, "observations": 2}


def test_count_react_steps_on_a_bare_answer():
    assert _strategy().count_react_steps("b") == {
        "thoughts": 0,
        "actions": 0,
        "observations": 0,
    }
    assert _strategy().count_react_steps(None) == {
        "thoughts": 0,
        "actions": 0,
        "observations": 0,
    }
