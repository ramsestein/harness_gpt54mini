# -*- coding: utf-8 -*-
"""The simulated critic-verifier strategy (generator + simulator + critic)."""

import dataclasses
import re
from pathlib import Path

import pytest

from harness import run as harness_run
from harness.critic_verifier_sim import simulator as sim
from src import config, dataset, harness_registry, llm, prompting

HERE = Path(config.HARNESS_DIR) / "critic_verifier_sim"
GENERATOR_PROMPT = HERE / "generator_prompt.txt"
CRITIC_PROMPT = HERE / "critic_prompt.txt"

GENERATOR_ANSWER = "Candidate 1: c\nCandidate 2: a"
CRITIC_ANSWER = "Window 2 keeps the BIS in range.\n\nAnswer: 2"


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("test", limit=1)[0]


@pytest.fixture(scope="module")
def engine():
    return sim.AnesthesiaSimulator()


@pytest.fixture(scope="module")
def increased_and_baseline(window, engine):
    """One calibration + the 'increase_hypnotic' and 'no_action' futures."""
    return engine.simulate(window.input, ["increase_hypnotic", "no_action"])


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


def _strategy(**kwargs):
    return harness_registry.load_strategy("critic_verifier_sim", **kwargs)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_strategy_metadata():
    strategy = _strategy()
    assert strategy.name == "critic_verifier_sim"
    assert strategy.description.strip()


def test_generator_prompt_asks_for_two_candidates():
    text = GENERATOR_PROMPT.read_text(encoding="utf-8")
    assert "{case}" in text
    assert "Candidate 1: <letter>" in text
    assert "Candidate 2: <letter>" in text
    assert "must be different" in text
    for letter, action in prompting.LETTER_TO_ACTION.items():
        assert f"{letter})" in text
        assert action in text
    assert "vasopressor" not in text


def test_critic_prompt_is_about_choosing_between_two_windows():
    text = CRITIC_PROMPT.read_text(encoding="utf-8")
    assert "{window_1}" in text and "{window_2}" in text
    assert "Answer: <1 or 2>" in text
    assert "between 40 and 60" in text
    assert "65 mmHg" in text


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


def test_calibration_reproduces_the_bis_and_map_of_the_window(window, increased_and_baseline):
    calibration, _ = increased_and_baseline
    assert abs(calibration.bis_residual) < 0.5, calibration.as_dict()
    assert abs(calibration.map_residual) < 0.5, calibration.as_dict()
    assert calibration.matched is True
    assert calibration.u_propo > 0 and calibration.u_remi > 0 and calibration.u_nore > 0


def test_no_action_reproduces_the_calibrated_state(increased_and_baseline):
    calibration, (increased, baseline) = increased_and_baseline
    assert baseline.action == "no_action"
    assert baseline.bis == pytest.approx(calibration.bis, abs=1.0)
    assert baseline.map == pytest.approx(calibration.map, abs=1.5)


def test_increase_hypnotic_deepens_the_anaesthesia(increased_and_baseline):
    calibration, (increased, baseline) = increased_and_baseline
    assert increased.u_propo == pytest.approx(calibration.u_propo * 1.5)
    assert increased.u_remi == pytest.approx(calibration.u_remi)
    assert increased.bis < baseline.bis


def test_reduce_hypnotic_lightens_the_anaesthesia(window, engine):
    calibration, (reduced, _) = engine.simulate(window.input, ["reduce_hypnotic", "no_action"])
    assert reduced.u_propo == pytest.approx(calibration.u_propo * 0.5)
    assert reduced.bis > calibration.bis


def test_an_unknown_action_is_rejected(window, engine):
    with pytest.raises(ValueError):
        engine.simulate(window.input, ["levitate"])


def test_the_outcome_carries_the_simulated_signals(window, increased_and_baseline):
    _, (increased, _) = increased_and_baseline
    signals = increased.signals()
    assert set(signals) == {"bis", "map", "co", "tol"}
    assert 0 < signals["bis"] < 100
    assert signals["tol"] >= 0


# --------------------------------------------------------------------------- #
# Missing baselines
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def quiet_engine():
    """A short horizon: imputation does not depend on how far ahead we look."""
    return sim.AnesthesiaSimulator(horizon_seconds=60)


def _without_baselines(window, **fields):
    """The same window with the given baselines removed (or overwritten)."""
    return dataclasses.replace(window, input={**window.input, **fields})


def test_an_observed_baseline_is_not_imputed(increased_and_baseline):
    calibration, _ = increased_and_baseline
    assert calibration.bis_source == "observed"
    assert calibration.map_source == "observed"
    assert calibration.imputed == ()


def test_a_window_without_baselines_is_calibrated_on_the_ideal_values(window, quiet_engine):
    gone = _without_baselines(window, bis_current=None, map_current=None)

    calibration, _ = quiet_engine.simulate(gone.input, ["no_action", "increase_hypnotic"])

    assert calibration.bis_source == "imputed" and calibration.map_source == "imputed"
    assert calibration.imputed == ("bis", "map")
    assert calibration.bis == pytest.approx(sim.DEFAULT_BIS, abs=0.5)
    assert calibration.map == pytest.approx(sim.DEFAULT_MAP, abs=0.5)
    assert calibration.matched is True
    assert calibration.as_dict()["bis_source"] == "imputed"


def test_only_the_missing_baseline_is_imputed(window, quiet_engine):
    partial = _without_baselines(window, bis_current=None)

    calibration, _ = quiet_engine.simulate(partial.input, ["no_action", "increase_hypnotic"])

    assert calibration.bis_source == "imputed"
    assert calibration.map_source == "observed"
    assert calibration.imputed == ("bis",)
    assert calibration.map == pytest.approx(float(window.input["map_current"]), abs=0.5)


def test_a_non_finite_baseline_counts_as_missing(window, quiet_engine):
    broken = _without_baselines(window, map_current=float("nan"))

    calibration, _ = quiet_engine.simulate(broken.input, ["no_action", "increase_hypnotic"])

    assert calibration.map_source == "imputed"
    assert calibration.map == pytest.approx(sim.DEFAULT_MAP, abs=0.5)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Candidate 1: c\nCandidate 2: a", ["increase_opioid", "increase_hypnotic"]),
        ("candidate 1: A\ncandidate 2: e", ["increase_hypnotic", "no_action"]),
        ("Candidate 1: a\nCandidate 2: a", ["increase_hypnotic"]),  # deduplicated
        ("Candidate 2: b\nCandidate 1: d", ["reduce_opioid", "reduce_hypnotic"]),
        ("Candidate 1: (b)\nCandidate 2: no_action", ["reduce_hypnotic", "no_action"]),
        ("Candidate 1: increase_opioid\nCandidate 2: e", ["increase_opioid", "no_action"]),
        ("nothing here", []),
        ("", []),
        (None, []),
    ],
)
def test_parse_candidates(text, expected):
    assert _strategy().parse_candidates(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: 2", 2),
        ("I prefer the second window.\n\nAnswer: 1", 1),
        ("Answer: (1)", 1),
        ("1", 1),
        ("2.", 2),
        ("no idea", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_choice(text, expected):
    assert _strategy().parse_choice(text) == expected


def test_present_order_is_deterministic_but_varies():
    strategy = _strategy()
    assert strategy.present_order("case1169_w16415") == strategy.present_order("case1169_w16415")
    orders = {tuple(strategy.present_order(f"case{index}_w1")) for index in range(100)}
    assert orders == {(0, 1), (1, 0)}


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #


def test_the_pipeline_maps_the_choice_back_to_the_action(window, monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, CRITIC_ANSWER])
    strategy = _strategy()

    prediction = strategy.predict(window)

    assert len(calls["messages"]) == 2  # one generator call, one critic call
    order = strategy.present_order(window.window_id)
    assert prediction.meta["candidates"] == ["increase_opioid", "increase_hypnotic"]
    assert prediction.meta["presented"] == [
        prediction.meta["candidates"][order[0]],
        prediction.meta["candidates"][order[1]],
    ]
    assert prediction.meta["choice"] == 2
    assert prediction.action == prediction.meta["presented"][1]
    assert prediction.meta["chosen"] == prediction.action
    assert prediction.meta["critic_parsed"] is True
    assert prediction.meta["fallback"] is False


def test_the_critic_sees_the_simulated_signals_blind(window, monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, CRITIC_ANSWER])

    prediction = _strategy().predict(window)

    generator_prompt = calls["messages"][0][-1]["content"]
    critic_prompt = calls["messages"][1][-1]["content"]
    assert prompting.render_case(window.input) in generator_prompt

    assert "=== WINDOW 1 ===" in critic_prompt
    assert "=== WINDOW 2 ===" in critic_prompt
    assert "{window_1}" not in critic_prompt and "{window_2}" not in critic_prompt

    # The two blocks carry the simulated signals of the two candidates. The prompt
    # rounds to one decimal and the metadata to two, so compare with tolerance.
    bis_values = [
        float(value)
        for value in re.findall(r"BIS \(bispectral index\): ([\d.]+)", critic_prompt)
    ]
    map_values = [
        float(value)
        for value in re.findall(r"MAP \(mean arterial pressure\): ([\d.]+) mmHg", critic_prompt)
    ]
    assert len(bis_values) == 2 and len(map_values) == 2
    for outcome in prediction.meta["simulated"]:
        assert any(abs(value - outcome["bis"]) < 0.06 for value in bis_values)
        assert any(abs(value - outcome["map"]) < 0.06 for value in map_values)

    # blind: neither the actions nor the window's own values may leak
    for action in prediction.meta["candidates"]:
        assert action not in critic_prompt
    assert prompting.render_case(window.input) not in critic_prompt


def test_the_calibration_is_recorded(window, monkeypatch):
    _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, CRITIC_ANSWER])

    prediction = _strategy().predict(window)

    calibration = prediction.meta["calibration"]
    assert calibration["bis_residual"] == pytest.approx(0.0, abs=0.5)
    assert calibration["map_residual"] == pytest.approx(0.0, abs=0.5)
    assert len(prediction.meta["simulated"]) == 2
    assert prediction.meta["latencies_s"].__len__() == 2


def test_an_unparseable_choice_falls_back_to_the_first_window(window, monkeypatch):
    _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, "I cannot decide."])

    prediction = _strategy().predict(window)

    assert prediction.meta["critic_parsed"] is False
    assert prediction.meta["fallback"] is True
    assert prediction.action == prediction.meta["presented"][0]


def test_a_single_candidate_is_reported_as_a_failure(window, monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, ["Candidate 1: a\nCandidate 2: a"])

    prediction = _strategy().predict(window)

    assert len(calls["messages"]) == 1  # no simulation, no critic call
    assert prediction.action == "no_action"
    assert prediction.meta["parsed"] is False
    assert "error" in prediction.meta
    # the launcher retries windows flagged this way with --resume
    assert harness_run.is_failed({"action": prediction.action, "meta": prediction.meta}) is True


def test_a_window_without_baselines_still_reaches_the_critic(window, monkeypatch):
    _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, CRITIC_ANSWER])
    gone = _without_baselines(window, bis_current=None, map_current=None)

    prediction = _strategy().predict(gone)

    assert "error" not in prediction.meta
    assert prediction.meta["calibration"]["bis_source"] == "imputed"
    assert prediction.meta["calibration"]["map_source"] == "imputed"
    assert prediction.action == prediction.meta["presented"][1]


def test_the_calls_use_the_strategy_generation_settings(window, monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, [GENERATOR_ANSWER, CRITIC_ANSWER])

    _strategy().predict(window)

    for kwargs in calls["kwargs"]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["reasoning_effort"] == "high"
