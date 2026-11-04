# -*- coding: utf-8 -*-
"""The zero_shot ablation with the effect-site fields cut at -30 s."""

import sys

from src import dataset, harness_registry, prompting
from src.control import control_view


def _raw(ce_remi):
    """A raw input whose remifentanil Ce series is given as {t_offset_sec: value}."""
    series = [{"t_offset_sec": t, "value": v} for t, v in sorted(ce_remi.items())]
    return {"drug_concentrations_json": {"remi_Ce": series, "propofol_Ce": []}}


class FakeWindow:
    def __init__(self, inp, raw):
        self.window_id = "w"
        self.input = inp
        self.raw_input = raw


def _strategy():
    return harness_registry.load_strategy("zero_shot_ce_cut")


def _cut_input(window):
    """``cut_input`` of the loaded strategy module (the registry loads it by path)."""
    return sys.modules[type(_strategy()).__module__].cut_input(window)


def test_it_loads_under_its_own_name():
    assert _strategy().name == "zero_shot_ce_cut"


def test_a_sample_after_the_cut_never_reaches_the_prompt():
    # the remifentanil Ce jumps only at t = 0 (the action)
    inp = {"ce_remi": 3.5, "remi_ce_current": 3.5, "remi_ce_trend": "rising",
           "remi_ce_trend_code": 1.0, "map_current": 80.0}
    flat = {-120: 2.0, -90: 2.0, -60: 2.0, -30: 2.0}
    strategy = _strategy()

    jump = strategy.build_messages(FakeWindow(inp, _raw({**flat, 0: 3.5})))
    other = strategy.build_messages(FakeWindow(inp, _raw({**flat, 0: 9.0})))

    assert jump == other
    text = jump[1]["content"]
    assert "effect-site 2" in text and "(stable)" in text
    assert "3.5" not in text and "rising" not in text


def test_a_drug_without_samples_before_the_cut_is_missing():
    # propofol has no samples at all: no effect-site value and no NaN in the prompt
    inp = {"ce_propofol": 1.2, "propofol_ce_current": 1.2, "map_current": 80.0}
    cut = _cut_input(FakeWindow(inp, _raw({-60: 2.0, -30: 2.0})))

    assert cut["ce_propofol"] is None and cut["propofol_ce_trend"] == "unknown"
    text = _strategy().build_messages(FakeWindow(inp, _raw({-60: 2.0})))[1]["content"]
    assert "nan" not in text.lower()


def test_only_the_effect_site_fields_change():
    window = next(iter(dataset.iter_split("test")))
    cut = _cut_input(window)

    changed = {key for key in set(cut) | set(window.input) if cut.get(key) != window.input.get(key)}
    assert changed
    assert all(control_view.is_ce_feature(key) or key.endswith("_ce_trend") for key in changed)


def test_the_prompt_is_zero_shot_s_on_the_cut_input():
    window = next(iter(dataset.iter_split("test")))
    strategy = _strategy()

    expected = prompting.build_messages_from_case(
        prompting.render_case(_cut_input(window)),
        prompting.load_template(strategy.prompt_file),
    )
    assert strategy.build_messages(window) == expected
    assert strategy.prompt_file == harness_registry.load_strategy("zero_shot").prompt_file
