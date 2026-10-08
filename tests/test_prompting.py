# -*- coding: utf-8 -*-
"""Building prompts from a dataset window."""

import json

from src import dataset, prompting


def test_letter_to_action_mapping():
    assert prompting.LETTER_TO_ACTION == {
        "a": "increase_hypnotic",
        "b": "reduce_hypnotic",
        "c": "increase_opioid",
        "d": "reduce_opioid",
        "e": "no_action",
    }
    for letter, action in prompting.LETTER_TO_ACTION.items():
        assert prompting.ACTION_TO_LETTER[action] == letter


def test_build_prompt_substitutes_the_template():
    w = dataset.load_split("test", limit=1)[0]
    prompt = prompting.build_prompt(w)
    assert isinstance(prompt, str) and prompt.strip()
    assert "{case}" not in prompt
    for action in dataset.ACTIONS:
        assert action in prompt
    assert "a)" in prompt and "e)" in prompt
    assert w.window_id not in prompt


def test_build_prompt_does_not_leak_identifiers():
    w = dataset.load_split("test", limit=1)[0]
    prompt = prompting.build_prompt(w)
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in prompt


def test_render_case_produces_clinical_prose():
    w = dataset.load_split("test", limit=1)[0]
    text = prompting.render_case(w.input)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "patient" in text.lower()


def test_build_messages_has_system_and_user():
    w = dataset.load_split("test", limit=1)[0]
    messages = prompting.build_messages(w)
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert prompting.build_prompt(w) in messages[-1]["content"]


def test_render_json_round_trips_the_window():
    w = dataset.load_split("test", limit=1)[0]
    parsed = json.loads(prompting.render_json(w.input))
    assert set(parsed) == set(w.input)
    assert prompting.render_json(parsed) == prompting.render_json(w.input)


def test_render_json_is_deterministic():
    w = dataset.load_split("test", limit=1)[0]
    shuffled = dict(reversed(list(w.input.items())))
    assert prompting.render_json(shuffled) == prompting.render_json(w.input)


def test_render_json_has_no_identifiers():
    w = dataset.load_split("test", limit=1)[0]
    text = prompting.render_json(w.input)
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in text


def test_render_json_keeps_the_nested_time_series():
    w = dataset.load_split("test", limit=1)[0]
    parsed = json.loads(prompting.render_json(w.input))
    assert isinstance(parsed["vitals_json"], dict)
