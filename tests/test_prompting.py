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


def _adjustment(offset, drug, old, new):
    return {
        "t_offset_sec": offset,
        "action_type": "tci_change",
        "drug": drug,
        "old_target": old,
        "new_target": new,
        "delta": new - old,
    }


def test_the_infusions_are_rendered_with_their_target_and_direction():
    """The five options are infusion-rate changes: the rate state must be visible."""
    inp = {
        "ce_propofol": 3.5,
        "ct_propofol": 9.9,  # the pump target at t = 0 is outside the shared view
        "propofol_ce_trend": "stable",
        "ce_remi": 4.99,
        "remi_ce_trend": "falling",
        "actions_history_json": [_adjustment(-600, "propofol", 3.0, 3.5),
                                 _adjustment(-390, "remifentanil", 11.76, 5.78)],
    }

    text = prompting.render_case(inp)

    # targets come from the past adjustments of each drug, never from ct_*
    assert "Infusions: propofol target 3.5, effect-site 3.5 (stable)" in text
    assert "9.9" not in text
    assert "remifentanil target 5.78, effect-site 4.99 (falling)" in text
    assert "Most recent infusion adjustment: remifentanil target 11.76 -> 5.78 (6.5 min ago)" in text


def test_the_last_adjustment_is_the_most_recent_one():
    inp = {
        "actions_history_json": [
            _adjustment(-510, "remifentanil", 0.0, 0.37),
            _adjustment(-120, "propofol", 3.0, 4.5),
        ]
    }

    text = prompting.render_case(inp)

    assert "Most recent infusion adjustment: propofol target 3 -> 4.5 (2 min ago)" in text
    # the older remifentanil adjustment only sets that drug's current target in the prose;
    # the adjustment itself is listed in the full record
    prose, _, record = text.partition("Full monitoring record")
    assert "remifentanil target 0.37" in prose
    assert "0 -> 0.37" not in prose
    assert "new_target 0.37" in record


def test_the_time_out_of_band_is_rendered_only_when_it_happened():
    text = prompting.render_case({"map_below_65_sec": 28, "bis_below_40_sec": 0, "map_below_55_sec": 10})

    assert "Time recently outside the target bands:" in text
    assert "MAP below 65 mmHg for 28 s" in text
    assert "MAP below 55 mmHg for 10 s" in text
    assert "BIS below 40" not in text
    assert "bis_below_40_sec 0" in text  # a zero is still part of the view: in the record


def test_the_trend_keeps_its_steepness():
    text = prompting.render_case({"bis_trend": "rising", "bis_slope_short": 3.82, "map_trend": "falling"})

    assert "BIS rising (3.82)" in text
    assert "MAP falling" in text


def test_the_induction_time_is_rendered_in_minutes():
    assert "Time since induction: 53.5 min" in prompting.render_case(
        {"time_since_induction_sec": 3213}
    )


def test_a_window_without_the_new_fields_adds_no_line():
    text = prompting.render_case({"pt_age": 70})

    assert "Infusions:" not in text
    assert "Most recent infusion adjustment:" not in text
    assert "Time recently outside the target bands:" not in text
    assert "Time since induction:" not in text


def test_the_render_is_deterministic_with_a_history():
    inp = {
        "ce_propofol": 3.5,
        "actions_history_json": [
            _adjustment(-510, "remifentanil", 0.0, 0.37),
            _adjustment(-390, "remifentanil", 11.76, 5.78),
        ],
    }

    assert prompting.render_case(inp) == prompting.render_case(inp)


def test_a_malformed_history_does_not_break_the_render():
    for value in (None, "not json", {}, 42, [{"drug": "propofol"}]):
        text = prompting.render_case({"ce_propofol": 3.5, "actions_history_json": value})
        assert isinstance(text, str)


def test_the_real_window_gets_the_new_blocks():
    w = dataset.load_split("test", limit=1)[0]

    text = prompting.render_case(w.input)

    assert "Infusions:" in text
    assert "Time since induction:" in text


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
