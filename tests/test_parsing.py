# -*- coding: utf-8 -*-
"""Parsing the model answer into an action from the vocabulary."""

import pytest

from src import parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a", "increase_hypnotic"),
        ("B", "reduce_hypnotic"),
        ("c).", "increase_opioid"),
        ("Answer: d", "reduce_opioid"),
        ("e", "no_action"),
        ("  b  ", "reduce_hypnotic"),
        ("reduce_opioid", "reduce_opioid"),
        ("increase_hypnotic", "increase_hypnotic"),
        ("no_action", "no_action"),
        ("b) reduce_hypnotic", "reduce_hypnotic"),
    ],
)
def test_parse_action(text, expected):
    assert parsing.parse_action(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "banana", "42", "x)", None])
def test_parse_action_returns_none_on_garbage(text):
    assert parsing.parse_action(text) is None


def test_normalize():
    assert parsing.normalize("  Answer: B. ") == "answer: b."
    assert parsing.normalize(None) == ""


def test_a_labelled_answer_wins_over_actions_named_in_the_reasoning():
    """Chain-of-thought names several actions before committing to one letter."""
    text = (
        "1. MAP is falling, so consider increase_opioid.\n"
        "2. BIS is high, so reduce_hypnotic is not needed.\n"
        "3. The opioid looks low.\n"
        "Answer: a"
    )
    assert parsing.parse_action(text) == "increase_hypnotic"


def test_a_labelled_answer_can_be_a_name_too():
    text = "The MAP is low, so increase_opioid is tempting.\nAnswer: reduce_hypnotic"
    assert parsing.parse_action(text) == "reduce_hypnotic"


def test_a_labelled_letter_on_its_own_line_still_works():
    assert parsing.parse_action("Reasoning here.\n\nAnswer: (c)") == "increase_opioid"


def test_the_last_label_wins_over_options_compared_in_the_reasoning():
    """A ReAct trace mentions options on the way; the closing line is the answer."""
    text = (
        "Thought: option c (increase_opioid) is tempting, but I need the BIS.\n"
        "Action: LOOKUP bis_current\n"
        "Observation: BIS is 74.9 and falling.\n"
        "Thought: I now have enough information to decide.\n"
        "Answer: a"
    )
    assert parsing.parse_action(text) == "increase_hypnotic"


def test_the_closing_answer_wins_even_when_an_option_is_named_after_it():
    text = "Answer: b\n(option a was rejected because the BIS is low)"
    assert parsing.parse_action(text) == "reduce_hypnotic"
