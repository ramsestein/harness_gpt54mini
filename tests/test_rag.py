# -*- coding: utf-8 -*-
"""The ``rag`` strategy: zero_shot plus the curated clinical context in the prompt."""

from pathlib import Path

import pytest

from src import dataset, prompting
from src.harness_base import Strategy

from harness import run as harness_run
from harness.rag import strategy as rag

#: A distinctive sentence of the curated context, to prove the whole file travels.
MARKER = "The target range of BIS values during general anaesthesia is 40 to 60"


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("dev", limit=1)[0]


@pytest.fixture
def strategy():
    return rag.RagStrategy()


# --------------------------------------------------------------------------- #
# It is a strategy
# --------------------------------------------------------------------------- #


def test_it_is_a_strategy(strategy):
    assert isinstance(strategy, Strategy)
    assert strategy.name == rag.RagStrategy.name == "rag"
    assert strategy.description.strip()
    assert strategy.uses_model is True


def test_it_constructs_without_touching_the_disk(tmp_path):
    """The structure suite loads every strategy with no arguments."""
    strategy = rag.RagStrategy(context_file=tmp_path / "nowhere.md")

    assert strategy.prompt_file.is_file()  # zero_shot's template, shared on purpose


def test_the_context_file_is_the_one_in_the_folder():
    assert rag.DEFAULT_CONTEXT_FILE.parent.name == "rag"
    assert rag.DEFAULT_CONTEXT_FILE.name == rag.DEFAULT_CONTEXT_FILE.name


def test_the_prompt_is_the_zero_shot_one():
    """Replicating zero_shot means sharing its wording, not copying it."""
    assert rag.DEFAULT_PROMPT_FILE == Path(rag.HERE).parent / "zero_shot" / "prompt.txt"


# --------------------------------------------------------------------------- #
# The reference material
# --------------------------------------------------------------------------- #


def test_the_context_is_read_from_the_file(strategy):
    context = strategy.context()

    assert MARKER in context
    assert len(context) > 10000  # the curated excerpts, not a stub


def test_the_context_is_read_once(strategy):
    """One read per process: the windows run in a thread pool."""
    assert strategy.context() is strategy.context()


def test_a_missing_context_file_says_which(tmp_path):
    strategy = rag.RagStrategy(context_file=tmp_path / "nope.md")

    with pytest.raises(FileNotFoundError, match="no context file"):
        strategy.context()


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def test_the_whole_context_travels_in_the_prompt(strategy, window):
    prompt = strategy.build_prompt(window)

    assert MARKER in prompt
    # the last block of the file too, so nothing is truncated
    assert "remifentanil blood concentrations fall as rapidly" in prompt
    assert strategy.context().strip() in prompt


def test_the_prompt_keeps_the_zero_shot_wording(strategy, window):
    prompt = strategy.build_prompt(window)

    for text in ("expert anesthesiologist", "Choose exactly one", "e) no_action",
                 "Output nothing else"):
        assert text in prompt
    assert "{case}" not in prompt


def test_the_patient_state_is_there_too(strategy, window):
    prompt = strategy.build_prompt(window)

    assert prompting.render_case(window.input) in prompt


def test_the_reference_is_announced_and_placed_after_the_state(strategy, window):
    prompt = strategy.build_prompt(window)
    state = prompting.render_case(window.input)

    assert rag.REFERENCE_HEADER in prompt
    # the state first, then the reference, then the options
    assert prompt.index(state[:40]) < prompt.index(rag.REFERENCE_HEADER)
    assert prompt.index(rag.REFERENCE_HEADER) < prompt.index("Choose exactly one")


def test_the_prompt_is_longer_than_zero_shots(strategy, window):
    assert len(strategy.build_prompt(window)) > len(prompting.build_prompt(window))


def test_the_messages_are_a_system_and_a_user_turn(strategy, window):
    messages = strategy.build_messages(window)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == prompting.SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# A run
# --------------------------------------------------------------------------- #


def test_the_launcher_records_the_reference(strategy, window, tmp_path, monkeypatch):
    monkeypatch.setattr(
        strategy,
        "ask",
        lambda messages: type("R", (), {"text": "e", "model": "fake", "latency_s": 0.0,
                                       "dropped_params": []})(),
    )

    outcome = harness_run.run(strategy, split="dev", limit=2, results_root=tmp_path, quiet=True)

    assert outcome["summary"]["errors"] == 0
    meta = outcome["predictions"][0]["meta"]
    assert meta["context_chars"] == len(strategy.context())
    assert meta["prompt_chars"] > meta["context_chars"]
    assert Path(meta["context_file"]).name == "context.md"


@pytest.mark.live
def test_it_answers_like_any_other_strategy(strategy, window):
    prediction = strategy.predict(window)

    assert prediction.action in dataset.ACTIONS
