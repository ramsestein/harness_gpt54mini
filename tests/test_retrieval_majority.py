# -*- coding: utf-8 -*-
"""The pure retrieval control (no model, no consensus labels).

Per window: retrieve the ``k`` most similar training windows with the few-shot
similarity tool -- the case window goes in raw -- and answer with the action the
clinicians most often took in that neighbourhood. Only ``result_real`` takes part:
that is the whole point of this control.
"""

import ast
import json
from pathlib import Path

import pytest

from harness import run as harness_run
from harness.few_shot import retrieval
from src import dataset, harness_registry
from src.control import baselines, retrieval_majority, run_controls
from src.harness_base import Strategy

NAME = "retrieval_majority"
ACTIONS = list(dataset.ACTIONS)


class FakeIndex:
    """Returns a prescribed neighbourhood, and records how it was queried."""

    def __init__(self, actions):
        self.neighbours = [
            retrieval.Neighbor(
                window_id=f"train{index}",
                case_id=f"case{index}",
                action=action,
                prompt="",
                distance=float(index),
            )
            for index, action in enumerate(actions)
        ]
        self.calls = []

    def query(self, window, k=10, exclude_case=None):
        self.calls.append({"window": window, "k": k, "exclude_case": exclude_case})
        return list(self.neighbours)

    def __len__(self):
        return len(self.neighbours)


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("test", limit=1)[0]


def _strategy(actions, **options):
    return retrieval_majority.RetrievalMajorityStrategy(index=FakeIndex(actions), **options)


def _ten(**counts):
    """Ten neighbours' real actions: ``_ten(no_action=4, increase_opioid=3, ...)``."""
    actions = []
    for action, count in counts.items():
        actions += [action] * count
    actions += ["filler"] * (10 - len(actions))
    return actions


# --------------------------------------------------------------------------- #
# Shape of the control
# --------------------------------------------------------------------------- #


def test_it_is_a_control_without_a_model():
    strategy = _strategy([])

    assert isinstance(strategy, Strategy)
    assert strategy.name == NAME
    assert strategy.description.strip()
    assert strategy.uses_model is False


def test_the_defaults_are_the_documented_rule():
    strategy = _strategy([])

    assert strategy.k == 10
    assert strategy.exclude_same_case is True


def test_it_is_not_a_harness_strategy():
    assert NAME not in harness_registry.EXPECTED_STRATEGIES
    assert Path(retrieval_majority.__file__).parent.name == "control"
    assert NAME not in run_controls.default_names()  # it needs the few-shot index
    assert NAME in run_controls.OPT_IN_NAMES


def _code_tokens(module_path):
    """Every identifier and every string literal that is not a docstring.

    The prose is deliberately excluded: the module documents the circularity it avoids,
    so its docstrings do mention the consensus labels. What must never mention them is
    the code.
    """
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    tokens = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            tokens.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                tokens.add(node.value)
    return tokens


def test_the_rule_never_looks_at_the_consensus_labels():
    """The whole point: only ``result_real`` may decide, never the consensus labels."""
    code = _code_tokens(retrieval_majority.__file__)

    for forbidden in ("result_1", "result_aux", "plausible"):
        assert not any(forbidden in token for token in code), forbidden


def test_the_corpus_carries_the_real_action_of_each_window():
    index_corpus = list(retrieval.train_windows(cases=1))

    index = retrieval.SimilarityIndex.build(index_corpus)

    assert index.actions == [window.real_action for window in index_corpus]


# --------------------------------------------------------------------------- #
# The vote
# --------------------------------------------------------------------------- #


def test_it_answers_with_the_most_frequent_real_action(window):
    actions = _ten(no_action=4, increase_opioid=3, reduce_opioid=3)
    strategy = _strategy(actions)

    prediction = strategy.predict(window)

    assert prediction.action == "no_action"
    assert prediction.meta["chosen"] == "no_action"
    assert prediction.meta["rate"] == pytest.approx(0.4)


def test_the_rate_is_the_share_of_the_neighbourhood(window):
    actions = _ten(no_action=6, reduce_hypnotic=2, increase_hypnotic=2)
    strategy = _strategy(actions)

    scores = {score["action"]: score for score in strategy.score_neighbours(strategy.index.neighbours)}

    assert scores["no_action"]["count"] == 6
    assert scores["no_action"]["rate"] == pytest.approx(0.6)
    assert scores["reduce_hypnotic"]["count"] == 2
    assert scores["increase_opioid"]["count"] == 0


def test_every_action_gets_a_score(window):
    actions = _ten(no_action=10)
    strategy = _strategy(actions)

    scores = strategy.score_neighbours(strategy.index.neighbours)

    assert [score["action"] for score in scores] == ACTIONS
    assert sum(score["count"] for score in scores) == 10


def test_ties_go_to_the_first_action_of_the_dataset_order(window):
    actions = _ten(increase_hypnotic=3, no_action=3, reduce_opioid=2, filler=2)
    strategy = _strategy(actions)

    prediction = strategy.predict(window)

    assert prediction.action == dataset.ACTIONS[0]
    assert set(prediction.meta["tied"]) == {"increase_hypnotic", "no_action"}


def test_the_rate_is_relative_to_the_neighbours_found(window):
    """A short neighbourhood must not be penalised: the rate is a mean, not a count."""
    strategy = _strategy(["reduce_opioid"] * 4)

    prediction = strategy.predict(window)

    assert prediction.action == "reduce_opioid"
    assert prediction.meta["rate"] == 1.0
    assert prediction.meta["n_neighbours"] == 4


def test_an_empty_neighbourhood_still_answers(window):
    strategy = _strategy([])

    prediction = strategy.predict(window)

    assert prediction.action in ACTIONS
    assert prediction.meta["rate"] == 0.0
    assert prediction.meta["n_neighbours"] == 0


# --------------------------------------------------------------------------- #
# The retrieval call
# --------------------------------------------------------------------------- #


def test_it_queries_once_with_the_documented_k(window):
    strategy = _strategy([])

    strategy.predict(window)

    assert len(strategy.index.calls) == 1
    assert strategy.index.calls[0]["k"] == 10
    assert strategy.index.calls[0]["exclude_case"] == window.case_id
    assert strategy.index.calls[0]["window"] is window  # the raw window


def test_the_case_exclusion_can_be_turned_off(window):
    strategy = _strategy([], exclude_same_case=False)

    strategy.predict(window)

    assert strategy.index.calls[0]["exclude_case"] is None


def test_the_k_is_configurable(window):
    strategy = _strategy(["no_action", "no_action", "reduce_opioid"], k=3)

    meta = strategy.predict(window).meta

    assert meta["k"] == 3
    assert meta["n_neighbours"] == 3
    assert strategy.index.calls[0]["k"] == 3


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_the_metadata_lists_the_neighbours_once_with_their_real_action(window):
    actions = _ten(no_action=4)
    strategy = _strategy(actions)

    meta = strategy.predict(window).meta

    assert meta["baseline"] == NAME
    assert len(meta["neighbours"]) == 10
    first = meta["neighbours"][0]
    assert first["window_id"] == "train0"
    assert first["action"] == "no_action"
    assert "distance" in first
    assert [score["action"] for score in meta["scores"]] == ACTIONS


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_index():
    return retrieval.SimilarityIndex.build(list(retrieval.train_windows(cases=4)))


def test_it_predicts_the_majority_of_the_real_neighbourhood(window, tiny_index):
    strategy = retrieval_majority.RetrievalMajorityStrategy(index=tiny_index)

    prediction = strategy.predict(window)

    expected = [neighbour.action for neighbour in tiny_index.query(window, k=10, exclude_case=window.case_id)]
    counts = {action: expected.count(action) for action in set(expected)}
    best = max(counts.values())
    assert counts[prediction.action] == best
    assert prediction.window_id == window.window_id


def test_the_launcher_runs_it(tmp_path, tiny_index):
    strategy = retrieval_majority.RetrievalMajorityStrategy(index=tiny_index)

    outcome = harness_run.run(strategy, split="test", limit=2, results_root=tmp_path, quiet=True)

    run_dir = Path(outcome["out_dir"])
    assert run_dir.parent.name == NAME
    assert outcome["summary"]["n_windows"] == 2
    assert outcome["summary"]["errors"] == 0
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["model"] is None


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


def test_the_cli_can_select_it():
    args = run_controls.build_parser().parse_args(["--only", NAME])
    assert run_controls.selected_names(args.only) == [NAME]


def test_it_is_not_in_the_cheap_sweep():
    assert NAME not in run_controls.default_names()
    assert set(run_controls.selected_names("all")) >= {NAME}
