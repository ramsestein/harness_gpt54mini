# -*- coding: utf-8 -*-
"""The few-shot strategy: retrieve similar train windows and show them as examples."""

from pathlib import Path

import numpy as np
import pytest

from src import config, dataset, harness_registry, llm, prompting

from harness.few_shot import retrieval

PROMPT_FILE = Path(config.HARNESS_DIR) / "few_shot" / "prompt.txt"


def _window(window_id, case_id, values, action="no_action", output=None):
    """A synthetic DecisionWindow for the retrieval tests."""
    return dataset.DecisionWindow(
        window_id=window_id,
        case_id=str(case_id),
        input=dict(values),
        output=output or {
            "result_1": {"action": action, "ratio": 0.5},
            "result_aux": {"action": "no_action", "ratio": 0.3},
            "result_real": action,
        },
        difficulty=None,
        raw_input=dict(values),
    )


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


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def test_is_feature_only_accepts_numbers():
    assert retrieval.is_feature(1)
    assert retrieval.is_feature(1.5)
    assert not retrieval.is_feature(True)      # a bool is not a measurement
    assert not retrieval.is_feature(None)
    assert not retrieval.is_feature("rising")
    assert not retrieval.is_feature({"HR": [1, 2]})
    assert not retrieval.is_feature([1, 2])


def test_feature_names_drop_identifiers_labels_and_non_numeric_fields():
    window = _window(
        "case7_w1",
        "7",
        {
            "hr_current": 80.0,
            "n_drug_points": 3,
            "pt_sex": "M",
            "monitoring_type": "ART",
            "hr_trend": "falling",
            "vitals_json": {"HR": [70, 80]},
            "pt_caseid": 7,
            "pt_subjectid": 7,
            "case_id": "7",
            "case_id_fill": "7",
            "flag": True,
            "missing": None,
        },
    )
    assert retrieval.feature_names([window]) == ["hr_current", "n_drug_points"]


def test_the_corpus_split_is_train():
    assert retrieval.SOURCE_SPLIT == "train"
    assert "few_shot" in str(retrieval.DEFAULT_INDEX_PATH)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #


def test_labels_never_influence_the_similarity():
    """Same variables, different labels: identical vectors, identical distance."""
    values = {"hr_current": 80.0, "map_current": 70.0}
    first = _window("case1_w1", "1", values, action="increase_hypnotic")
    second = _window("case2_w1", "2", values, action="no_action")

    index = retrieval.SimilarityIndex.build([first, second])

    assert np.allclose(index.vectors[0], index.vectors[1])
    neighbors = index.query(_window("q_w1", "9", values), k=2)
    assert neighbors[0].distance == pytest.approx(neighbors[1].distance)
    assert {n.action for n in neighbors} == {"increase_hypnotic", "no_action"}


def test_query_returns_the_closest_window_first():
    near = _window("case1_w1", "1", {"hr_current": 80.0, "map_current": 70.0})
    far = _window("case2_w1", "2", {"hr_current": 200.0, "map_current": 70.0})
    index = retrieval.SimilarityIndex.build([far, near])

    neighbors = index.query(_window("q_w1", "9", {"hr_current": 81.0, "map_current": 70.0}), k=2)

    assert [n.window_id for n in neighbors] == ["case1_w1", "case2_w1"]
    assert neighbors[0].distance < neighbors[1].distance


def test_missing_values_are_imputed_with_the_corpus_mean():
    """A null contributes nothing; it does not push the window away."""
    window = _window("case1_w1", "1", {"hr_current": 80.0, "lab_lac": None})
    index = retrieval.SimilarityIndex.build([window])

    assert np.allclose(index.vector_for({"hr_current": 80.0, "lab_lac": None}), 0.0)
    assert np.allclose(index.vector_for({"hr_current": 80.0, "lab_lac": 80.0}), 0.0)


def test_query_respects_k():
    corpus = [
        _window(f"case{i}_w1", str(i), {"hr_current": 60.0 + i}) for i in range(1, 6)
    ]
    index = retrieval.SimilarityIndex.build(corpus)
    query = _window("q_w1", "9", {"hr_current": 61.0})

    assert len(index.query(query, k=3)) == 3
    assert len(index.query(query, k=99)) == 5


def test_query_can_exclude_the_same_case():
    corpus = [
        _window("case1_w1", "1", {"hr_current": 80.0}),
        _window("case1_w2", "1", {"hr_current": 81.0}),
        _window("case2_w1", "2", {"hr_current": 82.0}),
    ]
    index = retrieval.SimilarityIndex.build(corpus)

    neighbors = index.query(_window("q_w1", "1", {"hr_current": 80.0}), k=3, exclude_case="1")

    assert [n.case_id for n in neighbors] == ["2"]


def test_save_and_load_round_trip(tmp_path):
    corpus = [
        _window("case1_w1", "1", {"hr_current": 80.0, "map_current": 70.0}),
        _window("case2_w1", "2", {"hr_current": 200.0, "map_current": 70.0}),
    ]
    index = retrieval.SimilarityIndex.build(corpus)
    path = tmp_path / "index.npz"

    index.save(path)
    loaded = retrieval.SimilarityIndex.load(path)

    assert loaded.names == index.names
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.prompts == index.prompts
    assert loaded.window_ids == index.window_ids
    assert loaded.actions == index.actions

    query = _window("q_w1", "9", {"hr_current": 81.0, "map_current": 70.0})
    assert [(n.window_id, n.action) for n in loaded.query(query, k=2)] == [
        (n.window_id, n.action) for n in index.query(query, k=2)
    ]


def test_a_partial_corpus_is_never_cached(tmp_path):
    """A debug-sized index must not be mistaken for the full one later."""
    path = tmp_path / "index.npz"

    index = retrieval.load_or_build(path=path, cases=1)

    assert len(index) > 0
    assert not path.exists()


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


def test_strategy_metadata():
    strategy = harness_registry.load_strategy("few_shot", index=retrieval.SimilarityIndex.build(
        [_window("case1_w1", "1", {"hr_current": 80.0})]
    ))
    assert strategy.name == "few_shot"
    assert strategy.description.strip()


def test_makes_a_single_call_per_window(monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))
    strategy = harness_registry.load_strategy("few_shot", index=index, k=2)

    strategy.predict(dataset.load_split("test", limit=1)[0])

    assert len(calls["messages"]) == 1


def test_the_prompt_shows_the_examples_with_their_real_actions(monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, ["b"])
    corpus = dataset.load_split("dev", limit=5)
    index = retrieval.SimilarityIndex.build(corpus)
    strategy = harness_registry.load_strategy("few_shot", index=index, k=2)
    window = dataset.load_split("test", limit=1)[0]

    prediction = strategy.predict(window)

    prompt = calls["messages"][0][-1]["content"]
    assert "Below are 2 similar past decision windows" in prompt
    assert "=== SIMILAR PAST WINDOWS ===" in prompt
    assert "=== CURRENT CASE ===" in prompt
    assert prompt.index("=== SIMILAR PAST WINDOWS ===") < prompt.index("=== CURRENT CASE ===")

    for neighbor in prediction.meta["neighbors"]:
        assert f"Adjustment the clinician made: {neighbor['action']}" in prompt
        source = next(w for w in corpus if w.window_id == neighbor["window_id"])
        assert prompting.render_case(source.input) in prompt

    assert prompting.render_case(window.input) in prompt
    assert "{examples}" not in prompt
    assert "{case}" not in prompt
    assert "{n_examples}" not in prompt


def test_the_prompt_offers_exactly_five_options():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    assert "{examples}" in text and "{case}" in text
    for letter, action in prompting.LETTER_TO_ACTION.items():
        assert f"{letter})" in text
        assert action in text
    assert "f)" not in text
    assert "vasopressor" not in text


def test_the_prompt_does_not_leak_identifiers(monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))
    strategy = harness_registry.load_strategy("few_shot", index=index, k=2)

    strategy.predict(dataset.load_split("test", limit=1)[0])

    prompt = calls["messages"][0][-1]["content"]
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in prompt


def test_never_retrieves_the_case_being_predicted(monkeypatch):
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=20))
    strategy = harness_registry.load_strategy("few_shot", index=index, k=3)
    window = dataset.load_split("dev", limit=1)[0]

    neighbors = strategy.retrieve(window)

    assert neighbors
    assert all(n.case_id != window.case_id for n in neighbors)


def test_the_neighbours_are_recorded_in_the_metadata(monkeypatch):
    _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))
    strategy = harness_registry.load_strategy("few_shot", index=index, k=2)

    prediction = strategy.predict(dataset.load_split("test", limit=1)[0])

    neighbors = prediction.meta["neighbors"]
    assert len(neighbors) == 2
    assert prediction.meta["k"] == 2
    for neighbor in neighbors:
        assert set(neighbor) == {"window_id", "case_id", "action", "distance"}
        assert neighbor["action"] in dataset.ACTIONS


def test_k_can_be_changed(monkeypatch):
    _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))

    strategy = harness_registry.load_strategy("few_shot", index=index, k=1)

    assert strategy.k == 1
    assert len(strategy.retrieve(dataset.load_split("test", limit=1)[0])) == 1


def test_n_samples_is_accepted_as_an_alias_for_k(monkeypatch):
    """So the launcher's --samples flag works without touching shared code."""
    _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))

    strategy = harness_registry.load_strategy("few_shot", index=index, n_samples=4)

    assert strategy.k == 4


def test_asks_for_the_same_generation_settings_as_zero_shot(monkeypatch):
    calls = _patch_llm_sequence(monkeypatch, ["b"])
    index = retrieval.SimilarityIndex.build(dataset.load_split("dev", limit=5))
    strategy = harness_registry.load_strategy("few_shot", index=index, k=1)

    strategy.predict(dataset.load_split("test", limit=1)[0])

    assert calls["kwargs"][0]["temperature"] == 0.0
    assert calls["kwargs"][0]["reasoning_effort"] == "high"
    assert "k" not in calls["kwargs"][0]
    assert "index" not in calls["kwargs"][0]


def test_format_examples_numbers_the_examples():
    index = retrieval.SimilarityIndex.build(
        [_window("case1_w1", "1", {"hr_current": 80.0}, action="reduce_opioid")]
    )
    neighbors = index.query(_window("q_w1", "9", {"hr_current": 80.0}), k=1)

    text = harness_registry.load_strategy("few_shot", index=index).format_examples(neighbors)

    assert text.startswith("--- example 1 ---")
    assert "Adjustment the clinician made: reduce_opioid" in text
