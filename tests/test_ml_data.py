# -*- coding: utf-8 -*-
"""The feature matrix and labels shared by the machine-learning controls."""

import numpy as np
import pytest

from harness.few_shot import retrieval
from src import dataset
from src.control import ml_data


@pytest.fixture(scope="module")
def tiny_index(tmp_path_factory):
    """A real index cache, but over a handful of training windows."""
    path = tmp_path_factory.mktemp("index") / "train_index.npz"
    index = retrieval.SimilarityIndex.build(list(retrieval.train_windows(cases=2)))
    index.save(path)
    return path


def test_the_feature_names_come_from_the_index_cache(tiny_index):
    names = ml_data.feature_names(tiny_index)

    assert names == retrieval.SimilarityIndex.load(tiny_index).names
    assert len(names) > 10


def test_the_feature_names_fall_back_to_a_corpus_sample(tmp_path):
    names = ml_data.feature_names(tmp_path / "no-index-here.npz")

    assert names
    assert names == sorted(names)


def test_the_matrix_is_nan_where_the_window_has_nothing():
    class FakeWindow:
        def __init__(self, inp):
            self.input = inp

    windows = [
        FakeWindow({"a": 1.0, "b": None, "c": 3.0}),
        FakeWindow({"a": 2.0, "b": 5.0}),
    ]

    matrix = ml_data.matrix(windows, ["a", "b", "c"])

    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    assert np.isnan(matrix[0, 1])  # b is null
    assert np.isnan(matrix[1, 2])  # c is absent
    assert matrix[0, 0] == 1.0 and matrix[1, 1] == 5.0


def test_a_non_numeric_field_is_not_a_feature():
    class FakeWindow:
        input = {"a": 1.0, "trend": "rising", "flag": True}

    matrix = ml_data.matrix([FakeWindow()], ["a", "trend", "flag"])

    assert matrix[0, 0] == 1.0
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[0, 2])


def test_an_empty_iterable_gives_an_empty_matrix():
    assert ml_data.matrix([], ["a", "b"]).shape == (0, 2)


def test_the_labels_follow_the_dataset_order():
    windows = dataset.load_split("dev", limit=5)

    labels = ml_data.labels(windows)

    assert list(labels) == [ml_data.ACTION_INDEX[window.real_action] for window in windows]
    assert set(ml_data.ACTION_LABEL.values()) == set(dataset.ACTIONS)


def test_load_split_returns_aligned_rows(tiny_index):
    X, y, ids = ml_data.load_split("dev", names=ml_data.feature_names(tiny_index), cases=1, limit=4)

    assert X.shape == (4, len(ml_data.feature_names(tiny_index)))
    assert len(y) == len(ids) == 4
    assert ids == [window.window_id for window in dataset.load_split("dev", limit=4)]


def test_load_split_produces_the_same_matrix_as_the_helper(tiny_index):
    names = ml_data.feature_names(tiny_index)
    windows = dataset.load_split("dev", limit=3)

    X, _, _ = ml_data.load_split("dev", names=names, cases=1, limit=3)

    assert np.allclose(X, ml_data.matrix(windows, names), equal_nan=True)


def test_the_case_ids_come_out_of_the_window_ids():
    assert list(ml_data.case_ids(["case12_w3", "case12_w4", "case99_w1"])) == [
        "case12",
        "case12",
        "case99",
    ]
