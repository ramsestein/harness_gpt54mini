# -*- coding: utf-8 -*-
"""Loading decision windows from ``dataset/data/``.

Specification of ``src.dataset``.
"""

import re

from src import dataset

WINDOW_ID_RE = re.compile(r"^case\d+_w\d+$")
RAW_INPUT_FIELDS = 280


def test_splits_and_action_vocabulary():
    assert dataset.SPLITS == ["dev", "train", "test"]
    assert dataset.ACTIONS == [
        "increase_hypnotic",
        "reduce_hypnotic",
        "increase_opioid",
        "reduce_opioid",
        "no_action",
    ]
    assert dataset.OPTION_ACTIONS == ["vasopressor"]
    assert dataset.ALL_ACTIONS == dataset.ACTIONS + ["vasopressor"]
    assert "pt_caseid" in dataset.IDENTIFIER_FIELDS
    assert "pt_subjectid" in dataset.IDENTIFIER_FIELDS


def test_all_three_splits_have_data(data_dir):
    for split in dataset.SPLITS:
        split_dir = data_dir / split
        assert split_dir.is_dir(), f"missing {split_dir}"
        assert list(split_dir.glob("case*.jsonl")), f"{split} has no case*.jsonl files"


def test_case_files_are_sorted_and_well_named():
    files = dataset.case_files("dev")
    assert files, "dev must contain case*.jsonl files"
    names = [f.name for f in files]
    assert names == sorted(names)
    assert all(n.startswith("case") and n.endswith(".jsonl") for n in names)


def test_window_contract():
    path = dataset.case_files("test")[0]
    windows = dataset.load_case(path)
    n_lines = len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
    assert len(windows) == n_lines

    w = windows[0]
    assert WINDOW_ID_RE.match(w.window_id), w.window_id
    assert w.case_id == path.stem.replace("case", "")
    assert isinstance(w.input, dict)
    assert set(w.output) >= {"result_1", "result_aux", "result_real"}

    assert w.real_action == w.output["result_real"]
    assert w.real_action in dataset.ACTIONS
    assert set(w.result_1) == {"action", "ratio"}
    assert set(w.result_aux) == {"action", "ratio"}


def test_input_is_sanitised_but_raw_is_preserved():
    w = dataset.load_split("test", limit=1)[0]
    assert len(w.raw_input) == RAW_INPUT_FIELDS
    for field in dataset.IDENTIFIER_FIELDS:
        assert field in w.raw_input
        assert field not in w.input
    assert len(w.input) == RAW_INPUT_FIELDS - len(dataset.IDENTIFIER_FIELDS)


def test_load_split_respects_limit_and_order():
    full = dataset.load_split("dev")
    limited = dataset.load_split("dev", limit=3)
    assert len(limited) == 3
    assert [w.window_id for w in limited] == [w.window_id for w in full[:3]]


def test_iter_split_is_lazy_and_matches_the_manifest():
    manifest = dataset.load_manifest()
    cases = [f.stem.replace("case", "") for f in dataset.case_files("dev")]
    missing = [c for c in cases if c not in manifest]
    assert not missing, f"cases missing from the manifest: {missing}"

    expected = sum(manifest[c]["n_windows"] for c in cases)
    assert sum(1 for _ in dataset.iter_split("dev")) == expected


def test_difficulty_of():
    assert dataset.difficulty_of("1001") == "hard"
    assert dataset.difficulty_of("1000") == "easy"
    assert dataset.difficulty_of("99999999") is None


def test_split_summary():
    summary = dataset.split_summary("dev")
    assert summary["n_cases"] == len(dataset.case_files("dev"))
    assert summary["n_windows"] > 0


def test_to_record_roundtrip():
    w = dataset.load_split("test", limit=1)[0]
    rec = w.to_record()
    assert set(rec) == {"window_id", "case_id", "input", "output"}
    assert rec["window_id"] == w.window_id
    for field in dataset.IDENTIFIER_FIELDS:
        assert field not in rec["input"]
