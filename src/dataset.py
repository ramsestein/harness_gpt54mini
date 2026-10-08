# -*- coding: utf-8 -*-
"""Loading decision windows from ``dataset/data/{dev,train,test}/``.

STATE: implemented. The contract is defined in
``tests/test_dataset_loading.py``.

Contract
-----------------
Constants
    ``SPLITS``             -> ["dev", "train", "test"]
    ``ACTIONS``            -> the 5 ground-truth classes (``result_real``)
    ``OPTION_ACTIONS``     -> ["vasopressor"] (valid option, never ground truth)
    ``ALL_ACTIONS``        -> ACTIONS + OPTION_ACTIONS
    ``IDENTIFIER_FIELDS``  -> ("pt_caseid", "pt_subjectid")

``DecisionWindow`` (frozen dataclass)
    ``window_id``, ``case_id``, ``input`` (**sanitised** dict), ``output``,
    ``difficulty`` (str | None), ``raw_input`` (original dict, 280 fields)
    Properties: ``real_action``, ``result_1``, ``result_aux``
    Method: ``to_record()`` -> ``{"window_id", "case_id", "input", "output"}``

Functions
    ``sanitize_input(inp)``            -> copy without ``IDENTIFIER_FIELDS``
    ``load_jsonl(path)``               -> list[dict]
    ``case_files(split)``              -> sorted list[Path]
    ``iter_split(split)``              -> Iterator[DecisionWindow] (lazy)
    ``load_split(split, limit=None)``  -> list[DecisionWindow]
    ``load_case(path)``                -> list[DecisionWindow]
    ``load_manifest()``                -> {case_id: {difficulty, split, n_windows, null_score}}
    ``difficulty_of(case_id)``         -> str | None
    ``split_summary(split)``           -> {"n_cases": int, "n_windows": int}
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from . import config

__all__ = [
    "SPLITS",
    "ACTIONS",
    "OPTION_ACTIONS",
    "ALL_ACTIONS",
    "IDENTIFIER_FIELDS",
    "DecisionWindow",
    "get_action",
    "get_ratio",
    "sanitize_input",
    "load_jsonl",
    "case_files",
    "iter_split",
    "load_split",
    "load_case",
    "load_manifest",
    "difficulty_of",
    "split_summary",
]


SPLITS = ["dev", "train", "test"]
ACTIONS = [
    "increase_hypnotic",
    "reduce_hypnotic",
    "increase_opioid",
    "reduce_opioid",
    "no_action",
]
OPTION_ACTIONS = ["vasopressor"]
ALL_ACTIONS = ACTIONS + OPTION_ACTIONS
IDENTIFIER_FIELDS = ("pt_caseid", "pt_subjectid")

_MANIFEST_CACHE = None


def get_action(value):
    """``result_real`` is a string; ``result_1`` / ``result_aux`` are ``{action, ratio}``."""
    if isinstance(value, dict):
        return value.get("action")
    return value


def get_ratio(value):
    """Model probability of a consensus action, or ``None``."""
    if isinstance(value, dict):
        ratio = value.get("ratio")
        if isinstance(ratio, (int, float)):
            return float(ratio)
    return None


def sanitize_input(inp):
    """Return a copy of ``inp`` without the identifier fields."""
    if not isinstance(inp, dict):
        return inp
    return {key: value for key, value in inp.items() if key not in IDENTIFIER_FIELDS}


@dataclass(frozen=True)
class DecisionWindow:
    """One decision window: everything observable before the decision, plus labels.

    ``input`` is already sanitised; ``raw_input`` keeps the original 280 fields
    (including identifiers) for auditing only.
    """

    window_id: str
    case_id: str
    input: dict
    output: dict
    difficulty: str | None = None
    raw_input: dict | None = None

    @property
    def real_action(self):
        """Ground-truth action (the one the clinician actually took)."""
        return get_action(self.output.get("result_real"))

    @property
    def result_1(self):
        """Primary consensus action as ``{action, ratio}``."""
        value = self.output.get("result_1")
        return value if isinstance(value, dict) else {}

    @property
    def result_aux(self):
        """Alternative consensus action as ``{action, ratio}``."""
        value = self.output.get("result_aux")
        return value if isinstance(value, dict) else {}

    def to_record(self):
        """Original record shape, with the identifiers stripped from ``input``."""
        return {
            "window_id": self.window_id,
            "case_id": self.case_id,
            "input": self.input,
            "output": self.output,
        }


def load_jsonl(path):
    """Read a JSONL file and return the list of dictionaries."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def case_files(split):
    """Return the sorted list of ``case*.jsonl`` files of a split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r} (expected one of {SPLITS})")
    return sorted(Path(config.DATA_DIR, split).glob("case*.jsonl"))


def _build_window(record, manifest):
    case_id = str(record.get("case_id"))
    raw_input = record.get("input")
    if not isinstance(raw_input, dict):
        raw_input = {}
    return DecisionWindow(
        window_id=record.get("window_id"),
        case_id=case_id,
        input=sanitize_input(raw_input),
        output=record.get("output") or {},
        difficulty=manifest.get(case_id, {}).get("difficulty"),
        raw_input=raw_input,
    )


def iter_split(split):
    """Lazily iterate the windows of a split as ``DecisionWindow``."""
    manifest = load_manifest()
    for path in case_files(split):
        for record in load_jsonl(path):
            yield _build_window(record, manifest)


def load_split(split, limit=None):
    """Load a whole split (or its first ``limit`` windows)."""
    windows = []
    for window in iter_split(split):
        windows.append(window)
        if limit is not None and len(windows) >= limit:
            break
    return windows


def load_case(path):
    """Load every window of a ``case<id>.jsonl`` file."""
    manifest = load_manifest()
    return [_build_window(record, manifest) for record in load_jsonl(path)]


def load_manifest():
    """Read ``reports/split_manifest.csv``, keyed by ``case_id``."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        manifest = {}
        path = Path(config.REPORTS_DIR) / "split_manifest.csv"
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                manifest[row["case_id"]] = {
                    "difficulty": row.get("difficulty"),
                    "split": row.get("split"),
                    "n_windows": int(row["n_windows"]) if row.get("n_windows") else None,
                    "null_score": float(row["null_score"]) if row.get("null_score") else None,
                }
        _MANIFEST_CACHE = manifest
    return _MANIFEST_CACHE


def difficulty_of(case_id):
    """Difficulty (easy/medium/hard) of a case, or ``None`` if missing."""
    return load_manifest().get(str(case_id), {}).get("difficulty")


def split_summary(split):
    """``{"n_cases": ..., "n_windows": ...}`` summary for a split."""
    return {
        "split": split,
        "n_cases": len(case_files(split)),
        "n_windows": sum(1 for _ in iter_split(split)),
    }
