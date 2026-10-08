# -*- coding: utf-8 -*-
"""Similarity search over the training split, for the few-shot strategy.

The corpus is every window of the ``train`` split. A window is represented by its
**numeric input features** only:

- the labels (``result_1``, ``result_aux``, ``result_real``) live in ``output`` and
  are never read here, so they cannot influence the similarity;
- identifiers (``pt_caseid``, ``pt_subjectid``, ``case_id``) are excluded;
- nested ``*_json`` series and the string/categorical fields (sex, procedure name,
  trend labels, monitoring type...) are excluded in this version.

Features are standardised with the corpus mean/std and missing values are imputed
with the corpus mean (so a null contributes nothing). Distance is Euclidean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config, dataset, prompting

#: Corpus split. Only training windows may be retrieved.
SOURCE_SPLIT = "train"

#: Bump when the extraction, the standardisation or the stored file layout changes.
INDEX_VERSION = 1

#: Fields that must never take part in the similarity: identifiers and labels.
EXCLUDED_FIELDS = frozenset(dataset.IDENTIFIER_FIELDS) | {
    "case_id",
    "case_id_fill",
    "window_id",
}

#: Cached index (the ``_index`` prefix keeps it apart from the run directories).
DEFAULT_INDEX_PATH = Path(config.RESULTS_DIR) / "few_shot" / "_index" / "train_index.npz"


def is_feature(value):
    """Scalars only: bools and ``None`` are not numeric features."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def feature_names(windows):
    """Sorted names of the numeric features present in ``windows``."""
    names = set()
    for window in windows:
        for name, value in window.input.items():
            if name not in EXCLUDED_FIELDS and is_feature(value):
                names.add(name)
    return sorted(names)


@dataclass
class Neighbor:
    """A retrieved training window."""

    window_id: str
    case_id: str
    action: str
    prompt: str
    distance: float


def train_windows(cases=None):
    """Stream the training windows used as the retrieval corpus."""
    if cases is None:
        yield from dataset.iter_split(SOURCE_SPLIT)
        return
    for path in dataset.case_files(SOURCE_SPLIT)[:cases]:
        yield from dataset.load_case(path)


def _prompts_path(path):
    return Path(path).with_suffix(".prompts.jsonl")


class SimilarityIndex:
    """k-nearest-neighbour index over the numeric features of the train windows."""

    def __init__(self, names, vectors, means, stds, window_ids, case_ids, actions, prompts):
        self.names = list(names)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.window_ids = list(window_ids)
        self.case_ids = list(case_ids)
        self.actions = list(actions)
        self.prompts = list(prompts)

    def __len__(self):
        return len(self.vectors)

    @classmethod
    def build(cls, windows, render=None):
        """Build the index from an iterable of ``DecisionWindow``."""
        render = render or prompting.render_case
        windows = list(windows)
        if not windows:
            raise ValueError("cannot build a similarity index from zero windows")

        names = feature_names(windows)
        raw = np.array(
            [
                [w.input.get(name) if is_feature(w.input.get(name)) else np.nan for name in names]
                for w in windows
            ],
            dtype=np.float32,
        )

        with np.errstate(invalid="ignore"):
            means = np.nanmean(raw, axis=0)
            stds = np.nanstd(raw, axis=0)
        means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
        stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0).astype(np.float32)

        filled = np.where(np.isfinite(raw), raw, means)
        vectors = ((filled - means) / stds).astype(np.float32)

        return cls(
            names=names,
            vectors=vectors,
            means=means,
            stds=stds,
            window_ids=[w.window_id for w in windows],
            case_ids=[str(w.case_id) for w in windows],
            actions=[w.real_action for w in windows],
            prompts=[render(w.input) for w in windows],
        )

    def vector_for(self, inp):
        """Standardised feature vector of one window input (mean-imputed)."""
        raw = np.array(
            [inp.get(name) if is_feature(inp.get(name)) else np.nan for name in self.names],
            dtype=np.float32,
        )
        filled = np.where(np.isfinite(raw), raw, self.means)
        return ((filled - self.means) / self.stds).astype(np.float32)

    def query(self, window, k=3, exclude_case=None):
        """The ``k`` closest training windows to ``window`` (nearest first)."""
        if len(self) == 0:
            return []
        distances = np.linalg.norm(self.vectors - self.vector_for(window.input), axis=1)
        neighbors = []
        for position in np.argsort(distances):
            index = int(position)
            if exclude_case is not None and self.case_ids[index] == str(exclude_case):
                continue
            neighbors.append(
                Neighbor(
                    window_id=self.window_ids[index],
                    case_id=self.case_ids[index],
                    action=self.actions[index],
                    prompt=self.prompts[index],
                    distance=float(distances[index]),
                )
            )
            if len(neighbors) >= k:
                break
        return neighbors

    def save(self, path):
        """Write the index (``.npz`` plus a ``.prompts.jsonl`` side file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            version=np.array(INDEX_VERSION),
            names=np.array(self.names),
            vectors=self.vectors,
            means=self.means,
            stds=self.stds,
            window_ids=np.array(self.window_ids),
            case_ids=np.array(self.case_ids),
            actions=np.array(self.actions),
        )
        with open(_prompts_path(path), "w", encoding="utf-8") as handle:
            for prompt in self.prompts:
                handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path):
        """Load an index written by :meth:`save`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != INDEX_VERSION:
                raise ValueError(
                    f"index version mismatch in {path}: "
                    f"{int(data['version'])} != {INDEX_VERSION}"
                )
            index = cls(
                names=[str(name) for name in data["names"]],
                vectors=data["vectors"],
                means=data["means"],
                stds=data["stds"],
                window_ids=[str(value) for value in data["window_ids"]],
                case_ids=[str(value) for value in data["case_ids"]],
                actions=[str(value) for value in data["actions"]],
                prompts=[
                    json.loads(line)
                    for line in _prompts_path(path).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
            )
        if len(index.prompts) != len(index):
            raise ValueError(f"corrupt index cache at {path}: prompts do not match vectors")
        return index


def load_or_build(path=None, cases=None, rebuild=False):
    """Load the cached index, or build it from the train split.

    A partial corpus (``cases`` given) is never cached, so it cannot be mistaken
    for the full index later.
    """
    path = Path(path or DEFAULT_INDEX_PATH)
    if not rebuild and cases is None and path.is_file() and _prompts_path(path).is_file():
        try:
            return SimilarityIndex.load(path)
        except (ValueError, KeyError, OSError):
            pass

    index = SimilarityIndex.build(train_windows(cases=cases))
    if cases is None:
        index.save(path)
    return index
