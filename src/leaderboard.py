# -*- coding: utf-8 -*-
"""Cross-strategy table: one row per generated run of every strategy.

``results_harness/leaderboard.csv`` lists every run found under ``results_harness/``
with what it is and what it scored:

- ``effort`` / ``k`` -- the reasoning effort and the neighbourhood size the run used,
  when it uses them (taken from its ``run.json``, or from its predictions for older runs).
- ``agreement_result_real`` -- the prediction equals the action the clinician took.
- ``agreement_result_1`` -- the prediction equals the consensus ``result_1``.
- ``agreement_result_plausible`` -- the prediction is ``result_1`` or ``result_aux``, i.e.
  an acceptable answer even when it is not the action the clinician took.

Runs are read from ``<strategy>/<run_id>/summary.json`` and, for the no-model
baselines, from ``control/<baseline>/<run_id>/summary.json``. Folders starting with
``_`` (caches, the control comparison tables) and dry runs are skipped. A run whose
summary predates ``agreement_result_plausible`` gets the value computed from its own
``predictions.jsonl``, so old runs need no recomputation.

    python -m src.leaderboard
    python -m src.leaderboard --results-root C:\\scratch --output table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script, e.g. ``python src/leaderboard.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, metrics, results  # noqa: E402

#: Name of the table inside the results folder.
TABLE_NAME = "leaderboard.csv"

#: The no-model baselines nest one level deeper: ``control/<baseline>/<run_id>/``.
CONTROL_DIRNAME = "control"
CONTROL_GROUP = "control"
HARNESS_GROUP = "harness"

#: Columns of the table. The run bookkeeping (id, window counts) is deliberately out:
#: the table compares what each strategy is and scores, not how it was launched.
COLUMNS = [
    "group",
    "strategy",
    "split",
    "effort",
    "k",
    "errors",
    "agreement_result_real",
    "agreement_result_1",
    "agreement_result_plausible",
    "run_dir",
]


def _root(path=None):
    """The results folder to scan."""
    return Path(path) if path is not None else Path(config.RESULTS_DIR)


def _subdirs(folder):
    """Child folders, sorted, ignoring the ``_``-prefixed caches."""
    if not folder.is_dir():
        return []
    return sorted(
        child
        for child in folder.iterdir()
        if child.is_dir() and not child.name.startswith(("_", "."))
    )


def _summary_of(run_dir):
    """The stored summary of a run, or ``None`` when it is not a finished run."""
    path = Path(run_dir) / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _run_json(run_dir):
    """The run metadata the launcher wrote (the generation settings live here)."""
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _finished_runs(folder):
    """The run folders of a strategy that hold a scored, non-dry run."""
    for run_dir in _subdirs(folder):
        summary = _summary_of(run_dir)
        if summary is not None and not summary.get("dry_run"):
            yield run_dir


def iter_runs(root=None):
    """``(group, strategy, run_dir)`` for every finished run under ``root``.

    ``root`` is normally the top of the results tree. Pointing it at ``control/`` also
    works: the runs belong to the control group either way.
    """
    base = _root(root)
    for strategy_dir in _subdirs(base):
        if strategy_dir.name == CONTROL_DIRNAME:
            for baseline_dir in _subdirs(strategy_dir):
                for run_dir in _finished_runs(baseline_dir):
                    yield CONTROL_GROUP, baseline_dir.name, run_dir
            continue
        group = CONTROL_GROUP if base.name == CONTROL_DIRNAME else HARNESS_GROUP
        for run_dir in _finished_runs(strategy_dir):
            yield group, strategy_dir.name, run_dir


def _plausible_from_predictions(run_dir):
    """Agreement with ``result_1`` or ``result_aux`` of a run that predates the metric."""
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        return None
    rows = [row for row in results.read_predictions(path) if row.get("action")]
    if not rows:
        return None
    return metrics.plausible_agreement(
        [row["action"] for row in rows],
        [(row.get("result_1"), row.get("result_aux")) for row in rows],
    )


def _first_meta(run_dir):
    """The metadata of the first stored prediction.

    Older runs keep their generation settings there (``reasoning_effort``, ``k``) instead
    of in ``run.json``; only the first line is read.
    """
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    return json.loads(line).get("meta") or {}
    except (OSError, ValueError):
        return {}
    return {}


def run_row(group, strategy, run_dir):
    """One row of the table, read from the run's own summary and settings."""
    run_dir = Path(run_dir)
    summary = _summary_of(run_dir) or {}
    settings = _run_json(run_dir)
    plausible = summary.get("agreement_result_plausible")
    if plausible is None:
        plausible = _plausible_from_predictions(run_dir)

    # ``run.json`` is the intended source; a run too old to carry the field falls back to
    # what its predictions recorded (and a strategy that uses no effort/k has neither).
    effort = settings.get("effort")
    k = settings.get("k")
    if effort is None or k is None:
        meta = _first_meta(run_dir)
        if effort is None:
            effort = meta.get("reasoning_effort")
        if k is None:
            k = meta.get("k")

    return {
        "group": group,
        "strategy": strategy,
        "split": summary.get("split"),
        "effort": effort,
        "k": k,
        "errors": summary.get("errors", 0),
        "agreement_result_real": summary.get("accuracy"),
        "agreement_result_1": summary.get("agreement_result_1"),
        "agreement_result_plausible": plausible,
        "run_dir": str(run_dir),
    }


def collect_rows(root=None):
    """Every row of the table, ordered by group, strategy and run id."""
    return [
        run_row(group, strategy, run_dir)
        for group, strategy, run_dir in iter_runs(root)
    ]


def write_leaderboard(root=None, output=None):
    """Write the table and return its path (``<root>/leaderboard.csv`` by default)."""
    base = _root(root)
    path = Path(output) if output is not None else base / TABLE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(collect_rows(base))
    return path


def build_parser():
    """Command-line interface of the table builder."""
    parser = argparse.ArgumentParser(
        description="Build the cross-strategy table from every run under the results folder."
    )
    parser.add_argument(
        "--results-root",
        default=str(config.RESULTS_DIR),
        help=f"folder holding the runs (default: {config.RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"CSV to write (default: <results-root>/{TABLE_NAME})",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = write_leaderboard(args.results_root, output=args.output)
    print(f"Leaderboard written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
