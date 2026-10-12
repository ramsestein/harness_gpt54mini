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

Two runs of the same strategy, split, effort and ``k`` are the same experiment run
twice: the table keeps the **most recent** one, so what is compared across strategies
is one configuration each. (``collect_rows(..., latest_only=False)`` still returns
everything, and the runs themselves are never touched.)

    python -m src.leaderboard
    python -m src.leaderboard --results-root C:\\scratch --output table.csv
    python -m src.leaderboard --all-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script, e.g. ``python src/leaderboard.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, dataset, metrics, results, view  # noqa: E402

#: Name of the table inside the results folder.
TABLE_NAME = "leaderboard.csv"

#: The no-model baselines nest one level deeper: ``control/<baseline>/<run_id>/``.
CONTROL_DIRNAME = "control"
CONTROL_GROUP = "control"
HARNESS_GROUP = "harness"

#: Columns of the table. The run bookkeeping (id, window counts, run folder) is
#: deliberately out: the table compares what each strategy is and scores, not how or when
#: it was launched.
#: ``ci95_low``/``ci95_high`` are the case-level bootstrap interval of the accuracy, and
#: the rest is what a corpus with two labels at ~27 % each demands: metrics that do not
#: reward leaning on the majority class (``balanced_accuracy``, ``macro_f1``, ``kappa``),
#: the recall of every label and the distribution of what the strategy answered.
COLUMNS = [
    "group",
    "strategy",
    "model",
    "split",
    "effort",
    "k",
    "view",
    "errors",
    "agreement_result_real",
    "ci95_low",
    "ci95_high",
    "agreement_result_1",
    "agreement_result_plausible",
    "balanced_accuracy",
    "macro_f1",
    "kappa",
    *[f"recall_{action}" for action in dataset.ACTIONS],
    *[f"predicted_{action}" for action in dataset.ACTIONS],
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


def _plausible_from_predictions(rows):
    """Agreement with ``result_1`` or ``result_aux`` of a run that predates the metric."""
    rows = [row for row in rows if row.get("action")]
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


def scored_rows(run_dir):
    """The rows of a run that count: a prediction and a truth to compare it with."""
    path = Path(run_dir) / "predictions.jsonl"
    if not path.is_file():
        return []
    return [
        row
        for row in results.read_predictions(path)
        if row.get("action") and row.get("real_action")
    ]


def run_analysis(rows, labels=None):
    """The uncertainty and prior-free metrics of a run, from its own predictions.

    Everything here is computed from the stored predictions, so a run recorded before
    these columns existed gets them anyway.
    """
    labels = list(labels if labels is not None else dataset.ACTIONS)
    columns = (
        ["ci95_low", "ci95_high", "balanced_accuracy", "macro_f1", "kappa"]
        + [f"recall_{label}" for label in labels]
        + [f"predicted_{label}" for label in labels]
    )
    if not rows:
        return {column: None for column in columns}

    preds = [row["action"] for row in rows]
    refs = [row["real_action"] for row in rows]
    hits = [int(pred == ref) for pred, ref in zip(preds, refs)]
    # The interval resamples whole cases: the windows of a case are one patient.
    low, high, _ = metrics.cluster_bootstrap_ci(
        [row.get("case_id") or row["window_id"] for row in rows], hits
    )
    analysis = {
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "balanced_accuracy": round(metrics.balanced_accuracy(preds, refs, labels), 4),
        "macro_f1": round(metrics.macro_f1(preds, refs, labels), 4),
        "kappa": round(metrics.cohen_kappa(preds, refs, labels), 4),
    }
    recalls = metrics.per_class_recall(preds, refs, labels)
    for label in labels:
        analysis[f"recall_{label}"] = round(recalls[label], 4)
        analysis[f"predicted_{label}"] = round(preds.count(label) / len(preds), 4)
    return analysis


def run_row(group, strategy, run_dir):
    """One row of the table, read from the run's own summary, settings and predictions."""
    run_dir = Path(run_dir)
    summary = _summary_of(run_dir) or {}
    settings = _run_json(run_dir)
    rows = scored_rows(run_dir)
    plausible = summary.get("agreement_result_plausible")
    if plausible is None:
        plausible = _plausible_from_predictions(rows)

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
        # the model the run was made with (``None`` for a strategy that calls no model)
        "model": settings.get("model"),
        "split": summary.get("split"),
        "effort": effort,
        "k": k,
        # the shared view the run saw (``src.view``); runs older than the view carry none
        "view": settings.get("view_version"),
        "errors": summary.get("errors", 0),
        "agreement_result_real": summary.get("accuracy"),
        "agreement_result_1": summary.get("agreement_result_1"),
        "agreement_result_plausible": plausible,
        **run_analysis(rows),
    }


def run_timestamp(run_dir):
    """The run id inside the folder name -- ``<YYYYmmdd-HHMMSS>_<split>``; it sorts by time.

    A folder that does not follow the convention keeps its whole name, which is still a
    stable ordering key.
    """
    return Path(run_dir).name.partition("_")[0]


def run_key(group, strategy, row):
    """What makes two rows the same experiment: same group, strategy, model, split, effort, k, view.

    The model is part of it: the same strategy on two models are two experiments, and their
    rows must not replace each other.
    """
    return (
        group,
        strategy,
        row.get("model"),
        row.get("split"),
        row.get("effort"),
        row.get("k"),
        row.get("view"),
    )


def latest_per_configuration(entries):
    """``entries`` as ``(group, strategy, timestamp, row)``, keeping the newest of each run
    configuration."""
    newest = {}
    for group, strategy, timestamp, row in entries:
        key = run_key(group, strategy, row)
        previous = newest.get(key)
        if previous is None or timestamp >= previous[2]:
            newest[key] = (group, strategy, timestamp, row)
    return list(newest.values())


#: ``view_version`` value meaning "keep the runs of every view".
ALL_VIEWS = "all"

#: Name of the table that holds only the runs made with another model.
MODELS_TABLE_NAME = "leaderboard_models.csv"


def collect_rows(root=None, latest_only=True, view_version=view.VIEW_VERSION, models_only=False,
                 model=None):
    """Every row of the table, ordered by group, strategy and run id.

    With ``latest_only`` (the default) two runs of the same configuration collapse into
    the most recent one: the table then holds one row per configuration, not per attempt.

    Only the runs made on ``view_version`` -- the current shared view by default -- are
    kept, so the table never mixes strategies that saw different windows; pass
    :data:`ALL_VIEWS` to keep every run (the ``view`` column then tells them apart).

    ``model`` keeps one model only; ``models_only`` keeps the runs made with any model
    other than the project's own (the point of the second table), which is also what
    excludes the no-model controls.
    """
    base = _root(root)
    entries = [
        (group, strategy, run_timestamp(run_dir), run_row(group, strategy, run_dir))
        for group, strategy, run_dir in iter_runs(base)
    ]
    if view_version != ALL_VIEWS:
        entries = [entry for entry in entries if entry[3].get("view") == view_version]
    if model is not None:
        entries = [entry for entry in entries if entry[3].get("model") == model]
    elif models_only:
        entries = [
            entry
            for entry in entries
            if entry[3].get("model") and entry[3].get("model") != config.DEFAULT_MODEL
        ]
    if latest_only:
        entries = latest_per_configuration(entries)
    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2], str(entry[3].get("split") or "")))
    return [row for *_, row in entries]


def write_leaderboard(root=None, output=None, latest_only=True, view_version=view.VIEW_VERSION,
                      models_only=False, model=None):
    """Write the table and return its path.

    By default it is ``<root>/leaderboard.csv``; with ``models_only`` it is
    ``<root>/leaderboard_models.csv``, holding only the runs made with another model.
    """
    base = _root(root)
    if output is not None:
        path = Path(output)
    else:
        path = base / (MODELS_TABLE_NAME if models_only else TABLE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(
            collect_rows(base, latest_only=latest_only, view_version=view_version,
                         models_only=models_only, model=model)
        )
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
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="keep every run instead of only the newest of each configuration",
    )
    parser.add_argument(
        "--all-views",
        action="store_true",
        help=f"keep the runs of every view of the window, not only the current one "
             f"({view.VIEW_VERSION}); the 'view' column tells them apart",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="keep only the runs made with this model (e.g. gpt-5.4-pro)",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help=(
            f"keep only the runs made with another model, and write "
            f"<results-root>/{MODELS_TABLE_NAME} unless --output says otherwise"
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = write_leaderboard(
        args.results_root,
        output=args.output,
        latest_only=not args.all_runs,
        view_version=ALL_VIEWS if args.all_views else view.VIEW_VERSION,
        models_only=args.models_only,
        model=args.model,
    )
    print(f"Leaderboard written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
