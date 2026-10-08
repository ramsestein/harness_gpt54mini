# -*- coding: utf-8 -*-
"""Run the control baselines (no model) over a split and compare them.

    python -m src.control.run_controls --split test
    python src/control/run_controls.py --split dev --cases 2
    python -m src.control.run_controls --split test --only sim_normality --workers 8
    python -m src.control.run_controls --split test --only retrieval_majority

Each baseline writes to ``results_harness/control/<baseline>/<run_id>/`` with the
same layout as a harness strategy run (``predictions.jsonl``, ``summary.json``,
``run.json``), and the invocation also writes one comparison table with a row per
baseline to ``results_harness/control/_comparison/<run_id>.csv``. The leading
underscore keeps that folder out of the way of the run lookups.

These baselines are the reference the model strategies are compared against:
the random one gives the chance level, and the five constants give the label
prevalence (the best of them is the majority-class baseline).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # executed as a script, e.g. ``python src/control/run_controls.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import run as harness_run  # noqa: E402
from src import config, dataset, leaderboard  # noqa: E402
from src.control import baselines, retrieval_majority, sim_normality  # noqa: E402

#: Controls a bare invocation runs: the cheap ones (no corpus, no simulation).
DEFAULT_NAMES = list(baselines.BASELINE_NAMES)

#: Controls that need something extra, so they are opt-in through ``--only``:
#: ``retrieval_majority`` loads the few-shot index (175 MB),
#: ``sim_normality`` spends seconds of CPU per window on the simulator.
OPT_IN_NAMES = [retrieval_majority.NAME, sim_normality.NAME]


def default_names():
    """The controls a bare invocation runs."""
    return list(DEFAULT_NAMES)


def selected_names(only=None):
    """The controls to run: the default set, or the ``--only`` selection (``all`` allowed)."""
    if only is None or not str(only).strip():
        return default_names()
    names = [name.strip() for name in str(only).split(",") if name.strip()]
    if names == ["all"]:
        return default_names() + list(OPT_IN_NAMES)
    return names


def control_strategy(name, seed=0, **options):
    """Build one control by name, with the options that control accepts."""
    if name == sim_normality.NAME:
        return sim_normality.SimNormalityStrategy(**options)
    if name == retrieval_majority.NAME:
        return retrieval_majority.RetrievalMajorityStrategy(**options)
    return baselines.build(name, seed=seed, **options)


def canonical_root(root):
    """The top of the results tree to refresh, when the sweep wrote inside it.

    The controls live in a sub-tree (``results_harness/control/``), so the launcher only
    refreshes the table of that sub-tree. ``None`` means "somewhere else" (a scratch or
    test folder): the canonical table must not be touched.
    """
    return Path(config.RESULTS_DIR) if Path(root) == Path(baselines.CONTROL_DIR) else None

#: Folder holding the comparison tables (the leading "_" keeps runs discoverable).
COMPARISON_DIRNAME = "_comparison"

#: Columns of the comparison table (same agreement vocabulary as the leaderboard).
COLUMNS = [
    "baseline",
    "split",
    "n_windows",
    "n_scored",
    "agreement_result_real",
    "agreement_result_1",
    "agreement_result_plausible",
    "chance_level",
    "ci95_halfwidth",
    "run_dir",
]


def build_parser():
    """Command-line interface of the control launcher."""
    parser = argparse.ArgumentParser(
        description="Run the control baselines (uniform random + the five constants).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", default="dev", choices=dataset.SPLITS, help="dataset split to run on"
    )
    parser.add_argument(
        "--cases", type=int, default=None, help="use only the first N cases of the split"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the total number of windows")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random baseline")
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "comma-separated controls to run (default: "
            f"{', '.join(DEFAULT_NAMES)}; 'all' adds {', '.join(OPT_IN_NAMES)})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="windows to process concurrently (only useful for the expensive controls)",
    )
    parser.add_argument("--results-root",
        default=str(baselines.CONTROL_DIR),
        help=f"override the results folder (default: {baselines.CONTROL_DIR})",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print one line per window (the controls answer the same thing thousands of times)",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing but the results")
    return parser


def ci95_halfwidth(accuracy, n):
    """Half-width of the 95% normal-approximation interval of a proportion."""
    if not n:
        return 0.0
    return 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n)


def comparison_row(outcome):
    """One row of the comparison table, from one baseline's run outcome."""
    summary = outcome["summary"]
    name = summary.get("strategy")
    n_scored = summary.get("n_scored") or 0
    accuracy = summary.get("accuracy")
    return {
        "baseline": name,
        "split": summary.get("split"),
        "n_windows": summary.get("n_windows"),
        "n_scored": n_scored,
        # the accuracy the random baseline is expected to reach by chance
        "chance_level": (1.0 / len(dataset.ACTIONS)) if name == baselines.RANDOM_NAME else "",
        "agreement_result_real": accuracy,
        "agreement_result_1": summary.get("agreement_result_1"),
        "agreement_result_plausible": summary.get("agreement_result_plausible"),
        "ci95_halfwidth": ci95_halfwidth(accuracy or 0.0, n_scored),
        "run_dir": outcome["out_dir"],
    }


def write_comparison(rows, root, split, started=None):
    """Write the comparison table of one invocation; returns its path."""
    stamp = f"{started or datetime.now():%Y%m%d-%H%M%S}_{split}"
    path = Path(root) / COMPARISON_DIRNAME / f"{stamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(rows):
    """Print the comparison as a small aligned table."""
    header = ("baseline", "n", "result_real", "result_1", "plausible")
    name_width = max(len(header[0]), *(len(str(row["baseline"])) for row in rows))
    n_width = max(len(header[1]), *(len(str(row["n_windows"])) for row in rows))
    lines = [
        f"{header[0]:<{name_width}}  {header[1]:>{n_width}}  "
        f"{header[2]:>11}  {header[3]:>9}  {header[4]:>9}"
    ]
    lines.append("-" * len(lines[0]))
    for row in rows:
        cells = [
            "n/a" if row[key] is None else f"{float(row[key]):.4f}"
            for key in ("agreement_result_real", "agreement_result_1", "agreement_result_plausible")
        ]
        lines.append(
            f"{row['baseline']:<{name_width}}  {row['n_windows']:>{n_width}}  "
            f"{cells[0]:>11}  {cells[1]:>9}  {cells[2]:>9}"
        )
    print("\n" + "\n".join(lines))


def run_controls(split="dev", cases=None, limit=None, seed=0, results_root=None, quiet=False,
                 progress=False, only=None, workers=None, **strategy_options):
    """Run the selected control baselines over ``split`` and write the comparison table.

    ``progress`` prints one line per window; the default output is the table alone,
    since a baseline answers the same action to every window. ``workers`` > 1 only helps
    the I/O of model calls: the simulator holds the GIL, so ``sim_normality`` ignores it.
    Returns ``{"root", "outcomes", "rows", "comparison"}``.
    """
    root = Path(results_root) if results_root is not None else Path(baselines.CONTROL_DIR)
    started = datetime.now()
    outcomes = []
    for name in selected_names(only):
        baseline = control_strategy(name, seed=seed, **strategy_options)
        if not quiet:
            print(f"\n=== {baseline.name} ===")
        outcomes.append(
            harness_run.run(
                baseline,
                split=split,
                cases=cases,
                limit=limit,
                workers=workers,
                results_root=root,
                quiet=quiet or not progress,
            )
        )
    rows = [comparison_row(outcome) for outcome in outcomes]
    comparison = write_comparison(rows, root, split, started=started)
    target = canonical_root(root)
    if target is not None:
        leaderboard.write_leaderboard(target)
    if not quiet:
        print_table(rows)
        print(f"\nComparison written to {comparison}")
    return {
        "root": str(root),
        "outcomes": outcomes,
        "rows": rows,
        "comparison": str(comparison),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_controls(
        split=args.split,
        cases=args.cases,
        limit=args.limit,
        seed=args.seed,
        results_root=args.results_root,
        quiet=args.quiet,
        progress=args.progress,
        only=args.only,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
