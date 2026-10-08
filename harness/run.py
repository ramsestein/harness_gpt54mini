# -*- coding: utf-8 -*-
"""Launcher for the strategy harnesses.

Runs one strategy over a slice of a dataset split and writes the results to
``results_harness/<strategy>/<run_id>/`` (predictions, summary and run metadata),
and refreshes the cross-strategy table ``results_harness/leaderboard.csv``.

Examples
--------
python harness/run.py --strategy zero_shot --split dev --cases 2
python harness/run.py --strategy self_consistency --split dev --cases 2 --samples 5 --workers 8
python harness/run.py --strategy zero_shot --split test --limit 20 --workers 8 --effort high
python harness/run.py --strategy zero_shot --split dev --cases 2 --dry-run

# Recompute only the missing/failed windows of the newest run of this strategy+split
python harness/run.py --strategy zero_shot --split test --workers 8 --resume
python harness/run.py --strategy zero_shot --split test --workers 8 --resume results_harness/zero_shot/20261008-134507_test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # executed as a script, e.g. ``python harness/run.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, dataset, harness_registry, leaderboard, metrics, prompting, results  # noqa: E402
from src.harness_base import Prediction, Strategy  # noqa: E402


def build_parser():
    """Command-line interface of the launcher."""
    parser = argparse.ArgumentParser(
        description="Run a harness strategy over a dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy", default="zero_shot", help="strategy name (a folder under harness/)"
    )
    parser.add_argument(
        "--split", default="dev", choices=dataset.SPLITS, help="dataset split to run on"
    )
    parser.add_argument(
        "--cases", type=int, default=None, help="use only the first N cases of the split"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the total number of windows")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="windows to process concurrently (default: 1, or the stored value when resuming)",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=["none", "low", "medium", "high"],
        help=(
            "reasoning effort for the model call (default: config, currently "
            "'high'). The accepted values are the ones the base model supports; "
            "'none' omits the parameter"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "number of samples for sample-based strategies such as "
            "self_consistency (default: the strategy's own default)"
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help=(
            "resume an existing run: only the missing or failed windows are computed "
            "again. Without a value, the newest run of this strategy and split"
        ),
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=f"override the results folder (default: {config.RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="render the prompts without calling the model"
    )
    parser.add_argument("--quiet", action="store_true", help="do not print progress")
    return parser


def select_windows(split, cases=None, limit=None):
    """Windows to evaluate: the first ``cases`` cases, capped at ``limit`` windows."""
    windows = []
    if cases is None:
        iterator = dataset.iter_split(split)
    else:
        iterator = (
            window
            for path in dataset.case_files(split)[:cases]
            for window in dataset.load_case(path)
        )
    for window in iterator:
        windows.append(window)
        if limit is not None and len(windows) >= limit:
            break
    return windows


def _confusion(pairs):
    """Nested ``{predicted: {real: count}}`` confusion matrix (JSON friendly)."""
    matrix = {}
    for predicted, real in pairs:
        row = matrix.setdefault(predicted, {})
        row[real] = row.get(real, 0) + 1
    return matrix


def build_summary(predictions, strategy=None, split=None, dry_run=False):
    """Aggregate metrics of a prediction list, scored against ``result_real``."""
    scored = [p for p in predictions if p.get("action") and p.get("real_action")]
    preds = [p["action"] for p in scored]
    refs = [p["real_action"] for p in scored]
    summary = {
        "strategy": strategy,
        "split": split,
        "dry_run": dry_run,
        "n_windows": len(predictions),
        "n_scored": len(scored),
        "action_counts": results.summarize(predictions)["action_counts"],
        "unparsed": sum(1 for p in predictions if p.get("meta", {}).get("parsed") is False),
        "errors": sum(1 for p in predictions if p.get("meta", {}).get("error")),
    }
    if scored:
        summary["accuracy"] = metrics.accuracy(preds, refs)
        summary["agreement_result_1"] = metrics.accuracy(preds, [p["result_1"] for p in scored])
        # the dataset's two acceptable actions: the prediction is one of them
        summary["agreement_result_plausible"] = metrics.plausible_agreement(
            preds, [(p["result_1"], p.get("result_aux")) for p in scored]
        )
        summary["accuracy_by_difficulty"] = metrics.per_difficulty(
            preds, refs, [p.get("difficulty") for p in scored]
        )
        summary["confusion"] = _confusion(zip(preds, refs))
    return summary


def _safe_predict(strategy, window):
    """Predict one window; a model failure becomes an error record, not a crash."""
    try:
        return strategy.predict(window)
    except Exception as exc:  # noqa: BLE001 - keep the rest of the run going
        return Prediction(
            window_id=window.window_id,
            action=None,
            raw_output="",
            meta={"error": f"{type(exc).__name__}: {exc}"},
        )


def _predict_windows(strategy, windows, workers=1, quiet=False):
    """Predict every window, optionally in parallel, preserving the input order."""
    outcomes = [None] * len(windows)
    if workers <= 1:
        for index, window in enumerate(windows):
            outcomes[index] = _safe_predict(strategy, window)
            _report(index + 1, len(windows), window, outcomes[index], quiet)
        return outcomes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_safe_predict, strategy, window): index
            for index, window in enumerate(windows)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            outcomes[index] = future.result()
            _report(done, len(windows), windows[index], outcomes[index], quiet)
    return outcomes


def _dry_run_prediction(strategy, window):
    """Render the prompt the strategy would send, without calling the model."""
    build = getattr(strategy, "build_messages", None)
    if callable(build):
        text = "\n\n".join(message["content"] for message in build(window))
    else:
        text = prompting.build_prompt(window)
    return Prediction(
        window_id=window.window_id,
        action=None,
        raw_output=text,
        meta={"dry_run": True},
    )


def _report(position, total, window, outcome, quiet):
    """Print one progress line (safe to call from the main thread only)."""
    if quiet:
        return
    print(
        f"[{position}/{total}] {window.window_id} -> {outcome.action} "
        f"(real: {window.real_action})",
        flush=True,
    )


def _prediction_record(window, outcome):
    """One row of ``predictions.jsonl``."""
    return {
        "window_id": outcome.window_id,
        "action": outcome.action,
        "raw_output": outcome.raw_output,
        "meta": outcome.meta,
        "case_id": window.case_id,
        "difficulty": window.difficulty,
        "real_action": window.real_action,
        "result_1": window.result_1.get("action"),
        "result_aux": window.result_aux.get("action"),
    }


def run_root(strategy_name, results_root=None):
    """Folder holding every run of a strategy."""
    base = Path(results_root if results_root is not None else config.RESULTS_DIR)
    return base / strategy_name


def latest_run_dir(strategy_name, split, results_root=None):
    """Newest ``<timestamp>_<split>`` run directory of a strategy.

    Directories starting with ``_`` (caches such as the few-shot index) are ignored.
    """
    root = run_root(strategy_name, results_root)
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith("_") and child.name.endswith(f"_{split}")
    ] if root.is_dir() else []
    if not candidates:
        raise SystemExit(f"no {split!r} run found for strategy {strategy_name!r} under {root}")
    return max(candidates, key=lambda child: child.name)


def resolve_run_dir(resume, strategy_name, split, results_root=None):
    """The run directory to resume: an explicit one, or the newest of the split."""
    if resume == "latest":
        return latest_run_dir(strategy_name, split, results_root)
    path = Path(resume)
    if not path.is_dir():
        raise SystemExit(f"cannot resume: {path} is not a directory")
    return path


def backup_predictions(out_dir):
    """Keep a one-off copy of ``predictions.jsonl`` before a resume rewrites it."""
    out_dir = Path(out_dir)
    path = out_dir / "predictions.jsonl"
    backup = out_dir / "predictions.jsonl.bak"
    if path.is_file() and not backup.exists():
        shutil.copy2(path, backup)
        return backup
    return None


def is_failed(prediction):
    """True when a stored prediction has to be recomputed."""
    if not prediction:
        return True
    meta = prediction.get("meta") or {}
    if meta.get("error"):
        return True
    return prediction.get("action") is None and not meta.get("dry_run")


def refresh_summary(out_dir, strategy_name, split, dry_run=False):
    """Rebuild ``summary.json`` from ``predictions.jsonl``.

    The predictions file is the single source of truth, so the metrics always
    reflect every row it holds -- including rows added by a later resume.
    """
    out_dir = Path(out_dir)
    predictions = results.read_predictions(out_dir / "predictions.jsonl")
    summary = build_summary(predictions, strategy=strategy_name, split=split, dry_run=dry_run)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def run(strategy_name="zero_shot", split="dev", cases=None, limit=None, workers=None,
        effort=None, samples=None, resume=None, results_root=None, dry_run=False,
        quiet=False, **options):
    """Run ``strategy_name`` over ``split`` and persist the results.

    ``strategy_name`` is the name of a folder under ``harness/``, or an already-built
    ``Strategy`` instance (``src/control/`` uses that for the baselines that never
    call the model; a strategy with ``uses_model = False`` records no model).

    ``workers`` > 1 calls the model for several windows concurrently.
    ``effort`` overrides the reasoning effort (``"none"`` omits the parameter).
    ``samples`` sets ``n_samples`` for sample-based strategies.
    ``resume`` continues an existing run (a directory, or ``"latest"``): the
    selection **and the generation settings** (effort, samples, workers) are taken
    from its ``run.json`` when not given again, only the missing or failed windows
    are computed, and the metrics are rebuilt from the merged
    ``predictions.jsonl``.
    Returns ``{"out_dir", "summary", "predictions"}``.
    """
    strategy_instance = strategy_name if isinstance(strategy_name, Strategy) else None
    if strategy_instance is not None:
        strategy_name = strategy_instance.name

    saved = {}
    existing = []

    if resume is not None:
        out_dir = resolve_run_dir(resume, strategy_name, split, results_root)
        run_file = out_dir / "run.json"
        if run_file.is_file():
            saved = json.loads(run_file.read_text(encoding="utf-8"))
        predictions_file = out_dir / "predictions.jsonl"
        if predictions_file.is_file():
            existing = results.read_predictions(predictions_file)
            backup = backup_predictions(out_dir)
            if backup is not None and not quiet:
                print(f"backup made: {backup.name}")

        # Reuse the stored settings so the completed run stays homogeneous: mixing
        # two efforts (or two sample counts) inside one run would invalidate it.
        if cases is None:
            cases = saved.get("cases")
        if limit is None:
            limit = saved.get("limit")
        if workers is None:
            workers = saved.get("workers") or 1
        if effort is None and "effort" in saved:
            # ``None`` is meaningful here: the run was launched with no reasoning effort.
            effort = saved["effort"]
        if samples is None and saved.get("samples"):
            samples = saved["samples"]

        started = datetime.now()
        if saved.get("started_at"):
            try:
                started = datetime.fromisoformat(saved["started_at"])
            except ValueError:
                pass
    else:
        started = datetime.now()
        out_dir = run_root(strategy_name, results_root) / f"{started:%Y%m%d-%H%M%S}_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if workers is None:
            workers = 1

    if effort is not None or (resume is not None and "effort" in saved):
        options["reasoning_effort"] = effort
    if samples is not None:
        options["n_samples"] = samples
    strategy = strategy_instance or harness_registry.load_strategy(strategy_name, **options)

    windows = select_windows(split, cases=cases, limit=limit)
    if not windows:
        raise SystemExit(f"no windows selected (split={split!r}, cases={cases!r})")

    by_id = {prediction.get("window_id"): prediction for prediction in existing}
    todo = [window for window in windows if is_failed(by_id.get(window.window_id))]

    if resume is not None and not quiet:
        print(f"resuming {out_dir}")
        print(f"  stored windows : {len(existing)}")
        print(f"  to compute now : {len(todo)}")

    if not todo:
        outcomes = []
    elif dry_run:
        outcomes = []
        for index, window in enumerate(todo, start=1):
            outcomes.append(_dry_run_prediction(strategy, window))
            _report(index, len(todo), window, outcomes[-1], quiet)
    else:
        outcomes = _predict_windows(strategy, todo, workers=workers, quiet=quiet)

    for window, outcome in zip(todo, outcomes):
        by_id[window.window_id] = _prediction_record(window, outcome)

    # Keep the stored order (so a resume does not shuffle the file) and append
    # anything new at the end.
    order = [prediction.get("window_id") for prediction in existing]
    order += [window.window_id for window in windows if window.window_id not in order]
    predictions = [by_id[window_id] for window_id in order if window_id in by_id]

    results.write_predictions(out_dir / "predictions.jsonl", predictions)
    summary = refresh_summary(out_dir, strategy.name, split, dry_run=dry_run)
    uses_model = getattr(strategy, "uses_model", True)

    run_info = {
        "strategy": strategy.name,
        "split": split,
        "cases": cases,
        "limit": limit,
        "workers": workers,
        "n_windows": len(predictions),
        "n_computed_now": len(todo),
        "n_failed": summary["errors"],
        "resumed": resume is not None,
        "dry_run": dry_run,
        "model": None if (dry_run or not uses_model) else config.get_settings().model,
        "effort": None if (dry_run or not uses_model) else getattr(strategy, "reasoning_effort", None),
        "samples": getattr(strategy, "n_samples", getattr(strategy, "k", None)),
        "temperature": None if (dry_run or not uses_model) else getattr(strategy, "temperature", None),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "options": options or {},
    }
    (out_dir / "run.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # The cross-strategy table is a derived view of every run: keep it fresh, but a
    # failure here must not invalidate the run that is already persisted.
    table = None
    try:
        table = str(leaderboard.write_leaderboard(results_root))
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"WARNING: could not refresh the leaderboard: {type(exc).__name__}: {exc}")

    if summary["errors"] and not quiet:
        print(
            f"\nWARNING: {summary['errors']} window(s) failed; "
            "see 'meta.error' in predictions.jsonl"
        )
    return {"out_dir": str(out_dir), "summary": summary, "predictions": predictions,
            "leaderboard": table}


def main(argv=None):
    args = build_parser().parse_args(argv)
    outcome = run(
        args.strategy,
        args.split,
        cases=args.cases,
        limit=args.limit,
        workers=args.workers,
        effort=args.effort,
        samples=args.samples,
        resume=args.resume,
        results_root=args.results_root,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
    if not args.quiet:
        print(f"\nResults written to {outcome['out_dir']}")
        if outcome.get("leaderboard"):
            print(f"Leaderboard: {outcome['leaderboard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
