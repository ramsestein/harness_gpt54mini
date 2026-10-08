# -*- coding: utf-8 -*-
"""Control baselines (no model): random choice and the five constant choices.

The controls live in ``src/control/`` (they are not harness strategies) and their
results go to ``results_harness/control/``. They exist to answer two questions
without any model: what does picking uniformly at random score, and what does
always answering the same option score?
"""

import csv
import json
from pathlib import Path

import pytest

from src import config, dataset, harness_registry, llm
from src.control import baselines, run_controls
from src.harness_base import Strategy

ACTIONS = list(dataset.ACTIONS)


class FakeWindow:
    """Only the field the controls read."""

    def __init__(self, window_id):
        self.window_id = window_id


@pytest.fixture(scope="module")
def fake_windows():
    return [FakeWindow(f"case{index:04d}_w{index}") for index in range(2000)]


@pytest.fixture(scope="module")
def real_windows():
    return dataset.load_split("dev", limit=12)


# --------------------------------------------------------------------------- #
# Where the controls live
# --------------------------------------------------------------------------- #


def test_the_controls_live_outside_the_harness_folder():
    assert (Path(config.PROJECT_ROOT) / "src" / "control").is_dir()
    # they are not harness strategies: the harness roadmap must stay untouched
    assert not set(baselines.BASELINE_NAMES) & set(harness_registry.EXPECTED_STRATEGIES)
    assert not set(baselines.BASELINE_NAMES) & set(harness_registry.strategy_dirs())


def test_the_results_go_to_the_control_folder():
    assert Path(baselines.CONTROL_DIR) == Path(config.RESULTS_DIR) / "control"


# --------------------------------------------------------------------------- #
# The baselines
# --------------------------------------------------------------------------- #


def test_every_baseline_is_a_strategy():
    baselines_ = baselines.all_baselines()
    assert len(baselines_) == 2 + len(ACTIONS)  # random + one per action + the human one
    assert [baseline.name for baseline in baselines_] == baselines.BASELINE_NAMES
    for baseline in baselines_:
        assert isinstance(baseline, Strategy)
        assert baseline.name
        assert baseline.description.strip()
    assert len({baseline.name for baseline in baselines_}) == len(baselines_)


def test_the_constant_baselines_cover_every_action_exactly_once():
    constants = baselines.constant_baselines()
    assert [baseline.action for baseline in constants] == ACTIONS
    assert [baseline.name for baseline in constants] == [f"always_{a}" for a in ACTIONS]


def test_the_random_baseline_is_named_and_seeded():
    baseline = baselines.random_baseline(seed=7)
    assert baseline.name == "random"
    assert baseline.seed == 7


def test_an_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        baselines.ConstantBaseline("levitate")


def test_a_constant_baseline_always_answers_its_action(real_windows):
    for action in ACTIONS:
        baseline = baselines.ConstantBaseline(action)
        predictions = baseline.predict_many(real_windows)
        assert [prediction.action for prediction in predictions] == [action] * len(real_windows)
        assert [prediction.window_id for prediction in predictions] == [
            window.window_id for window in real_windows
        ]


# --------------------------------------------------------------------------- #
# The random baseline
# --------------------------------------------------------------------------- #


def test_the_random_baseline_only_predicts_known_actions(fake_windows):
    actions = {p.action for p in baselines.random_baseline().predict_many(fake_windows)}
    assert actions <= set(ACTIONS)


def test_the_random_baseline_is_deterministic(fake_windows):
    first = baselines.random_baseline(seed=3).predict_many(fake_windows)
    second = baselines.random_baseline(seed=3).predict_many(fake_windows)
    assert [p.action for p in first] == [p.action for p in second]


def test_the_random_baseline_ignores_the_window_order(fake_windows):
    """A window's draw must depend on its id, not on its position in the batch."""
    shuffling = fake_windows[1::2] + fake_windows[0::2]
    ordered = {p.window_id: p.action for p in baselines.random_baseline().predict_many(fake_windows)}
    shuffled = {
        p.window_id: p.action for p in baselines.random_baseline().predict_many(shuffling)
    }
    assert ordered == shuffled


def test_another_seed_draws_differently(fake_windows):
    zero = [p.action for p in baselines.random_baseline(seed=0).predict_many(fake_windows)]
    one = [p.action for p in baselines.random_baseline(seed=1).predict_many(fake_windows)]
    assert zero != one


def test_the_random_baseline_spreads_evenly_over_the_actions(fake_windows):
    predictions = baselines.random_baseline().predict_many(fake_windows)
    counts = {action: 0 for action in ACTIONS}
    for prediction in predictions:
        counts[prediction.action] += 1
    for action, count in counts.items():
        share = count / len(predictions)
        assert 0.15 < share < 0.25, f"{action} drawn {share:.3f} of the time"


def test_the_random_baseline_records_its_draw(fake_windows):
    for prediction in baselines.random_baseline(seed=5).predict_many(fake_windows[:50]):
        assert prediction.meta["baseline"] == "random"
        assert prediction.meta["seed"] == 5
        assert ACTIONS[prediction.meta["draw"]] == prediction.action


def test_a_constant_baseline_records_what_it_answered(real_windows):
    prediction = baselines.ConstantBaseline("increase_opioid").predict(real_windows[0])
    assert prediction.meta["baseline"] == "always"
    assert prediction.meta["action"] == "increase_opioid"


def test_the_controls_do_not_use_the_model():
    assert Strategy.uses_model is True
    for baseline in baselines.all_baselines():
        assert baseline.uses_model is False


# --------------------------------------------------------------------------- #
# The human baseline
# --------------------------------------------------------------------------- #


def test_the_human_baseline_replays_the_clinician(real_windows):
    predictions = baselines.human_baseline().predict_many(real_windows)

    assert [p.action for p in predictions] == [w.real_action for w in real_windows]
    assert all(p.meta["baseline"] == "human" for p in predictions)


def test_the_human_baseline_is_perfect_against_the_real_action(control_run):
    assert float(_rows(control_run)["human"]["agreement_result_real"]) == 1.0


def test_the_human_baseline_measures_the_consensus(control_run, real_windows):
    """Its agreements are exactly how often the clinician matches the consensus top-1 / top-2."""
    real = [window.real_action for window in real_windows]
    first = [window.result_1["action"] for window in real_windows]
    pair = [(window.result_1["action"], window.result_aux["action"]) for window in real_windows]

    row = _rows(control_run)["human"]

    assert float(row["agreement_result_1"]) == pytest.approx(
        sum(a == b for a, b in zip(real, first)) / len(real)
    )
    assert float(row["agreement_result_plausible"]) == pytest.approx(
        sum(a in options for a, options in zip(real, pair)) / len(real)
    )


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def control_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("control-results")
    return run_controls.run_controls(split="dev", limit=12, results_root=root, quiet=True)


def _rows(control_run):
    return {row["baseline"]: row for row in control_run["rows"]}


def test_the_runner_writes_one_run_per_baseline_under_control(control_run):
    root = Path(control_run["root"])
    assert root.name.startswith("control-results")  # the caller decides the root
    for name in baselines.BASELINE_NAMES:
        runs = [child for child in (root / name).iterdir() if child.is_dir()]
        assert len(runs) == 1, f"{name} should have exactly one run"
        run_dir = runs[0]
        assert run_dir.name.endswith("_dev")
        assert (run_dir / "predictions.jsonl").is_file()
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "run.json").is_file()


def test_every_baseline_scores_the_same_windows(control_run):
    for row in control_run["rows"]:
        assert int(row["n_windows"]) == 12
        assert int(row["n_scored"]) == 12


def test_the_constant_baselines_score_the_label_prevalence(control_run, real_windows):
    truth = [window.real_action for window in real_windows]
    for action in ACTIONS:
        expected = truth.count(action) / len(truth)
        row = _rows(control_run)[f"always_{action}"]
        assert float(row["agreement_result_real"]) == pytest.approx(expected)


def test_the_constant_accuracies_add_up_to_one(control_run):
    total = sum(
        float(_rows(control_run)[f"always_{action}"]["agreement_result_real"])
        for action in ACTIONS
    )
    assert total == pytest.approx(1.0)


def test_the_comparison_table_lists_every_baseline_once(control_run):
    table = Path(control_run["comparison"])
    assert table.is_file()
    assert table.parent.name == "_comparison"
    with open(table, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["baseline"] for row in rows] == run_controls.default_names()


def test_the_comparison_table_carries_the_run_folders(control_run):
    for row in control_run["rows"]:
        assert Path(row["run_dir"]).is_dir()
        assert (Path(row["run_dir"]) / "predictions.jsonl").is_file()


def test_the_random_baseline_stays_within_chance(control_run):
    row = _rows(control_run)["random"]
    accuracy = float(row["agreement_result_real"])
    assert 0.0 <= accuracy <= 1.0
    assert float(row["chance_level"]) == pytest.approx(1 / len(ACTIONS))


def test_the_comparison_table_carries_the_three_agreements(control_run):
    for row in control_run["rows"]:
        assert row["agreement_result_real"] is not None
        assert row["agreement_result_1"] is not None
        assert row["agreement_result_plausible"] is not None


def test_the_run_json_does_not_claim_a_model(control_run):
    for name in baselines.BASELINE_NAMES:
        runs = [child for child in (Path(control_run["root"]) / name).iterdir() if child.is_dir()]
        info = json.loads((runs[0] / "run.json").read_text(encoding="utf-8"))
        assert info["strategy"] == name
        assert info["model"] is None
        assert info["effort"] is None
        assert info["temperature"] is None


def test_the_controls_never_call_the_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a control baseline must not call the model")

    monkeypatch.setattr(llm, "complete_messages", forbidden)
    monkeypatch.setattr(llm, "complete", forbidden)

    outcome = run_controls.run_controls(split="dev", limit=12, results_root=tmp_path, quiet=True)

    for run in outcome["outcomes"]:
        assert run["summary"]["errors"] == 0
        assert all(prediction["action"] for prediction in run["predictions"])


def test_the_runner_keeps_the_window_order(control_run):
    run = control_run["outcomes"][0]
    assert [p["window_id"] for p in run["predictions"]] == [
        window.window_id for window in dataset.load_split("dev", limit=12)
    ]


def test_only_the_real_results_tree_refreshes_the_canonical_table():
    assert run_controls.canonical_root(baselines.CONTROL_DIR) == Path(config.RESULTS_DIR)
    assert run_controls.canonical_root(Path("C:/somewhere/else")) is None


def test_a_scratch_root_never_touches_the_canonical_table(tmp_path, monkeypatch):
    refreshed = []
    monkeypatch.setattr(
        run_controls.leaderboard,
        "write_leaderboard",
        lambda root, **options: refreshed.append(Path(root)),
    )

    run_controls.run_controls(split="dev", limit=2, results_root=tmp_path, quiet=True)

    assert refreshed, "the launcher refreshes the table of the tree it wrote into"
    assert Path(config.RESULTS_DIR) not in refreshed


def test_the_default_output_is_the_table_not_every_window(tmp_path, capsys):
    run_controls.run_controls(split="dev", limit=3, results_root=tmp_path)

    printed = capsys.readouterr().out
    assert "case1169" not in printed, "a per-window line is not printed by default"
    assert "baseline" in printed and "always_no_action" in printed


def test_the_progress_flag_prints_every_window(tmp_path, capsys):
    run_controls.run_controls(split="dev", limit=3, results_root=tmp_path, progress=True)

    printed = capsys.readouterr().out
    assert printed.count("case1169") == 3 * len(run_controls.default_names())


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_the_cli_defaults_to_the_control_results_folder():
    args = run_controls.build_parser().parse_args([])
    assert Path(args.results_root) == Path(baselines.CONTROL_DIR)
    assert args.split == "dev"
    assert args.seed == 0


def test_the_cli_accepts_the_selection_flags():
    args = run_controls.build_parser().parse_args(
        ["--split", "test", "--cases", "3", "--limit", "20", "--seed", "9", "--quiet"]
    )
    assert (args.split, args.cases, args.limit, args.seed) == ("test", 3, 20, 9)
    assert args.quiet is True


def test_the_cli_defaults_to_no_per_window_progress():
    assert run_controls.build_parser().parse_args([]).progress is False


def test_the_cli_rejects_an_unknown_split():
    with pytest.raises(SystemExit):
        run_controls.build_parser().parse_args(["--split", "validation"])
