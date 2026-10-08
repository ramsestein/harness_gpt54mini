# -*- coding: utf-8 -*-
"""The simulator + normality control (no model).

Per window: calibrate the patient, simulate the five actions five minutes ahead, and
answer with the future closest to the centre of the normal range -- every signal one
dimension, normalised by its tolerance.
"""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from harness import run as harness_run
from harness.critic_verifier_sim import simulator as sim
from src import dataset, harness_registry
from src.control import baselines, run_controls, sim_normality
from src.harness_base import Strategy

NAME = "sim_normality"


def _outcome(action, bis=50.0, map=77.5, co=6.0, tol=0.0):
    """One simulated future, centred on the normal range unless told otherwise."""
    return sim.SimulatedOutcome(
        action=action, bis=bis, map=map, co=co, tol=tol, u_propo=0.1, u_remi=0.05, u_nore=1.0
    )


class FakeSimulator:
    """Returns prescribed futures, and records what it was asked for."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def simulate(self, inp, actions):
        self.calls.append({"input": inp, "actions": list(actions)})
        calibration = sim.Calibration(
            u_propo=0.1, u_remi=0.05, u_nore=1.0, bis=50.0, map=77.5,
            bis_residual=0.0, map_residual=0.0,
        )
        return calibration, list(self.outcomes)


def _futures(filler=None, **deviations):
    """A future per action: at the centre, moved by ``filler``, or by ``deviations``.

    ``_futures(filler={"bis": 100.0}, reduce_hypnotic={"map": 140.0})`` puts every
    action at the centre except the named one and the filler group.
    """
    filler = filler or {}
    outcomes = []
    for action in dataset.ACTIONS:
        moved = deviations.get(action, filler if action not in deviations else {})
        outcomes.append(
            _outcome(
                action,
                bis=moved.get("bis", 50.0),
                map=moved.get("map", 77.5),
                co=moved.get("co", 6.0),
                tol=moved.get("tol", 0.0),
            )
        )
    return outcomes


@pytest.fixture(scope="module")
def window():
    return dataset.load_split("test", limit=1)[0]


@pytest.fixture(scope="module")
def fast_simulator():
    """A short horizon: the rule does not depend on how far ahead we look."""
    return sim.AnesthesiaSimulator(horizon_seconds=60)


def _strategy(simulator=None, **options):
    return sim_normality.SimNormalityStrategy(simulator=simulator, **options)


# --------------------------------------------------------------------------- #
# Shape of the control
# --------------------------------------------------------------------------- #


def test_it_is_a_control_without_a_model():
    strategy = _strategy()

    assert isinstance(strategy, Strategy)
    assert strategy.name == NAME
    assert strategy.description.strip()
    assert strategy.uses_model is False


def test_it_is_not_a_harness_strategy():
    assert NAME not in harness_registry.EXPECTED_STRATEGIES
    assert Path(sim_normality.__file__).parent.name == "control"
    assert NAME not in baselines.BASELINE_NAMES  # too slow for the cheap sweep
    assert NAME in run_controls.OPT_IN_NAMES


def test_the_normal_table_is_the_documented_one():
    assert sim_normality.SIGNALS == ("bis", "map", "co", "tol")
    assert sim_normality.TARGETS == {"bis": 50.0, "map": 77.5, "co": 6.0, "tol": 0.0}
    assert sim_normality.TOLERANCES == {"bis": 10.0, "map": 12.5, "co": 2.0, "tol": 1.0}


def test_a_positive_tolerance_is_required():
    with pytest.raises(ValueError):
        _strategy(tolerances={"map": 0.0})


# --------------------------------------------------------------------------- #
# The score
# --------------------------------------------------------------------------- #


def test_a_future_at_the_centre_scores_zero():
    strategy = _strategy()

    candidate = strategy.candidate(_outcome("no_action"))

    assert candidate["score"] == 0.0
    assert all(value == 0.0 for value in candidate["deviations"].values())


def test_every_signal_is_one_dimension():
    """Being one tolerance away from the centre costs 1.0 in every signal."""
    strategy = _strategy()

    for signal, target, tolerance in (
        ("bis", 50.0, 10.0),
        ("map", 77.5, 12.5),
        ("co", 6.0, 2.0),
        ("tol", 0.0, 1.0),
    ):
        moved = _outcome("no_action", **{signal: target + tolerance})
        assert strategy.candidate(moved)["score"] == pytest.approx(1.0)


def test_the_score_is_the_euclidean_distance_in_tolerance_units():
    strategy = _strategy()

    one = strategy.candidate(_outcome("no_action", bis=30.0))  # two tolerances off
    two = strategy.candidate(_outcome("no_action", bis=60.0, tol=1.0))  # one each

    assert one["score"] == pytest.approx(2.0)
    assert two["score"] == pytest.approx(math.sqrt(2))


def test_the_score_is_in_tolerance_units():
    """Raw magnitudes do not matter: 40 mmHg is 3.2 tolerances and 40 BIS is 4.0."""
    strategy = _strategy()

    assert strategy.candidate(_outcome("no_action", map=117.5))["score"] == pytest.approx(40 / 12.5)
    assert strategy.candidate(_outcome("no_action", bis=90.0))["score"] == pytest.approx(40 / 10)


def test_the_breakdown_names_every_signal():
    strategy = _strategy()

    candidate = strategy.candidate(_outcome("no_action", bis=50.0, map=90.0, co=8.0, tol=0.5))

    assert set(candidate["signals"]) == set(sim_normality.SIGNALS)
    assert set(candidate["deviations"]) == set(sim_normality.SIGNALS)
    assert candidate["signals"]["map"] == 90.0
    assert candidate["deviations"]["map"] == pytest.approx(1.0)
    assert candidate["score"] == pytest.approx(
        math.sqrt(0.0**2 + 1.0**2 + 1.0**2 + 0.5**2)
    )


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


def test_it_picks_the_future_closest_to_the_centre(window):
    simulator = FakeSimulator(
        _futures(
            **{
                dataset.ACTIONS[0]: {"bis": 20.0},
                dataset.ACTIONS[1]: {"map": 140.0},
                dataset.ACTIONS[2]: {"bis": 45.0},  # the least far from the centre
                dataset.ACTIONS[3]: {"tol": 1.0},
                dataset.ACTIONS[4]: {"bis": 90.0},
            }
        )
    )
    strategy = _strategy(simulator=simulator)

    prediction = strategy.predict(window)

    assert prediction.action == dataset.ACTIONS[2]
    assert prediction.meta["chosen"] == dataset.ACTIONS[2]
    assert [candidate["action"] for candidate in prediction.meta["candidates"]] == list(
        dataset.ACTIONS
    )


def test_it_simulates_the_five_actions(window):
    simulator = FakeSimulator(_futures())
    strategy = _strategy(simulator=simulator)

    strategy.predict(window)

    assert len(simulator.calls) == 1
    assert simulator.calls[0]["actions"] == list(dataset.ACTIONS)
    assert simulator.calls[0]["input"] == window.input


def test_ties_go_to_the_first_action_of_the_dataset_order(window):
    simulator = FakeSimulator(_futures())
    strategy = _strategy(simulator=simulator)

    prediction = strategy.predict(window)

    assert prediction.action == dataset.ACTIONS[0]
    assert prediction.meta["tied"] == list(dataset.ACTIONS)
    assert prediction.meta["margin"] == 0.0


def test_the_margin_is_the_gap_to_the_runner_up(window):
    futures = _futures(
        **{
            dataset.ACTIONS[0]: {"bis": 60.0},  # 1.0
            dataset.ACTIONS[1]: {"bis": 70.0},  # 2.0
            dataset.ACTIONS[2]: {"bis": 80.0},  # 3.0
            dataset.ACTIONS[3]: {"bis": 90.0},  # 4.0
            dataset.ACTIONS[4]: {"bis": 50.0},  # 0.0
        }
    )
    strategy = _strategy(simulator=FakeSimulator(futures))

    prediction = strategy.predict(window)

    assert prediction.action == dataset.ACTIONS[4]
    assert prediction.meta["score"] == 0.0
    assert prediction.meta["margin"] == pytest.approx(1.0)


def test_a_custom_target_moves_the_winner(window):
    """Moving the TOL target to 1 turns the tolerating future into the good one."""
    futures = _futures(**{dataset.ACTIONS[0]: {"tol": 1.0}})
    default = _strategy(simulator=FakeSimulator(futures))
    custom = _strategy(simulator=FakeSimulator(futures), targets={"tol": 1.0})

    assert default.predict(window).action == dataset.ACTIONS[1]
    assert custom.predict(window).action == dataset.ACTIONS[0]
    assert custom.predict(window).meta["targets"]["tol"] == 1.0


def test_a_custom_tolerance_moves_the_winner(window):
    """The normalisation decides the winner: widening the map tolerance flips it."""
    futures = _futures(
        filler={"bis": 100.0},  # every other future is far from the centre (5.0)
        **{
            dataset.ACTIONS[0]: {"bis": 60.0},  # one BIS tolerance off -> 1.0
            dataset.ACTIONS[1]: {"map": 140.0},  # five MAP tolerances off -> 5.0
        }
    )
    strict = _strategy(simulator=FakeSimulator(futures))
    loose = _strategy(simulator=FakeSimulator(futures), tolerances={"map": 100.0})

    assert strict.predict(window).action == dataset.ACTIONS[0]
    assert loose.predict(window).action == dataset.ACTIONS[1]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def test_the_metadata_records_the_calibration_and_the_table(window, fast_simulator):
    strategy = _strategy(simulator=fast_simulator)

    meta = strategy.predict(window).meta

    assert meta["baseline"] == NAME
    assert meta["targets"] == sim_normality.TARGETS
    assert meta["tolerances"] == sim_normality.TOLERANCES
    assert set(meta["calibration"]) >= {"u_propo", "u_remi", "u_nore", "bis", "map"}
    assert len(meta["candidates"]) == len(dataset.ACTIONS)
    for candidate in meta["candidates"]:
        assert candidate["score"] >= 0.0
        assert set(candidate["signals"]) == set(sim_normality.SIGNALS)


def test_the_metadata_flags_an_imputed_baseline(window, fast_simulator):
    strategy = _strategy(simulator=fast_simulator)
    gone = dataclasses.replace(window, input={**window.input, "bis_current": None})

    assert strategy.predict(gone).meta["calibration"]["bis_source"] == "imputed"


def test_it_predicts_a_real_window(window, fast_simulator):
    prediction = _strategy(simulator=fast_simulator).predict(window)

    assert prediction.action in dataset.ACTIONS
    assert prediction.window_id == window.window_id


def test_the_launcher_runs_it(tmp_path, fast_simulator):
    strategy = _strategy(simulator=fast_simulator)

    outcome = harness_run.run(strategy, split="test", limit=2, results_root=tmp_path, quiet=True)

    run_dir = Path(outcome["out_dir"])
    assert run_dir.parent.name == NAME
    assert outcome["summary"]["n_windows"] == 2
    assert outcome["summary"]["errors"] == 0
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["model"] is None


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


def test_the_cheap_sweep_does_not_run_it():
    assert NAME not in run_controls.default_names()
    assert NAME in run_controls.OPT_IN_NAMES


def test_the_cli_can_select_it():
    args = run_controls.build_parser().parse_args(["--only", NAME, "--workers", "8"])
    assert run_controls.selected_names(args.only) == [NAME]
    assert args.workers == 8


def test_the_cli_can_run_all_of_them():
    names = run_controls.selected_names("all")
    assert set(names) >= set(run_controls.OPT_IN_NAMES)
    assert set(names) >= set(baselines.BASELINE_NAMES)
