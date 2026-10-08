# -*- coding: utf-8 -*-
"""The harness launcher (``harness/run.py``)."""

import json
from pathlib import Path

import pytest

from harness import run as harness_run
from src import dataset, llm, results


class FakeResponse:
    """Stand-in for ``src.llm.LLMResponse``."""

    def __init__(self, text="b"):
        self.text = text
        self.model = "gpt-5.4-mini"
        self.usage = {}
        self.latency_s = 0.0
        self.dropped_params = []


def _patch_llm(monkeypatch, text="b"):
    """Replace the model call; returns the list of captured message lists."""
    calls = []

    def fake_complete_messages(messages, **kwargs):
        calls.append(messages)
        return FakeResponse(text)

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    return calls


def _capture_llm_kwargs(monkeypatch, text="b"):
    """Replace the model call; returns the last keyword arguments it received."""
    captured = {}

    def fake_complete_messages(messages, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return FakeResponse(text)

    monkeypatch.setattr(llm, "complete_messages", fake_complete_messages)
    return captured


def test_select_windows_by_cases_and_limit():
    windows = harness_run.select_windows("dev", cases=2)
    assert windows
    expected_cases = {p.stem.replace("case", "") for p in dataset.case_files("dev")[:2]}
    assert {w.case_id for w in windows} == expected_cases

    limited = harness_run.select_windows("dev", cases=2, limit=3)
    assert len(limited) == 3
    assert [w.window_id for w in limited] == [w.window_id for w in windows[:3]]


def test_build_summary_reports_the_three_agreements():
    """The summary carries the real action, `result_1` and the plausible answer."""
    predictions = [
        {"action": "a", "real_action": "a", "result_1": "a", "result_aux": "z"},
        {"action": "b", "real_action": "c", "result_1": "d", "result_aux": "b"},
        {"action": "e", "real_action": "e", "result_1": "f", "result_aux": "g"},
    ]

    summary = harness_run.build_summary(predictions, strategy="fake", split="dev")

    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["agreement_result_1"] == pytest.approx(1 / 3)
    assert summary["agreement_result_plausible"] == pytest.approx(2 / 3)


def test_run_writes_predictions_summary_and_run_info(tmp_path, monkeypatch):
    calls = _patch_llm(monkeypatch, "b")

    outcome = harness_run.run(
        "zero_shot", "dev", cases=1, limit=2, results_root=tmp_path, quiet=True
    )

    run_dirs = list((tmp_path / "zero_shot").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir.name.endswith("_dev")
    assert len(calls) == 2

    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    predictions = [json.loads(line) for line in lines]
    assert len(predictions) == 2
    assert all(p["action"] == "reduce_hypnotic" for p in predictions)
    assert all(p["real_action"] in dataset.ACTIONS for p in predictions)

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["strategy"] == "zero_shot"
    assert summary["split"] == "dev"
    assert summary["n_windows"] == 2
    assert summary["n_scored"] == 2
    assert 0.0 <= summary["accuracy"] <= 1.0

    run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["cases"] == 1
    assert run_info["limit"] == 2
    assert run_info["workers"] == 1
    assert run_info["dry_run"] is False
    assert outcome["out_dir"] == str(run_dir)


def test_parallel_run_preserves_the_window_order(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")

    outcome = harness_run.run(
        "zero_shot", "dev", cases=1, limit=6, workers=4, results_root=tmp_path, quiet=True
    )

    windows = harness_run.select_windows("dev", cases=1, limit=6)
    assert [p["window_id"] for p in outcome["predictions"]] == [w.window_id for w in windows]
    assert outcome["summary"]["n_windows"] == 6
    run_dir = Path(outcome["out_dir"])
    run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["workers"] == 4


def test_parallel_matches_serial_results(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "c")

    serial = harness_run.run(
        "zero_shot", "dev", cases=1, limit=4, workers=1,
        results_root=tmp_path / "serial", quiet=True,
    )
    parallel = harness_run.run(
        "zero_shot", "dev", cases=1, limit=4, workers=4,
        results_root=tmp_path / "parallel", quiet=True,
    )

    assert [p["action"] for p in serial["predictions"]] == [
        p["action"] for p in parallel["predictions"]
    ]
    assert serial["summary"]["accuracy"] == parallel["summary"]["accuracy"]


def test_failing_windows_are_recorded_as_errors(tmp_path, monkeypatch):
    def boom(messages, **kwargs):
        raise RuntimeError("api exploded")

    monkeypatch.setattr(llm, "complete_messages", boom)

    outcome = harness_run.run(
        "zero_shot", "dev", cases=1, limit=3, workers=2, results_root=tmp_path, quiet=True
    )

    assert outcome["summary"]["errors"] == 3
    assert outcome["summary"]["n_scored"] == 0
    assert "accuracy" not in outcome["summary"]
    assert all("api exploded" in p["meta"]["error"] for p in outcome["predictions"])


def test_dry_run_does_not_call_the_model(tmp_path, monkeypatch):
    calls = _patch_llm(monkeypatch)

    outcome = harness_run.run(
        "zero_shot", "dev", cases=1, limit=1, results_root=tmp_path, dry_run=True, quiet=True
    )

    assert calls == []
    assert outcome["summary"]["unparsed"] == 0
    prediction = json.loads(
        (Path(outcome["out_dir"]) / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["action"] is None
    assert prediction["real_action"] in dataset.ACTIONS


def test_cli_passes_the_workers_flag(tmp_path, monkeypatch):
    """Regression: the CLI must forward ``--workers`` to ``run``."""
    _patch_llm(monkeypatch, "b")

    exit_code = harness_run.main(
        [
            "--strategy", "zero_shot",
            "--split", "dev",
            "--cases", "1",
            "--limit", "2",
            "--workers", "3",
            "--results-root", str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    run_dir = next((tmp_path / "zero_shot").iterdir())
    run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["workers"] == 3
    assert run_info["n_windows"] == 2


def test_cli_passes_the_effort_flag(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")

    exit_code = harness_run.main(
        [
            "--strategy", "zero_shot",
            "--split", "dev",
            "--cases", "1",
            "--limit", "1",
            "--effort", "low",
            "--results-root", str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    run_dir = next((tmp_path / "zero_shot").iterdir())
    run_info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_info["effort"] == "low"
    prediction = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip())
    assert prediction["meta"]["reasoning_effort"] == "low"


def test_effort_defaults_to_high(tmp_path, monkeypatch):
    captured = _capture_llm_kwargs(monkeypatch)

    harness_run.run("zero_shot", "dev", cases=1, limit=1, results_root=tmp_path, quiet=True)

    assert captured["reasoning_effort"] == "high"


def test_effort_none_omits_the_parameter(tmp_path, monkeypatch):
    captured = _capture_llm_kwargs(monkeypatch)

    harness_run.run(
        "zero_shot", "dev", cases=1, limit=1, effort="none", results_root=tmp_path, quiet=True
    )

    assert captured["reasoning_effort"] is None


def test_unknown_strategy_raises(tmp_path):
    with pytest.raises(KeyError):
        harness_run.run("does_not_exist", "dev", limit=1, results_root=tmp_path, quiet=True)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prediction,expected",
    [
        (None, True),
        ({"action": "no_action", "meta": {}}, False),
        ({"action": None, "meta": {"error": "LLMError: model call failed"}}, True),
        ({"action": None, "meta": {}}, True),
        ({"action": None, "meta": {"dry_run": True}}, False),
    ],
)
def test_is_failed(prediction, expected):
    assert harness_run.is_failed(prediction) is expected


def test_latest_run_dir_ignores_underscore_folders(tmp_path):
    root = tmp_path / "zero_shot"
    (root / "_index").mkdir(parents=True)
    (root / "20260101-000000_dev").mkdir()
    (root / "20260102-000000_dev").mkdir()
    (root / "20260103-000000_test").mkdir()

    assert harness_run.latest_run_dir("zero_shot", "dev", tmp_path).name == "20260102-000000_dev"
    with pytest.raises(SystemExit):
        harness_run.latest_run_dir("zero_shot", "train", tmp_path)


def test_resume_rejects_a_missing_directory(tmp_path):
    with pytest.raises(SystemExit):
        harness_run.resolve_run_dir(str(tmp_path / "nope"), "zero_shot", "dev", tmp_path)


def test_resume_recomputes_only_the_failed_windows(tmp_path, monkeypatch):
    def boom(messages, **kwargs):
        raise RuntimeError("api exploded")

    monkeypatch.setattr(llm, "complete_messages", boom)
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=4, workers=2, results_root=tmp_path, quiet=True
    )
    assert first["summary"]["errors"] == 4
    assert first["summary"]["n_scored"] == 0

    calls = _patch_llm(monkeypatch, "b")  # the model works again
    second = harness_run.run(
        "zero_shot", "dev", workers=2, results_root=tmp_path, quiet=True,
        resume=first["out_dir"],
    )

    assert len(calls) == 4  # only the failed windows
    assert second["out_dir"] == first["out_dir"]
    assert second["summary"]["errors"] == 0
    assert second["summary"]["n_scored"] == 4
    assert all(row["action"] == "reduce_hypnotic" for row in second["predictions"])
    assert all("error" not in (row["meta"] or {}) for row in second["predictions"])

    run_info = json.loads(
        (Path(second["out_dir"]) / "run.json").read_text(encoding="utf-8")
    )
    assert run_info["resumed"] is True
    assert run_info["n_computed_now"] == 4
    assert run_info["n_failed"] == 0


def test_resume_keeps_the_successful_windows(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=3, results_root=tmp_path, quiet=True
    )

    calls = _patch_llm(monkeypatch, "c")
    second = harness_run.run("zero_shot", "dev", results_root=tmp_path, quiet=True, resume="latest")

    assert calls == []  # nothing to recompute
    assert [row["action"] for row in second["predictions"]] == ["reduce_hypnotic"] * 3
    assert second["summary"]["accuracy"] == first["summary"]["accuracy"]


def test_resume_reuses_the_saved_selection(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=2, results_root=tmp_path, quiet=True
    )

    calls = _patch_llm(monkeypatch, "c")
    second = harness_run.run(
        "zero_shot", "dev", results_root=tmp_path, quiet=True, resume=first["out_dir"]
    )

    assert calls == []
    assert [row["window_id"] for row in second["predictions"]] == [
        row["window_id"] for row in first["predictions"]
    ]


def test_resume_reuses_the_stored_generation_settings(tmp_path, monkeypatch):
    """Mixing two efforts inside one run would invalidate it."""

    def boom(messages, **kwargs):
        raise RuntimeError("api exploded")

    monkeypatch.setattr(llm, "complete_messages", boom)
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=2, effort="low", workers=3,
        results_root=tmp_path, quiet=True,
    )

    captured = _capture_llm_kwargs(monkeypatch, "b")
    second = harness_run.run(
        "zero_shot", "dev", results_root=tmp_path, quiet=True, resume=first["out_dir"]
    )

    assert captured["reasoning_effort"] == "low"
    run_info = json.loads((Path(second["out_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert run_info["effort"] == "low"
    assert run_info["workers"] == 3


def test_resume_reuses_a_disabled_reasoning_effort(tmp_path, monkeypatch):
    """``--effort none`` is stored as null and must stay null, not fall back to high."""

    def boom(messages, **kwargs):
        raise RuntimeError("api exploded")

    monkeypatch.setattr(llm, "complete_messages", boom)
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=1, effort="none", results_root=tmp_path, quiet=True
    )

    captured = _capture_llm_kwargs(monkeypatch, "b")
    harness_run.run("zero_shot", "dev", results_root=tmp_path, quiet=True, resume=first["out_dir"])

    assert captured["reasoning_effort"] is None


def test_resume_backs_up_the_predictions_once(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")
    first = harness_run.run(
        "zero_shot", "dev", cases=1, limit=2, results_root=tmp_path, quiet=True
    )
    run_dir = Path(first["out_dir"])
    original = (run_dir / "predictions.jsonl").read_text(encoding="utf-8")

    harness_run.run("zero_shot", "dev", results_root=tmp_path, quiet=True, resume=first["out_dir"])
    backup = run_dir / "predictions.jsonl.bak"
    assert backup.read_text(encoding="utf-8") == original

    # A later resume must not overwrite the original backup.
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    harness_run.run("zero_shot", "dev", results_root=tmp_path, quiet=True, resume=first["out_dir"])
    assert backup.read_text(encoding="utf-8") == original


def test_summary_is_rebuilt_from_the_predictions_file(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, "b")
    outcome = harness_run.run(
        "zero_shot", "dev", cases=1, limit=3, results_root=tmp_path, quiet=True
    )
    run_dir = Path(outcome["out_dir"])
    assert outcome["summary"]["accuracy"] <= 1.0

    # Hand-edit the file: every prediction now matches the real action.
    predictions_file = run_dir / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["action"] = row["real_action"]
    results.write_predictions(predictions_file, rows)

    summary = harness_run.refresh_summary(run_dir, "zero_shot", "dev")

    assert summary["accuracy"] == 1.0
    assert summary["n_windows"] == 3
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == summary
