# -*- coding: utf-8 -*-
"""The model probe: which models answer, and which take the ``reasoning_effort`` parameter."""

import csv

import pytest

from src import model_probe


class FakePage:
    def __init__(self, ids):
        self.data = [type("M", (), {"id": model_id})() for model_id in ids]


class FakeModels:
    def __init__(self, ids=None, error=None):
        self._ids = ids or []
        self._error = error

    def list(self):
        if self._error:
            raise self._error
        return FakePage(self._ids)


class FakeResponse:
    def __init__(self, text="ok", model="m", dropped=None):
        self.text = text
        self.model = model
        self.latency_s = 0.5
        self.dropped_params = list(dropped or [])


class FakeClient:
    """A client that answers, or that rejects the effort parameter like a real model does."""

    def __init__(self, settings=None, ids=None, error=None, rejects_effort=False,
                 fails=None):
        self.settings = settings
        self.client = type("C", (), {"models": FakeModels(ids, error)})()
        self._rejects_effort = rejects_effort
        self._fails = fails
        self.calls = []

    def complete_messages(self, messages, **kwargs):
        self.calls.append(kwargs)
        if self._fails:
            raise self._fails
        dropped = []
        if kwargs.get("reasoning_effort") and self._rejects_effort:
            # ``LLMResponse.dropped_params`` holds parameter names, not pairs.
            dropped = ["reasoning_effort"]
        return FakeResponse(model=kwargs.get("model"), dropped=dropped)


def factory(**kwargs):
    """A client factory that records the kwargs it was built with."""
    holder = {"clients": []}

    def build(settings=None, client=None):
        instance = FakeClient(settings=settings, **kwargs)
        holder["clients"].append(instance)
        return instance

    build.holder = holder
    return build


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


def test_the_listing_returns_the_ids():
    ids, error = model_probe.list_models(client_factory=factory(ids=["a", "b"]))

    assert ids == ["a", "b"]
    assert error is None


def test_an_endpoint_that_refuses_to_list_is_reported_not_raised():
    ids, error = model_probe.list_models(
        client_factory=factory(error=RuntimeError("401 Incorrect API key"))
    )

    assert ids is None
    assert "401" in error


# --------------------------------------------------------------------------- #
# Probing one model
# --------------------------------------------------------------------------- #


def test_a_model_that_answers_is_ok():
    row = model_probe.probe_model("m", client_factory=factory())

    assert row["ok"] is True
    assert row["model"] == "m"
    assert row["effort"] == "high"
    assert row["dropped"] == ""
    assert row["answered"] == "ok"
    assert row["error"] == ""


def test_a_model_that_rejects_the_effort_is_flagged():
    """The client drops the parameter when the model rejects it; the probe reports that."""
    row = model_probe.probe_model("m", client_factory=factory(rejects_effort=True))

    assert row["ok"] is True
    assert row["dropped"] == "reasoning_effort"


def test_omitting_the_effort_is_recorded_as_such():
    row = model_probe.probe_model("m", effort=None, client_factory=factory())

    assert row["effort"] == "(omitted)"


def test_a_model_that_never_answers_reports_the_error():
    row = model_probe.probe_model("m", client_factory=factory(fails=RuntimeError("boom")))

    assert row["ok"] is False
    assert "boom" in row["error"]


def test_the_key_variable_is_recorded_not_the_key():
    row = model_probe.probe_model("m", key_env="CLAUDE_API_KEY", client_factory=factory())

    assert row["key_env"] == "CLAUDE_API_KEY"
    assert "sk-" not in ",".join(str(value) for value in row.values())


# --------------------------------------------------------------------------- #
# The whole report
# --------------------------------------------------------------------------- #


def test_every_model_is_tried_with_and_without_the_effort():
    rows = model_probe.probe(["a", "b"], client_factory=factory())

    assert len(rows) == 4
    assert [row["model"] for row in rows] == ["a", "a", "b", "b"]
    assert [row["effort"] for row in rows[:2]] == ["high", "(omitted)"]


def test_the_report_is_written(tmp_path):
    out_dir, rows, problem = model_probe.run(
        models=["a"], key_env="CLAUDE_API_KEY", out_dir=tmp_path, client_factory=factory()
    )

    assert problem is None
    assert len(rows) == 2
    path = out_dir / "probe_CLAUDE_API_KEY.csv"
    written = list(csv.DictReader(path.open(encoding="utf-8")))
    assert list(written[0]) == model_probe.COLUMNS
    assert len(written) == 2


def test_without_models_it_asks_the_endpoint(tmp_path):
    _, rows, problem = model_probe.run(
        out_dir=tmp_path, client_factory=factory(ids=["only-one"])
    )

    assert problem is None
    assert {row["model"] for row in rows} == {"only-one"}


def test_an_endpoint_that_cannot_be_asked_is_a_clean_failure(tmp_path):
    _, rows, problem = model_probe.run(
        out_dir=tmp_path, client_factory=factory(error=RuntimeError("nope"))
    )

    assert rows == []
    assert "nope" in problem


def test_the_cli_defaults_to_the_project_key_variable():
    args = model_probe.build_parser().parse_args([])

    assert args.api_key_env is None
    assert args.list_only is False
    assert args.effort == "high"
