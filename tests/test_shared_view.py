# -*- coding: utf-8 -*-
"""The shared view: every strategy and every control sees the same window.

``src.view`` decides which fields of a window record a model may see; ``dataset`` applies it
on load, ``prompting.render_case`` renders all of it, and the retrieval index and the
gradient-boosting control read its numeric part. These tests pin that contract down:

- the hidden set holds what the reviewer excluded, and nothing hidden reaches a window;
- no time-series sample after the decision point survives;
- ``render_case`` renders every visible field that has a value;
- every language-model strategy sends that same rendering;
- the machine-learning controls read only fields of the view.
"""

import json
import re

import pytest

from src import dataset, harness_registry, llm, prompting, view
from harness.few_shot import retrieval
from src.control import ml_data

#: What the reviewer asked to hide (2026-10-09), on top of the future/outcome fields.
REVIEWER_HIDDEN = {
    "case_id", "case_id_fill", "causal_depth_json", "decision_time_abs", "exp_sevo", "mac",
    "remi_causal_flag", "schema_version", "staleness_json", "staleness_max", "depth_score",
    "pt_adm", "pt_anestart", "pt_opstart", "pt_casestart", "window_id",
    "drug_concentrations_json",
}

#: Future, outcome and identifier fields.
NEVER_VISIBLE = {
    "pt_caseid", "pt_subjectid", "pt_caseend", "pt_opend", "pt_aneend", "surgery_pct",
    "pt_dis", "pt_icu_days", "pt_death_inhosp",
}


@pytest.fixture(scope="module")
def windows():
    return dataset.load_split("test", limit=300)


# --------------------------------------------------------------------------- #
# The view itself
# --------------------------------------------------------------------------- #


def test_the_hidden_set_holds_everything_the_reviewer_excluded():
    for name in REVIEWER_HIDDEN | NEVER_VISIBLE:
        assert view.is_hidden(name), name


def test_counters_totals_plasma_concentrations_and_targets_are_hidden_by_prefix():
    for name in ("n_events", "n_actions_hist", "n_drug_points", "n_stale_9999",
                 "pt_intraop_ppf", "pt_intraop_ftn", "pt_intraop_ebl",
                 "cp_remi", "cp_propofol", "ct_propofol"):
        assert view.is_hidden(name), name


def test_clinical_fields_stay_visible():
    for name in ("ce_remi", "remi_ce_slope_short", "MAP_mean",
                 "bis_current", "lab_hb", "pt_age", "vitals_json", "events_json",
                 "actions_history_json", "pkpd_residuals_json", "history_summary"):
        assert not view.is_hidden(name), name


def test_no_hidden_field_reaches_a_window(windows):
    for window in windows:
        assert view.hidden_in(window.input) == []
        assert set(view.hidden_in(window.raw_input)) >= {"pt_caseid", "surgery_pct"}


def test_no_sample_after_the_decision_point_survives(windows):
    for window in windows:
        for field in ("vitals_json", "events_json", "actions_history_json"):
            value = window.input.get(field)
            series = value.values() if isinstance(value, dict) else [value or []]
            for points in series:
                for point in points if isinstance(points, list) else []:
                    if isinstance(point, dict) and "t_offset_sec" in point:
                        assert point["t_offset_sec"] <= 0, (window.window_id, field)


def test_the_view_drops_future_samples_but_keeps_the_rest():
    raw = {"vitals_json": {"HR": [{"value": 80, "t_offset_sec": 1},
                                  {"value": 78, "t_offset_sec": 0},
                                  {"value": 75, "t_offset_sec": -30}]},
           "surgery_pct": 0.4, "hr_current": 78}
    shown = view.visible_input(raw)
    assert shown == {"vitals_json": {"HR": [{"value": 78, "t_offset_sec": 0},
                                            {"value": 75, "t_offset_sec": -30}]},
                     "hr_current": 78}


# --------------------------------------------------------------------------- #
# The rendering
# --------------------------------------------------------------------------- #

#: The record block names a field by its code, except these grouped ones.
_GROUPED = {"vitals_json": "Vital-sign samples", "events_json": "Clinical events",
            "actions_history_json": "Infusion adjustments", "pkpd_residuals_json": "PK/PD model fit"}


def test_render_case_renders_every_visible_field_with_a_value(windows):
    for window in windows:
        text, prose = prompting._render(window.input)
        for field in prompting.rendered_fields(window.input):
            if field in prose:
                continue
            if field in _GROUPED:
                assert _GROUPED[field] in text, (window.window_id, field)
            elif field.startswith("lab_"):
                test = field[4:-4] if field.endswith("_age") else field[4:]
                assert re.search(rf"\b{re.escape(test)}\b", text), (window.window_id, field)
            else:
                assert re.search(rf"\b{re.escape(field)} ", text), (window.window_id, field)


def test_render_case_never_names_a_hidden_field(windows):
    coded = sorted(name for name in view.HIDDEN_FIELDS if "_" in name)
    for window in windows[:100]:
        text = prompting.render_case(window.raw_input)  # even fed the raw record
        for name in coded:
            assert not re.search(rf"\b{re.escape(name)}\b", text), (window.window_id, name)
        assert not re.search(r"\bn_[a-z]", text)
        assert "pt_intraop_" not in text


def test_render_case_is_the_same_for_the_raw_record_and_the_view(windows):
    for window in windows[:50]:
        assert prompting.render_case(window.raw_input) == prompting.render_case(window.input)


# --------------------------------------------------------------------------- #
# Every language-model strategy sends the same rendering
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.model = "fake"
        self.usage = {}
        self.latency_s = 0.0
        self.dropped_params = []


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(messages, **kwargs):
        calls.append(messages)
        return _FakeResponse("Candidate 1: a\nCandidate 2: c\nAnswer: 1\nb")

    monkeypatch.setattr(llm, "complete_messages", fake)
    return calls


@pytest.fixture(scope="module")
def ranking_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("ranking") / "shap_importance.csv"
    path.write_text("feature,mean_abs_shap,rank\nremi_ce_current,0.5,1\nce_remi,0.4,2\n", encoding="utf-8")
    return path


def _options(name, ranking_file, tmp_path):
    if name == "few_shot":
        return {"index_cases": 2, "index_path": tmp_path / "index.npz", "k": 2}
    if name == "shap_prompt":
        return {"importance_path": ranking_file}
    if name == "self_consistency":
        return {"n_samples": 2}
    return {}


#: Strategies that call no model (they reuse stored answers).
_NO_MODEL = {"support"}


@pytest.mark.parametrize("name", sorted(set(harness_registry.available_strategies()) - _NO_MODEL))
def test_every_language_model_strategy_sends_the_shared_rendering(name, captured, ranking_file,
                                                                   tmp_path):
    window = dataset.load_split("test", limit=1)[0]
    case = prompting.render_case(window.input)

    harness_registry.load_strategy(name, **_options(name, ranking_file, tmp_path)).predict(window)

    assert captured, f"{name} called no model"
    first_user = [m["content"] for m in captured[0] if m["role"] == "user"][-1]
    if name == "structured_json":
        assert json.loads(first_user)["patient_state"] == case
    else:
        assert case in first_user, name


# --------------------------------------------------------------------------- #
# The machine-learning controls read the same view
# --------------------------------------------------------------------------- #


def test_the_feature_set_is_the_numeric_part_of_the_view(windows):
    names = retrieval.feature_names(windows)
    assert view.hidden_in(names) == []
    visible_numeric = {
        k for w in windows for k, v in w.input.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert set(names) == visible_numeric


def test_the_ml_controls_never_read_a_hidden_field(tmp_path):
    names = ml_data.feature_names(index_path=tmp_path / "absent.npz")
    assert names and view.hidden_in(names) == []


def test_a_model_fitted_on_another_view_is_refused(tmp_path):
    from src.control import xgboost_baseline

    with pytest.raises(ValueError, match="view"):
        xgboost_baseline.check_view({"names": ["ce_remi", "surgery_pct"], "view_version": None},
                                    tmp_path / "old.json")
    xgboost_baseline.check_view({"names": ["ce_remi"], "view_version": view.VIEW_VERSION},
                                tmp_path / "new.json")
