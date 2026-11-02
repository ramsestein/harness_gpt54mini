# -*- coding: utf-8 -*-
"""The control view: effect-site features without the samples that can reflect the decision."""

import math

import numpy as np

from src import dataset
from src.control import control_view, ml_data


def _raw(ce_remi, ce_prop=None):
    """A raw input whose Ce series are given as {t_offset_sec: value}."""
    def series(d):
        return [{"t_offset_sec": t, "value": v} for t, v in sorted((d or {}).items())]
    return {"drug_concentrations_json": {"remi_Ce": series(ce_remi), "propofol_Ce": series(ce_prop)}}


class FakeWindow:
    def __init__(self, inp, raw):
        self.input = inp
        self.raw_input = raw


def test_the_t0_sample_never_reaches_the_features():
    # the remifentanil Ce jumps only at t = 0 (the action): the cut features must not see it
    raw = _raw({-120: 2.0, -90: 2.0, -60: 2.0, -30: 2.0, 0: 3.5})

    feats = control_view.ce_features(raw)

    assert feats["ce_remi"] == 2.0 and feats["remi_ce_current"] == 2.0
    assert abs(feats["remi_ce_slope_short"]) < 1e-9
    assert feats["remi_ce_trend_code"] == 0.0
    assert feats["remi_ce_n_pts_short"] == 4.0
    # with no cut the jump is there: that is the leak the control view removes
    assert control_view.ce_features(raw, cut=0)["ce_remi"] == 3.5


def test_a_drug_without_kept_samples_is_missing_not_zero():
    feats = control_view.ce_features(_raw({0: 3.0}))

    assert math.isnan(feats["ce_remi"]) and math.isnan(feats["remi_ce_slope_short"])
    assert math.isnan(feats["ce_propofol"])


def test_the_prior_target_is_the_last_past_adjustment_of_each_drug():
    inp = {"actions_history_json": [
        {"t_offset_sec": -600, "drug": "remifentanil", "new_target": 4.0},
        {"t_offset_sec": -240, "drug": "remifentanil", "new_target": 5.5},
        {"t_offset_sec": -90, "drug": "propofol", "new_target": 3.0},
    ]}

    out = control_view.prior_targets(inp)

    assert out["prior_target_remi"] == 5.5
    assert out["prior_target_propofol"] == 3.0
    assert set(out) == set(control_view.PRIOR_TARGET_NAMES)  # no age: it encodes the sampling


def test_the_matrix_reads_the_control_view_over_the_shared_view():
    window = FakeWindow({"ce_remi": 3.5, "hr_current": 70.0},
                        _raw({-60: 2.0, -30: 2.0, 0: 3.5}))

    X = ml_data.matrix([window], ["ce_remi", "hr_current", "prior_target_remi"])

    assert X[0, 0] == 2.0          # recomputed without the t0 sample
    assert X[0, 1] == 70.0         # untouched shared-view field
    assert np.isnan(X[0, 2])       # no past remifentanil adjustment


def test_with_no_cut_the_recomputation_matches_the_dataset_fields():
    for window in dataset.load_split("dev", limit=20):
        feats = control_view.ce_features(window.raw_input, cut=0)
        # the trend family is computed from the series; the scalar ce_* comes from its own track
        for name in ("remi_ce_current", "remi_ce_slope_short", "remi_ce_delta_abs", "remi_ce_delta_pct",
                     "propofol_ce_slope_long", "propofol_ce_trend_code"):
            given = window.input.get(name)
            if isinstance(given, (int, float)) and math.isfinite(feats[name]):
                assert abs(given - feats[name]) <= 2e-3 * max(1.0, abs(given)), name
