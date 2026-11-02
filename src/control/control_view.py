# -*- coding: utf-8 -*-
"""The numeric features the machine-learning controls read, on top of the shared view.

The shared view (``src.view``, view 4) still carries the effect-site concentration *at* the
decision point. For remifentanil the decision point falls at or seconds after the pump's
target change, so the t0 sample of the Ce series (and every Ce feature computed with it:
current value, short/long slope, deltas, trend code) already moves with the action. Measured on
the test split (2026-11-01), XGBoost on view 4 reaches 0.748; recomputing the Ce features
without the t0 sample drops it to 0.629, and every further 30 s cut costs only 1.5-2 points
(-60 s 0.614, -90 s 0.595, -120 s 0.548): the jump is the leak, the slope after it is legitimate
information that ages. The language models do not exploit that sample (on the 567 test windows
only the t0 sample makes answerable, the 50 stored LLM runs average 0.190 against 0.237 on the
windows both trees miss, and no run gains more than +0.042 there, against +0.35 for the tree),
so the LLM runs keep view 4; the tree controls do not get that benefit of the doubt, because
they read the sample directly.

This module therefore gives the ML controls (XGBoost, ``support``, the failure predictor, the
SHAP analysis behind ``shap_prompt``) a *control view*:

- every effect-site feature (``ce_*``, ``propofol_ce_*``, ``remi_ce_*``) recomputed from the raw
  Ce series using only samples at ``t <= CE_CUT_SEC`` (same names, so the SHAP ranking still
  names fields the prompt shows);
- the target of the most recent *past* infusion adjustment of each drug
  (``prior_target_*``): what the anaesthetist reads on the pump before deciding, given
  explicitly instead of being inferred from the Ce trajectory. The prompt already shows it in
  prose ("propofol target 3.5"), so the controls and the models see the same information.
  Its *age* is deliberately left out: the no_action windows were sampled away from recent
  adjustments, so "seconds since the last adjustment" mostly encodes how the benchmark was
  built (it became the top SHAP feature when tried, 2026-11-01), not the patient.

Bump :data:`CONTROL_VIEW_VERSION` whenever the cut or the derived features change; trained
models record it and stale ones are refused.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CE_CUT_SEC",
    "CONTROL_VIEW_VERSION",
    "CE_PREFIXES",
    "PRIOR_TARGET_NAMES",
    "is_ce_feature",
    "ce_features",
    "prior_targets",
    "control_features",
    "describe",
]

#: Last Ce sample the controls may read, in seconds relative to the decision point.
CE_CUT_SEC = -30

#: Version of the control view. Bump on any change to the cut or the derived features.
CONTROL_VIEW_VERSION = 2

#: Effect-site features recomputed from the cut series.
CE_PREFIXES = ("ce_", "propofol_ce_", "remi_ce_")

#: Window of the short slope, counted back from the last kept sample (seconds).
SHORT_WINDOW_SEC = 120

#: |slope| (per minute) under which the trend code is 0.
TREND_THRESHOLD = 0.05

#: Raw series per drug, the scalar effect-site field and the prefix of its trend family.
_DRUGS = (
    ("propofol", "propofol_Ce", "ce_propofol", "propofol_ce_", "propofol"),
    ("remi", "remi_Ce", "ce_remi", "remi_ce_", "remifentanil"),
)

#: Explicit pre-decision targets, per drug (their age is left out on purpose, see above).
PRIOR_TARGET_NAMES = (
    "prior_target_propofol",
    "prior_target_remi",
)


def is_ce_feature(name):
    """Whether a feature is one of the effect-site features the control view recomputes."""
    return str(name).startswith(CE_PREFIXES)


def _series(raw_input, key, cut):
    """``[(t_offset_sec, value)]`` of one raw concentration series up to ``cut``, sorted."""
    series = ((raw_input or {}).get("drug_concentrations_json") or {}).get(key) or []
    points = []
    for item in series:
        if not isinstance(item, dict):
            continue
        t, v = item.get("t_offset_sec"), item.get("value")
        if isinstance(t, bool) or isinstance(v, bool):
            continue
        if isinstance(t, (int, float)) and isinstance(v, (int, float)) and t <= cut:
            if math.isfinite(t) and math.isfinite(v):
                points.append((float(t), float(v)))
    return sorted(points)


def _slope_per_min(points):
    if len(points) < 2:
        return math.nan
    t = np.array([p[0] for p in points]) / 60.0
    v = np.array([p[1] for p in points])
    if np.ptp(t) == 0:
        return math.nan
    return float(np.polyfit(t, v, 1)[0])


def ce_features(raw_input, cut=CE_CUT_SEC):
    """The effect-site features of one window, from the raw Ce series up to ``cut`` seconds."""
    out = {}
    for _, key, scalar, prefix, _ in _DRUGS:
        points = _series(raw_input, key, cut)
        if not points:
            values = dict.fromkeys(("current", "slope_short", "slope_long", "delta_abs",
                                    "delta_pct", "trend_code", "n_pts_short"), math.nan)
        else:
            current, first = points[-1][1], points[0][1]
            short = [p for p in points if p[0] >= points[-1][0] - SHORT_WINDOW_SEC]
            slope_short = _slope_per_min(short)
            values = {
                "current": current,
                "slope_short": slope_short,
                "slope_long": _slope_per_min(points),
                "delta_abs": current - first,
                "delta_pct": (current - first) / first if first else math.nan,  # a fraction, as in the dataset
                "trend_code": (float(np.sign(slope_short))
                               if math.isfinite(slope_short) and abs(slope_short) > TREND_THRESHOLD
                               else 0.0),
                "n_pts_short": float(len(short)),
            }
        out[scalar] = values["current"]
        for name, value in values.items():
            out[prefix + name] = value
    return out


def prior_targets(inp):
    """Target set by the most recent past adjustment of each drug (NaN when there is none)."""
    history = [e for e in ((inp or {}).get("actions_history_json") or [])
               if isinstance(e, dict) and isinstance(e.get("t_offset_sec"), (int, float))
               and not isinstance(e.get("t_offset_sec"), bool) and e["t_offset_sec"] <= 0]
    out = {}
    for short, _, _, _, drug in _DRUGS:
        events = [e for e in history if str(e.get("drug", "")).strip().lower() == drug]
        last = max(events, key=lambda e: e["t_offset_sec"]) if events else None
        target = last.get("new_target") if last else None
        out[f"prior_target_{short}"] = (float(target) if isinstance(target, (int, float))
                                        and not isinstance(target, bool) else math.nan)
    return out


def control_features(window):
    """Overrides of the control view for one window: recomputed Ce + explicit prior targets.

    Reads ``window.raw_input`` for the Ce series only (the samples after the cut never leave
    this function) and ``window.input`` (the shared view) for the action history.
    """
    raw = getattr(window, "raw_input", None) or {}
    out = ce_features(raw)
    out.update(prior_targets(getattr(window, "input", None)))
    return out


def describe():
    """What a trained model records about the control view it was fitted on."""
    return {"control_view_version": CONTROL_VIEW_VERSION, "ce_cut_sec": CE_CUT_SEC,
            "prior_targets": True}
