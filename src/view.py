# -*- coding: utf-8 -*-
"""The one view of a decision window that every strategy and every control sees.

A window record carries 280 input fields. Some of them must never reach a model:

- identifiers and pipeline bookkeeping (case/window ids, schema version, point counters,
  signal staleness, the absolute session clock and the case timestamps);
- information from the future or from the outcome (case/operation/anaesthesia end, the
  fraction of surgery completed, the whole-case intra-operative totals, ICU stay, discharge,
  in-hospital death);
- derived pipeline products the reviewer chose to exclude (causal-depth analysis and its
  depth score, the remifentanil causal flag, the drug concentration time series);
- volatile-agent variables (``exp_sevo``, ``mac``): the task excludes volatile changes;
- the plasma concentrations ``cp_*`` and the pump targets ``ct_*``: for remifentanil the
  decision point falls at or just after the pump's target change, so the plasma
  concentration already carries the action (it moves with the label in ~74 % of
  ``increase_opioid`` and ~86 % of ``reduce_opioid`` test windows, against <8 % of
  ``no_action``; checked 2026-10-09), and the target changes even earlier than the plasma
  concentration does. View 3 hid ``cp_*``; view 4 hides ``ct_*`` too. The targets of
  *past* adjustments stay visible through the action history.

:func:`visible_input` removes all of them and also drops any time-series sample stamped
after the decision point (``t_offset_sec > 0``). ``dataset`` applies it when a window is
loaded, so ``window.input`` *is* the view: the prompt renderer, the retrieval index, the
gradient-boosting control and the SHAP analysis all read the same fields. The untouched
record stays in ``window.raw_input`` for auditing only.

Bump :data:`VIEW_VERSION` whenever the hidden set or the filtering changes: run metadata,
the retrieval index and the trained models record it, and stale artefacts are refused.
"""

from __future__ import annotations

__all__ = [
    "VIEW_VERSION",
    "HIDDEN_FIELDS",
    "HIDDEN_PREFIXES",
    "is_hidden",
    "visible_input",
    "hidden_in",
]

#: Version of the shared view. Bump on any change to the hidden set or the filtering.
VIEW_VERSION = 4

#: Fields no model may see, by name.
HIDDEN_FIELDS = frozenset(
    {
        # identifiers
        "pt_caseid",
        "pt_subjectid",
        "case_id",
        "case_id_fill",
        "window_id",
        # pipeline bookkeeping
        "schema_version",
        "staleness_json",
        "staleness_max",
        "decision_time_abs",
        "pt_casestart",
        "pt_adm",
        "pt_anestart",
        "pt_opstart",
        # derived pipeline products excluded by the reviewer
        "causal_depth_json",
        "depth_score",
        "remi_causal_flag",
        "drug_concentrations_json",
        # volatile agent (outside the task's action space)
        "exp_sevo",
        "mac",
        # future and outcome
        "pt_caseend",
        "pt_opend",
        "pt_aneend",
        "surgery_pct",
        "pt_dis",
        "pt_icu_days",
        "pt_death_inhosp",
    }
)

#: Fields no model may see, by prefix: point/event counters, whole-case intra-op totals, and
#: the plasma concentrations and pump targets (they already carry the action at t = 0).
HIDDEN_PREFIXES = ("n_", "pt_intraop_", "cp_", "ct_")

#: Nested fields whose items carry a ``t_offset_sec`` relative to the decision point.
_TIMED_FIELDS = ("vitals_json", "events_json", "actions_history_json")


def is_hidden(name):
    """Whether a field is outside the shared view."""
    name = str(name)
    return name in HIDDEN_FIELDS or name.startswith(HIDDEN_PREFIXES)


def _is_future(item):
    offset = item.get("t_offset_sec") if isinstance(item, dict) else None
    return isinstance(offset, (int, float)) and not isinstance(offset, bool) and offset > 0


def _past_only(value):
    """A time series (dict of lists, or list) without the samples after the decision point."""
    if isinstance(value, list):
        return [item for item in value if not _is_future(item)]
    if isinstance(value, dict):
        return {
            key: ([item for item in series if not _is_future(item)] if isinstance(series, list) else series)
            for key, series in value.items()
        }
    return value


def visible_input(inp):
    """The shared view of a window input: hidden fields removed, future samples dropped."""
    if not isinstance(inp, dict):
        return inp
    view = {}
    for key, value in inp.items():
        if is_hidden(key):
            continue
        view[key] = _past_only(value) if key in _TIMED_FIELDS else value
    return view


def hidden_in(names):
    """The names of an iterable that are outside the view (empty when it is clean)."""
    return sorted(name for name in names if is_hidden(name))
