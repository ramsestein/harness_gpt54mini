# -*- coding: utf-8 -*-
"""Rendering a window into clinical prose and building prompts.

STATE: implemented. The contract is defined in
``tests/test_prompting.py``.

Contract
-----------------
Constants
    ``LETTER_TO_ACTION`` -> {"a": "increase_hypnotic", "b": "reduce_hypnotic",
                             "c": "increase_opioid", "d": "reduce_opioid",
                             "e": "no_action"}
    ``ACTION_TO_LETTER`` -> inverse of ``LETTER_TO_ACTION``
    ``PROMPT_TEMPLATE``  -> template from ``dataset/scripts/prompt.txt``
    ``SYSTEM_PROMPT``    -> system instructions

Functions
    ``render_case(input_dict) -> str``             the shared view (``src.view``): clinical
                                                   prose plus the full record block, every
                                                   visible field with a value exactly once
    ``render_json(input_dict, indent=None) -> str`` the window as a JSON object,
                                                   deterministic key order
    ``load_template(path) -> str``
    ``build_prompt(window, template=None) -> str``  template with ``{case}`` filled in
    ``build_prompt_from_case(case, template=None) -> str``
    ``build_messages(window, template=None) -> list[dict]``
        ``[{"role": "system", ...}, {"role": "user", ...}]``
    ``build_messages_from_case(case, template=None) -> list[dict]``
"""

from __future__ import annotations

import json
from pathlib import Path

from .view import visible_input

__all__ = [
    "LETTER_TO_ACTION",
    "ACTION_TO_LETTER",
    "PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    "load_template",
    "render_case",
    "render_json",
    "rendered_fields",
    "build_prompt",
    "build_prompt_from_case",
    "build_messages",
    "build_messages_from_case",
]

LETTER_TO_ACTION = {
    "a": "increase_hypnotic",
    "b": "reduce_hypnotic",
    "c": "increase_opioid",
    "d": "reduce_opioid",
    "e": "no_action",
}
ACTION_TO_LETTER = {action: letter for letter, action in LETTER_TO_ACTION.items()}

MONITORING = {"ART": "arterial line", "NIBP": "non-invasive blood pressure", "NONE": "none"}

KEY_LABS = ["hb", "hct", "plt", "wbc", "na", "k", "gluc", "cr", "lac", "ph", "hco3", "be"]

LAB_UNITS = {
    "hb": "g/dL",
    "hct": "%",
    "plt": "K/uL",
    "wbc": "K/uL",
    "na": "mEq/L",
    "k": "mEq/L",
    "gluc": "mg/dL",
    "cr": "mg/dL",
    "lac": "mmol/L",
    "ph": "",
    "hco3": "mEq/L",
    "be": "mEq/L",
}

SYSTEM_PROMPT = (
    "You are an expert anesthesiologist in the middle of a real surgical case. "
    "You must choose the single best next anesthetic adjustment among five options."
)

PROMPT_TEMPLATE = """You are an expert anesthesiologist in the middle of a real surgical case. Below is the current state of the patient, derived from continuous intraoperative monitoring.

{case}

Based ONLY on the information above, decide the single best next anesthetic adjustment.

Choose exactly one of the following five options:

a) increase_hypnotic
b) reduce_hypnotic
c) increase_opioid
d) reduce_opioid
e) no_action

Respond with exactly one character: the letter of your choice (a, b, c, d, or e).
Output nothing else.
"""


def _fmt(value):
    """Format a value for prose: ``None`` stays ``None``, floats trimmed to 2 decimals."""
    if value is None:
        return None
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(round(value, 2))
    return str(value)


def _last_adjustment(inp, drug=None):
    """The most recent infusion adjustment of the window's own action history (of ``drug``).

    The history is a list of events (``drug``, ``old_target``, ``new_target``, the time
    offset). Only the most recent one is rendered: it is the decision this one follows.
    """
    history = inp.get("actions_history_json")
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except ValueError:
            return None
    if not isinstance(history, list):
        return None
    events = [
        event
        for event in history
        if isinstance(event, dict)
        and isinstance(event.get("t_offset_sec"), (int, float))
        and not isinstance(event.get("t_offset_sec"), bool)
        and (drug is None or str(event.get("drug", "")).strip().lower() == drug)
    ]
    return max(events, key=lambda event: event["t_offset_sec"]) if events else None


def _infusions(inp, shown):
    """The infusion state: where each drug is, where it is heading and what it is set to.

    The actions the model must choose between are *infusion-rate changes*, so the rate
    state is the decision-relevant part of the window: the target the pump is set to, the
    effect-site concentration reached and the direction it is moving in.
    """
    adjustment = _last_adjustment(inp)
    parts = []

    # The pump targets at t = 0 (``ct_*``) are outside the shared view: the target shown is
    # the one the most recent *past* adjustment of that drug set (the action history).
    propofol_last = _last_adjustment(inp, drug="propofol")
    propofol_target = _fmt(propofol_last.get("new_target")) if propofol_last else None
    propofol_effect = _fmt(inp.get("ce_propofol"))
    propofol_trend = inp.get("propofol_ce_trend")
    if propofol_effect is not None or propofol_target is not None:
        rendered = "propofol"
        if propofol_target is not None:
            rendered += f" target {propofol_target}"
        if propofol_effect is not None:
            rendered += f", effect-site {propofol_effect}"
            shown.add("ce_propofol")
        if propofol_trend:
            rendered += f" ({propofol_trend})"
            shown.add("propofol_ce_trend")
        parts.append(rendered)

    remi_effect = _fmt(inp.get("ce_remi"))
    remi_trend = inp.get("remi_ce_trend")
    remi_last = _last_adjustment(inp, drug="remifentanil")
    remi_target = _fmt(remi_last.get("new_target")) if remi_last else None
    if remi_effect is not None or remi_target is not None:
        rendered = "remifentanil"
        if remi_target is not None:
            rendered += f" target {remi_target}"
        if remi_effect is not None:
            rendered += f", effect-site {remi_effect}"
            shown.add("ce_remi")
        if remi_trend:
            rendered += f" ({remi_trend})"
            shown.add("remi_ce_trend")
        parts.append(rendered)

    return parts


def _time_out_of_band(inp, shown):
    """Seconds recently spent outside the bands an anesthesiologist cares about."""
    parts = []
    for field, label in (
        ("bis_below_40_sec", "BIS below 40"),
        ("bis_above_60_sec", "BIS above 60"),
        ("map_below_65_sec", "MAP below 65 mmHg"),
        ("map_below_55_sec", "MAP below 55 mmHg"),
        ("spo2_below_90_sec", "SpO2 below 90 %"),
    ):
        value = inp.get(field)
        if _is_number(value) and value > 0:  # a zero is left to the full record block
            parts.append(f"{label} for {_fmt(float(value))} s")
            shown.add(field)
    return parts


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _present(value):
    """A value worth rendering: not missing, not an empty string/list/dict."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return False
    return True


def _value(value):
    """Any value as compact text: numbers trimmed, nested objects as compact JSON."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return _fmt(value)


#: The vital signs of the window, as the two families of field names spell them.
_SIGNALS = (("HR", "hr"), ("MAP", "map"), ("SBP", "sbp"), ("DBP", "dbp"),
            ("SpO2", "spo2"), ("EtCO2", "etco2"), ("BIS", "bis"))


def _series(points):
    """``[{value, t_offset_sec}, ...]`` as ``value@offset`` pairs, most recent first."""
    timed = [p for p in points if isinstance(p, dict) and _is_number(p.get("t_offset_sec"))]
    timed.sort(key=lambda p: -p["t_offset_sec"])
    return ", ".join(f"{_fmt(p.get('value'))}@{_fmt(p['t_offset_sec'])}s" for p in timed)


def _events(items):
    """An event or action list, one compact entry per item, oldest first."""
    timed = [item for item in items if isinstance(item, dict)]
    timed.sort(key=lambda item: item.get("t_offset_sec") if _is_number(item.get("t_offset_sec")) else 0)
    rendered = []
    for item in timed:
        offset = item.get("t_offset_sec")
        rest = ", ".join(
            f"{key} {_value(value)}" for key, value in item.items()
            if key != "t_offset_sec" and _present(value)
        )
        rendered.append(f"[{_fmt(offset)} s] {rest}")
    return "; ".join(rendered)


def _full_record(inp, shown):
    """Every field of the view the clinical prose has not rendered, grouped by kind.

    The prose above is the readable summary; this block guarantees that the model sees
    the *whole* shared view (``src.view``), exactly the fields the gradient-boosting
    control and the retrieval index read. Field names are the dataset's own codes.
    """
    remaining = {k: v for k, v in inp.items() if k not in shown and _present(v)}
    sections = []

    def take(keys):
        taken = [(k, remaining.pop(k)) for k in keys if k in remaining]
        return ", ".join(f"{k} {_value(v)}" for k, v in taken)

    # drug pharmacology: plasma/effect-site concentrations, their dynamics, PK/PD fit
    drug_keys = sorted(k for k in remaining if k.startswith(("ce_", "propofol_ce_", "remi_ce_")))
    line = take(drug_keys)
    if line:
        sections.append("Drug concentrations and dynamics: " + line + ".")
    residuals = remaining.pop("pkpd_residuals_json", None)
    if isinstance(residuals, dict) and residuals:
        sections.append("PK/PD model fit: " + ", ".join(
            f"{k} {_value(v)}" for k, v in residuals.items() if _present(v)) + ".")
    elif _present(residuals):
        sections.append(f"PK/PD model fit: {_value(residuals)}.")

    # vital signs: window statistics and short/long-window trend descriptors
    rows = []
    for upper, lower in _SIGNALS:
        keys = sorted(k for k in remaining if k.startswith((upper + "_", lower + "_")))
        line = take(keys)
        if line:
            rows.append(f"- {upper}: {line}")
    for key in sorted(k for k in remaining if k.endswith(("_below_40_sec", "_above_60_sec", "_below_55_sec", "_below_65_sec", "_below_90_sec"))):
        rows.append(f"- {key} {_value(remaining.pop(key))}")
    if rows:
        sections.append("Vital-sign statistics over the window (current, min, mean, max, slopes, deltas, trend codes):\n" + "\n".join(rows))

    vitals = remaining.pop("vitals_json", None)
    if isinstance(vitals, dict):
        rows = [f"- {signal}: {_series(points)}" for signal, points in vitals.items()
                if isinstance(points, list) and _series(points)]
        if rows:
            sections.append("Vital-sign samples (value@seconds relative to now, most recent first):\n" + "\n".join(rows))

    for key, title in (("actions_history_json", "Infusion adjustments in the last 20 min"),
                       ("events_json", "Clinical events in the last 20 min")):
        items = remaining.pop(key, None)
        if isinstance(items, list) and items:
            sections.append(f"{title} (seconds relative to now): {_events(items)}.")

    # laboratory: every test with the age of the sample
    tests = sorted({k[4:-4] if k.endswith("_age") else k[4:] for k in remaining if k.startswith("lab_")})
    labs = []
    for test in tests:
        value = remaining.pop(f"lab_{test}", None)
        age = remaining.pop(f"lab_{test}_age", None)
        if _present(value):
            entry = f"{test} {_value(value)}"
        elif f"lab_{test}" in shown:
            entry = f"{test} (value above)"
        else:
            entry = f"{test} (no value)"
        if _present(age):
            entry += f" [age {_value(age)} s]"
        labs.append(entry)
    if labs:
        sections.append("Laboratory (all recorded tests): " + ", ".join(labs) + ".")

    patient = take(sorted(k for k in remaining if k.startswith("pt_")))
    if patient:
        sections.append("Patient record: " + patient + ".")

    other = take(sorted(remaining))
    if other:
        sections.append("Other recorded values: " + other + ".")
    return sections


def render_case(input_dict):
    """Render a decision window as clinical prose, followed by the rest of the shared view.

    The input goes through :func:`src.view.visible_input` first, so the rendering is the
    shared view whatever the caller passes. Every field of the view with a value appears
    once: the clinically salient ones in the prose, all the others in the record block.
    """
    return _render(input_dict)[0]


def _render(input_dict):
    """``(text, prose_fields)``: the rendering and the fields the prose part rendered."""
    inp = visible_input(input_dict if isinstance(input_dict, dict) else {})
    shown = set()
    lines = []

    sex = {"M": "male", "F": "female"}.get(inp.get("pt_sex"), inp.get("pt_sex"))
    age = _fmt(inp.get("pt_age"))
    descriptor = []
    if age:
        descriptor.append(f"{age}-year-old")
        shown.add("pt_age")
    if sex:
        descriptor.append(str(sex))
        shown.add("pt_sex")
    descriptor.append("patient")
    demo = " ".join(descriptor)

    attrs = []
    for field, unit in (("pt_weight", "kg"), ("pt_height", "cm")):
        value = _fmt(inp.get(field))
        if value is not None:
            attrs.append(f"{value} {unit}")
            shown.add(field)
    if _fmt(inp.get("pt_bmi")) is not None:
        attrs.append(f"BMI {_fmt(inp['pt_bmi'])}")
        shown.add("pt_bmi")
    if _fmt(inp.get("pt_asa")) is not None:
        attrs.append(f"ASA {_fmt(inp['pt_asa'])}")
        shown.add("pt_asa")
    if attrs:
        demo += f" ({', '.join(attrs)})"

    surgery = str(inp["pt_opname"]) if inp.get("pt_opname") else "an unspecified procedure"
    if inp.get("pt_opname"):
        shown.add("pt_opname")
    context = []
    for field in ("pt_optype", "pt_approach", "pt_position", "pt_ane_type"):
        if inp.get(field):
            context.append(str(inp[field]))
            shown.add(field)
    kind = "emergency" if inp.get("pt_emop") else "elective"
    if "pt_emop" in inp:
        shown.add("pt_emop")
    opening = f"A {demo} is undergoing {surgery}"
    if context:
        opening += f" ({', '.join(context)})"
    opening += f", {kind}."
    lines.append(opening)

    for field, label in (("pt_dx", "Diagnosis"), ("pt_department", "Department")):
        if inp.get(field):
            lines.append(f"{label}: {inp[field]}.")
            shown.add(field)
    if inp.get("monitoring_type"):
        lines.append(f"Monitoring: {MONITORING.get(inp['monitoring_type'], inp['monitoring_type'])}.")
        shown.add("monitoring_type")

    vitals = []
    for field, label, unit in (
        ("hr_current", "HR", "bpm"),
        ("map_current", "MAP", "mmHg"),
        ("spo2_current", "SpO2", "%"),
        ("etco2_current", "EtCO2", "mmHg"),
        ("bis_current", "BIS", ""),
    ):
        value = _fmt(inp.get(field))
        if value is not None:
            vitals.append(f"{label} {value}" + (f" {unit}" if unit else ""))
            shown.add(field)
    if vitals:
        lines.append("Current vitals: " + ", ".join(vitals) + ".")

    trends = []
    for field, label, slope_field in (
        ("hr_trend", "HR", "hr_slope_short"),
        ("map_trend", "MAP", "map_slope_short"),
        ("bis_trend", "BIS", "bis_slope_short"),
        ("spo2_trend", "SpO2", "spo2_slope_short"),
        ("etco2_trend", "EtCO2", "etco2_slope_short"),
    ):
        value = inp.get(field)
        if value is None:
            continue
        shown.add(field)
        # the steepness of the recent window, so "falling" is not all the model gets
        steepness = _fmt(inp.get(slope_field))
        if steepness is not None:
            shown.add(slope_field)
        trends.append(f"{label} {value}" + (f" ({steepness})" if steepness is not None else ""))
    if trends:
        lines.append("Recent trends: " + ", ".join(trends) + ".")

    # The five options are infusion-rate changes, so the rate state is rendered here:
    # the target the pump is set to, the concentration reached and where it is heading.
    infusions = _infusions(inp, shown)
    if infusions:
        lines.append("Infusions: " + "; ".join(infusions) + ".")

    adjustment = _last_adjustment(inp)
    if adjustment is not None:
        ago = _fmt(round(abs(float(adjustment["t_offset_sec"])) / 60.0, 1))
        old = _fmt(adjustment.get("old_target"))
        new = _fmt(adjustment.get("new_target"))
        change = f" target {old} -> {new}" if old is not None and new is not None else ""
        lines.append(
            f"Most recent infusion adjustment: {adjustment.get('drug', 'unknown')}"
            f"{change} ({ago} min ago)."
        )

    out_of_band = _time_out_of_band(inp, shown)
    if out_of_band:
        lines.append("Time recently outside the target bands: " + ", ".join(out_of_band) + ".")

    induction = inp.get("time_since_induction_sec")
    if _is_number(induction):
        lines.append(f"Time since induction: {_fmt(round(induction / 60.0, 1))} min.")
        shown.add("time_since_induction_sec")

    ventilator = []
    for field, label in (
        ("set_fio2", "FiO2"),
        ("set_peep", "PEEP"),
        ("set_tv", "tidal volume"),
        ("set_rr", "respiratory rate"),
    ):
        value = _fmt(inp.get(field))
        if value is not None:
            ventilator.append(f"{label} {value}")
            shown.add(field)
    if ventilator:
        lines.append("Ventilator: " + ", ".join(ventilator) + ".")

    labs = []
    for test in KEY_LABS:
        value = _fmt(inp.get(f"lab_{test}"))
        if value is not None:
            unit = LAB_UNITS.get(test, "")
            labs.append(f"{test} {value}" + (f" {unit}" if unit else ""))
            shown.add(f"lab_{test}")
    if labs:
        lines.append("Preoperative labs: " + ", ".join(labs) + ".")

    history = inp.get("history_summary")
    if history:
        lines.append(f"Recent course: {history}")
        shown.add("history_summary")

    prose_fields = set(shown)
    record = _full_record(inp, shown)
    if record:
        lines.append("")
        lines.append("Full monitoring record (dataset field codes):")
        lines.extend(record)
    return "\n".join(lines), prose_fields


def rendered_fields(input_dict):
    """The fields of the shared view that :func:`render_case` renders (all with a value)."""
    inp = visible_input(input_dict if isinstance(input_dict, dict) else {})
    return {k for k, v in inp.items() if _present(v)}


def render_json(input_dict, indent=None):
    """Render a decision window as a JSON object.

    Keys are sorted so the output is deterministic (same window, same prompt).
    Nested ``*_json`` fields come out as real JSON objects/arrays.
    """
    inp = visible_input(input_dict if isinstance(input_dict, dict) else {})
    return json.dumps(inp, ensure_ascii=False, sort_keys=True, indent=indent, default=str)


def load_template(path):
    """Read a prompt template from disk (``{case}`` is the placeholder)."""
    return Path(path).read_text(encoding="utf-8")


def build_prompt_from_case(case, template=None):
    """Build the final prompt from an already rendered case."""
    template = PROMPT_TEMPLATE if template is None else template
    return template.replace("{case}", case)


def build_prompt(window, template=None):
    """Build the final prompt by replacing ``{case}`` in the template."""
    return build_prompt_from_case(render_case(getattr(window, "input", window)), template)


def build_messages_from_case(case, template=None):
    """Return the chat messages (system + user) for an already rendered case."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt_from_case(case, template)},
    ]


def build_messages(window, template=None):
    """Return the chat messages (system + user)."""
    return build_messages_from_case(render_case(getattr(window, "input", window)), template)
