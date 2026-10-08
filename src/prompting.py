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
    ``render_case(input_dict) -> str``             clinical prose WITHOUT identifiers
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

__all__ = [
    "LETTER_TO_ACTION",
    "ACTION_TO_LETTER",
    "PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    "load_template",
    "render_case",
    "render_json",
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


def _last_adjustment(inp):
    """The most recent infusion adjustment of the window's own action history.

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
    ]
    return max(events, key=lambda event: event["t_offset_sec"]) if events else None


def _infusions(inp):
    """The infusion state: where each drug is, where it is heading and what it is set to.

    The actions the model must choose between are *infusion-rate changes*, so the rate
    state is the decision-relevant part of the window: the target the pump is set to, the
    effect-site concentration reached and the direction it is moving in.
    """
    adjustment = _last_adjustment(inp)
    parts = []

    propofol_target = _fmt(inp.get("ct_propofol"))
    propofol_effect = _fmt(inp.get("ce_propofol"))
    propofol_trend = inp.get("propofol_ce_trend")
    if propofol_effect is not None or propofol_target is not None:
        rendered = "propofol"
        if propofol_target is not None:
            rendered += f" target {propofol_target}"
        if propofol_effect is not None:
            rendered += f", effect-site {propofol_effect}"
        if propofol_trend:
            rendered += f" ({propofol_trend})"
        parts.append(rendered)

    remi_effect = _fmt(inp.get("ce_remi"))
    remi_trend = inp.get("remi_ce_trend")
    remi_target = None
    if adjustment and str(adjustment.get("drug", "")).strip().lower() == "remifentanil":
        remi_target = _fmt(adjustment.get("new_target"))
    if remi_effect is not None or remi_target is not None:
        rendered = "remifentanil"
        if remi_target is not None:
            rendered += f" target {remi_target}"
        if remi_effect is not None:
            rendered += f", effect-site {remi_effect}"
        if remi_trend:
            rendered += f" ({remi_trend})"
        parts.append(rendered)

    return parts


def _time_out_of_band(inp):
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
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            parts.append(f"{label} for {_fmt(float(value))} s")
    return parts


def render_case(input_dict):
    """Render a decision window as clinical prose (without identifiers)."""
    inp = input_dict if isinstance(input_dict, dict) else {}
    lines = []

    sex = {"M": "male", "F": "female"}.get(inp.get("pt_sex"), inp.get("pt_sex"))
    age = _fmt(inp.get("pt_age"))
    descriptor = []
    if age:
        descriptor.append(f"{age}-year-old")
    if sex:
        descriptor.append(str(sex))
    descriptor.append("patient")
    demo = " ".join(descriptor)

    attrs = []
    for field, unit in (("pt_weight", "kg"), ("pt_height", "cm")):
        value = _fmt(inp.get(field))
        if value is not None:
            attrs.append(f"{value} {unit}")
    if _fmt(inp.get("pt_bmi")) is not None:
        attrs.append(f"BMI {_fmt(inp['pt_bmi'])}")
    if _fmt(inp.get("pt_asa")) is not None:
        attrs.append(f"ASA {_fmt(inp['pt_asa'])}")
    if attrs:
        demo += f" ({', '.join(attrs)})"

    surgery = str(inp["pt_opname"]) if inp.get("pt_opname") else "an unspecified procedure"
    context = [
        str(inp[field])
        for field in ("pt_optype", "pt_approach", "pt_position", "pt_ane_type")
        if inp.get(field)
    ]
    kind = "emergency" if inp.get("pt_emop") else "elective"
    opening = f"A {demo} is undergoing {surgery}"
    if context:
        opening += f" ({', '.join(context)})"
    opening += f", {kind}."
    lines.append(opening)

    if inp.get("pt_dx"):
        lines.append(f"Diagnosis: {inp['pt_dx']}.")
    if inp.get("pt_department"):
        lines.append(f"Department: {inp['pt_department']}.")
    if inp.get("monitoring_type"):
        lines.append(f"Monitoring: {MONITORING.get(inp['monitoring_type'], inp['monitoring_type'])}.")

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
        # the steepness of the recent window, so "falling" is not all the model gets
        steepness = _fmt(inp.get(slope_field))
        trends.append(f"{label} {value}" + (f" ({steepness})" if steepness is not None else ""))
    if trends:
        lines.append("Recent trends: " + ", ".join(trends) + ".")

    drugs = []
    for field, label in (
        ("ce_propofol", "propofol effect-site concentration"),
        ("ce_remi", "remifentanil effect-site concentration"),
        ("exp_sevo", "sevoflurane expired concentration"),
        ("mac", "MAC"),
    ):
        value = _fmt(inp.get(field))
        if value is not None:
            drugs.append(f"{label} {value}")
    if drugs:
        lines.append("Drugs: " + ", ".join(drugs) + ".")

    # The five options are infusion-rate changes, so the rate state is rendered here:
    # the target the pump is set to, the concentration reached and where it is heading.
    infusions = _infusions(inp)
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

    out_of_band = _time_out_of_band(inp)
    if out_of_band:
        lines.append("Time recently outside the target bands: " + ", ".join(out_of_band) + ".")

    induction = inp.get("time_since_induction_sec")
    if isinstance(induction, (int, float)) and not isinstance(induction, bool) and induction > 0:
        lines.append(f"Time since induction: {_fmt(round(induction / 60.0, 1))} min.")

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
    if ventilator:
        lines.append("Ventilator: " + ", ".join(ventilator) + ".")

    labs = []
    for test in KEY_LABS:
        value = _fmt(inp.get(f"lab_{test}"))
        if value is not None:
            unit = LAB_UNITS.get(test, "")
            labs.append(f"{test} {value}" + (f" {unit}" if unit else ""))
    if labs:
        lines.append("Preoperative labs: " + ", ".join(labs) + ".")

    history = inp.get("history_summary")
    if history:
        lines.append(f"Recent course: {history}")

    return "\n".join(lines)


def render_json(input_dict, indent=None):
    """Render a decision window as a JSON object.

    Keys are sorted so the output is deterministic (same window, same prompt).
    Nested ``*_json`` fields come out as real JSON objects/arrays.
    """
    inp = input_dict if isinstance(input_dict, dict) else {}
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
