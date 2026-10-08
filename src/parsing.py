# -*- coding: utf-8 -*-
"""Parsing the model answer into an action from the vocabulary.

STATE: implemented. The contract is defined in
``tests/test_parsing.py``.

Contract
-----------------
``normalize(text) -> str``        lowercase + strip (tolerates ``None`` -> "")
``parse_action(text) -> str | None``
    Rules, in order:
      1. The last letter labelled with ``answer`` -- the canonical closing marker.
      2. The last letter labelled with ``option``/``choice``/``letter`` -- weaker
         hints that a reasoning trace contains while comparing alternatives.
      3. A lone letter ``a``-``e`` (with ``)`` ``.`` ``:`` or surrounding
         parentheses).
      4. An explicit action name anywhere in the text.
    Anything else returns ``None`` -- the parser never guesses.
"""

from __future__ import annotations

import re

from .prompting import LETTER_TO_ACTION

__all__ = ["normalize", "parse_action"]

_ACTION_ORDER = [
    "increase_hypnotic",
    "reduce_hypnotic",
    "increase_opioid",
    "reduce_opioid",
    "no_action",
]

_ALIASES = (
    ("increase hypnotic", "increase_hypnotic"),
    ("reduce hypnotic", "reduce_hypnotic"),
    ("decrease hypnotic", "reduce_hypnotic"),
    ("increase opioid", "increase_opioid"),
    ("reduce opioid", "reduce_opioid"),
    ("decrease opioid", "reduce_opioid"),
    ("no action", "no_action"),
)

_LONE_LETTER = re.compile(r"\(?\s*([a-e])\s*\)?\s*[.):]?")

#: ``Answer: b`` is the canonical closing marker; ``option``/``choice``/``letter``
#: are weaker hints that the reasoning contains while comparing alternatives.
_ANSWER_LETTER = re.compile(r"\banswer\b[^a-e\n]{0,12}?\b([a-e])\b")
_HINT_LETTER = re.compile(r"\b(?:option|choice|letter)\b[^a-e\n]{0,12}?\b([a-e])\b")


def normalize(text):
    """Normalise the text for parsing (``None`` -> "")."""
    if text is None:
        return ""
    return str(text).strip().lower()


def parse_action(text):
    """Extract the predicted action from the model text, or ``None``.

    Rules, in order:
      1. the last letter labelled with ``answer`` -- the canonical closing marker,
         which is what makes a chain-of-thought or ReAct trace parseable;
      2. the last letter labelled with ``option``/``choice``/``letter``;
      3. a lone letter ``a``-``e``;
      4. an explicit action name anywhere in the text.
    Anything else returns ``None`` -- the parser never guesses.
    """
    normalized = normalize(text)
    if not normalized:
        return None
    for phrase, action in _ALIASES:
        normalized = normalized.replace(phrase, action)

    for pattern in (_ANSWER_LETTER, _HINT_LETTER):
        labelled = list(pattern.finditer(normalized))
        if labelled:
            return LETTER_TO_ACTION[labelled[-1].group(1)]
    match = _LONE_LETTER.fullmatch(normalized)
    if match:
        return LETTER_TO_ACTION[match.group(1)]
    for action in _ACTION_ORDER:
        if action in normalized:
            return action
    return None
