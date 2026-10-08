# -*- coding: utf-8 -*-
"""Chain-of-thought / ReAct strategy: reason first, then commit to one letter.

Same single call as ``zero_shot`` -- the same clinical prose, the same five
options and the same one-letter answer -- but the prompt asks the model to work
the case with a **ReAct** loop (``Thought`` / ``Action`` / ``Observation``) over
the three physiological axes (haemodynamics, hypnosis, analgesia) before closing
with an ``Answer: <letter>`` line.

Because the trace names actions and options on the way, the decision is read from
the closing label (``src.parsing.parse_action`` makes ``answer`` the authoritative
marker). The trace itself is kept in ``meta`` for inspection.
"""

import re
from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = HERE / "prompt.txt"

_ANSWER_LINE = re.compile(r"^\s*answer\b", flags=re.IGNORECASE)

#: ReAct trace markers, counted per response so the prompt can be audited.
_REACT_MARKERS = {
    "thoughts": re.compile(r"^\s*thought\b", flags=re.IGNORECASE | re.MULTILINE),
    "actions": re.compile(r"^\s*action\b", flags=re.IGNORECASE | re.MULTILINE),
    "observations": re.compile(r"^\s*observation\b", flags=re.IGNORECASE | re.MULTILINE),
}


class CotStrategy(Strategy):
    """Ask the model to reason briefly, then answer with one letter."""

    name = "cot"
    description = (
        "Single call per window: the zero-shot prompt rewritten as a ReAct trace "
        "(Thought/Action/Observation over haemodynamics, hypnosis and analgesia) "
        "before the closing 'Answer: <letter>' line."
    )

    def __init__(self, prompt_file=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE

    def build_messages(self, window):
        """Chat messages for one window, with ``{case}`` filled from the template."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_messages(window, template=template)

    @staticmethod
    def extract_reasoning(text):
        """Everything before the final ``Answer:`` line (``""`` when there is none)."""
        if not text:
            return ""
        lines = [line for line in text.splitlines() if line.strip()]
        for position in range(len(lines) - 1, -1, -1):
            if _ANSWER_LINE.match(lines[position]):
                return "\n".join(lines[:position]).strip()
        return ""

    @staticmethod
    def count_react_steps(text):
        """How many ``Thought`` / ``Action`` / ``Observation`` lines the trace has."""
        text = text or ""
        return {name: len(pattern.findall(text)) for name, pattern in _REACT_MARKERS.items()}

    def predict(self, window) -> Prediction:
        response = self.ask(self.build_messages(window))
        prediction = self.prediction_from(
            window, response, parsing.parse_action(response.text)
        )
        prediction.meta.update(
            reasoning=self.extract_reasoning(response.text),
            react=self.count_react_steps(response.text),
        )
        return prediction
