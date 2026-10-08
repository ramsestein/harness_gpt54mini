# -*- coding: utf-8 -*-
"""Structured-JSON strategy: the whole prompt is a JSON document.

Unlike ``zero_shot`` (clinical prose) this strategy sends the model a single
JSON object that holds both the instructions and the decision window, injected
under ``patient_state``. The answer format is the same: one letter, parsed the
same way.

The instruction half of the prompt lives in ``prompt.json`` (this folder);
``patient_state`` is added by the strategy, so the window is a *nested JSON
object*, not a string.
"""

import json
from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = HERE / "prompt.json"


class StructuredJsonStrategy(Strategy):
    """Ask the model once per window, with the prompt itself written as JSON."""

    name = "structured_json"
    description = (
        "Single call per window: instructions and decision window are sent as one "
        "JSON document (window nested under 'patient_state'); the answer is the "
        "same single letter as zero_shot."
    )

    def __init__(self, prompt_file=None, indent=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        self.indent = indent

    def load_template(self):
        """The instruction half of the prompt, as a dict."""
        template = json.loads(self.prompt_file.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise ValueError(f"the prompt template must be a JSON object: {self.prompt_file}")
        return template

    def build_prompt(self, window):
        """The user prompt: one JSON document with the window nested inside."""
        payload = self.load_template()
        payload["patient_state"] = window.input
        return json.dumps(payload, ensure_ascii=False, indent=self.indent, default=str)

    def build_messages(self, window):
        """Chat messages for one window (system + the JSON prompt)."""
        return [
            {"role": "system", "content": prompting.SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(window)},
        ]

    def predict(self, window) -> Prediction:
        response = self.ask(self.build_messages(window))
        return self.prediction_from(window, response, parsing.parse_action(response.text))
