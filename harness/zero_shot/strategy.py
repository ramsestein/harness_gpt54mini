# -*- coding: utf-8 -*-
"""Zero-shot strategy: one direct call to the model per decision window.

The window is rendered as clinical prose by ``src.prompting`` and injected into
``prompt.txt`` (five options; the model answers with a single letter). The
``temperature`` / ``reasoning_effort`` settings come from ``Strategy``.
"""

from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = HERE / "prompt.txt"


class ZeroShotStrategy(Strategy):
    """Ask the model once per window: clinical prose in, one letter out."""

    name = "zero_shot"
    description = (
        "Single call per window: the decision window rendered as clinical prose "
        "plus a five-option prompt (a-e); the model answers with one letter."
    )

    def __init__(self, prompt_file=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE

    def build_messages(self, window):
        """Chat messages for one window, with ``{case}`` filled from the template."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_messages(window, template=template)

    def predict(self, window) -> Prediction:
        response = self.ask(self.build_messages(window))
        return self.prediction_from(window, response, parsing.parse_action(response.text))
