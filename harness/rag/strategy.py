# -*- coding: utf-8 -*-
"""RAG strategy: the zero-shot call plus a reference file of clinical guidance.

``zero_shot`` renders the decision window as clinical prose and asks for one of five actions.
This strategy does exactly the same and then hands the model the **whole content of
``context.md``** as reference material: verbatim excerpts of published guidance (TIVA and BIS
monitoring recommendations, the propofol and remifentanil prescribing information, the
intraoperative arterial-pressure consensus, the notes on older and frail patients), each block
attributed to its source.

The prompt wording and the five options are ``zero_shot``'s own template
(``harness/zero_shot/prompt.txt``); the reference is inserted right after the patient state, so
it is part of "the information above" the instructions already refer to.
"""

import threading
from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent

#: The reference material, curated by hand in this folder.
DEFAULT_CONTEXT_FILE = HERE / "context.md"

#: The prompt is ``zero_shot``'s, so the two runs differ in the reference and nothing else.
DEFAULT_PROMPT_FILE = HERE.parent / "zero_shot" / "prompt.txt"

#: How the reference is announced to the model.
REFERENCE_HEADER = (
    "Reference information: published clinical guidance relevant to this decision. "
    "Use it together with the patient state above."
)


class RagStrategy(Strategy):
    """Ask the model once per window, with the curated clinical context in the prompt."""

    name = "rag"
    description = (
        "Single call per window: the zero_shot clinical prose and options, preceded by the "
        "whole curated clinical context (guideline excerpts with their sources) as reference "
        "material. Same prompt as zero_shot otherwise."
    )

    def __init__(self, prompt_file=None, context_file=None, n_samples=None, k=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        self.context_file = Path(context_file) if context_file else DEFAULT_CONTEXT_FILE
        self._context = None
        self._lock = threading.Lock()

    def context(self):
        """The reference material, read once (thread-safe: windows run in a pool)."""
        if self._context is None:
            with self._lock:
                if self._context is None:
                    path = Path(self.context_file)
                    if not path.is_file():
                        raise FileNotFoundError(f"no context file at {path}")
                    self._context = path.read_text(encoding="utf-8")
        return self._context

    def case_block(self, window):
        """The ``{case}`` block: the patient state, then the reference material."""
        state = prompting.render_case(getattr(window, "input", window))
        return f"{state}\n\n{REFERENCE_HEADER}\n\n{self.context().strip()}"

    def build_prompt(self, window):
        """The final prompt: the zero_shot template with the extended case block."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_prompt_from_case(self.case_block(window), template)

    def build_messages(self, window):
        """Chat messages for one window (system + case + reference)."""
        return [
            {"role": "system", "content": prompting.SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(window)},
        ]

    def predict(self, window) -> Prediction:
        prompt = self.build_prompt(window)
        response = self.ask([
            {"role": "system", "content": prompting.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        prediction = self.prediction_from(window, response, parsing.parse_action(response.text))
        prediction.meta.update(
            context_file=str(self.context_file),
            context_chars=len(self.context()),
            prompt_chars=len(prompt),
        )
        return prediction
