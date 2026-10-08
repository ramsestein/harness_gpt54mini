# -*- coding: utf-8 -*-
"""Few-shot strategy: retrieve the most similar training windows as examples.

For every window to predict (from ``dev`` or ``test``) the strategy looks up the
``k`` most similar windows of the ``train`` split -- similarity over the window's
numeric variables only, never over the labels -- and shows them in the prompt
together with the adjustment the clinician actually made.

The retrieval index is built once from ``train`` and cached under
``results_harness/few_shot/_index/``; ``harness/few_shot/build_index.py`` can
prebuild it.
"""

import threading
from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

from harness.few_shot import retrieval

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = HERE / "prompt.txt"


class FewShotStrategy(Strategy):
    """Ask the model once per window, with its nearest training windows as examples."""

    name = "few_shot"
    description = (
        "Single call per window: the k most similar training windows (numeric-feature "
        "similarity, labels excluded) are shown as labelled examples before the "
        "current case, plus the same five-option prompt (a-e)."
    )

    def __init__(self, prompt_file=None, k=3, index=None, index_path=None, index_cases=None,
                 exclude_same_case=True, n_samples=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        # ``n_samples`` is accepted as an alias so the launcher's --samples flag works.
        if n_samples is not None:
            k = n_samples
        self.k = max(1, int(k))
        self.exclude_same_case = bool(exclude_same_case)
        self.index_path = Path(index_path) if index_path else retrieval.DEFAULT_INDEX_PATH
        self.index_cases = index_cases
        self._index = index
        self._lock = threading.Lock()

    def get_index(self):
        """The retrieval index, built from ``train`` on first use (thread-safe)."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = retrieval.load_or_build(
                        path=self.index_path, cases=self.index_cases
                    )
        return self._index

    def retrieve(self, window):
        """The ``k`` nearest training windows for this window."""
        return self.get_index().query(
            window,
            k=self.k,
            exclude_case=window.case_id if self.exclude_same_case else None,
        )

    @staticmethod
    def format_examples(neighbors):
        """Render the retrieved windows as prompt blocks, with their real action."""
        blocks = []
        for position, neighbor in enumerate(neighbors, start=1):
            blocks.append(
                f"--- example {position} ---\n"
                f"{neighbor.prompt}\n"
                f"Adjustment the clinician made: {neighbor.action}"
            )
        return "\n\n".join(blocks)

    def build_prompt(self, window, neighbors=None):
        """The user prompt: examples first, then the current case."""
        neighbors = self.retrieve(window) if neighbors is None else list(neighbors)
        template = prompting.load_template(self.prompt_file)
        prompt = template.replace("{n_examples}", str(len(neighbors)))
        prompt = prompt.replace("{examples}", self.format_examples(neighbors))
        return prompt.replace("{case}", prompting.render_case(window.input))

    def build_messages(self, window, neighbors=None):
        """Chat messages for one window (system + examples + current case)."""
        return [
            {"role": "system", "content": prompting.SYSTEM_PROMPT},
            {"role": "user", "content": self.build_prompt(window, neighbors=neighbors)},
        ]

    def predict(self, window) -> Prediction:
        neighbors = self.retrieve(window)
        response = self.ask(self.build_messages(window, neighbors=neighbors))
        prediction = self.prediction_from(window, response, parsing.parse_action(response.text))
        prediction.meta.update(
            k=self.k,
            neighbors=[
                {
                    "window_id": neighbor.window_id,
                    "case_id": neighbor.case_id,
                    "action": neighbor.action,
                    "distance": round(neighbor.distance, 4),
                }
                for neighbor in neighbors
            ],
        )
        return prediction
