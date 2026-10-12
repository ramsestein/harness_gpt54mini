# -*- coding: utf-8 -*-
"""SHAP-prompt strategy: the zero-shot call plus the SHAP analysis of the tree model.

``zero_shot`` renders the decision window as clinical prose and asks for one of five actions.
This strategy does exactly the same and hands the model, as reference information, the
**results of the SHAP analysis already run on the gradient-boosting control**: the ranked list
of the window variables by how much they move that model's answer, with their average absolute
contribution.

The model keeps the full picture of the patient (the shared ``render_case``, like every
strategy) *and* the analysis, so it can use the ranking as advice.

The ranking is read at run time from
``results_harness/control/_analysis/shap_importance.csv`` (written by
``python -m src.control.xgboost_diagnostics``), so it always follows the current model. The
whole ranking is pasted, every variable of the shared view the model was fitted on; the
launcher's ``--samples`` flag is accepted and ignored, so no run can cut it.
"""

import csv
import threading
from pathlib import Path

from src import parsing, prompting, view
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent

#: The SHAP analysis of the tree control.
DEFAULT_IMPORTANCE_PATH = (
    HERE.parents[1] / "results_harness" / "control" / "_analysis" / "shap_importance.csv"
)

#: The prompt is ``zero_shot``'s, so the two runs differ in the reference and nothing else.
DEFAULT_PROMPT_FILE = HERE.parent / "zero_shot" / "prompt.txt"

#: How the analysis is announced to the model.
REFERENCE_HEADER = (
    "What a model trained on thousands of past decisions of this kind pays attention to:\n"
    "the variables below are ranked by how much they change that model's answer, most "
    "influential first. The number is the average size of the variable's contribution to the "
    "model output (a mean absolute SHAP value), so it says how much the variable matters, not "
    "which decision it favours. Use it as a guide to what to look at in the patient state above."
)


class ShapPromptStrategy(Strategy):
    """Ask the model once per window, with the SHAP ranking of the tree model as reference."""

    name = "shap_prompt"
    description = (
        "Single call per window: the zero_shot clinical prose and options, preceded by the "
        "results of the SHAP analysis of the XGBoost control (the variables ranked by how much "
        "they influence that model). Same prompt as zero_shot otherwise."
    )

    def __init__(self, prompt_file=None, importance_path=None, rows=None, n_samples=None, k=None,
                 **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        self.importance_path = (
            Path(importance_path) if importance_path else DEFAULT_IMPORTANCE_PATH
        )
        # ``n_samples``/``k`` (the launcher's --samples) are accepted and ignored: the whole
        # ranking is always pasted. ``rows`` stays for tests and one-off experiments.
        self.rows = None if rows is None else max(1, int(rows))
        self._ranking = None
        self._lock = threading.Lock()

    def ranking(self):
        """``[(feature, value)]`` in ranked order, read once from the SHAP report."""
        if self._ranking is None:
            with self._lock:
                if self._ranking is None:
                    path = Path(self.importance_path)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"no SHAP report at {path}: run "
                            "'python -m src.control.xgboost_diagnostics' first"
                        )
                    entries = []
                    with path.open(encoding="utf-8", newline="") as handle:
                        for row in csv.DictReader(handle):
                            try:
                                rank = int(row["rank"])
                                value = float(row["mean_abs_shap"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            entries.append((rank, row["feature"], value))
                    entries.sort()
                    hidden = view.hidden_in(feature for _, feature, _ in entries)
                    if hidden:
                        # a report of a model fitted on another view: its numbers describe
                        # fields no strategy may see, so it is not advice for this one
                        raise ValueError(
                            f"{path} ranks fields outside the shared view "
                            f"({', '.join(hidden[:5])}...): retrain the tree control and rerun "
                            "'python -m src.control.xgboost_diagnostics'"
                        )
                    self._ranking = [(feature, value) for _, feature, value in entries]
        return self._ranking

    def shown(self):
        """The rows the prompt shows: the top ``rows`` of the ranking, or all of them."""
        ranking = self.ranking()
        return ranking[: self.rows] if self.rows is not None else ranking

    def render_reference(self):
        """The reference block: the ranked variables and their contributions, as a table."""
        lines = [f"{position}. {feature}: {value:.4f}"
                 for position, (feature, value) in enumerate(self.shown(), start=1)]
        return f"{REFERENCE_HEADER}\n\n" + "\n".join(lines)

    def case_block(self, window):
        """The ``{case}`` block: the patient state, then the SHAP analysis."""
        state = prompting.render_case(getattr(window, "input", window))
        return f"{state}\n\n{self.render_reference()}"

    def build_prompt(self, window):
        """The final prompt: the zero_shot template with the extended case block."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_prompt_from_case(self.case_block(window), template)

    def build_messages(self, window):
        """Chat messages for one window (system + case + SHAP ranking)."""
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
            importance_path=str(self.importance_path),
            rows_shown=len(self.shown()),
            k=len(self.shown()),  # the leaderboard's knob column: how much of the analysis
            prompt_chars=len(prompt),
        )
        return prediction
