# -*- coding: utf-8 -*-
"""Simulated critic-verifier strategy.

A variation of ``critic_verifier`` in which the second step is a **simulation**
instead of a second opinion on the same question:

1. **Generator** -- ``generator_prompt.txt`` asks for the **two** most appropriate
   candidate actions instead of the single best one.
2. **Simulator** -- both candidates are applied to a patient calibrated from the
   window (see ``simulator.py``, built on the Python Anesthesia Simulator) and
   simulated five minutes ahead.
3. **Critic** -- ``critic_prompt.txt`` shows the two resulting windows **blind**
   (simulated signals only, in an order that varies per window) and asks which one
   is more beneficial for the patient. The chosen window points back to the action
   that produced it, and that action is the prediction.

This strategy uses its own system messages: its two roles (propose two, judge two)
do not match the single-call wording the other strategies share.
"""

import hashlib
import re
from pathlib import Path

from src import prompting
from src.harness_base import Prediction, Strategy

from harness.critic_verifier_sim import simulator as simulator_module

HERE = Path(__file__).resolve().parent
DEFAULT_GENERATOR_PROMPT = HERE / "generator_prompt.txt"
DEFAULT_CRITIC_PROMPT = HERE / "critic_prompt.txt"

GENERATOR_SYSTEM_PROMPT = (
    "You are an expert anesthesiologist in the middle of a real surgical case. "
    "You must propose the two most appropriate next anesthetic adjustments among "
    "the five options."
)

CRITIC_SYSTEM_PROMPT = (
    "You are an expert anesthesiologist judging the outcome of an anesthetic "
    "decision. You must state which of two simulated windows is more beneficial "
    "for the patient."
)

_CANDIDATE = re.compile(r"candidate\s*([12])\s*[:\-]?\s*\(?([A-Za-z_]+)\)?", re.IGNORECASE)
_ANSWER_CHOICE = re.compile(r"\banswer\b[^0-9\r\n]{0,12}?([12])\b", re.IGNORECASE)
_BARE_CHOICE = re.compile(r"\s*\(?([12])\)?\s*[.):]?\s*")


class SimulatedCriticStrategy(Strategy):
    """Propose two actions, simulate both, and let the critic pick the better future."""

    name = "critic_verifier_sim"
    description = (
        "Two calls per window plus a simulation: the generator proposes two "
        "candidate actions, both are simulated five minutes ahead on a patient "
        "calibrated from the window, and the critic blindly chooses the more "
        "beneficial simulated window; that choice is the prediction."
    )

    def __init__(self, generator_prompt_file=None, critic_prompt_file=None,
                 simulator=None, **options):
        super().__init__(**options)
        self.generator_prompt_file = (
            Path(generator_prompt_file) if generator_prompt_file else DEFAULT_GENERATOR_PROMPT
        )
        self.critic_prompt_file = (
            Path(critic_prompt_file) if critic_prompt_file else DEFAULT_CRITIC_PROMPT
        )
        self.simulator = simulator or simulator_module.AnesthesiaSimulator()

    # -- prompts ---------------------------------------------------------- #

    def build_generator_messages(self, window):
        """First call: the window plus a request for two candidate actions."""
        template = prompting.load_template(self.generator_prompt_file)
        return [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompting.build_prompt(window, template=template)},
        ]

    @staticmethod
    def format_outcome(outcome):
        """The blind block the critic sees: simulated signals only, no action name."""
        signals = outcome.signals()
        return (
            f"BIS (bispectral index): {signals['bis']}\n"
            f"MAP (mean arterial pressure): {signals['map']} mmHg\n"
            f"CO (cardiac output): {signals['co']} L/min\n"
            f"TOL (tolerance of laryngoscopy): {signals['tol']}"
        )

    def build_critic_messages(self, outcomes, order):
        """Second call: the two simulated windows, blind, in the given order."""
        prompt = prompting.load_template(self.critic_prompt_file)
        for position, index in enumerate(order, start=1):
            prompt = prompt.replace(
                f"{{window_{position}}}", self.format_outcome(outcomes[index])
            )
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def build_messages(self, window):
        """Messages of the first call; used by the launcher's ``--dry-run``."""
        return self.build_generator_messages(window)

    # -- parsing ---------------------------------------------------------- #

    @staticmethod
    def _as_action(token):
        """A letter (``c``) or an action name (``increase_opioid``) as an action."""
        token = (token or "").strip().lower()
        if token in prompting.LETTER_TO_ACTION:
            return prompting.LETTER_TO_ACTION[token]
        return token if token in prompting.ACTION_TO_LETTER else None

    @classmethod
    def parse_candidates(cls, text):
        """The distinct actions after ``Candidate 1:`` / ``Candidate 2:``, in order."""
        found = {}
        for match in _CANDIDATE.finditer(text or ""):
            found.setdefault(int(match.group(1)), cls._as_action(match.group(2)))
        candidates = []
        for position in sorted(found):
            action = found[position]
            if action and action not in candidates:
                candidates.append(action)
        return candidates

    @staticmethod
    def parse_choice(text):
        """The ``1``/``2`` the critic picked, or ``None``."""
        text = text or ""
        labelled = list(_ANSWER_CHOICE.finditer(text))
        if labelled:
            return int(labelled[-1].group(1))
        bare = _BARE_CHOICE.fullmatch(text)
        return int(bare.group(1)) if bare else None

    @staticmethod
    def present_order(window_id):
        """Which outcome goes first, per window.

        Deterministic (so a run is reproducible) but not constant, so a position
        bias in the critic cannot become a systematic bias in the results.
        """
        digest = hashlib.sha256(str(window_id).encode("utf-8")).hexdigest()
        return [0, 1] if int(digest[:8], 16) % 2 == 0 else [1, 0]

    # -- prediction ------------------------------------------------------- #

    def predict(self, window) -> Prediction:
        generator = self.ask(self.build_generator_messages(window))
        candidates = self.parse_candidates(generator.text)

        if len(candidates) < 2:
            prediction = self.prediction_from(window, generator, None)
            prediction.meta.update(
                generator_raw=generator.text,
                candidates=candidates,
                error="the generator did not propose two different candidates",
            )
            return prediction

        calibration, outcomes = self.simulator.simulate(window.input, candidates)
        order = self.present_order(window.window_id)
        critic = self.ask(self.build_critic_messages(outcomes, order))
        choice = self.parse_choice(critic.text)

        # Without a usable choice the first presented window is kept, and the
        # fallback is flagged so it can be filtered out during analysis.
        index = order[choice - 1] if choice is not None else order[0]
        chosen = outcomes[index]

        prediction = self.prediction_from(window, critic, chosen.action)
        prediction.meta.update(
            generator_raw=generator.text,
            candidates=candidates,
            calibration=calibration.as_dict(),
            simulated=[outcome.as_dict() for outcome in outcomes],
            presented=[outcomes[position].action for position in order],
            critic_raw=critic.text,
            choice=choice,
            chosen=chosen.action,
            critic_parsed=choice is not None,
            fallback=choice is None,
            latencies_s=[round(generator.latency_s, 3), round(critic.latency_s, 3)],
            total_latency_s=round(generator.latency_s + critic.latency_s, 3),
            dropped_params=sorted(
                set(generator.dropped_params) | set(critic.dropped_params)
            ),
        )
        return prediction
