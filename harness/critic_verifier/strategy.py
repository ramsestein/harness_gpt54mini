# -*- coding: utf-8 -*-
"""Critic-verifier strategy: generate, then critically review.

Two calls per window:

1. **Generator** -- the very same prompt as ``zero_shot`` (shared file, so the
   proposal is directly comparable with the zero-shot baseline) produces a letter.
2. **Critic** -- ``critic_prompt.txt`` receives the same window plus the proposed
   action and either confirms it or corrects it, again with one letter.

The critic has the final say. If its answer cannot be parsed, the proposal is
kept; if neither parses, the prediction falls back to ``no_action``.
"""

from pathlib import Path

from src import parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent

#: Same prompt as zero_shot on purpose: the generator must be the baseline.
DEFAULT_PROMPT_FILE = HERE.parent / "zero_shot" / "prompt.txt"
DEFAULT_CRITIC_PROMPT_FILE = HERE / "critic_prompt.txt"


class CriticVerifierStrategy(Strategy):
    """Propose with the zero-shot prompt, then let a critic confirm or correct."""

    name = "critic_verifier"
    description = (
        "Two calls per window: the zero-shot prompt proposes an action and a second "
        "critic prompt confirms or corrects it; the critic's answer wins."
    )

    def __init__(self, prompt_file=None, critic_prompt_file=None, **options):
        super().__init__(**options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        self.critic_prompt_file = (
            Path(critic_prompt_file) if critic_prompt_file else DEFAULT_CRITIC_PROMPT_FILE
        )

    def build_generator_messages(self, window):
        """First call: identical to zero_shot."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_messages(window, template=template)

    def build_critic_messages(self, window, proposal):
        """Second call: the same window plus the proposal to review."""
        template = prompting.load_template(self.critic_prompt_file)
        prompt = template.replace("{case}", prompting.render_case(window.input))
        prompt = prompt.replace(
            "{proposal}", proposal or "none (the first answer could not be parsed)"
        )
        return [
            {"role": "system", "content": prompting.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def build_messages(self, window):
        """Messages of the first call; used by the launcher's ``--dry-run``."""
        return self.build_generator_messages(window)

    def predict(self, window) -> Prediction:
        generator = self.ask(self.build_generator_messages(window))
        proposal = parsing.parse_action(generator.text)

        critic = self.ask(self.build_critic_messages(window, proposal))
        reviewed = parsing.parse_action(critic.text)
        final = reviewed if reviewed is not None else proposal

        prediction = self.prediction_from(window, critic, final)
        prediction.meta.update(
            proposal=proposal,
            final=final,
            changed=(proposal is not None and reviewed is not None and proposal != reviewed),
            generator_parsed=proposal is not None,
            critic_parsed=reviewed is not None,
            generator_raw=generator.text,
            critic_raw=critic.text,
            latencies_s=[round(generator.latency_s, 3), round(critic.latency_s, 3)],
            total_latency_s=round(generator.latency_s + critic.latency_s, 3),
            dropped_params=sorted(set(generator.dropped_params) | set(critic.dropped_params)),
        )
        return prediction
