# -*- coding: utf-8 -*-
"""Self-consistency strategy: zero_shot sampled N times + majority vote.

Everything is the same as ``zero_shot`` -- the prompt, the five options, the
single-letter answer, the parsing -- except the aggregation: the model is called
``n_samples`` times and the prediction is the majority over the parsed answers.

The prompt file is deliberately shared with ``zero_shot``
(``harness/zero_shot/prompt.txt``), so the only difference between the two
strategies is the consistency step.
"""

from pathlib import Path

from src import dataset, parsing, prompting
from src.harness_base import Prediction, Strategy

HERE = Path(__file__).resolve().parent

#: Same prompt as zero_shot on purpose: only the aggregation must differ.
DEFAULT_PROMPT_FILE = HERE.parent / "zero_shot" / "prompt.txt"


class SelfConsistencyStrategy(Strategy):
    """Sample the zero-shot prompt N times and take the majority action."""

    name = "self_consistency"
    description = (
        "Same call as zero_shot, repeated n_samples times; the prediction is the "
        "majority vote over the parsed answers."
    )

    def __init__(self, prompt_file=None, n_samples=5, temperature=1.0, **options):
        # Sampling needs stochasticity, so this strategy asks for temperature 1.0
        # by default. Models that reject any non-default temperature drop it
        # (see ``LLMResponse.dropped_params``) and diversity comes from the model.
        super().__init__(temperature=temperature, **options)
        self.prompt_file = Path(prompt_file) if prompt_file else DEFAULT_PROMPT_FILE
        self.n_samples = max(1, int(n_samples))

    def build_messages(self, window):
        """Chat messages for one window (identical to zero_shot's)."""
        template = prompting.load_template(self.prompt_file)
        return prompting.build_messages(window, template=template)

    @staticmethod
    def vote(actions):
        """Most frequent action (``None`` samples do not vote).

        Ties are resolved by ``dataset.ACTIONS`` order, so the result is
        deterministic. Returns ``(winner, counts)``.
        """
        counts = {}
        for action in actions:
            if action is not None:
                counts[action] = counts.get(action, 0) + 1
        if not counts:
            return None, counts

        def rank(action):
            return counts[action], -dataset.ACTIONS.index(action)

        return max(counts, key=rank), counts

    def predict(self, window) -> Prediction:
        messages = self.build_messages(window)
        samples = [self.ask(messages) for _ in range(self.n_samples)]
        actions = [parsing.parse_action(sample.text) for sample in samples]
        winner, votes = self.vote(actions)
        parsed = [action for action in actions if action is not None]

        # ``raw_output`` is the first sample that voted for the winning action.
        representative = next(
            (sample for sample, action in zip(samples, actions) if action == winner),
            samples[0],
        )
        prediction = self.prediction_from(window, representative, winner)
        prediction.meta.update(
            n_samples=self.n_samples,
            parsed_samples=len(parsed),
            votes=votes,
            agreement=(votes.get(winner, 0) / len(parsed)) if parsed else 0.0,
            samples=[sample.text for sample in samples],
            latencies_s=[round(sample.latency_s, 3) for sample in samples],
            total_latency_s=round(sum(sample.latency_s for sample in samples), 3),
            dropped_params=sorted(
                {param for sample in samples for param in sample.dropped_params}
            ),
        )
        return prediction
