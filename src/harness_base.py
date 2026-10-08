# -*- coding: utf-8 -*-
"""Common interface that every harness strategy must implement.

The contract is defined in ``tests/test_harness_structure.py``.
The interface (``Prediction`` + ``Strategy``) is the contract specified by the
tests; the concrete strategies live under ``harness/<name>/strategy.py``.

Contract
--------
``Prediction`` (dataclass)
    ``window_id: str``, ``action: str``, ``raw_output: str``,
    ``meta: dict = field(default_factory=dict)``

``Strategy`` (ABC)
    class attributes: ``name: str``, ``description: str``
    ``__init__(self, model=_UNSET, temperature=_UNSET, reasoning_effort=_UNSET, model_options=None, **options)``
        generation settings shared by every strategy; they default to
        ``config.get_settings()`` and an explicit ``None`` omits the parameter
        from the request. ``model_options`` holds extra model-call kwargs
        (e.g. ``max_tokens``); any other keyword belongs to the strategy and is
        **not** forwarded to the model.
    ``generation_options() -> dict`` model-call kwargs from those settings
    ``ask(messages) -> LLMResponse``  one model call with those settings
    ``prediction_from(window, response, action) -> Prediction``
        standard ``Prediction`` including the generation metadata
    ``predict(self, window) -> Prediction``
    ``predict_many(self, windows) -> list[Prediction]``
        default implementation: ``[self.predict(w) for w in windows]``

Every ``harness/<name>/strategy.py`` must expose exactly one concrete subclass
of ``Strategy`` whose ``name`` matches the folder name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from . import config, llm

__all__ = ["Prediction", "Strategy", "get_strategy_class", "UNSET"]

#: Sentinel meaning "take this generation setting from ``config.get_settings()``".
UNSET = object()
_UNSET = UNSET


def _normalise_effort(value):
    """``"none"``/``"off"``/empty mean "do not send ``reasoning_effort``"."""
    if isinstance(value, str) and value.strip().lower() in ("none", "off", ""):
        return None
    return value


@dataclass
class Prediction:
    """A strategy's result for a single decision window."""

    window_id: str
    action: str
    raw_output: str
    meta: dict = field(default_factory=dict)


class Strategy(ABC):
    """Interface implemented by every strategy under ``harness/``.

    The constructor centralises the *generation settings* so every strategy
    behaves the same. ``model``, ``temperature`` and ``reasoning_effort`` default
    to ``config.get_settings()`` (temperature 0, reasoning "high"); passing an
    explicit ``None`` omits that parameter from the request.
    """

    name: str = ""
    description: str = ""
    #: ``False`` for the control baselines (``src/control/``): they answer without
    #: a model, so the run metadata must not claim one.
    uses_model: bool = True

    def __init__(self, model=_UNSET, temperature=_UNSET, reasoning_effort=_UNSET,
                 model_options=None, **options):
        settings = config.get_settings()
        self.model = settings.model if model is _UNSET or model is None else model
        self.temperature = settings.temperature if temperature is _UNSET else temperature
        if reasoning_effort is _UNSET:
            reasoning_effort = settings.reasoning_effort
        self.reasoning_effort = _normalise_effort(reasoning_effort)
        #: Extra keyword arguments for the model call.
        self.model_options = dict(model_options or {})
        #: Strategy-specific options; never forwarded to the model.
        self.options = options

    def generation_options(self):
        """Model-call keyword arguments derived from the generation settings."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    def ask(self, messages):
        """Make one model call with this strategy's generation settings."""
        return llm.complete_messages(messages, **self.generation_options(), **self.model_options)

    def prediction_from(self, window, response, action):
        """Build the standard ``Prediction``, including the generation metadata."""
        return Prediction(
            window_id=window.window_id,
            action=action or "no_action",
            raw_output=response.text,
            meta={
                "model": response.model or self.model,
                "latency_s": response.latency_s,
                "parsed": action is not None,
                "temperature": self.temperature,
                "reasoning_effort": self.reasoning_effort,
                "dropped_params": list(getattr(response, "dropped_params", []) or []),
            },
        )

    @abstractmethod
    def predict(self, window) -> Prediction:
        """Predict the action for a decision window."""
        raise NotImplementedError

    def predict_many(self, windows) -> list:
        """Predict a list of windows (sequential by default)."""
        return [self.predict(w) for w in windows]


def get_strategy_class(module):
    """Return the concrete ``Strategy`` subclass defined in ``module``.

    Raises ``LookupError`` when there is not exactly one concrete subclass.
    """
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, Strategy)
        and value is not Strategy
        and value.__module__ == getattr(module, "__name__", None)
    ]
    if len(candidates) != 1:
        raise LookupError(
            f"expected exactly one Strategy subclass in "
            f"{getattr(module, '__name__', module)!r}, found {len(candidates)}"
        )
    return candidates[0]
