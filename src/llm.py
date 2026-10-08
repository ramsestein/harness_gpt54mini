# -*- coding: utf-8 -*-
"""LLM client (OpenAI-compatible API) used by every harness.

STATE: implemented. The contract is defined in
``tests/test_llm_client.py`` and ``tests/test_llm_live.py``.

Contract
-----------------
``LLMError(Exception)``                    project error, wraps SDK failures.

``LLMResponse`` (dataclass)
    ``text: str``, ``model: str``, ``usage: dict``, ``latency_s: float``,
    ``raw: object``, ``dropped_params: list[str]``

``LLMClient``
    ``__init__(self, settings=None, client=None)``  ``client`` allows injecting a
        double (the tests use a fake exposing ``.chat.completions.create``).
    ``build_payload(messages, **kwargs) -> dict``  request payload
    ``complete(self, prompt, **kwargs) -> LLMResponse``
        sends ``messages=[{"role": "user", "content": prompt}]``
    ``complete_messages(self, messages, **kwargs) -> LLMResponse``
    relevant kwargs: ``model``, ``temperature``, ``reasoning_effort``, ``max_tokens``,
        ``response_format``, ``extra_body``

Generation settings
    ``temperature`` defaults to 0 (deterministic) and ``reasoning_effort`` to
    "high" (as much reasoning as the model allows). ``reasoning_effort`` travels
    inside ``extra_body`` so it works with every SDK version. Models that reject
    a parameter (reasoning models, for instance, usually accept only their default
    temperature) have it dropped on the first "unsupported/unknown parameter"
    error; the rejection is remembered by the client, so only the first call pays
    for the failed request. Whatever was dropped is reported in
    ``LLMResponse.dropped_params``.

``get_client() -> LLMClient``    default client (does NOT hit the network on construction)
``complete(prompt, **kwargs) -> str``   shortcut -> ``get_client().complete(...).text``

SDK errors are retried up to ``settings.max_retries`` times and, if they persist,
re-raised as ``LLMError``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from . import config

__all__ = [
    "LLMError",
    "LLMResponse",
    "LLMClient",
    "get_client",
    "complete_messages",
    "complete",
]


class LLMError(Exception):
    """Error raised when invoking the model."""


@dataclass
class LLMResponse:
    """A single model completion."""

    text: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: object = None
    dropped_params: list = field(default_factory=list)  # rejected by the model


#: Request parameters some models reject. They are dropped on the first
#: "unsupported/unknown parameter" error instead of failing the whole run.
_DROPPABLE_PARAMS = ("temperature", "reasoning_effort")

_UNSUPPORTED_HINTS = (
    "unsupported",
    "not supported",
    "does not support",
    "unknown",
    "unrecognized",
    "unexpected",
    "invalid",
)

_UNSET = object()


def _strip_unsupported(payload, error):
    """Drop the request parameter the API rejected.

    Returns ``(name, value)`` for the parameter removed, or ``None``.
    """
    message = str(error).lower()
    if not any(hint in message for hint in _UNSUPPORTED_HINTS):
        return None
    if "temperature" in payload and "temperature" in message:
        return "temperature", payload.pop("temperature")
    extra = payload.get("extra_body")
    if isinstance(extra, dict) and "reasoning_effort" in extra and "reasoning" in message:
        value = extra.pop("reasoning_effort")
        if not extra:
            payload.pop("extra_body", None)
        return "reasoning_effort", value
    return None


class LLMClient:
    """Thin wrapper over the OpenAI-compatible chat completions API."""

    def __init__(self, settings=None, client=None):
        self.settings = settings if settings is not None else config.get_settings()
        self._client = client
        self._lock = threading.Lock()
        #: Parameters this model has already rejected, with the rejected value
        #: (e.g. ``{"temperature": 0}`` on a reasoning model): that exact value is
        #: not sent again, but other values still are.
        self._unsupported = {}

    @property
    def client(self):
        """Lazily build the SDK client (no network traffic at construction time).

        Thread-safe: the harness runs windows in parallel and the client is shared.
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - the dependency is installed
            raise LLMError("the 'openai' package is required to call the model") from exc
        if not self.settings.api_key:
            raise LLMError("no API key: set OPENAI_API_KEY in .env or the environment")
        kwargs = {"api_key": self.settings.api_key, "timeout": self.settings.timeout}
        if self.settings.base_url:
            kwargs["base_url"] = self.settings.base_url
        return OpenAI(**kwargs)

    def build_payload(self, messages, model=None, temperature=_UNSET,
                      reasoning_effort=_UNSET, extra_body=None, **kwargs):
        """Build the chat-completions payload, applying the generation settings."""
        payload = {"model": model or self.settings.model, "messages": messages}

        if temperature is _UNSET:
            temperature = self.settings.temperature
        if temperature is not None:
            payload["temperature"] = temperature

        if reasoning_effort is _UNSET:
            reasoning_effort = self.settings.reasoning_effort
        extra = dict(extra_body or {})
        if reasoning_effort:
            extra.setdefault("reasoning_effort", reasoning_effort)

        payload.update(kwargs)
        if extra:
            payload["extra_body"] = extra
        return payload

    def complete(self, prompt, **kwargs):
        """Complete a single user prompt."""
        return self.complete_messages([{"role": "user", "content": prompt}], **kwargs)

    def complete_messages(self, messages, **kwargs):
        """Complete a chat ``messages`` list and return an ``LLMResponse``."""
        payload = self.build_payload(messages, **kwargs)
        dropped = self._drop_known_unsupported(payload)
        attempts = max(1, int(self.settings.max_retries) + 1)
        last_error = None
        for _ in range(attempts):
            started = time.perf_counter()
            try:
                response = self._create(payload, dropped)
            except Exception as exc:
                last_error = exc
                continue
            return _to_response(response, time.perf_counter() - started, dropped)
        raise LLMError(f"model call failed after {attempts} attempt(s): {last_error}") from last_error

    def _drop_known_unsupported(self, payload):
        """Remove the parameters (with the rejected value) from earlier calls."""
        with self._lock:
            known = dict(self._unsupported)
        if not known:
            return []
        dropped = []
        if "temperature" in payload and known.get("temperature") == payload["temperature"]:
            payload.pop("temperature", None)
            dropped.append("temperature")
        extra = payload.get("extra_body")
        if isinstance(extra, dict) and "reasoning_effort" in extra:
            if known.get("reasoning_effort") == extra["reasoning_effort"]:
                extra.pop("reasoning_effort", None)
                dropped.append("reasoning_effort")
                if not extra:
                    payload.pop("extra_body", None)
        return dropped

    def _create(self, payload, dropped):
        """POST the request, dropping parameters the model rejects."""
        for _ in range(len(_DROPPABLE_PARAMS) + 1):
            try:
                return self.client.chat.completions.create(**payload)
            except Exception as exc:
                rejected = _strip_unsupported(payload, exc)
                if rejected is None:
                    raise
                name, value = rejected
                with self._lock:
                    self._unsupported[name] = value
                dropped.append(name)
        raise LLMError("request rejected even after dropping unsupported parameters")


def _to_response(response, latency_s, dropped=None):
    """Normalise a chat completion into an ``LLMResponse``."""
    choices = getattr(response, "choices", None) or []
    text = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) or ""
    usage = getattr(response, "usage", None)
    usage_dict = {}
    if isinstance(usage, dict):
        usage_dict = dict(usage)
    elif usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if value is not None:
                usage_dict[key] = value
    return LLMResponse(
        text=text,
        model=getattr(response, "model", "") or "",
        usage=usage_dict,
        latency_s=latency_s,
        raw=response,
        dropped_params=list(dropped or []),
    )


_DEFAULT_CLIENT = None


def get_client():
    """Return the default LLM client (no network on construction)."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT


def complete_messages(messages, **kwargs):
    """Shortcut: complete a chat message list with the default client."""
    return get_client().complete_messages(messages, **kwargs)


def complete(prompt, **kwargs):
    """Shortcut: return the text of the model response."""
    return get_client().complete(prompt, **kwargs).text
