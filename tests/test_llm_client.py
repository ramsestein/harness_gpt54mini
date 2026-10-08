# -*- coding: utf-8 -*-
"""LLM client with an injected double (no network)."""

import pytest

from src import config, llm


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 1
    total_tokens = 11


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.model = "gpt-5.4-mini"
        self.usage = _FakeUsage()
        self.id = "resp_1"


class _FakeCompletions:
    def __init__(self, reply="a", error=None):
        self.reply = reply
        self.error = error
        self.last_kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.reply)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    """Double of the official client: only ``.chat.completions.create``."""

    def __init__(self, reply="a", error=None):
        self.completions = _FakeCompletions(reply=reply, error=error)
        self.chat = _FakeChat(self.completions)

    @property
    def last_kwargs(self):
        return self.completions.last_kwargs

    @property
    def calls(self):
        return self.completions.calls


class _FakeCompletionsRejecting:
    """Raises a queue of errors and then answers; records every call."""

    def __init__(self, errors, reply="a"):
        self.errors = list(errors)
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.errors:
            raise self.errors.pop(0)
        return _FakeResponse(self.reply)


class FakeOpenAIRejecting:
    """Double that rejects parameters before answering."""

    def __init__(self, errors, reply="a"):
        self.completions = _FakeCompletionsRejecting(errors, reply=reply)
        self.chat = _FakeChat(self.completions)


@pytest.fixture
def settings():
    return config.Settings(
        model="gpt-5.4-mini",
        api_key="test-key",
        base_url=None,
        timeout=30.0,
        max_retries=0,
    )


def test_default_base_model():
    assert config.DEFAULT_MODEL == "gpt-5.4-mini"
    assert config.get_settings().model


def test_complete_uses_the_default_model(settings):
    fake = FakeOpenAI(reply="b")
    client = llm.LLMClient(settings=settings, client=fake)

    resp = client.complete("hello")

    assert fake.last_kwargs["model"] == "gpt-5.4-mini"
    assert fake.last_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert resp.text == "b"


def test_complete_messages_passes_messages_through(settings):
    fake = FakeOpenAI(reply="c")
    client = llm.LLMClient(settings=settings, client=fake)
    messages = [
        {"role": "system", "content": "you are an anesthesiologist"},
        {"role": "user", "content": "decide"},
    ]

    client.complete_messages(messages)

    assert fake.last_kwargs["messages"] == messages


def test_model_can_be_overridden(settings):
    fake = FakeOpenAI()
    client = llm.LLMClient(settings=settings, client=fake)

    client.complete("hello", model="another-model")

    assert fake.last_kwargs["model"] == "another-model"


def test_response_metadata(settings):
    fake = FakeOpenAI(reply="e")
    resp = llm.LLMClient(settings=settings, client=fake).complete("hello")

    assert resp.model == "gpt-5.4-mini"
    assert resp.usage["total_tokens"] == 11
    assert resp.latency_s >= 0.0
    assert resp.raw is not None


def test_sdk_errors_are_wrapped(settings):
    fake = FakeOpenAI(error=RuntimeError("boom"))
    client = llm.LLMClient(settings=settings, client=fake)

    with pytest.raises(llm.LLMError):
        client.complete("hello")


def test_get_client_does_not_touch_the_network():
    client = llm.get_client()
    assert isinstance(client, llm.LLMClient)


def test_default_generation_settings_are_deterministic_and_high_reasoning():
    defaults = config.Settings()
    assert config.DEFAULT_TEMPERATURE == 0.0
    assert config.DEFAULT_REASONING_EFFORT == "high"
    assert defaults.temperature == 0.0
    assert defaults.reasoning_effort == "high"


def test_generation_settings_are_sent(settings):
    fake = FakeOpenAI(reply="a")

    llm.LLMClient(settings=settings, client=fake).complete("hello")

    assert fake.last_kwargs["temperature"] == 0.0
    assert fake.last_kwargs["extra_body"] == {"reasoning_effort": "high"}


def test_generation_settings_can_be_overridden(settings):
    fake = FakeOpenAI(reply="a")
    client = llm.LLMClient(settings=settings, client=fake)

    client.complete("hello", temperature=0.7, reasoning_effort="low")

    assert fake.last_kwargs["temperature"] == 0.7
    assert fake.last_kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_rejected_parameters_are_dropped_and_reported(settings):
    errors = [
        RuntimeError("Unsupported value: 'temperature' does not support 0 with this model"),
        RuntimeError("Unknown parameter: 'reasoning_effort'"),
    ]
    fake = FakeOpenAIRejecting(errors, reply="b")
    client = llm.LLMClient(settings=settings, client=fake)

    response = client.complete("hello")

    assert response.text == "b"
    assert response.dropped_params == ["temperature", "reasoning_effort"]
    assert len(fake.completions.calls) == 3
    assert "temperature" not in fake.completions.calls[-1]
    assert "extra_body" not in fake.completions.calls[-1]


def test_rejected_parameters_are_remembered_across_calls(settings):
    errors = [RuntimeError("Unsupported value: 'temperature' does not support 0 with this model")]
    fake = FakeOpenAIRejecting(errors, reply="b")
    client = llm.LLMClient(settings=settings, client=fake)

    first = client.complete("hello")
    second = client.complete("hello again")

    assert first.dropped_params == ["temperature"]
    assert second.dropped_params == ["temperature"]
    assert len(fake.completions.calls) == 3  # one rejected, then two clean
    assert "temperature" not in fake.completions.calls[-1]


def test_only_the_rejected_value_is_remembered(settings):
    errors = [RuntimeError("Unsupported value: 'temperature' does not support 0 with this model")]
    fake = FakeOpenAIRejecting(errors, reply="b")
    client = llm.LLMClient(settings=settings, client=fake)

    client.complete("first")
    client.complete("second", temperature=0.5)

    # 0.0 was rejected, but 0.5 was never tried and must still be sent.
    assert fake.completions.calls[-1]["temperature"] == 0.5
