# -*- coding: utf-8 -*-
"""Smoke tests against the real model (marked ``live``).

Run them with ``pytest -m live``; disable them with ``ANESLLM_SKIP_LIVE=1``.
"""

import pytest

from src import config, dataset, harness_registry, llm

pytestmark = pytest.mark.live


def test_api_key_is_available():
    settings = config.get_settings()
    assert settings.api_key, "OPENAI_API_KEY is missing (.env or environment)"
    assert settings.model


def test_model_smoke_call():
    response = llm.get_client().complete("Reply with exactly the letter a.")
    assert response.text.strip() != ""
    assert response.model


def test_zero_shot_strategy_returns_a_valid_action():
    window = dataset.load_split("test", limit=1)[0]
    strategy = harness_registry.load_strategy("zero_shot")

    prediction = strategy.predict(window)

    assert prediction.window_id == window.window_id
    assert prediction.action in dataset.ACTIONS
    assert prediction.raw_output.strip() != ""
