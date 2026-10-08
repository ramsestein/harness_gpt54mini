# -*- coding: utf-8 -*-
"""Global configuration: paths, ``.env`` loading and model settings.

Implemented. The contract is defined in ``tests/``.

Expected contract
-----------------
Constants
    ``DEFAULT_MODEL``   -> "gpt-5.4-mini" (the project's base model)
    ``PROJECT_ROOT``    -> repository root
    ``DATASET_DIR``     -> <root>/dataset
    ``DATA_DIR``        -> <root>/dataset/data
    ``REPORTS_DIR``     -> <root>/dataset/reports
    ``SCRIPTS_DIR``     -> <root>/dataset/scripts
    ``HARNESS_DIR``     -> <root>/harness
    ``RESULTS_DIR``     -> <root>/results_harness
    ``TESTS_DIR``       -> <root>/tests

Functions
    ``load_env(path=None) -> dict``   loads ``.env`` into ``os.environ`` without
                                      overwriting already-set variables.
    ``get_settings() -> Settings``
    ``Settings``                      frozen dataclass with fields, in this
                                      order: ``model``, ``api_key``,
                                      ``base_url``, ``timeout``,
                                      ``max_retries``, ``temperature``,
                                      ``reasoning_effort``.

Recognised environment variables
    ``ANESLLM_MODEL``             -> model (default ``DEFAULT_MODEL``)
    ``OPENAI_API_KEY``            -> api_key
    ``ANESLLM_BASE_URL``          -> base_url (falls back to ``OPENAI_BASE_URL``)
    ``ANESLLM_TIMEOUT``           -> timeout in seconds (default 60.0)
    ``ANESLLM_MAX_RETRIES``       -> max_retries (default 3)
    ``ANESLLM_TEMPERATURE``       -> temperature (default 0.0; ``none`` omits it)
    ``ANESLLM_REASONING_EFFORT``  -> reasoning_effort (default "high"; ``none`` omits it)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_REASONING_EFFORT",
    "PROJECT_ROOT",
    "DATASET_DIR",
    "DATA_DIR",
    "REPORTS_DIR",
    "SCRIPTS_DIR",
    "HARNESS_DIR",
    "RESULTS_DIR",
    "TESTS_DIR",
    "Settings",
    "load_env",
    "get_settings",
]

DEFAULT_MODEL = "gpt-5.4-mini"

#: Deterministic sampling: the project always asks for temperature 0.
DEFAULT_TEMPERATURE = 0.0

#: As much reasoning as the model allows.
DEFAULT_REASONING_EFFORT = "high"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = DATASET_DIR / "data"
REPORTS_DIR = DATASET_DIR / "reports"
SCRIPTS_DIR = DATASET_DIR / "scripts"
HARNESS_DIR = PROJECT_ROOT / "harness"
RESULTS_DIR = PROJECT_ROOT / "results_harness"
TESTS_DIR = PROJECT_ROOT / "tests"

ENV_PATH = PROJECT_ROOT / ".env"

_ENV_LOADED = False


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by the model client."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float | None = DEFAULT_TEMPERATURE
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_optional_float(name, default):
    """Float from the environment; ``none``/``null``/``off``/empty mean "omit it"."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("", "none", "null", "off"):
        return None
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_str(name, default):
    """String from the environment; ``none``/``null``/``off``/empty mean "omit it"."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in ("", "none", "null", "off"):
        return None
    return raw


def load_env(path=None):
    """Load the variables from ``.env`` into ``os.environ`` (without overwriting).

    Returns the mapping of variables found in the file.
    """
    env_path = Path(path) if path is not None else ENV_PATH
    loaded = {}
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def get_settings():
    """Return a ``Settings`` instance built from the environment."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_env()
        _ENV_LOADED = True
    return Settings(
        model=os.environ.get("ANESLLM_MODEL", DEFAULT_MODEL),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("ANESLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        timeout=_env_float("ANESLLM_TIMEOUT", 60.0),
        max_retries=_env_int("ANESLLM_MAX_RETRIES", 3),
        temperature=_env_optional_float("ANESLLM_TEMPERATURE", DEFAULT_TEMPERATURE),
        reasoning_effort=_env_optional_str("ANESLLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
    )
