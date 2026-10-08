# -*- coding: utf-8 -*-
"""Fixtures shared by the test suite."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root."""
    return ROOT


@pytest.fixture(scope="session")
def dataset_dir(repo_root: Path) -> Path:
    """The ``dataset/`` folder."""
    return repo_root / "dataset"


@pytest.fixture(scope="session")
def data_dir(dataset_dir: Path) -> Path:
    """The ``dataset/data/`` folder."""
    return dataset_dir / "data"


@pytest.fixture(scope="session")
def harness_dir(repo_root: Path) -> Path:
    """The ``harness/`` folder."""
    return repo_root / "harness"


@pytest.fixture(autouse=True)
def _skip_live_if_requested(request):
    """Allow skipping the tests that call the real model.

    ``ANESLLM_SKIP_LIVE=1`` disables the tests marked with ``live``.
    """
    if request.node.get_closest_marker("live") is None:
        return
    if os.environ.get("ANESLLM_SKIP_LIVE", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("ANESLLM_SKIP_LIVE is set")
