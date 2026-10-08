# -*- coding: utf-8 -*-
"""Reusable utilities shared by every strategy harness.

This package is the *shared core*: dataset loading, prompt building, LLM client,
answer parsing and metrics. The harnesses under ``harness/<strategy>/`` must not
reimplement any of this.

Current state: **skeleton**. The modules declare the expected API and raise
``NotImplementedError``; the tests in ``tests/`` are the specification.
"""
