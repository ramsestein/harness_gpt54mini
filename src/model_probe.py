# -*- coding: utf-8 -*-
"""Which models the keys in ``.env`` reach, and which of them accept ``reasoning_effort``.

The harness sends ``reasoning_effort`` with every call and the client drops it when the model
rejects it (recording that in the prediction's metadata). Before choosing models for a
comparison, this asks the endpoint what each key serves and then makes one tiny call per model,
twice: **with** the effort parameter and **without** it. That separates three cases:

- the model answers both times: it takes the parameter;
- it answers only when the parameter is omitted: it rejects ``reasoning_effort``;
- it never answers: the key does not reach that model (or the id is wrong).

    python -m src.model_probe --api-key-env CLAUDE_API_KEY
    python -m src.model_probe --api-key-env CLAUDE_API_KEY --list-only
    python -m src.model_probe --api-key-env CLAUDE_API_KEY --models model-a,model-b

The key itself is never printed or written: only the name of the variable holding it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, llm  # noqa: E402

#: The prompt of the probe: one word out, no thinking required.
PING = "Reply with the single word: ok"

#: Where the report goes (the leading "_" keeps it out of the run scan).
DEFAULT_OUT_DIR = Path(config.RESULTS_DIR) / "_models"

COLUMNS = ["key_env", "model", "effort", "ok", "dropped", "answered", "latency_s", "error"]


def list_models(key_env=None, base_url=None, client_factory=None):
    """``(ids, error)``: what the endpoint serves for that key.

    ``ids`` is ``None`` when the endpoint does not implement the listing (some
    OpenAI-compatible gateways do not), in which case the models must be named by hand.
    """
    settings = config.settings_for(api_key_env=key_env, base_url=base_url)
    factory = client_factory or llm.LLMClient
    try:
        client = factory(settings=settings)
        page = client.client.models.list()
    except Exception as exc:  # noqa: BLE001 - any failure is reported, never raised
        return None, f"{type(exc).__name__}: {exc}"
    ids = []
    for entry in getattr(page, "data", page) or []:
        model_id = getattr(entry, "id", None) or (entry.get("id") if isinstance(entry, dict) else None)
        if model_id:
            ids.append(str(model_id))
    return ids, None


def probe_model(model, key_env=None, base_url=None, effort="high", prompt=PING,
                client_factory=None):
    """One tiny call to ``model``; reports whether ``effort`` was accepted.

    ``temperature`` is deliberately omitted: the probe is about the effort parameter, and
    every other generation setting is irrelevant to it.
    """
    settings = config.settings_for(model=model, api_key_env=key_env, base_url=base_url)
    factory = client_factory or llm.LLMClient
    row = {
        "key_env": key_env or config.DEFAULT_API_KEY_ENV,
        "model": model,
        "effort": effort or "(omitted)",
        "ok": False,
        "dropped": "",
        "answered": "",
        "latency_s": "",
        "error": "",
    }
    try:
        client = factory(settings=settings)
        response = client.complete_messages(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=None,
            reasoning_effort=effort,
        )
    except Exception as exc:  # noqa: BLE001 - a failing model is a result, not an error
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    dropped = list(getattr(response, "dropped_params", []) or [])
    row["ok"] = True
    row["dropped"] = ",".join(str(name) for name in dropped) if dropped else ""
    row["answered"] = " ".join((response.text or "").split())[:40]
    row["latency_s"] = round(float(getattr(response, "latency_s", 0.0) or 0.0), 2)
    return row


def probe(models, key_env=None, base_url=None, effort="high", client_factory=None):
    """``with`` and ``without`` the effort parameter for every model: the full report."""
    rows = []
    for model in models:
        for value in (effort, None):
            rows.append(
                probe_model(model, key_env=key_env, base_url=base_url, effort=value,
                            client_factory=client_factory)
            )
    return rows


def run(models=None, key_env=None, base_url=None, effort="high", out_dir=None,
        client_factory=None):
    """Probe the models (or the ones the endpoint lists) and write the report."""
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    listing_error = None
    if not models:
        models, listing_error = list_models(key_env=key_env, base_url=base_url,
                                            client_factory=client_factory)
        if not models:
            return out_dir, [], listing_error or "the endpoint listed no models"

    rows = probe(models, key_env=key_env, base_url=base_url, effort=effort,
                 client_factory=client_factory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe_{key_env or config.DEFAULT_API_KEY_ENV}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return out_dir, rows, None


def build_parser():
    """Command-line interface of the probe."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=f"variable holding the key (default: {config.DEFAULT_API_KEY_ENV})",
    )
    parser.add_argument("--base-url", default=None, help="endpoint, when it is not the default")
    parser.add_argument("--models", default=None, help="comma-separated ids (default: ask)")
    parser.add_argument("--effort", default="high", help="effort value to test (default: high)")
    parser.add_argument("--list-only", action="store_true", help="only list what the key reaches")
    parser.add_argument("--out-dir", default=None, help="where to write the report")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    models = [name.strip() for name in (args.models or "").split(",") if name.strip()]

    if args.list_only:
        ids, error = list_models(key_env=args.api_key_env, base_url=args.base_url)
        if ids is None:
            print(f"the endpoint did not list its models: {error}")
            return 1
        print(f"{len(ids)} model(s) reachable with {args.api_key_env or config.DEFAULT_API_KEY_ENV}:")
        for model_id in ids:
            print(f"  {model_id}")
        return 0

    out_dir, rows, problem = run(models=models or None, key_env=args.api_key_env,
                                 base_url=args.base_url, effort=args.effort,
                                 out_dir=args.out_dir)
    if problem:
        print(problem)
        return 1
    print(f"{'model':38s} {'effort':10s} {'ok':3s} {'dropped':10s} {'s':>6s}  answer")
    for row in rows:
        print(
            f"{row['model']:38s} {row['effort']:10s} {str(row['ok']):3s} "
            f"{row['dropped']:10s} {str(row['latency_s']):>6s}  "
            f"{row['answered'] or row['error'][:40]}"
        )
    print(f"\nreport: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
