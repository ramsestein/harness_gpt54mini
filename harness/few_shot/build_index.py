# -*- coding: utf-8 -*-
"""Prebuild the few-shot retrieval index from the ``train`` split.

Building it takes a few minutes (the whole train split is parsed and rendered);
the strategy then reuses the cache.

    python harness/few_shot/build_index.py
    python harness/few_shot/build_index.py --rebuild
    python harness/few_shot/build_index.py --cases 50 --out scratch.npz
"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.few_shot import retrieval  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Build the few-shot retrieval index.")
    parser.add_argument("--cases", type=int, default=None,
                        help="use only the first N cases of train (never cached)")
    parser.add_argument("--out", default=None,
                        help=f"output path (default: {retrieval.DEFAULT_INDEX_PATH})")
    parser.add_argument("--rebuild", action="store_true", help="ignore any existing cache")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = Path(args.out) if args.out else retrieval.DEFAULT_INDEX_PATH

    print(f"corpus    : {retrieval.SOURCE_SPLIT}"
          + (f" (first {args.cases} cases)" if args.cases else " (all cases)"))
    print(f"index file: {path}")

    index = retrieval.load_or_build(path=path, cases=args.cases, rebuild=args.rebuild)

    print(f"windows   : {len(index)}")
    print(f"features  : {len(index.names)}")
    print("saved" if args.cases is None else "not cached (partial corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
