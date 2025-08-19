#!/usr/bin/env python3
#  Copyright (c) 2025, StepCast Team.

"""
Backfill model_descriptor.json (and optionally tensor_index.json) for existing
model directories per RFC-0007.

Usage:
  python -m scstore.tools.backfill_descriptor <model_dir> [--write-index] [--recursive]

Behavior:
  - Computes mi2: model_id using canonical index and tree hash over the
    normalized linear stream (supports standard partitions and safetensors).
  - Writes model_descriptor.json.
  - If --write-index is set and the target is a safetensors directory lacking
    a canonical index, writes tensor_index.json as canonical JSON.
  - With --recursive, walks the directory and backfills each subdirectory
    containing a recognizable model layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scstore._C import inspect_or_generate_descriptor


def _maybe_write_index_for_safetensors(model_dir: Path) -> bool:
    """Deprecated: Index generation now handled in C++; keep no-op placeholder."""
    return False


def backfill_one(model_dir: Path, write_index: bool) -> None:
    desc = inspect_or_generate_descriptor(str(model_dir))

    wrote_index = False
    if write_index:
        wrote_index = _maybe_write_index_for_safetensors(model_dir)

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "model_id": desc["model_id"],
                "index_multihash": desc["index_multihash"],
                "data_multihash": desc["data_multihash"],
                "wrote_index": wrote_index,
            },
            ensure_ascii=False,
        )
    )


def detect_model_dirs(root: Path) -> list[Path]:
    """Heuristically detect model directories under root.

    A directory is considered a candidate if it contains any of:
      - tensor.data or tensor.data_* files
      - *.safetensors files
    """
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        has_partitions = any(
            (p / "tensor.data").exists() or list(p.glob("tensor.data_*"))
        )
        has_safetensors = any(p.glob("*.safetensors"))
        if has_partitions or has_safetensors:
            candidates.append(p)
    return candidates


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=str, help="Model directory or root path")
    parser.add_argument("--write-index", action="store_true", help="Write tensor_index.json for safetensors if missing")
    parser.add_argument("--recursive", action="store_true", help="Recursively process all detected model directories")
    args = parser.parse_args(argv)

    root = Path(args.model_dir)
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    if args.recursive:
        dirs = detect_model_dirs(root)
        if not dirs:
            print("No model directories detected", file=sys.stderr)
            return 1
        for d in dirs:
            try:
                backfill_one(d, args.write_index)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"model_dir": str(d), "error": str(e)}), file=sys.stderr)
        return 0

    # Single directory
    try:
        backfill_one(root, args.write_index)
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"model_dir": str(root), "error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    # python -m scstore.tools.backfill_descriptor /path/to/model_dir --write-index
    # python -m scstore.tools.backfill_descriptor /datasets/checkpoints --write-index --recursive
    raise SystemExit(main(sys.argv[1:]))


