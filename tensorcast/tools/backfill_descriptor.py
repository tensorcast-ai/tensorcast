#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

"""
Backfill artifact_descriptor.json for existing artifact directories per RFC-0007.

Usage:
  python -m tensorcast.tools.backfill_descriptor <artifact_path> [--recursive]

Behavior:
  - Delegates to the unified C++ pipeline to compute/inspect the descriptor.
  - Writes artifact_descriptor.json when missing (and may persist canonical index
    for safetensors directories if the core decides to do so).
  - With --recursive, walks the directory and backfills each subdirectory
    containing a recognizable artifact layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorcast._C import inspect_or_generate_descriptor


def backfill_one(artifact_path: Path) -> None:
    desc = inspect_or_generate_descriptor(str(artifact_path))

    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "artifact_id": desc["artifact_id"],
                "index_multihash": desc["index_multihash"],
                "data_multihash": desc["data_multihash"],
            },
            ensure_ascii=False,
        )
    )


def detect_model_dirs(root: Path) -> list[Path]:
    """Heuristically detect artifact directories under root.

    A directory is considered a candidate if it contains any of:
      - tensor.data or tensor.data_* files
      - *.safetensors files
    """
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        has_partitions = (p / "tensor.data").exists() or any(p.glob("tensor.data_*"))
        has_safetensors = any(p.glob("*.safetensors"))
        if has_partitions or has_safetensors:
            candidates.append(p)
    return candidates


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_path", type=str, help="Artifact directory or root path")
    parser.add_argument("--recursive", action="store_true", help="Recursively process all detected artifact directories")
    args = parser.parse_args(argv)

    root = Path(args.artifact_path)
    if not root.exists():
        print(f"Path does not exist: {root}", file=sys.stderr)
        return 2

    if args.recursive:
        dirs = detect_model_dirs(root)
        if not dirs:
            print("No artifact directories detected", file=sys.stderr)
            return 1
        for d in dirs:
            try:
                backfill_one(d)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"artifact_path": str(d), "error": str(e)}), file=sys.stderr)
        return 0

    # Single directory
    try:
        backfill_one(root)
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"artifact_path": str(root), "error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    # python -m tensorcast.tools.backfill_descriptor /path/to/artifact --write-index
    # python -m tensorcast.tools.backfill_descriptor /datasets/checkpoints --write-index --recursive
    raise SystemExit(main(sys.argv[1:]))


