#  Copyright (c) 2025, TensorCast Team.

"""
Content-addressing helper for RFC-0007.

This module provides a thin wrapper that delegates to the C++ core
implementation for descriptor inspection/generation.
"""

from __future__ import annotations

from pathlib import Path

from tensorcast._C import inspect_or_generate_descriptor


def generate_artifact_id_from_path(artifact_path: str | Path) -> dict:
    """Generate descriptor via C++ pipeline for an existing directory."""
    d = Path(str(artifact_path))
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Invalid artifact_path: {d}")

    desc = inspect_or_generate_descriptor(str(d))
    # Return as a plain dict
    return {
        "artifact_id": desc["artifact_id"],
        "index_multihash": desc["index_multihash"],
        "data_multihash": desc["data_multihash"],
        "schema_version": desc.get("schema_version", "v2"),
        "encoding": desc.get("encoding", "json"),
        "total_size": int(desc.get("total_size", 0)),
    }


__all__ = ["generate_artifact_id_from_path"]
