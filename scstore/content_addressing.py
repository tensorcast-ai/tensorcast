#  Copyright (c) 2025, StepCast Team.

"""
Content-addressing helper for RFC-0007.

This module provides a thin wrapper that delegates to the C++ core
implementation for descriptor inspection/generation.
"""

from __future__ import annotations

from pathlib import Path

from scstore._C import inspect_or_generate_descriptor


def generate_model_id_from_path(model_dir: str | Path) -> dict:
    """Generate descriptor via C++ pipeline for an existing directory."""
    d = Path(str(model_dir))
    if not d.exists() or not d.is_dir():
        raise ValueError(f"Invalid model_dir: {d}")

    desc = inspect_or_generate_descriptor(str(d))
    # Return as a plain dict
    return {
        "model_id": desc["model_id"],
        "index_multihash": desc["index_multihash"],
        "data_multihash": desc["data_multihash"],
        "schema_version": desc.get("schema_version", "v2"),
        "encoding": desc.get("encoding", "json"),
        "total_size": int(desc.get("total_size", 0)),
    }


__all__ = ["generate_model_id_from_path"]
