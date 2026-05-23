#  Copyright (c) 2026, TensorCast Team.

"""TracePlan cache and debug dump helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tensorcast.serving.builder.trace_ir import (
    TracePlan,
    trace_plan_from_dict,
    trace_plan_to_dict,
)

TRACE_PLAN_CACHE_PAYLOAD_VERSION = 1


def load_trace_plan_cache(
    cache_path: str | os.PathLike[str] | None,
) -> TracePlan | None:
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            return None
        if "trace_plan" not in data:
            return None
        version = int(data.get("version", 0) or 0)
        if version != TRACE_PLAN_CACHE_PAYLOAD_VERSION:
            return None
        data = data["trace_plan"]
        return trace_plan_from_dict(dict(data))
    except Exception:
        return None


def write_trace_plan_cache(
    cache_path: str | os.PathLike[str],
    trace_plan: TracePlan,
) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": TRACE_PLAN_CACHE_PAYLOAD_VERSION,
        "trace_plan": trace_plan_to_dict(trace_plan),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def trace_plan_debug_payload(
    trace_plan: TracePlan,
    *,
    cache_path: str | os.PathLike[str] | None,
    cache_hit: bool,
    trace_cache_schema_version: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "cache_hit": bool(cache_hit),
        "cache_path": None if cache_path is None else str(cache_path),
        "trace_cache_schema_version": int(trace_cache_schema_version),
        "trace_plan_cache_payload_version": TRACE_PLAN_CACHE_PAYLOAD_VERSION,
        **trace_plan_to_dict(trace_plan),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def dump_trace_plan_debug(
    trace_plan: TracePlan,
    *,
    output_dir: str | os.PathLike[str] | None,
    filename: str,
    cache_path: str | os.PathLike[str] | None,
    cache_hit: bool,
    trace_cache_schema_version: int,
    extra: Mapping[str, Any] | None = None,
) -> Path | None:
    if output_dir is None:
        return None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{os.getpid()}"
    payload = trace_plan_debug_payload(
        trace_plan,
        cache_path=cache_path,
        cache_hit=cache_hit,
        trace_cache_schema_version=trace_cache_schema_version,
        extra=extra,
    )
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)
    return output_path


__all__ = [
    "TRACE_PLAN_CACHE_PAYLOAD_VERSION",
    "dump_trace_plan_debug",
    "load_trace_plan_cache",
    "trace_plan_debug_payload",
    "write_trace_plan_cache",
]
