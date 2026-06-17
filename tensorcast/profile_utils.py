#  Copyright (c) 2025-2026, TensorCast Team.
"""Opt-in JSONL profiling helpers for TensorCast debugging."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

_PROFILE_DIR_ENV = "TENSORCAST_PROFILE_DIR"
_PROFILE_LOG_ENV = "TENSORCAST_PROFILE_LOG"


def tensorcast_profile_enabled() -> bool:
    return bool(os.getenv(_PROFILE_DIR_ENV) or os.getenv(_PROFILE_LOG_ENV))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _profile_file(component: str) -> Optional[Path]:
    raw_dir = os.getenv(_PROFILE_DIR_ENV)
    if not raw_dir:
        return None
    profile_dir = Path(raw_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    safe_component = str(component).replace(os.sep, "_")
    return profile_dir / f"{safe_component}_pid{os.getpid()}.jsonl"


def _rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    with contextlib.suppress(Exception):
        return int(psutil.Process().memory_info().rss)
    return None


def _normalize_cuda_device(device: Any) -> Optional[torch.device]:
    if device is None:
        if not torch.cuda.is_available():
            return None
        with contextlib.suppress(Exception):
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return None
    with contextlib.suppress(Exception):
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return None
        if resolved.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return resolved
    return None


def _cuda_snapshot(device: Any) -> Optional[dict[str, Any]]:
    resolved = _normalize_cuda_device(device)
    if resolved is None:
        return None
    try:
        index = int(
            resolved.index
            if resolved.index is not None
            else torch.cuda.current_device()
        )
        stats = torch.cuda.memory_stats(index)
        return {
            "device": str(resolved),
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "allocation_count": int(stats.get("allocation.all.current", 0)),
            "segment_count": int(stats.get("segment.all.current", 0)),
        }
    except Exception:
        return None


def _maybe_sync_cuda(device: Any) -> float:
    resolved = _normalize_cuda_device(device)
    if resolved is None:
        return 0.0
    start = time.perf_counter()
    torch.cuda.synchronize(resolved)
    return time.perf_counter() - start


def emit_tensorcast_profile_event(
    component: str,
    stage: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    logger: Any = None,
) -> None:
    if not tensorcast_profile_enabled():
        return
    record: dict[str, Any] = {
        "component": str(component),
        "stage": str(stage),
        "kind": "event",
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "ts_unix": time.time(),
    }
    for env_key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
        value = os.getenv(env_key)
        if value:
            record[env_key.lower()] = value
    if payload:
        record.update(_jsonable(payload))
    profile_file = _profile_file(component)
    if profile_file is not None:
        with profile_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    if logger is not None and os.getenv(_PROFILE_LOG_ENV):
        logger.info("TensorCast profile: %s", json.dumps(record, sort_keys=True))


class TensorcastProfileStage(dict[str, Any]):
    """Mutable payload surface for profile scopes."""


@contextlib.contextmanager
def tensorcast_profile_stage(
    component: str,
    stage: str,
    *,
    logger: Any = None,
    device: Any = None,
    sync_cuda: bool = False,
    extra: Optional[dict[str, Any]] = None,
):
    if not tensorcast_profile_enabled():
        yield None
        return

    resolved_device = _normalize_cuda_device(device)
    stage_payload = TensorcastProfileStage()
    sync_before_sec = _maybe_sync_cuda(resolved_device) if sync_cuda else 0.0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = _rss_bytes()
    cuda_start = _cuda_snapshot(resolved_device)
    error_payload: Optional[dict[str, str]] = None
    try:
        yield stage_payload
    except Exception as exc:
        error_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        sync_after_sec = _maybe_sync_cuda(resolved_device) if sync_cuda else 0.0
        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        rss_end = _rss_bytes()
        cuda_end = _cuda_snapshot(resolved_device)

        wall_sec = wall_end - wall_start
        cpu_sec = cpu_end - cpu_start
        record: dict[str, Any] = {
            "component": str(component),
            "stage": str(stage),
            "kind": "scope",
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "ts_unix": time.time(),
            "wall_sec": wall_sec,
            "cpu_sec": cpu_sec,
            "wait_sec": max(0.0, wall_sec - cpu_sec),
            "cpu_util_pct": (cpu_sec / wall_sec * 100.0) if wall_sec > 0 else 0.0,
            "cuda_sync_before_sec": sync_before_sec,
            "cuda_sync_after_sec": sync_after_sec,
        }
        for env_key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
            value = os.getenv(env_key)
            if value:
                record[env_key.lower()] = value
        if rss_start is not None:
            record["rss_start_bytes"] = rss_start
        if rss_end is not None:
            record["rss_end_bytes"] = rss_end
        if rss_start is not None and rss_end is not None:
            record["rss_delta_bytes"] = rss_end - rss_start
        if cuda_start is not None:
            record["cuda_device"] = cuda_start["device"]
        if cuda_start is not None and cuda_end is not None:
            for key in (
                "allocated_bytes",
                "reserved_bytes",
                "active_bytes",
                "allocation_count",
                "segment_count",
            ):
                start_value = int(cuda_start.get(key, 0))
                end_value = int(cuda_end.get(key, 0))
                record[f"cuda_{key}_start"] = start_value
                record[f"cuda_{key}_end"] = end_value
                record[f"cuda_{key}_delta"] = end_value - start_value
        if extra:
            record.update(_jsonable(extra))
        if stage_payload:
            record.update(_jsonable(dict(stage_payload)))
        if error_payload is not None:
            record["error"] = error_payload
        emit_tensorcast_profile_event(component, stage, payload=record, logger=logger)
