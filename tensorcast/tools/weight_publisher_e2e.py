#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""End-to-end harness for tensorcast.tools.weight_publisher.

Scenarios:
1) single-host: run publisher and receiver concurrently on one node.
2) publisher: distributed publisher role.
3) receiver: distributed receiver role.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

import tensorcast as tc
from tensorcast import GetArtifactOptions
from tensorcast.api.store import CopyPlanEntry, Range
from tensorcast.api.store import artifact as resolve_artifact
from tensorcast.api.store.runtime import get_context as get_store_context
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

SUPPORTED_PAYLOAD_MODES = {"probe", "version_fill", "tp_ranked"}
SUPPORTED_RECEIVER_APPLY_MODES = {
    "tensor_dict",
    "binding_swap",
    "tp_bind_into_swap",
    "tp4_bind_into_swap",
}
SUPPORTED_TRANSPORT_GROUP_MODES = {"none", "tp_version"}
SUPPORTED_TP_DEVICE_MAP_POLICIES = {"auto", "strict", "modulo"}

TP4_COL_BLOCK_ROWS = 4
TP4_COL_WIDTH = 8
TP4_ROW_HEIGHT = 8
TP4_ROW_BLOCK_COLS = 4
FLOAT32_BYTES = 4
TP_RANKED_SOURCE_ELEMS_LIMIT = (1 << 31) - 1
TP_FULL_VALIDATION_MAX_BYTES = 4 * 1024**3
TP_SAMPLED_VALIDATION_POINTS = 4096
RECEIVER_PROGRESS_LOG_INTERVAL_S = 10.0
RECEIVER_RESOLVE_SLOW_THRESHOLD_MS = 1000.0
RECEIVER_TENSOR_DICT_UNLOAD_RETRIES = 3
RECEIVER_TENSOR_DICT_UNLOAD_RETRY_INTERVAL_S = 0.05
TP_BIND_PER_RANK_TIMEOUT_MIN_S = 20.0
TP_BIND_PER_RANK_TIMEOUT_FLOOR_S = 8.0
TP_BIND_PER_RANK_TIMEOUT_GIB_FACTOR = 12.0


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "not_found" in msg
        or "not found" in msg
        or "key not found" in msg
        or "statuscode.not_found" in msg
        or "no available replicas" in msg
    )


def _is_version_unavailable_during_apply_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if _is_not_found_error(exc):
        return True
    return (
        "region is poisoned" in msg
        or "tensor not found" in msg
        or (
            "artifact id" in msg
            and ("not found" in msg or "was not found" in msg)
        )
    )


def _is_non_retryable_transport_group_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "group contract violation" in msg:
        return True
    return (
        "duplicate part_id in transport history" in msg
        or "artifact/view/total_parts mismatch" in msg
    )


def _should_clear_tp_pending_target_on_apply_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "region is poisoned" in msg:
        return True
    if "deadline exceeded" in msg:
        return True
    return (
        "already used with different payload" in msg
        or "client_request_id already used with a different payload" in msg
    )


def _compact_error_text(exc: Exception) -> str:
    compact = str(exc).replace("\n", " ").strip()
    if len(compact) > 200:
        compact = compact[:200] + "..."
    return f"{type(exc).__name__}:{compact}"


def _is_cuda_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


@dataclass(frozen=True)
class PublishEvent:
    version: int
    key: str
    artifact_id: str
    export_dir: str
    publish_device: str
    published_at_s: float
    publish_latency_s: float
    publish_payload_bytes: int
    publish_throughput_gib_s: float
    put_throughput_gib_s: float
    publish_breakdown_s: dict[str, float]


@dataclass(frozen=True)
class ReceiveEvent:
    version: int
    key: str
    artifact_id: str
    received_at_s: float
    materialize_latency_s: float
    apply_mode: str
    apply_operation: str
    pointer_stable: bool | None


class VersionDroppedError(RuntimeError):
    def __init__(
        self,
        *,
        version: int,
        key: str,
        artifact_id: str,
        message: str,
        newer_version: int | None = None,
    ) -> None:
        super().__init__(message)
        self.version = int(version)
        self.key = str(key)
        self.artifact_id = str(artifact_id)
        self.newer_version = int(newer_version) if newer_version is not None else None


def _ensure_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _ensure_non_negative(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _ensure_non_negative_float(name: str, value: float) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return float(value)


def _read_process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if not line.startswith("VmRSS:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                return None
            return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_cuda_allocated_bytes(device: str) -> int | None:
    if not str(device).startswith("cuda:"):
        return None
    if not torch.cuda.is_available():
        return None
    try:
        index = int(str(device).split(":", 1)[1])
        return int(torch.cuda.memory_allocated(index))
    except Exception:
        return None


def _publish_memory_log(*, stage: str, version: int, publish_device: str) -> None:
    rss_bytes = _read_process_rss_bytes()
    cuda_bytes = _read_cuda_allocated_bytes(publish_device)
    tokens = [
        "[publisher][mem]",
        f"stage={stage}",
        f"version={version}",
    ]
    if rss_bytes is not None:
        tokens.append(f"rss_gib={float(rss_bytes) / float(1024**3):.2f}")
    if cuda_bytes is not None:
        tokens.append(f"cuda_alloc_gib={float(cuda_bytes) / float(1024**3):.2f}")
    print(" ".join(tokens), flush=True)


def _estimate_tensor_payload_bytes(tensors: dict[str, torch.Tensor]) -> int:
    total_bytes = 0
    for tensor in tensors.values():
        total_bytes += int(tensor.numel()) * int(tensor.element_size())
    return int(max(0, total_bytes))


def _bytes_throughput_gib_s(*, bytes_total: int, duration_s: float) -> float:
    if int(bytes_total) <= 0 or float(duration_s) <= 0.0:
        return 0.0
    return float(bytes_total) / float(1024**3) / float(duration_s)


def _sanitize_group_token(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:")
    return "".join(ch if ch in allowed else "_" for ch in value)


def _build_key(*, model_name: str, key_template: str, version: int) -> str:
    key = key_template.format(
        model_name=model_name,
        version=version,
        weight_version=version,
    ).strip()
    if not key:
        raise ValueError("resolved key is empty")
    return key


def _materialization_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        # Fake CUDA exposes virtual GPU ordinals and avoids CPU shared-memory
        # prerequisites that may be disabled in lightweight daemon configs.
        return "cuda:0"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _publisher_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_tp_rank_devices(
    *,
    tp_world_size: int,
    tp_device_base_index: int,
    visible_device_count: int,
    map_policy: str,
) -> tuple[dict[int, str], str]:
    if visible_device_count <= 0:
        raise ValueError(
            "tp_bind_into_swap requires at least one visible CUDA device, "
            f"got {visible_device_count}"
        )
    if tp_device_base_index < 0:
        raise ValueError(
            "tp_device_base_index must be >= 0, "
            f"got {tp_device_base_index}"
        )
    if tp_device_base_index >= visible_device_count:
        raise ValueError(
            "tp_device_base_index is out of range for visible CUDA devices: "
            f"base={tp_device_base_index}, visible={visible_device_count}"
        )
    available_from_base = visible_device_count - tp_device_base_index
    if available_from_base >= tp_world_size:
        return (
            {
                rank: f"cuda:{tp_device_base_index + rank}"
                for rank in range(tp_world_size)
            },
            "contiguous",
        )
    if map_policy == "strict":
        raise ValueError(
            "tp strict device mapping requires enough visible CUDA devices from base index: "
            f"tp_world_size={tp_world_size}, base={tp_device_base_index}, "
            f"visible={visible_device_count}, available_from_base={available_from_base}"
        )
    if map_policy not in {"auto", "modulo"}:
        raise ValueError(
            f"unsupported tp device map policy: {map_policy!r}"
        )
    return (
        {
            rank: f"cuda:{tp_device_base_index + (rank % available_from_base)}"
            for rank in range(tp_world_size)
        },
        "modulo",
    )
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_payload_mode(payload_mode: str) -> str:
    mode = str(payload_mode).strip()
    if mode not in SUPPORTED_PAYLOAD_MODES:
        raise ValueError(
            f"unsupported payload_mode={mode!r}, "
            f"allowed={sorted(SUPPORTED_PAYLOAD_MODES)}"
        )
    return mode


def _tp_rank_value(*, version: int, rank: int) -> float:
    return float(version) + (float(rank) / 10.0)


def _normalize_tp_total_bytes(tp_total_bytes: int) -> int:
    value = int(tp_total_bytes)
    if value < 0:
        raise ValueError(f"tp_total_bytes must be >= 0, got {value}")
    return value


def _tp_ranked_chunk_elems(
    *,
    tp_world_size: int,
    tp_total_bytes: int,
) -> list[int]:
    total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    if total_bytes == 0:
        return []
    bytes_per_rank_elem = FLOAT32_BYTES * 2 * int(tp_world_size)
    if total_bytes % bytes_per_rank_elem != 0:
        raise ValueError(
            "tp_total_bytes must be divisible by "
            f"{bytes_per_rank_elem} for tp_world_size={tp_world_size}, got {total_bytes}"
        )
    total_rank_elems = total_bytes // bytes_per_rank_elem
    max_rank_elems_per_chunk = TP_RANKED_SOURCE_ELEMS_LIMIT // int(tp_world_size)
    if max_rank_elems_per_chunk <= 0:
        raise ValueError(
            f"invalid tp_world_size for chunk planning: tp_world_size={tp_world_size}"
        )
    chunks: list[int] = []
    remaining = int(total_rank_elems)
    while remaining > 0:
        chunk = min(remaining, max_rank_elems_per_chunk)
        chunks.append(int(chunk))
        remaining -= int(chunk)
    return chunks


def _tp_col_tensor_name(chunk_index: int) -> str:
    return f"tp_col_weight_{chunk_index}"


def _tp_row_tensor_name(chunk_index: int) -> str:
    return f"tp_row_weight_{chunk_index}"


def _tp_rank_col_target_name(chunk_index: int) -> str:
    return f"rank_col_weight_{chunk_index}"


def _tp_rank_row_target_name(chunk_index: int) -> str:
    return f"rank_row_weight_{chunk_index}"


def _build_tp_ranked_publish_tensors(
    *,
    version: int,
    device: str,
    tp_world_size: int,
    tp_total_bytes: int,
) -> dict[str, torch.Tensor]:
    _ensure_positive("tp_world_size", tp_world_size)
    normalized_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    if normalized_total_bytes > 0:
        tensors: dict[str, torch.Tensor] = {}
        chunk_elems = _tp_ranked_chunk_elems(
            tp_world_size=tp_world_size,
            tp_total_bytes=normalized_total_bytes,
        )
        for chunk_index, rank_block_elems in enumerate(chunk_elems):
            col_name = _tp_col_tensor_name(chunk_index)
            row_name = _tp_row_tensor_name(chunk_index)
            tp_col_weight = torch.empty(
                (tp_world_size, rank_block_elems),
                dtype=torch.float32,
                device=device,
            )
            tp_row_weight = torch.empty(
                (1, tp_world_size * rank_block_elems),
                dtype=torch.float32,
                device=device,
            )
            for rank in range(tp_world_size):
                expected = _tp_rank_value(version=version, rank=rank)
                row_start = rank * rank_block_elems
                row_end = row_start + rank_block_elems
                tp_col_weight[rank : rank + 1, :].fill_(expected)
                tp_row_weight[:, row_start:row_end].fill_(expected)
            tensors[col_name] = tp_col_weight
            tensors[row_name] = tp_row_weight
        return tensors

    tp_col_weight = torch.empty(
        (tp_world_size * TP4_COL_BLOCK_ROWS, TP4_COL_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    tp_row_weight = torch.empty(
        (TP4_ROW_HEIGHT, tp_world_size * TP4_ROW_BLOCK_COLS),
        dtype=torch.float32,
        device=device,
    )
    for rank in range(tp_world_size):
        expected = _tp_rank_value(version=version, rank=rank)
        col_start = rank * TP4_COL_BLOCK_ROWS
        col_end = col_start + TP4_COL_BLOCK_ROWS
        row_start = rank * TP4_ROW_BLOCK_COLS
        row_end = row_start + TP4_ROW_BLOCK_COLS
        tp_col_weight[col_start:col_end, :].fill_(expected)
        tp_row_weight[:, row_start:row_end].fill_(expected)
    return {
        "tp_col_weight": tp_col_weight,
        "tp_row_weight": tp_row_weight,
    }


def _build_publish_tensors(
    *,
    version: int,
    device: str,
    payload_mode: str,
    tp_world_size: int,
    tp_total_bytes: int,
) -> dict[str, torch.Tensor]:
    mode = _normalize_payload_mode(payload_mode)
    if mode == "tp_ranked":
        return _build_tp_ranked_publish_tensors(
            version=version,
            device=device,
            tp_world_size=tp_world_size,
            tp_total_bytes=tp_total_bytes,
        )
    if mode == "version_fill":
        return {
            "version_marker": torch.full(
                (1,),
                int(version),
                dtype=torch.int64,
                device=device,
            ),
            "weight_probe": torch.full(
                (4, 4),
                float(version),
                dtype=torch.float32,
                device=device,
            ),
            "rolling_checksum": torch.full(
                (3,),
                int(version),
                dtype=torch.int64,
                device=device,
            ),
        }

    version_marker = torch.tensor([version], dtype=torch.int64)
    weight_probe = torch.arange(16, dtype=torch.float32).reshape(4, 4) + float(version)
    weight_probe[0, 0] = float(version)
    checksum_seed = int(version * 9973 + 17)
    rolling_checksum = torch.tensor(
        [checksum_seed, checksum_seed + 1, checksum_seed + 2],
        dtype=torch.int64,
    )
    return {
        "version_marker": version_marker.to(device=device),
        "weight_probe": weight_probe.to(device=device),
        "rolling_checksum": rolling_checksum.to(device=device),
    }


def _assert_tensor_all_equals(
    *,
    name: str,
    tensor: torch.Tensor,
    expected_int: int,
    expected_float: float,
) -> None:
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.numel() == 0:
        raise AssertionError(f"{name} is empty")
    if cpu_tensor.dtype.is_floating_point:
        expected = torch.tensor(expected_float, dtype=cpu_tensor.dtype)
    else:
        expected = torch.tensor(expected_int, dtype=cpu_tensor.dtype)
    if bool(torch.all(cpu_tensor == expected).item()):
        return
    first = cpu_tensor.reshape(-1)[0].item()
    raise AssertionError(
        f"{name} contains unexpected value: expected={expected.item()}, actual_first={first}"
    )


def _assert_tensor_allclose_scalar(
    *,
    name: str,
    tensor: torch.Tensor,
    expected: float,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    detached = tensor.detach()
    if detached.numel() == 0:
        raise AssertionError(f"{name} is empty")
    expected_tensor = torch.tensor(
        float(expected),
        dtype=detached.dtype,
        device=detached.device,
    )
    if bool(torch.allclose(detached, expected_tensor, atol=atol, rtol=rtol)):
        return
    actual = float(detached.reshape(-1)[0].item())
    raise AssertionError(
        f"{name} scalar mismatch: expected={expected}, actual_first={actual}"
    )


def _build_sample_indices(
    *,
    total: int,
    points: int,
    device: torch.device,
) -> torch.Tensor:
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    if points <= 0:
        raise ValueError(f"points must be > 0, got {points}")
    if points == 1:
        return torch.zeros((1,), dtype=torch.long, device=device)
    # Use integer math to avoid float rounding drift on large tensors.
    base = torch.arange(points, dtype=torch.long, device=device)
    return (base * (total - 1)) // (points - 1)


def _assert_tensor_sampled_allclose_scalar(
    *,
    name: str,
    tensor: torch.Tensor,
    expected: float,
    sample_points: int = TP_SAMPLED_VALIDATION_POINTS,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    detached = tensor.detach()
    if detached.numel() == 0:
        raise AssertionError(f"{name} is empty")
    flat = detached.reshape(-1)
    total = int(flat.numel())
    points = min(total, max(1, int(sample_points)))
    if points >= total:
        _assert_tensor_allclose_scalar(
            name=name,
            tensor=detached,
            expected=expected,
            atol=atol,
            rtol=rtol,
        )
        return
    indices = _build_sample_indices(total=total, points=points, device=flat.device)
    sampled = flat.index_select(0, indices)
    expected_tensor = torch.tensor(
        float(expected),
        dtype=sampled.dtype,
        device=sampled.device,
    )
    if bool(torch.allclose(sampled, expected_tensor, atol=atol, rtol=rtol)):
        return
    actual = float(sampled.reshape(-1)[0].item())
    raise AssertionError(
        f"{name} sampled scalar mismatch: expected={expected}, actual_first_sample={actual}"
    )


def _validate_tp_ranked_payload(
    *,
    version: int,
    tensors: dict[str, torch.Tensor],
    tp_world_size: int,
    tp_total_bytes: int,
) -> None:
    normalized_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    if normalized_total_bytes > 0:
        chunk_elems = _tp_ranked_chunk_elems(
            tp_world_size=tp_world_size,
            tp_total_bytes=normalized_total_bytes,
        )
        expected = {
            name
            for index in range(len(chunk_elems))
            for name in (_tp_col_tensor_name(index), _tp_row_tensor_name(index))
        }
        missing = expected - set(tensors)
        if missing:
            raise AssertionError(
                f"missing tensors in tp_ranked payload: {sorted(missing)}"
            )
        for chunk_index, rank_block_elems in enumerate(chunk_elems):
            col_name = _tp_col_tensor_name(chunk_index)
            row_name = _tp_row_tensor_name(chunk_index)
            tp_col_weight = tensors[col_name]
            tp_row_weight = tensors[row_name]
            expected_col_shape = (tp_world_size, rank_block_elems)
            expected_row_shape = (1, tp_world_size * rank_block_elems)
            if tuple(tp_col_weight.shape) != expected_col_shape:
                raise AssertionError(
                    f"{col_name} shape mismatch: "
                    f"expected={expected_col_shape}, actual={tuple(tp_col_weight.shape)}"
                )
            if tuple(tp_row_weight.shape) != expected_row_shape:
                raise AssertionError(
                    f"{row_name} shape mismatch: "
                    f"expected={expected_row_shape}, actual={tuple(tp_row_weight.shape)}"
                )
            for rank in range(tp_world_size):
                expected_value = _tp_rank_value(version=version, rank=rank)
                row_start = rank * rank_block_elems
                row_end = row_start + rank_block_elems
                _assert_tensor_allclose_scalar(
                    name=f"{col_name}[rank={rank}]",
                    tensor=tp_col_weight[rank : rank + 1, :],
                    expected=expected_value,
                )
                _assert_tensor_allclose_scalar(
                    name=f"{row_name}[rank={rank}]",
                    tensor=tp_row_weight[:, row_start:row_end],
                    expected=expected_value,
                )
        return

    expected = {"tp_col_weight", "tp_row_weight"}
    missing = expected - set(tensors)
    if missing:
        raise AssertionError(f"missing tensors in tp_ranked payload: {sorted(missing)}")
    tp_col_weight = tensors["tp_col_weight"]
    tp_row_weight = tensors["tp_row_weight"]
    expected_col_shape = (tp_world_size * TP4_COL_BLOCK_ROWS, TP4_COL_WIDTH)
    expected_row_shape = (TP4_ROW_HEIGHT, tp_world_size * TP4_ROW_BLOCK_COLS)
    if tuple(tp_col_weight.shape) != expected_col_shape:
        raise AssertionError(
            "tp_col_weight shape mismatch: "
            f"expected={expected_col_shape}, actual={tuple(tp_col_weight.shape)}"
        )
    if tuple(tp_row_weight.shape) != expected_row_shape:
        raise AssertionError(
            "tp_row_weight shape mismatch: "
            f"expected={expected_row_shape}, actual={tuple(tp_row_weight.shape)}"
        )
    for rank in range(tp_world_size):
        expected_value = _tp_rank_value(version=version, rank=rank)
        col_start = rank * TP4_COL_BLOCK_ROWS
        col_end = col_start + TP4_COL_BLOCK_ROWS
        row_start = rank * TP4_ROW_BLOCK_COLS
        row_end = row_start + TP4_ROW_BLOCK_COLS
        _assert_tensor_allclose_scalar(
            name=f"tp_col_weight[rank={rank}]",
            tensor=tp_col_weight[col_start:col_end, :],
            expected=expected_value,
        )
        _assert_tensor_allclose_scalar(
            name=f"tp_row_weight[rank={rank}]",
            tensor=tp_row_weight[:, row_start:row_end],
            expected=expected_value,
        )


def _build_tp4_rank_copy_plan(
    *,
    rank: int,
    tp_world_size: int,
    tp_total_bytes: int,
) -> tuple[CopyPlanEntry, ...]:
    normalized_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    if normalized_total_bytes > 0:
        entries: list[CopyPlanEntry] = []
        chunk_elems = _tp_ranked_chunk_elems(
            tp_world_size=tp_world_size,
            tp_total_bytes=normalized_total_bytes,
        )
        for chunk_index, rank_block_elems in enumerate(chunk_elems):
            row_start = rank * rank_block_elems
            row_end = row_start + rank_block_elems
            entries.append(
                CopyPlanEntry(
                    ckpt_name=_tp_col_tensor_name(chunk_index),
                    ckpt_range=Range(dim=0, start=rank, end=rank + 1),
                    dst_name=_tp_rank_col_target_name(chunk_index),
                    dst_range=Range(dim=0, start=0, end=1),
                )
            )
            entries.append(
                CopyPlanEntry(
                    ckpt_name=_tp_row_tensor_name(chunk_index),
                    ckpt_range=Range(dim=1, start=row_start, end=row_end),
                    dst_name=_tp_rank_row_target_name(chunk_index),
                    dst_range=Range(dim=1, start=0, end=rank_block_elems),
                )
            )
        return tuple(entries)

    col_start = rank * TP4_COL_BLOCK_ROWS
    col_end = col_start + TP4_COL_BLOCK_ROWS
    row_start = rank * TP4_ROW_BLOCK_COLS
    row_end = row_start + TP4_ROW_BLOCK_COLS
    return (
        CopyPlanEntry(
            ckpt_name="tp_col_weight",
            ckpt_range=Range(dim=0, start=col_start, end=col_end),
            dst_name="rank_col_weight",
            dst_range=Range(dim=0, start=0, end=TP4_COL_BLOCK_ROWS),
        ),
        CopyPlanEntry(
            ckpt_name="tp_row_weight",
            ckpt_range=Range(dim=1, start=row_start, end=row_end),
            dst_name="rank_row_weight",
            dst_range=Range(dim=1, start=0, end=TP4_ROW_BLOCK_COLS),
        ),
    )


def _allocate_tp4_rank_targets(
    *,
    device: str,
    tp_world_size: int,
    tp_total_bytes: int,
) -> dict[str, torch.Tensor]:
    normalized_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    if normalized_total_bytes > 0:
        targets: dict[str, torch.Tensor] = {}
        chunk_elems = _tp_ranked_chunk_elems(
            tp_world_size=tp_world_size,
            tp_total_bytes=normalized_total_bytes,
        )
        for chunk_index, rank_block_elems in enumerate(chunk_elems):
            targets[_tp_rank_col_target_name(chunk_index)] = torch.empty(
                (1, rank_block_elems),
                dtype=torch.float32,
                device=device,
            )
            targets[_tp_rank_row_target_name(chunk_index)] = torch.empty(
                (1, rank_block_elems),
                dtype=torch.float32,
                device=device,
            )
        return targets

    return {
        "rank_col_weight": torch.empty(
            (TP4_COL_BLOCK_ROWS, TP4_COL_WIDTH),
            dtype=torch.float32,
            device=device,
        ),
        "rank_row_weight": torch.empty(
            (TP4_ROW_HEIGHT, TP4_ROW_BLOCK_COLS),
            dtype=torch.float32,
            device=device,
        ),
    }


def _validate_tp4_rank_targets(
    *,
    version: int,
    rank: int,
    tensors: dict[str, torch.Tensor],
    tp_world_size: int,
    tp_total_bytes: int,
) -> None:
    normalized_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
    use_sampled_validation = normalized_total_bytes > TP_FULL_VALIDATION_MAX_BYTES
    if normalized_total_bytes > 0:
        chunk_elems = _tp_ranked_chunk_elems(
            tp_world_size=tp_world_size,
            tp_total_bytes=normalized_total_bytes,
        )
        expected = {
            name
            for index in range(len(chunk_elems))
            for name in (
                _tp_rank_col_target_name(index),
                _tp_rank_row_target_name(index),
            )
        }
        missing = expected - set(tensors)
        if missing:
            raise AssertionError(f"missing rank tensors: {sorted(missing)}")
        expected_value = _tp_rank_value(version=version, rank=rank)
        for chunk_index, rank_block_elems in enumerate(chunk_elems):
            col_name = _tp_rank_col_target_name(chunk_index)
            row_name = _tp_rank_row_target_name(chunk_index)
            col_tensor = tensors[col_name]
            row_tensor = tensors[row_name]
            expected_shape = (1, rank_block_elems)
            if tuple(col_tensor.shape) != expected_shape:
                raise AssertionError(
                    f"{col_name} shape mismatch: "
                    f"expected={expected_shape}, actual={tuple(col_tensor.shape)}"
                )
            if tuple(row_tensor.shape) != expected_shape:
                raise AssertionError(
                    f"{row_name} shape mismatch: "
                    f"expected={expected_shape}, actual={tuple(row_tensor.shape)}"
                )
            if use_sampled_validation:
                _assert_tensor_sampled_allclose_scalar(
                    name=f"{col_name}[rank={rank}]",
                    tensor=col_tensor,
                    expected=expected_value,
                )
                _assert_tensor_sampled_allclose_scalar(
                    name=f"{row_name}[rank={rank}]",
                    tensor=row_tensor,
                    expected=expected_value,
                )
            else:
                _assert_tensor_allclose_scalar(
                    name=f"{col_name}[rank={rank}]",
                    tensor=col_tensor,
                    expected=expected_value,
                )
                _assert_tensor_allclose_scalar(
                    name=f"{row_name}[rank={rank}]",
                    tensor=row_tensor,
                    expected=expected_value,
                )
        return

    expected = {"rank_col_weight", "rank_row_weight"}
    missing = expected - set(tensors)
    if missing:
        raise AssertionError(f"missing rank tensors: {sorted(missing)}")
    expected_value = _tp_rank_value(version=version, rank=rank)
    _assert_tensor_allclose_scalar(
        name=f"rank_col_weight[rank={rank}]",
        tensor=tensors["rank_col_weight"],
        expected=expected_value,
    )
    _assert_tensor_allclose_scalar(
        name=f"rank_row_weight[rank={rank}]",
        tensor=tensors["rank_row_weight"],
        expected=expected_value,
    )


def _validate_payload(
    *,
    version: int,
    tensors: dict[str, torch.Tensor],
    payload_mode: str,
    tp_world_size: int,
    tp_total_bytes: int,
) -> None:
    mode = _normalize_payload_mode(payload_mode)
    if mode == "tp_ranked":
        _validate_tp_ranked_payload(
            version=version,
            tensors=tensors,
            tp_world_size=tp_world_size,
            tp_total_bytes=tp_total_bytes,
        )
        return

    expected = {"version_marker", "weight_probe", "rolling_checksum"}
    missing = expected - set(tensors)
    if missing:
        raise AssertionError(f"missing tensors in payload: {sorted(missing)}")

    if mode == "version_fill":
        _assert_tensor_all_equals(
            name="version_marker",
            tensor=tensors["version_marker"],
            expected_int=int(version),
            expected_float=float(version),
        )
        _assert_tensor_all_equals(
            name="weight_probe",
            tensor=tensors["weight_probe"],
            expected_int=int(version),
            expected_float=float(version),
        )
        _assert_tensor_all_equals(
            name="rolling_checksum",
            tensor=tensors["rolling_checksum"],
            expected_int=int(version),
            expected_float=float(version),
        )
        return

    marker = int(tensors["version_marker"].reshape(-1)[0].cpu().item())
    if marker != version:
        raise AssertionError(
            f"version_marker mismatch: expected={version}, actual={marker}"
        )

    probe_origin = float(tensors["weight_probe"][0, 0].cpu().item())
    if probe_origin != float(version):
        raise AssertionError(
            f"weight_probe[0,0] mismatch: expected={float(version)}, actual={probe_origin}"
        )

    checksum_seed = int(version * 9973 + 17)
    checksum_sum = int(tensors["rolling_checksum"].sum().cpu().item())
    expected_sum = checksum_seed + (checksum_seed + 1) + (checksum_seed + 2)
    if checksum_sum != expected_sum:
        raise AssertionError(
            f"rolling_checksum sum mismatch: expected={expected_sum}, actual={checksum_sum}"
        )


class WeightUpdatePublisher:
    def __init__(
        self,
        *,
        model_name: str,
        key_template: str,
        keep_last: int,
        pre_publish_trim_margin: int,
        history_path: Path,
        run_root: Path,
        check_poll_interval_s: float,
        check_timeout_s: float,
        strict_drop_check: bool,
        payload_mode: str,
        tp_world_size: int,
        tp_total_bytes: int,
        publish_device: str,
    ) -> None:
        self._model_name = model_name
        self._key_template = key_template
        self._run_root = run_root
        self._check_poll_interval_s = check_poll_interval_s
        self._check_timeout_s = check_timeout_s
        self._strict_drop_check = bool(strict_drop_check)
        self._publish_device = _publisher_device(str(publish_device))
        self._check_device = _materialization_device("auto")
        self._payload_mode = _normalize_payload_mode(payload_mode)
        self._tp_world_size = _ensure_positive("tp_world_size", tp_world_size)
        self._tp_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
        if self._tp_total_bytes > 0 and self._payload_mode != "tp_ranked":
            raise ValueError("tp_total_bytes requires payload_mode=tp_ranked")
        gc_drain_timeout_s = (
            120.0
            if self._payload_mode == "tp_ranked"
            and self._tp_total_bytes > TP_FULL_VALIDATION_MAX_BYTES
            else 30.0
        )
        self._config = WeightPublisherConfig(
            model_name=model_name,
            keep_last=keep_last,
            pre_publish_trim_margin=pre_publish_trim_margin,
            history_path=str(history_path),
            policy="pinned",
            overflow_policy="reject",
            key_template=key_template,
            trigger_reload=False,
            verify_key_mapping=True,
            wait_persistence=False,
            gc_drain_timeout_s=gc_drain_timeout_s,
            gc_require_drained=True,
            stage_on_gpu=False,
        )
        self._publisher = WeightPublisher(self._config)

    def publish_versions(
        self,
        *,
        start_version: int,
        num_versions: int,
        publish_interval_s: float,
    ) -> list[PublishEvent]:
        _ensure_positive("num_versions", num_versions)
        _ensure_positive("start_version", start_version)

        events: list[PublishEvent] = []
        for offset in range(num_versions):
            version = start_version + offset
            self.publish_one_version(version=version, events=events)
            if offset + 1 < num_versions:
                time.sleep(max(0.0, publish_interval_s))
        return events

    def publish_one_version(
        self,
        *,
        version: int,
        events: list[PublishEvent],
    ) -> PublishEvent:
        key = _build_key(
            model_name=self._model_name,
            key_template=self._key_template,
            version=version,
        )
        publish_device = self._publish_device
        _publish_memory_log(
            stage="before_build",
            version=version,
            publish_device=publish_device,
        )
        tensors = _build_publish_tensors(
            version=version,
            device=publish_device,
            payload_mode=self._payload_mode,
            tp_world_size=self._tp_world_size,
            tp_total_bytes=self._tp_total_bytes,
        )
        _publish_memory_log(
            stage="after_build",
            version=version,
            publish_device=publish_device,
        )
        publish_payload_bytes = _estimate_tensor_payload_bytes(tensors)
        publish_start = time.monotonic()
        artifact_id = self._publisher.publish(tensors, version=version)
        publish_latency_s = time.monotonic() - publish_start
        publish_breakdown_s = self._publisher.last_publish_breakdown_s()
        put_s = float(publish_breakdown_s.get("put_s", 0.0))
        publish_throughput_gib_s = _bytes_throughput_gib_s(
            bytes_total=publish_payload_bytes,
            duration_s=float(publish_latency_s),
        )
        put_throughput_gib_s = _bytes_throughput_gib_s(
            bytes_total=publish_payload_bytes,
            duration_s=put_s,
        )
        _publish_memory_log(
            stage="after_publish",
            version=version,
            publish_device=publish_device,
        )
        # Release publisher-side source tensors before retention checks so
        # large payloads do not stay resident across GC polling.
        del tensors
        if self._tp_total_bytes > TP_FULL_VALIDATION_MAX_BYTES and str(
            publish_device
        ).startswith("cpu"):
            gc.collect()
        _publish_memory_log(
            stage="after_source_release",
            version=version,
            publish_device=publish_device,
        )
        event = PublishEvent(
            version=version,
            key=key,
            artifact_id=artifact_id,
            export_dir=f"in_memory://{publish_device}",
            publish_device=publish_device,
            published_at_s=time.time(),
            publish_latency_s=publish_latency_s,
            publish_payload_bytes=int(publish_payload_bytes),
            publish_throughput_gib_s=float(publish_throughput_gib_s),
            put_throughput_gib_s=float(put_throughput_gib_s),
            publish_breakdown_s=publish_breakdown_s,
        )
        events.append(event)
        self._verify_retention_window(events=events)
        print(
            "[publisher] published",
            f"version={version}",
            f"key={key}",
            f"artifact_id={artifact_id}",
            f"publish_latency_s={publish_latency_s:.3f}",
            f"payload_gib={float(publish_payload_bytes)/float(1024**3):.3f}",
            f"publish_bw_gib_s={publish_throughput_gib_s:.3f}",
            f"put_bw_gib_s={put_throughput_gib_s:.3f}",
            flush=True,
        )
        return event

    def _verify_retention_window(self, *, events: list[PublishEvent]) -> None:
        keep_last = int(self._config.keep_last)
        if keep_last <= 0 or not events:
            return
        kept_versions = {event.version for event in events[-keep_last:]}
        heavy_materialization_probe = (
            self._strict_drop_check
            or self._tp_total_bytes <= TP_FULL_VALIDATION_MAX_BYTES
            or self._payload_mode != "tp_ranked"
        )
        for event in events:
            self._wait_key_mapping_state(
                key=event.key,
                expected_artifact_id=event.artifact_id,
            )
            expected_materializable = event.version in kept_versions
            if expected_materializable or self._strict_drop_check:
                if heavy_materialization_probe:
                    self._wait_materialization_state(
                        key=event.key,
                        version=event.version,
                        expected_materializable=expected_materializable,
                    )
                self._wait_artifact_registry_state(
                    artifact_id=event.artifact_id,
                    expected_exists=expected_materializable,
                )

    def _probe_artifact_exists(self, artifact_id: str) -> bool:
        try:
            return resolve_artifact(artifact_id=artifact_id).exists()
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return False
            raise

    def _resolve_key_mapping(self, key: str) -> str | None:
        try:
            mapping = get_store_context().ensure_client().resolve_key_mapping(key)
            resolved = str(mapping.artifact_id or "").strip()
            if not resolved:
                return None
            return resolved
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return None
            raise

    def _wait_key_mapping_state(self, *, key: str, expected_artifact_id: str) -> None:
        deadline = time.monotonic() + self._check_timeout_s
        last_resolved: str | None = None
        while time.monotonic() < deadline:
            resolved = self._resolve_key_mapping(key)
            last_resolved = resolved
            if resolved == expected_artifact_id:
                return
            time.sleep(self._check_poll_interval_s)
        raise AssertionError(
            f"key mapping mismatch: key={key}, "
            f"expected_artifact_id={expected_artifact_id}, last_resolved={last_resolved}"
        )

    def _probe_materializable(self, *, key: str, version: int) -> bool:
        try:
            artifact = resolve_artifact(key=key)
            tensors = artifact.tensor_dict(
                device=self._check_device,
                options=self._check_options(),
            )
            _validate_payload(
                version=version,
                tensors=tensors,
                payload_mode=self._payload_mode,
                tp_world_size=self._tp_world_size,
                tp_total_bytes=self._tp_total_bytes,
            )
            return True
        except Exception:
            return False

    def _check_options(self) -> GetArtifactOptions:
        return GetArtifactOptions(source="local_only")

    def _wait_materialization_state(
        self,
        *,
        key: str,
        version: int,
        expected_materializable: bool,
    ) -> None:
        deadline = time.monotonic() + self._check_timeout_s
        last_state: bool | None = None
        while time.monotonic() < deadline:
            materializable = self._probe_materializable(key=key, version=version)
            last_state = materializable
            if materializable == expected_materializable:
                return
            time.sleep(self._check_poll_interval_s)
        raise AssertionError(
            f"materialization retention mismatch: key={key}, version={version}, "
            f"expected_materializable={expected_materializable}, last_materializable={last_state}"
        )

    def _wait_artifact_registry_state(
        self,
        *,
        artifact_id: str,
        expected_exists: bool,
    ) -> None:
        deadline = time.monotonic() + self._check_timeout_s
        last_exists: bool | None = None
        while time.monotonic() < deadline:
            exists = self._probe_artifact_exists(artifact_id)
            last_exists = exists
            if exists == expected_exists:
                return
            time.sleep(self._check_poll_interval_s)
        raise AssertionError(
            f"artifact registry retention mismatch: artifact_id={artifact_id}, "
            f"expected_exists={expected_exists}, last_exists={last_exists}"
        )


class WeightUpdateReceiver:
    def __init__(
        self,
        *,
        model_name: str,
        key_template: str,
        poll_interval_s: float,
        per_version_timeout_s: float,
        materialize_device: str,
        apply_mode: str,
        allow_version_skip: bool,
        payload_mode: str,
        tp_world_size: int,
        tp_device_base_index: int,
        tp_device_map_policy: str,
        tp_total_bytes: int,
        tp_materialize_deadline_s: float,
        transport_group_mode: str,
        transport_group_kind: str,
        transport_group_namespace: str,
        transport_group_total_parts: int,
        transport_group_receiver_index: int,
        transport_group_priority: int,
        transport_group_epoch: int,
    ) -> None:
        self._model_name = model_name
        self._key_template = key_template
        self._poll_interval_s = poll_interval_s
        self._per_version_timeout_s = per_version_timeout_s
        self._materialize_options = GetArtifactOptions(
            source={
                "preference": "prefer_p2p",
                "allow_p2p": True,
                "allow_disk": False,
            },
            verify_checksums=False,
        )
        self._materialize_device = _materialization_device(materialize_device)
        self._apply_mode = str(apply_mode).strip()
        self._allow_version_skip = bool(allow_version_skip)
        self._payload_mode = _normalize_payload_mode(payload_mode)
        self._tp_world_size = _ensure_positive("tp_world_size", tp_world_size)
        self._tp_total_bytes = _normalize_tp_total_bytes(tp_total_bytes)
        self._tp_materialize_deadline_s = _ensure_non_negative_float(
            "tp_materialize_deadline_s",
            float(tp_materialize_deadline_s),
        )
        self._transport_group_mode = str(transport_group_mode).strip().lower()
        if self._transport_group_mode not in SUPPORTED_TRANSPORT_GROUP_MODES:
            raise ValueError(
                "unsupported transport_group_mode="
                f"{self._transport_group_mode!r}, "
                f"allowed={sorted(SUPPORTED_TRANSPORT_GROUP_MODES)}"
            )
        self._transport_group_kind = _sanitize_group_token(transport_group_kind)
        self._transport_group_namespace = _sanitize_group_token(
            transport_group_namespace
        )
        self._transport_group_total_parts = _ensure_non_negative(
            "transport_group_total_parts",
            int(transport_group_total_parts),
        )
        self._transport_group_receiver_index = _ensure_non_negative(
            "transport_group_receiver_index",
            int(transport_group_receiver_index),
        )
        self._transport_group_priority = _ensure_non_negative(
            "transport_group_priority",
            int(transport_group_priority),
        )
        self._transport_group_epoch = _ensure_non_negative(
            "transport_group_epoch",
            int(transport_group_epoch),
        )
        self._tp_device_base_index = _ensure_non_negative(
            "tp_device_base_index",
            tp_device_base_index,
        )
        self._tp_device_map_policy = str(tp_device_map_policy).strip().lower()
        if self._tp_device_map_policy not in SUPPORTED_TP_DEVICE_MAP_POLICIES:
            raise ValueError(
                "unsupported tp_device_map_policy="
                f"{self._tp_device_map_policy!r}, "
                f"allowed={sorted(SUPPORTED_TP_DEVICE_MAP_POLICIES)}"
            )
        self._tp_visible_device_count = 0
        self._tp_device_map_effective_mode = "inactive"
        self._tp_rank_devices: dict[int, str] = {}
        if self._tp_total_bytes > 0 and self._payload_mode != "tp_ranked":
            raise ValueError("tp_total_bytes requires payload_mode=tp_ranked")
        if self._apply_mode not in SUPPORTED_RECEIVER_APPLY_MODES:
            raise ValueError(f"unsupported receiver apply mode: {self._apply_mode}")
        if (
            self._apply_mode == "binding_swap"
            and not self._materialize_device.startswith("cuda:")
        ):
            raise ValueError("binding_swap mode requires a CUDA materialization device")
        if self._apply_mode in {"tp_bind_into_swap", "tp4_bind_into_swap"}:
            if self._apply_mode == "tp4_bind_into_swap" and self._tp_world_size != 4:
                raise ValueError(
                    f"tp4_bind_into_swap requires tp_world_size=4, got {self._tp_world_size}"
                )
            if self._payload_mode != "tp_ranked":
                raise ValueError(f"{self._apply_mode} requires payload_mode=tp_ranked")
            if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
                raise ValueError(f"{self._apply_mode} requires real CUDA backend")
            if not torch.cuda.is_available():
                raise ValueError(f"{self._apply_mode} requires CUDA devices")
            self._tp_visible_device_count = int(torch.cuda.device_count())
            (
                self._tp_rank_devices,
                self._tp_device_map_effective_mode,
            ) = _resolve_tp_rank_devices(
                tp_world_size=self._tp_world_size,
                tp_device_base_index=self._tp_device_base_index,
                visible_device_count=self._tp_visible_device_count,
                map_policy=self._tp_device_map_policy,
            )
            map_tokens = " ".join(
                f"r{rank}->{device}"
                for rank, device in self._tp_rank_devices.items()
            )
            print(
                "[receiver][tp-device-map]",
                f"policy={self._tp_device_map_policy}",
                f"effective_mode={self._tp_device_map_effective_mode}",
                f"tp_world_size={self._tp_world_size}",
                f"visible_cuda_devices={self._tp_visible_device_count}",
                f"base_index={self._tp_device_base_index}",
                map_tokens,
                flush=True,
            )
            if self._tp_device_map_effective_mode == "modulo":
                print(
                    "[receiver][tp-device-map]",
                    "fallback=modulo",
                    "reason=insufficient_visible_cuda_devices_for_contiguous_rank_map",
                    f"available_from_base={self._tp_visible_device_count - self._tp_device_base_index}",
                    flush=True,
                )
        if self._transport_group_mode == "tp_version":
            if self._transport_group_total_parts <= 0:
                raise ValueError(
                    "transport_group_total_parts must be > 0 when "
                    "transport_group_mode=tp_version"
                )
            if not self._transport_group_kind:
                raise ValueError(
                    "transport_group_kind must be non-empty when "
                    "transport_group_mode=tp_version"
                )
            if not self._transport_group_namespace:
                raise ValueError(
                    "transport_group_namespace must be non-empty when "
                    "transport_group_mode=tp_version"
                )
        self._binding: Any | None = None
        self._binding_ptrs: dict[str, int] | None = None
        self._tp_bindings: dict[int, Any] = {}
        self._tp_binding_ptrs: dict[int, dict[str, int]] = {}
        self._tp_pending_targets: dict[int, dict[str, torch.Tensor]] = {}
        self._progress_log_interval_s = RECEIVER_PROGRESS_LOG_INTERVAL_S
        self._resolve_slow_threshold_ms = RECEIVER_RESOLVE_SLOW_THRESHOLD_MS
        self._last_resolve_log_state = ""
        self._last_resolve_log_mono = 0.0

    def tp_device_map_info(self) -> dict[str, Any]:
        policy = str(getattr(self, "_tp_device_map_policy", "auto"))
        effective_mode = str(
            getattr(self, "_tp_device_map_effective_mode", "inactive")
        )
        visible_device_count = int(getattr(self, "_tp_visible_device_count", 0))
        rank_device_map = dict(getattr(self, "_tp_rank_devices", {}))
        return {
            "policy": policy,
            "effective_mode": effective_mode,
            "visible_device_count": visible_device_count,
            "rank_device_map": rank_device_map,
        }

    def receive_versions(
        self,
        *,
        start_version: int,
        num_versions: int,
        on_event: Callable[[ReceiveEvent], None] | None = None,
    ) -> list[ReceiveEvent]:
        _ensure_positive("num_versions", num_versions)
        _ensure_positive("start_version", start_version)

        events: list[ReceiveEvent] = []
        previous_artifact_id: str | None = None
        for offset in range(num_versions):
            version = start_version + offset
            key = _build_key(
                model_name=self._model_name,
                key_template=self._key_template,
                version=version,
            )
            try:
                event = self._wait_one(
                    version=version,
                    key=key,
                    max_version=start_version + num_versions - 1,
                )
            except VersionDroppedError as dropped:
                if not self._allow_version_skip:
                    raise RuntimeError(
                        "receiver observed dropped version while version skip is disabled: "
                        f"version={dropped.version}, key={dropped.key}, "
                        f"artifact_id={dropped.artifact_id}, "
                        f"newer_version={dropped.newer_version}"
                    ) from dropped
                print(
                    "[receiver] skipped",
                    f"version={dropped.version}",
                    f"key={dropped.key}",
                    f"artifact_id={dropped.artifact_id}",
                    (
                        f"newer_version={dropped.newer_version}"
                        if dropped.newer_version is not None
                        else "newer_version=n/a"
                    ),
                    "reason=version_deregistered",
                    flush=True,
                )
                continue
            if previous_artifact_id == event.artifact_id:
                raise AssertionError(
                    f"artifact_id reused across versions: version={version}, artifact_id={event.artifact_id}"
                )
            previous_artifact_id = event.artifact_id
            events.append(event)
            if on_event is not None:
                on_event(event)
            print(
                "[receiver] received",
                f"version={version}",
                f"key={key}",
                f"artifact_id={event.artifact_id}",
                f"latency_s={event.materialize_latency_s:.3f}",
                f"mode={event.apply_mode}",
                f"op={event.apply_operation}",
                (
                    f"pointer_stable={event.pointer_stable}"
                    if event.pointer_stable is not None
                    else "pointer_stable=n/a"
                ),
                flush=True,
            )
        return events

    def _wait_one(self, *, version: int, key: str, max_version: int) -> ReceiveEvent:
        if self._apply_mode in {"tp_bind_into_swap", "tp4_bind_into_swap"}:
            return self._wait_one_tp4_binding(
                version=version,
                key=key,
                max_version=max_version,
            )
        if self._apply_mode == "binding_swap":
            return self._wait_one_binding(
                version=version, key=key, max_version=max_version
            )
        return self._wait_one_tensor_dict(
            version=version, key=key, max_version=max_version
        )

    def _maybe_log_resolve(
        self,
        *,
        key: str,
        status: str,
        latency_ms: float,
        artifact_id: str | None,
        error_detail: str | None,
    ) -> None:
        now = time.monotonic()
        should_emit = (
            status != self._last_resolve_log_state
            or latency_ms >= self._resolve_slow_threshold_ms
            or (now - self._last_resolve_log_mono) >= self._progress_log_interval_s
        )
        if not should_emit:
            return
        tokens = [
            "[receiver][resolve]",
            f"status={status}",
            f"key={key}",
            f"latency_ms={latency_ms:.1f}",
        ]
        if artifact_id:
            tokens.append(f"artifact_id={artifact_id}")
        if error_detail:
            compact_error = str(error_detail).replace("\n", " ").strip()
            if len(compact_error) > 200:
                compact_error = compact_error[:200] + "..."
            tokens.append(f"error={compact_error}")
        print(" ".join(tokens), flush=True)
        self._last_resolve_log_state = status
        self._last_resolve_log_mono = now

    def _maybe_log_wait_progress(
        self,
        *,
        mode: str,
        version: int,
        key: str,
        start_monotonic: float,
        deadline: float,
        last_state: str,
        last_error: str | None,
        next_log_at: float,
    ) -> float:
        now = time.monotonic()
        if now < next_log_at:
            return next_log_at
        error_text = "none" if last_error is None else last_error
        print(
            "[receiver][wait]",
            f"mode={mode}",
            f"version={version}",
            f"key={key}",
            f"state={last_state}",
            f"elapsed_s={now - start_monotonic:.1f}",
            f"remaining_s={max(0.0, deadline - now):.1f}",
            f"last_error={error_text}",
            flush=True,
        )
        return now + self._progress_log_interval_s

    def _resolve_artifact_for_key(self, *, key: str) -> Any | None:
        artifact_id = self._resolve_artifact_id_for_key(key=key)
        if not artifact_id:
            return None
        return resolve_artifact(artifact_id=artifact_id)

    def _resolve_artifact_id_for_key(self, *, key: str) -> str | None:
        started = time.monotonic()
        try:
            mapping = get_store_context().ensure_client().resolve_key_mapping(key)
            artifact_id = str(mapping.artifact_id or "").strip()
            latency_ms = (time.monotonic() - started) * 1000.0
            if not artifact_id:
                self._maybe_log_resolve(
                    key=key,
                    status="empty",
                    latency_ms=latency_ms,
                    artifact_id=None,
                    error_detail=None,
                )
                return None
            self._maybe_log_resolve(
                key=key,
                status="ok",
                latency_ms=latency_ms,
                artifact_id=artifact_id,
                error_detail=None,
            )
            return artifact_id
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - started) * 1000.0
            if _is_not_found_error(exc):
                self._maybe_log_resolve(
                    key=key,
                    status="not_found",
                    latency_ms=latency_ms,
                    artifact_id=None,
                    error_detail=str(exc),
                )
                return None
            self._maybe_log_resolve(
                key=key,
                status="error",
                latency_ms=latency_ms,
                artifact_id=None,
                error_detail=f"{type(exc).__name__}:{exc}",
            )
            raise

    def _probe_artifact_exists(self, *, artifact_id: str) -> bool:
        try:
            return bool(resolve_artifact(artifact_id=artifact_id).exists())
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return False
            raise

    def _probe_version_materializable(self, *, version: int) -> bool:
        key = _build_key(
            model_name=self._model_name,
            key_template=self._key_template,
            version=version,
        )
        try:
            artifact = self._resolve_artifact_for_key(key=key)
            if artifact is None:
                return False
            tensors = artifact.tensor_dict(
                device=self._materialize_device,
                options=self._materialize_options,
            )
            _validate_payload(
                version=version,
                tensors=tensors,
                payload_mode=self._payload_mode,
                tp_world_size=self._tp_world_size,
                tp_total_bytes=self._tp_total_bytes,
            )
            return True
        except Exception:
            return False

    def _find_newer_materializable_version(
        self,
        *,
        version: int,
        max_version: int,
    ) -> int | None:
        for candidate in range(version + 1, max_version + 1):
            if self._probe_version_materializable(version=candidate):
                return candidate
        return None

    def _find_newer_resolved_version(
        self,
        *,
        version: int,
        max_version: int,
    ) -> int | None:
        for candidate in range(version + 1, max_version + 1):
            key = _build_key(
                model_name=self._model_name,
                key_template=self._key_template,
                version=candidate,
            )
            if self._resolve_artifact_id_for_key(key=key):
                return candidate
        return None

    def _maybe_raise_version_dropped(
        self,
        *,
        version: int,
        key: str,
        max_version: int,
        exc: Exception,
        treat_apply_unavailable_error_as_dropped: bool = False,
        prefer_resolved_newer_version: bool = False,
    ) -> None:
        looks_unavailable = _is_not_found_error(exc)
        if treat_apply_unavailable_error_as_dropped:
            looks_unavailable = looks_unavailable or _is_version_unavailable_during_apply_error(
                exc
            )
        if not looks_unavailable:
            return
        artifact_id = self._resolve_artifact_id_for_key(key=key)
        newer_version: int | None = None
        if prefer_resolved_newer_version:
            newer_version = self._find_newer_resolved_version(
                version=version,
                max_version=max_version,
            )
        if newer_version is None:
            newer_version = self._find_newer_materializable_version(
                version=version,
                max_version=max_version,
            )
        if artifact_id and newer_version is not None:
            raise VersionDroppedError(
                version=version,
                key=key,
                artifact_id=artifact_id,
                message=(
                    "target version already deregistered before receiver applied it"
                ),
                newer_version=newer_version,
            ) from exc

    def _wait_one_tensor_dict(
        self,
        *,
        version: int,
        key: str,
        max_version: int,
    ) -> ReceiveEvent:
        start_monotonic = time.monotonic()
        deadline = start_monotonic + self._per_version_timeout_s
        last_error: str | None = None
        last_state = "key_mapping_absent"
        next_log_at = start_monotonic

        while time.monotonic() < deadline:
            try:
                artifact = self._resolve_artifact_for_key(key=key)
                if artifact is None:
                    last_state = "key_mapping_absent"
                    next_log_at = self._maybe_log_wait_progress(
                        mode="tensor_dict",
                        version=version,
                        key=key,
                        start_monotonic=start_monotonic,
                        deadline=deadline,
                        last_state=last_state,
                        last_error=last_error,
                        next_log_at=next_log_at,
                    )
                    time.sleep(self._poll_interval_s)
                    continue
                start = time.monotonic()
                materialized = artifact.tensor_dict_with_diagnostics(
                    device=self._materialize_device,
                    options=self._materialize_options,
                )
                latency_s = time.monotonic() - start
                tensors = materialized.tensors
                diagnostics = materialized.diagnostics
                try:
                    _validate_payload(
                        version=version,
                        tensors=tensors,
                        payload_mode=self._payload_mode,
                        tp_world_size=self._tp_world_size,
                        tp_total_bytes=self._tp_total_bytes,
                    )
                finally:
                    self._release_tensor_dict_replica_after_apply(
                        replica_uuid=str(diagnostics.replica_uuid),
                        disk_path=(
                            str(diagnostics.disk_path)
                            if diagnostics.disk_path is not None
                            else ""
                        ),
                        tensors=tensors,
                    )
                    materialized = None
                return ReceiveEvent(
                    version=version,
                    key=key,
                    artifact_id=artifact.artifact_id,
                    received_at_s=time.time(),
                    materialize_latency_s=latency_s,
                    apply_mode="tensor_dict",
                    apply_operation="materialize",
                    pointer_stable=None,
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(
                    exc, RuntimeError
                ) and "failed to unload tensor_dict replica after apply" in str(exc):
                    raise
                last_error = _compact_error_text(exc)
                if _is_cuda_oom_error(exc):
                    raise RuntimeError(
                        "receiver encountered CUDA OOM in tensor_dict path "
                        f"version={version}, key={key}, detail={last_error}"
                    ) from exc
                self._maybe_raise_version_dropped(
                    version=version,
                    key=key,
                    max_version=max_version,
                    exc=exc,
                )
                last_state = "materialize_failed"
                next_log_at = self._maybe_log_wait_progress(
                    mode="tensor_dict",
                    version=version,
                    key=key,
                    start_monotonic=start_monotonic,
                    deadline=deadline,
                    last_state=last_state,
                    last_error=last_error,
                    next_log_at=next_log_at,
                )
                time.sleep(self._poll_interval_s)

        raise TimeoutError(
            "receiver timeout "
            f"version={version}, key={key}, last_state={last_state}, last_error={last_error}"
        )

    def _release_tensor_dict_replica_after_apply(
        self,
        *,
        replica_uuid: str,
        disk_path: str,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if not replica_uuid:
            return
        tensors.clear()
        gc.collect()
        client = get_store_context().ensure_client()
        unloaded = False
        for _ in range(RECEIVER_TENSOR_DICT_UNLOAD_RETRIES):
            if client.unload_replica(replica_uuid, disk_path=disk_path):
                unloaded = True
                break
            gc.collect()
            time.sleep(RECEIVER_TENSOR_DICT_UNLOAD_RETRY_INTERVAL_S)
        if unloaded:
            return
        raise RuntimeError(
            "receiver failed to unload tensor_dict replica after apply "
            f"replica_uuid={replica_uuid}, disk_path={disk_path or '<memory>'}"
        )

    def _wait_one_binding(
        self,
        *,
        version: int,
        key: str,
        max_version: int,
    ) -> ReceiveEvent:
        start_monotonic = time.monotonic()
        deadline = start_monotonic + self._per_version_timeout_s
        attempt_seq = 0
        last_error: str | None = None
        last_state = "key_mapping_absent"
        next_log_at = start_monotonic

        while time.monotonic() < deadline:
            try:
                artifact = self._resolve_artifact_for_key(key=key)
                if artifact is None:
                    last_state = "key_mapping_absent"
                    next_log_at = self._maybe_log_wait_progress(
                        mode="binding_swap",
                        version=version,
                        key=key,
                        start_monotonic=start_monotonic,
                        deadline=deadline,
                        last_state=last_state,
                        last_error=last_error,
                        next_log_at=next_log_at,
                    )
                    time.sleep(self._poll_interval_s)
                    continue
                start = time.monotonic()
                pointer_stable: bool | None = None
                apply_operation = "swap"
                remaining_s = max(0.001, deadline - time.monotonic())
                materialize_ctx = self._make_tp_materialize_ctx(
                    version=version,
                    rank=0,
                    remaining_s=remaining_s,
                    attempt=attempt_seq,
                )
                attempt_seq += 1
                if self._binding is None:
                    self._binding = artifact.bind(
                        device=self._materialize_device,
                        packing="byte_space",
                        options=self._materialize_options,
                        ctx=materialize_ctx,
                    )
                    self._binding_ptrs = {
                        name: tensor.data_ptr()
                        for name, tensor in self._binding.tensors.items()
                    }
                    pointer_stable = True
                    apply_operation = "bind"
                else:
                    if self._binding_ptrs is None:
                        raise AssertionError("binding pointer baseline is missing")
                    self._binding.swap(
                        artifact,
                        options=self._materialize_options,
                        ctx=materialize_ctx,
                    )
                    latest_ptrs = {
                        name: tensor.data_ptr()
                        for name, tensor in self._binding.tensors.items()
                    }
                    if set(latest_ptrs) != set(self._binding_ptrs):
                        raise AssertionError(
                            "binding tensor set changed across swap: "
                            f"before={sorted(self._binding_ptrs.keys())}, "
                            f"after={sorted(latest_ptrs.keys())}"
                        )
                    pointer_stable = True
                    for name, expected_ptr in self._binding_ptrs.items():
                        actual_ptr = latest_ptrs.get(name)
                        if actual_ptr != expected_ptr:
                            pointer_stable = False
                            raise AssertionError(
                                "binding pointer changed across swap: "
                                f"name={name}, before={expected_ptr}, after={actual_ptr}"
                            )
                latency_s = time.monotonic() - start
                active_tensors = dict(self._binding.tensors)
                _validate_payload(
                    version=version,
                    tensors=active_tensors,
                    payload_mode=self._payload_mode,
                    tp_world_size=self._tp_world_size,
                    tp_total_bytes=self._tp_total_bytes,
                )
                return ReceiveEvent(
                    version=version,
                    key=key,
                    artifact_id=str(self._binding.artifact_id),
                    received_at_s=time.time(),
                    materialize_latency_s=latency_s,
                    apply_mode="binding_swap",
                    apply_operation=apply_operation,
                    pointer_stable=pointer_stable,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = _compact_error_text(exc)
                if _is_cuda_oom_error(exc):
                    raise RuntimeError(
                        "receiver encountered CUDA OOM in binding_swap path "
                        f"version={version}, key={key}, detail={last_error}"
                    ) from exc
                self._maybe_raise_version_dropped(
                    version=version,
                    key=key,
                    max_version=max_version,
                    exc=exc,
                )
                last_state = "binding_apply_failed"
                next_log_at = self._maybe_log_wait_progress(
                    mode="binding_swap",
                    version=version,
                    key=key,
                    start_monotonic=start_monotonic,
                    deadline=deadline,
                    last_state=last_state,
                    last_error=last_error,
                    next_log_at=next_log_at,
                )
                time.sleep(self._poll_interval_s)

        raise TimeoutError(
            "receiver timeout for binding mode "
            f"version={version}, key={key}, last_state={last_state}, last_error={last_error}"
        )

    def _tp_rank_device(self, rank: int) -> str:
        rank_devices = getattr(self, "_tp_rank_devices", None)
        resolved = (
            rank_devices.get(rank)
            if isinstance(rank_devices, dict)
            else None
        )
        if resolved is not None:
            return resolved
        return f"cuda:{self._tp_device_base_index + rank}"

    def _make_tp_materialize_ctx(
        self,
        *,
        version: int,
        rank: int,
        remaining_s: float,
        attempt: int = 0,
    ) -> Any | None:
        if self._tp_materialize_deadline_s <= 0.0:
            return None
        bounded_s = max(
            0.001,
            min(float(self._tp_materialize_deadline_s), float(remaining_s)),
        )
        safe_attempt = max(0, int(attempt))
        tags: dict[str, bool | int | float | str] | None = None
        if self._transport_group_mode == "tp_version":
            group_id = _sanitize_group_token(
                f"{self._transport_group_namespace}:v{int(version)}"
            )
            part_id = _sanitize_group_token(
                f"rx{self._transport_group_receiver_index}:r{int(rank)}"
            )
            # Keep request_id stable across retries so transport-group idempotency
            # deduplicates replays for the same logical part.
            request_id = _sanitize_group_token(f"{group_id}:{part_id}")
            tags = {
                "tc.transport.group.kind": self._transport_group_kind,
                "tc.transport.group.id": group_id,
                "tc.transport.group.total_parts": int(
                    self._transport_group_total_parts
                ),
                "tc.transport.group.part_id": part_id,
                "tc.transport.group.priority": int(self._transport_group_priority),
                "tc.transport.group.epoch": int(self._transport_group_epoch),
                "tc.transport.request_id": request_id,
                "tc.transport.request_attempt": safe_attempt,
                "tc.tp.no_progress_remaining_s": float(max(0.0, remaining_s)),
            }
        return tc.context(
            qos="interactive",
            deadline_ms=max(1, int(bounded_s * 1000.0)),
            tags=tags,
        )

    def _tp_rank_attempt_timeout_s(
        self,
        *,
        remaining_s: float,
        attempt: int = 0,
        completed_ranks_count: int = 0,
    ) -> float:
        per_rank_gib = (
            float(max(0, int(self._tp_total_bytes))) / float(1024**3)
        ) / float(max(1, int(self._tp_world_size)))
        budget_s = max(
            TP_BIND_PER_RANK_TIMEOUT_MIN_S,
            TP_BIND_PER_RANK_TIMEOUT_FLOOR_S
            + per_rank_gib * TP_BIND_PER_RANK_TIMEOUT_GIB_FACTOR,
        )
        safe_attempt = max(0, int(attempt))
        completed = max(0, int(completed_ranks_count))
        if safe_attempt > 0 and completed > 0:
            tp_world_size = max(1, int(self._tp_world_size))
            tail_ratio = min(1.0, float(completed) / float(tp_world_size))
            budget_s *= 1.0 + 0.75 * tail_ratio
            if self._transport_group_mode == "tp_version":
                budget_s *= 1.1
        return max(0.001, min(float(remaining_s), float(budget_s)))

    def _reset_tp_bindings_after_apply_failure(
        self,
        *,
        version: int,
        clear_pending_targets: bool = False,
    ) -> None:
        if not self._tp_bindings and (
            not clear_pending_targets or not self._tp_pending_targets
        ):
            return
        recycled = 0
        for rank, binding in list(self._tp_bindings.items()):
            tensors = getattr(binding, "tensors", None)
            if isinstance(tensors, dict) and tensors:
                self._tp_pending_targets[rank] = dict(tensors)
                recycled += 1
            with contextlib.suppress(Exception):
                binding.close()
        self._tp_bindings.clear()
        self._tp_binding_ptrs.clear()
        cleared_pending = 0
        if clear_pending_targets and self._tp_pending_targets:
            pending_targets = list(self._tp_pending_targets.values())
            cleared_pending = len(pending_targets)
            self._tp_pending_targets.clear()
            for tensors in pending_targets:
                tensors.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            "[receiver][tp]",
            f"version={version}",
            "phase=reset_after_failure",
            f"recycled_ranks={recycled}",
            f"cleared_pending_ranks={cleared_pending}",
            flush=True,
        )

    def _reset_tp_rank_after_apply_failure(
        self,
        *,
        version: int,
        rank: int,
        clear_pending_target: bool = False,
    ) -> None:
        binding = self._tp_bindings.pop(rank, None)
        self._tp_binding_ptrs.pop(rank, None)
        recycled = 0
        if binding is not None:
            tensors = getattr(binding, "tensors", None)
            if isinstance(tensors, dict) and tensors:
                self._tp_pending_targets[rank] = dict(tensors)
                recycled = 1
            with contextlib.suppress(Exception):
                binding.close()
        cleared_pending = 0
        if clear_pending_target:
            pending = self._tp_pending_targets.pop(rank, None)
            if isinstance(pending, dict):
                pending.clear()
                cleared_pending = 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            "[receiver][tp]",
            f"version={version}",
            "phase=reset_rank_after_failure",
            f"rank={rank}",
            f"recycled={recycled}",
            f"cleared_pending_target={cleared_pending}",
            flush=True,
        )

    def _maybe_publish_tp_binding(
        self,
        *,
        binding: Any,
        version: int,
        rank: int,
        phase: str,
        ctx: Any | None,
    ) -> None:
        publish_kwargs: dict[str, Any] = {}
        if ctx is not None:
            publish_kwargs["ctx"] = ctx
        try:
            binding.publish_replica(**publish_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(
                "[receiver][tp]",
                f"version={version}",
                f"rank={rank}",
                f"phase={phase}",
                "publish=skipped",
                f"reason={_compact_error_text(exc)}",
                flush=True,
            )

    def _apply_tp4_rank(
        self,
        *,
        version: int,
        rank: int,
        artifact: Any,
        ctx: Any | None,
    ) -> tuple[str, bool]:
        binding = self._tp_bindings.get(rank)
        if binding is None:
            target_tensors = self._tp_pending_targets.get(rank)
            if target_tensors is None:
                try:
                    target_tensors = _allocate_tp4_rank_targets(
                        device=self._tp_rank_device(rank),
                        tp_world_size=self._tp_world_size,
                        tp_total_bytes=self._tp_total_bytes,
                    )
                except Exception:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                self._tp_pending_targets[rank] = target_tensors
            bind_kwargs: dict[str, Any] = {}
            if ctx is not None:
                bind_kwargs["ctx"] = ctx
            try:
                binding = artifact.bind_into(
                    target_tensors=target_tensors,
                    mapping=_build_tp4_rank_copy_plan(
                        rank=rank,
                        tp_world_size=self._tp_world_size,
                        tp_total_bytes=self._tp_total_bytes,
                    ),
                    packing="byte_space",
                    options=self._materialize_options,
                    **bind_kwargs,
                )
            except Exception:
                # Keep preallocated rank targets across retries so transient
                # resolve/materialize races do not repeatedly allocate tens of
                # GiBs and trigger allocator-driven OOM.
                raise
            self._tp_bindings[rank] = binding
            self._tp_binding_ptrs[rank] = {
                name: tensor.data_ptr() for name, tensor in binding.tensors.items()
            }
            self._tp_pending_targets.pop(rank, None)
            _validate_tp4_rank_targets(
                version=version,
                rank=rank,
                tensors=dict(binding.tensors),
                tp_world_size=self._tp_world_size,
                tp_total_bytes=self._tp_total_bytes,
            )
            self._maybe_publish_tp_binding(
                binding=binding,
                version=version,
                rank=rank,
                phase="bind_into",
                ctx=ctx,
            )
            return ("bind_into", True)

        pointer_baseline = self._tp_binding_ptrs.get(rank)
        if pointer_baseline is None:
            raise AssertionError(f"missing pointer baseline for rank={rank}")
        swap_kwargs: dict[str, Any] = {"publish": False}
        if ctx is not None:
            swap_kwargs["ctx"] = ctx
        binding.swap(artifact, options=self._materialize_options, **swap_kwargs)
        latest_ptrs = {
            name: tensor.data_ptr() for name, tensor in binding.tensors.items()
        }
        if set(latest_ptrs) != set(pointer_baseline):
            raise AssertionError(
                "tp4 tensor set changed across swap: "
                f"rank={rank}, before={sorted(pointer_baseline.keys())}, "
                f"after={sorted(latest_ptrs.keys())}"
            )
        for name, expected_ptr in pointer_baseline.items():
            actual_ptr = latest_ptrs.get(name)
            if actual_ptr != expected_ptr:
                raise AssertionError(
                    "tp4 pointer changed across swap: "
                    f"rank={rank}, name={name}, before={expected_ptr}, after={actual_ptr}"
                )
        _validate_tp4_rank_targets(
            version=version,
            rank=rank,
            tensors=dict(binding.tensors),
            tp_world_size=self._tp_world_size,
            tp_total_bytes=self._tp_total_bytes,
        )
        self._maybe_publish_tp_binding(
            binding=binding,
            version=version,
            rank=rank,
            phase="swap",
            ctx=ctx,
        )
        return ("swap", True)

    def _wait_one_tp4_binding(
        self,
        *,
        version: int,
        key: str,
        max_version: int,
    ) -> ReceiveEvent:
        start_monotonic = time.monotonic()
        last_progress_monotonic = start_monotonic
        no_progress_deadline = start_monotonic + self._per_version_timeout_s
        last_error: str | None = None
        last_state = "key_mapping_absent"
        next_log_at = start_monotonic
        rank_attempt_seq: dict[int, int] = dict.fromkeys(
            range(self._tp_world_size), 0
        )
        completed_ranks: set[int] = set()
        active_rank: int | None = None
        pointer_stable = True
        operation = "swap"
        cold_start_bindings = not self._tp_bindings
        artifact_id: str | None = None

        while time.monotonic() < no_progress_deadline:
            try:
                artifact = self._resolve_artifact_for_key(key=key)
                if artifact is None:
                    last_state = "key_mapping_absent"
                    next_log_at = self._maybe_log_wait_progress(
                        mode=self._apply_mode,
                        version=version,
                        key=key,
                        start_monotonic=start_monotonic,
                        deadline=no_progress_deadline,
                        last_state=last_state,
                        last_error=last_error,
                        next_log_at=next_log_at,
                    )
                    time.sleep(self._poll_interval_s)
                    continue
                for rank in range(self._tp_world_size):
                    if rank in completed_ranks:
                        binding = self._tp_bindings.get(rank)
                        if binding is None:
                            completed_ranks.discard(rank)
                            continue
                        current_artifact_id = str(binding.artifact_id)
                        if artifact_id is None:
                            artifact_id = current_artifact_id
                        elif artifact_id != current_artifact_id:
                            raise AssertionError(
                                "tp4 ranks resolved different artifact ids: "
                                f"first={artifact_id}, rank={rank}, current={current_artifact_id}"
                            )
                        continue
                    active_rank = rank
                    rank_start = time.monotonic()
                    rank_remaining_s = max(
                        0.001, no_progress_deadline - time.monotonic()
                    )
                    rank_attempt = rank_attempt_seq.get(rank, 0)
                    rank_attempt_seq[rank] = rank_attempt + 1
                    rank_attempt_timeout_s = self._tp_rank_attempt_timeout_s(
                        remaining_s=rank_remaining_s,
                        attempt=rank_attempt,
                        completed_ranks_count=len(completed_ranks),
                    )
                    rank_ctx = self._make_tp_materialize_ctx(
                        version=version,
                        rank=rank,
                        remaining_s=rank_attempt_timeout_s,
                        attempt=rank_attempt,
                    )
                    print(
                        "[receiver][tp]",
                        f"version={version}",
                        f"rank={rank}",
                        f"attempt={rank_attempt}",
                        f"device={self._tp_rank_device(rank)}",
                        "phase=start",
                        f"deadline_s={rank_attempt_timeout_s:.3f}",
                        flush=True,
                    )
                    rank_operation, rank_pointer_stable = self._apply_tp4_rank(
                        version=version,
                        rank=rank,
                        artifact=artifact,
                        ctx=rank_ctx,
                    )
                    if rank_operation == "bind_into" and cold_start_bindings:
                        operation = "bind_into"
                    if not rank_pointer_stable:
                        pointer_stable = False
                    last_progress_monotonic = time.monotonic()
                    no_progress_deadline = (
                        last_progress_monotonic + self._per_version_timeout_s
                    )
                    completed_ranks.add(rank)
                    binding = self._tp_bindings.get(rank)
                    if binding is None:
                        raise AssertionError(f"tp4 binding missing for rank={rank}")
                    current_artifact_id = str(binding.artifact_id)
                    if artifact_id is None:
                        artifact_id = current_artifact_id
                    elif artifact_id != current_artifact_id:
                        raise AssertionError(
                            "tp4 ranks resolved different artifact ids: "
                            f"first={artifact_id}, rank={rank}, current={current_artifact_id}"
                        )
                    print(
                        "[receiver][tp]",
                        f"version={version}",
                        f"rank={rank}",
                        f"device={self._tp_rank_device(rank)}",
                        "phase=done",
                        f"op={rank_operation}",
                        f"latency_ms={(time.monotonic() - rank_start) * 1000.0:.1f}",
                        f"pointer_stable={rank_pointer_stable}",
                        flush=True,
                    )
                    active_rank = None
                if len(completed_ranks) != self._tp_world_size:
                    continue
                latency_s = time.monotonic() - start_monotonic
                if artifact_id is None:
                    raise AssertionError("tp4 artifact_id is empty after apply")
                return ReceiveEvent(
                    version=version,
                    key=key,
                    artifact_id=artifact_id,
                    received_at_s=time.time(),
                    materialize_latency_s=latency_s,
                    apply_mode=self._apply_mode,
                    apply_operation=operation,
                    pointer_stable=pointer_stable,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = _compact_error_text(exc)
                clear_pending_target = _should_clear_tp_pending_target_on_apply_failure(
                    exc
                )
                if _is_cuda_oom_error(exc):
                    raise RuntimeError(
                        "receiver encountered CUDA OOM in tp_bind_into_swap path "
                        f"version={version}, key={key}, detail={last_error}"
                    ) from exc
                try:
                    self._maybe_raise_version_dropped(
                        version=version,
                        key=key,
                        max_version=max_version,
                        exc=exc,
                        treat_apply_unavailable_error_as_dropped=True,
                        prefer_resolved_newer_version=True,
                    )
                except VersionDroppedError:
                    self._reset_tp_bindings_after_apply_failure(
                        version=version,
                        clear_pending_targets=True,
                    )
                    completed_ranks.clear()
                    raise
                if (
                    self._transport_group_mode == "tp_version"
                    and _is_non_retryable_transport_group_error(exc)
                ):
                    artifact_id = self._resolve_artifact_id_for_key(key=key)
                    self._reset_tp_bindings_after_apply_failure(
                        version=version,
                        clear_pending_targets=True,
                    )
                    completed_ranks.clear()
                    raise VersionDroppedError(
                        version=version,
                        key=key,
                        artifact_id=artifact_id,
                        message=(
                            "non-retryable tp transport-group apply failure "
                            f"version={version}, key={key}, artifact_id={artifact_id}, "
                            f"detail={last_error}"
                        ),
                        newer_version=None,
                    ) from exc
                if active_rank is not None:
                    self._reset_tp_rank_after_apply_failure(
                        version=version,
                        rank=active_rank,
                        clear_pending_target=clear_pending_target,
                    )
                    completed_ranks.discard(active_rank)
                    active_rank = None
                if completed_ranks:
                    print(
                        "[receiver][tp]",
                        f"version={version}",
                        "phase=retry_preserve_completed_ranks",
                        f"completed_ranks={sorted(completed_ranks)}",
                        f"last_error={last_error}",
                        flush=True,
                    )
                else:
                    self._reset_tp_bindings_after_apply_failure(
                        version=version,
                        clear_pending_targets=clear_pending_target,
                    )
                    completed_ranks.clear()
                    pointer_stable = True
                    operation = "swap"
                    artifact_id = None
                last_state = "tp_binding_apply_failed"
                next_log_at = self._maybe_log_wait_progress(
                    mode=self._apply_mode,
                    version=version,
                    key=key,
                    start_monotonic=start_monotonic,
                    deadline=no_progress_deadline,
                    last_state=last_state,
                    last_error=last_error,
                    next_log_at=next_log_at,
                )
                time.sleep(self._poll_interval_s)

        no_progress_elapsed_s = max(0.0, time.monotonic() - last_progress_monotonic)
        raise TimeoutError(
            "receiver timeout for tp binding mode "
            f"version={version}, key={key}, last_state={last_state}, "
            f"last_error={last_error}, no_progress_elapsed_s={no_progress_elapsed_s:.3f}"
        )

    def close(self) -> None:
        binding = self._binding
        self._binding = None
        self._binding_ptrs = None
        if binding is not None:
            binding.close()
        tp_bindings = list(self._tp_bindings.values())
        self._tp_bindings.clear()
        self._tp_binding_ptrs.clear()
        for rank_binding in tp_bindings:
            rank_binding.close()
        pending_targets = list(self._tp_pending_targets.values())
        self._tp_pending_targets.clear()
        for tensors in pending_targets:
            tensors.clear()
        if pending_targets and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _init_tensorcast(args: argparse.Namespace) -> None:
    init_mode = str(args.init_mode)
    if init_mode == "connect":
        if args.connect_address:
            tc.init(mode="connect", address=str(args.connect_address))
        else:
            tc.init(mode="connect")
        return
    tc.init(
        mode=init_mode,  # pyright: ignore[reportArgumentType]
        daemon_config_path=str(args.daemon_config_path)
        if args.daemon_config_path
        else None,
        global_store_mode=args.global_store_mode,  # pyright: ignore[reportArgumentType]
        global_store_address=str(args.global_store_address)
        if args.global_store_address
        else None,
        global_store_config_path=str(args.global_store_config_path)
        if args.global_store_config_path
        else None,
        cluster_id=str(args.cluster_id) if args.cluster_id else None,
        allow_gs_fallback=False,
    )


def _resolve_run_root(args: argparse.Namespace) -> Path:
    base = Path(str(args.weights_root)).expanduser().resolve()
    run_id = str(args.run_id).strip() if args.run_id else ""
    if not run_id:
        run_id = f"run-{int(time.time())}-{os.getpid()}"
    run_root = base / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _resolve_model_name(args: argparse.Namespace, *, run_root: Path | None) -> str:
    if args.model_name:
        return str(args.model_name).strip()
    if run_root is None:
        raise ValueError("model_name is required when run root is unavailable")
    suffix = run_root.name.replace("_", "-")
    return f"weight-publisher-e2e-{suffix}"


def _to_jsonable(
    items: list[PublishEvent] | list[ReceiveEvent],
) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def _write_summary(path: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_ready_file(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    ready_path = Path(path).expanduser().resolve()
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _maybe_hold_after_finish(args: argparse.Namespace, *, mode: str) -> None:
    hold_s = max(0.0, float(args.hold_after_finish_s))
    if hold_s <= 0:
        return
    print(
        f"[{mode}] hold_after_finish_s={hold_s:.1f}, "
        "keep runtime alive for external probes",
        flush=True,
    )
    time.sleep(hold_s)


def _build_publisher_runner(
    args: argparse.Namespace,
    *,
    run_root: Path,
    model_name: str,
) -> WeightUpdatePublisher:
    history_path = (
        Path(str(args.history_path)).expanduser().resolve()
        if args.history_path
        else run_root / "publisher_history.json"
    )
    return WeightUpdatePublisher(
        model_name=model_name,
        key_template=str(args.key_template),
        keep_last=_ensure_non_negative("keep_last", int(args.keep_last)),
        pre_publish_trim_margin=_ensure_non_negative(
            "pre_publish_trim_margin",
            int(args.pre_publish_trim_margin),
        ),
        history_path=history_path,
        run_root=run_root,
        check_poll_interval_s=float(args.poll_interval_s),
        check_timeout_s=float(args.retention_timeout_s),
        strict_drop_check=bool(args.strict_retention_drop_check),
        payload_mode=str(args.payload_mode),
        tp_world_size=int(args.tp_world_size),
        tp_total_bytes=int(args.tp_total_bytes),
        publish_device=str(args.publish_device),
    )


def _build_receiver_runner(
    args: argparse.Namespace,
    *,
    model_name: str,
) -> WeightUpdateReceiver:
    return WeightUpdateReceiver(
        model_name=model_name,
        key_template=str(args.key_template),
        poll_interval_s=float(args.poll_interval_s),
        per_version_timeout_s=float(args.receiver_timeout_s),
        materialize_device=str(args.materialize_device),
        apply_mode=str(args.receiver_apply_mode),
        allow_version_skip=bool(args.allow_version_skip),
        payload_mode=str(args.payload_mode),
        tp_world_size=int(args.tp_world_size),
        tp_device_base_index=int(args.tp_device_base_index),
        tp_device_map_policy=str(args.tp_device_map_policy),
        tp_total_bytes=int(args.tp_total_bytes),
        tp_materialize_deadline_s=float(args.tp_materialize_deadline_s),
        transport_group_mode=str(args.transport_group_mode),
        transport_group_kind=str(args.transport_group_kind),
        transport_group_namespace=str(args.transport_group_namespace),
        transport_group_total_parts=int(args.transport_group_total_parts),
        transport_group_receiver_index=int(args.transport_group_receiver_index),
        transport_group_priority=int(args.transport_group_priority),
        transport_group_epoch=int(args.transport_group_epoch),
    )


def _run_publisher(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    try:
        run_root = _resolve_run_root(args)
        model_name = _resolve_model_name(args, run_root=run_root)
        publisher = _build_publisher_runner(
            args, run_root=run_root, model_name=model_name
        )
        events = publisher.publish_versions(
            start_version=int(args.start_version),
            num_versions=int(args.num_versions),
            publish_interval_s=float(args.publish_interval_s),
        )
        summary = {
            "mode": "publisher",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "keep_last": int(args.keep_last),
            "pre_publish_trim_margin": int(args.pre_publish_trim_margin),
            "strict_retention_drop_check": bool(args.strict_retention_drop_check),
            "payload_mode": str(args.payload_mode),
            "tp_world_size": int(args.tp_world_size),
            "tp_total_bytes": int(args.tp_total_bytes),
            "publish_device": _publisher_device(str(args.publish_device)),
            "run_root": str(run_root),
            "published": _to_jsonable(events),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else run_root / "publisher_summary.json"
        )
        _write_summary(output, summary)
        _maybe_hold_after_finish(args, mode="publisher")
        return 0
    finally:
        tc.shutdown()


def _run_receiver(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    receiver: WeightUpdateReceiver | None = None
    try:
        model_name = _resolve_model_name(args, run_root=None)
        receiver = _build_receiver_runner(args, model_name=model_name)
        tp_device_map_info = receiver.tp_device_map_info()
        _write_ready_file(
            str(args.ready_file) if args.ready_file else None,
            {
                "mode": "receiver",
                "model_name": model_name,
                "receiver_apply_mode": str(args.receiver_apply_mode),
                "payload_mode": str(args.payload_mode),
                "tp_world_size": int(args.tp_world_size),
                "tp_total_bytes": int(args.tp_total_bytes),
                "tp_device_base_index": int(args.tp_device_base_index),
                "tp_device_map_policy": str(args.tp_device_map_policy),
                "tp_device_map_effective_mode": tp_device_map_info.get(
                    "effective_mode", "inactive"
                ),
                "tp_visible_device_count": int(
                    tp_device_map_info.get("visible_device_count", 0)
                ),
                "tp_rank_device_map": tp_device_map_info.get("rank_device_map", {}),
                "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
                "transport_group_mode": str(args.transport_group_mode),
                "transport_group_kind": str(args.transport_group_kind),
                "transport_group_namespace": str(args.transport_group_namespace),
                "transport_group_total_parts": int(args.transport_group_total_parts),
                "transport_group_receiver_index": int(
                    args.transport_group_receiver_index
                ),
                "transport_group_priority": int(args.transport_group_priority),
                "transport_group_epoch": int(args.transport_group_epoch),
                "pid": os.getpid(),
                "ready_at_s": time.time(),
            },
        )
        events = receiver.receive_versions(
            start_version=int(args.start_version),
            num_versions=int(args.num_versions),
        )
        summary = {
            "mode": "receiver",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "materialize_device": _materialization_device(str(args.materialize_device)),
            "receiver_apply_mode": str(args.receiver_apply_mode),
            "payload_mode": str(args.payload_mode),
            "tp_world_size": int(args.tp_world_size),
            "tp_total_bytes": int(args.tp_total_bytes),
            "tp_device_base_index": int(args.tp_device_base_index),
            "tp_device_map_policy": str(args.tp_device_map_policy),
            "tp_device_map_effective_mode": tp_device_map_info.get(
                "effective_mode", "inactive"
            ),
            "tp_visible_device_count": int(
                tp_device_map_info.get("visible_device_count", 0)
            ),
            "tp_rank_device_map": tp_device_map_info.get("rank_device_map", {}),
            "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
            "transport_group_mode": str(args.transport_group_mode),
            "transport_group_kind": str(args.transport_group_kind),
            "transport_group_namespace": str(args.transport_group_namespace),
            "transport_group_total_parts": int(args.transport_group_total_parts),
            "transport_group_receiver_index": int(args.transport_group_receiver_index),
            "transport_group_priority": int(args.transport_group_priority),
            "transport_group_epoch": int(args.transport_group_epoch),
            "received": _to_jsonable(events),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else None
        )
        _write_summary(output, summary)
        _maybe_hold_after_finish(args, mode="receiver")
        return 0
    finally:
        if receiver is not None:
            receiver.close()
        tc.shutdown()


def _run_single_host(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    receiver: WeightUpdateReceiver | None = None
    try:
        run_root = _resolve_run_root(args)
        model_name = _resolve_model_name(args, run_root=run_root)
        publisher = _build_publisher_runner(
            args, run_root=run_root, model_name=model_name
        )
        receiver = _build_receiver_runner(args, model_name=model_name)
        tp_device_map_info = receiver.tp_device_map_info()
        _write_ready_file(
            str(args.ready_file) if args.ready_file else None,
            {
                "mode": "single-host",
                "model_name": model_name,
                "receiver_apply_mode": str(args.receiver_apply_mode),
                "payload_mode": str(args.payload_mode),
                "tp_world_size": int(args.tp_world_size),
                "tp_total_bytes": int(args.tp_total_bytes),
                "tp_device_base_index": int(args.tp_device_base_index),
                "tp_device_map_policy": str(args.tp_device_map_policy),
                "tp_device_map_effective_mode": tp_device_map_info.get(
                    "effective_mode", "inactive"
                ),
                "tp_visible_device_count": int(
                    tp_device_map_info.get("visible_device_count", 0)
                ),
                "tp_rank_device_map": tp_device_map_info.get("rank_device_map", {}),
                "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
                "pid": os.getpid(),
                "ready_at_s": time.time(),
            },
        )

        receiver_holder: dict[str, Any] = {"events": None, "error": None}
        ack_condition = threading.Condition()
        acked_version = int(args.start_version) - 1

        def _on_receive_event(event: ReceiveEvent) -> None:
            nonlocal acked_version
            with ack_condition:
                acked_version = max(acked_version, event.version)
                ack_condition.notify_all()

        def _wait_for_receiver_ack(version: int, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            with ack_condition:
                while acked_version < version:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"receiver did not ack version={version} within {timeout_s}s"
                        )
                    ack_condition.wait(timeout=min(1.0, remaining))

        def _receiver_worker() -> None:
            try:
                receiver_holder["events"] = receiver.receive_versions(
                    start_version=int(args.start_version),
                    num_versions=int(args.num_versions),
                    on_event=_on_receive_event,
                )
            except Exception as exc:  # noqa: BLE001
                receiver_holder["error"] = exc
            finally:
                receiver.close()

        worker = threading.Thread(
            target=_receiver_worker,
            name="weight-publisher-e2e-receiver",
            daemon=True,
        )
        worker.start()
        time.sleep(max(0.0, float(args.receiver_start_delay_s)))

        published: list[PublishEvent] = []
        start_version = int(args.start_version)
        num_versions = int(args.num_versions)
        for offset in range(num_versions):
            version = start_version + offset
            publisher.publish_one_version(version=version, events=published)
            _wait_for_receiver_ack(
                version=version,
                timeout_s=float(args.receiver_timeout_s),
            )
            if offset + 1 < num_versions:
                time.sleep(max(0.0, float(args.publish_interval_s)))

        join_timeout = max(
            5.0,
            float(args.receiver_timeout_s) * float(args.num_versions) + 10.0,
        )
        worker.join(timeout=join_timeout)
        if worker.is_alive():
            raise TimeoutError("receiver worker did not finish in expected time")
        if receiver_holder["error"] is not None:
            raise RuntimeError(
                f"receiver failed: {receiver_holder['error']}"
            ) from receiver_holder["error"]
        received = receiver_holder["events"]
        if received is None:
            raise RuntimeError("receiver produced no result")

        expected_versions = list(
            range(
                int(args.start_version),
                int(args.start_version) + int(args.num_versions),
            )
        )
        actual_versions = [event.version for event in received]
        if expected_versions != actual_versions:
            raise AssertionError(
                f"receiver version sequence mismatch: expected={expected_versions}, actual={actual_versions}"
            )

        summary = {
            "mode": "single-host",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "keep_last": int(args.keep_last),
            "pre_publish_trim_margin": int(args.pre_publish_trim_margin),
            "strict_retention_drop_check": bool(args.strict_retention_drop_check),
            "receiver_apply_mode": str(args.receiver_apply_mode),
            "payload_mode": str(args.payload_mode),
            "tp_world_size": int(args.tp_world_size),
            "tp_total_bytes": int(args.tp_total_bytes),
            "tp_device_base_index": int(args.tp_device_base_index),
            "tp_device_map_policy": str(args.tp_device_map_policy),
            "tp_device_map_effective_mode": tp_device_map_info.get(
                "effective_mode", "inactive"
            ),
            "tp_visible_device_count": int(
                tp_device_map_info.get("visible_device_count", 0)
            ),
            "tp_rank_device_map": tp_device_map_info.get("rank_device_map", {}),
            "tp_materialize_deadline_s": float(args.tp_materialize_deadline_s),
            "publish_device": _publisher_device(str(args.publish_device)),
            "run_root": str(run_root),
            "published": _to_jsonable(published),
            "received": _to_jsonable(received),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else run_root / "single_host_summary.json"
        )
        _write_summary(output, summary)
        _maybe_hold_after_finish(args, mode="single-host")
        return 0
    finally:
        if receiver is not None:
            receiver.close()
        tc.shutdown()


def _add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_init_mode: str,
    default_global_store_mode: str,
) -> None:
    parser.add_argument(
        "--init-mode",
        choices=["connect", "create", "auto"],
        default=default_init_mode,
        help="TensorCast init mode.",
    )
    parser.add_argument(
        "--connect-address",
        default=None,
        help="Daemon address for connect mode, e.g. 127.0.0.1:8073.",
    )
    parser.add_argument(
        "--daemon-config-path",
        default=None,
        help="Daemon config path for create/auto mode.",
    )
    parser.add_argument(
        "--global-store-mode",
        choices=["none", "connect", "start"],
        default=default_global_store_mode,
        help="Global Store orchestration mode for create/auto mode.",
    )
    parser.add_argument(
        "--global-store-address",
        default=None,
        help="Global Store address for global_store_mode=connect.",
    )
    parser.add_argument(
        "--global-store-config-path",
        default=None,
        help="Global Store config path for global_store_mode=start.",
    )
    parser.add_argument(
        "--cluster-id",
        default=None,
        help="Optional cluster token to enforce for Global Store.",
    )
def _add_common_stream_args(
    parser: argparse.ArgumentParser,
    *,
    require_model_name: bool,
) -> None:
    parser.add_argument(
        "--model-name",
        required=require_model_name,
        help="Logical model name used by key_template.",
    )
    parser.add_argument(
        "--key-template",
        default="model:{model_name}:v{weight_version}",
        help="Versioned key template.",
    )
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--num-versions", type=int, default=3)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--receiver-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--payload-mode",
        choices=sorted(SUPPORTED_PAYLOAD_MODES),
        default="probe",
        help=(
            "Payload generator/validator mode. "
            "'version_fill' enforces all tensor elements equal to version. "
            "'tp_ranked' publishes TP-sharded source tensors with rank-tagged values."
        ),
    )
    parser.add_argument(
        "--materialize-device",
        default="auto",
        help="Receiver materialization device: auto/cpu/cuda:0...",
    )
    parser.add_argument(
        "--receiver-apply-mode",
        choices=sorted(SUPPORTED_RECEIVER_APPLY_MODES),
        default="tensor_dict",
        help=(
            "Receiver apply path: tensor_dict materialization, or "
            "Binding-based in-place update via binding.swap. "
            "'tp_bind_into_swap' runs TP rank-local bind_into/swap updates with copy plans. "
            "'tp4_bind_into_swap' is kept as a strict tp_world_size=4 alias."
        ),
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        default=1,
        help="Tensor-parallel world size for tp_ranked/tp_bind_into_swap modes.",
    )
    parser.add_argument(
        "--tp-total-bytes",
        type=int,
        default=0,
        help=(
            "Total tp_ranked payload bytes across all published tensors. "
            "0 keeps the default tiny TP payload."
        ),
    )
    parser.add_argument(
        "--tp-device-base-index",
        type=int,
        default=0,
        help="CUDA device base index for TP ranks (rank i uses cuda:{base+i}).",
    )
    parser.add_argument(
        "--tp-device-map-policy",
        choices=sorted(SUPPORTED_TP_DEVICE_MAP_POLICIES),
        default="auto",
        help=(
            "TP rank-to-device mapping policy. "
            "'auto' uses contiguous map when possible, else falls back to modulo reuse. "
            "'strict' requires contiguous map. "
            "'modulo' always reuses visible devices with modulo."
        ),
    )
    parser.add_argument(
        "--tp-materialize-deadline-s",
        type=float,
        default=600.0,
        help=(
            "RPC deadline for TP bind_into/swap path in seconds. "
            "Use larger values for large multi-node payloads."
        ),
    )
    parser.add_argument(
        "--transport-group-mode",
        choices=sorted(SUPPORTED_TRANSPORT_GROUP_MODES),
        default="none",
        help=(
            "Transport scheduling-group hint mode for TP receive path. "
            "'tp_version' tags each TP rank request into one per-version group."
        ),
    )
    parser.add_argument(
        "--transport-group-kind",
        default="tp_version",
        help="Group kind label used when --transport-group-mode=tp_version.",
    )
    parser.add_argument(
        "--transport-group-namespace",
        default="",
        help=(
            "Group id namespace prefix used when --transport-group-mode=tp_version. "
            "Should be unique per benchmark run."
        ),
    )
    parser.add_argument(
        "--transport-group-total-parts",
        type=int,
        default=0,
        help=("Expected total parts per group when --transport-group-mode=tp_version."),
    )
    parser.add_argument(
        "--transport-group-receiver-index",
        type=int,
        default=0,
        help="Receiver index encoded into group part id when group mode is enabled.",
    )
    parser.add_argument(
        "--transport-group-priority",
        type=int,
        default=0,
        help="Group priority hint.",
    )
    parser.add_argument(
        "--transport-group-epoch",
        type=int,
        default=0,
        help="Group epoch hint.",
    )
    parser.add_argument(
        "--publish-device",
        default="auto",
        help="Publisher tensor device: auto/cpu/cuda:<index>.",
    )
    parser.add_argument(
        "--allow-version-skip",
        action="store_true",
        help=(
            "Allow receiver to skip a version if that version is already deregistered "
            "before it can be applied."
        ),
    )
    parser.add_argument(
        "--hold-after-finish-s",
        type=float,
        default=0.0,
        help="Keep runtime alive for external probes before shutdown.",
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="Optional readiness marker path written after role init.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output summary path.",
    )


def _add_publisher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-interval-s", type=float, default=2.0)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument(
        "--pre-publish-trim-margin",
        type=int,
        default=1,
        help=(
            "Pre-publish retention trim margin. "
            "effective_pre_publish_keep = max(0, keep_last - margin)."
        ),
    )
    parser.add_argument("--retention-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--strict-retention-drop-check",
        action="store_true",
        help=(
            "When set, assert dropped versions are immediately non-materializable "
            "and absent from registry. For distributed runs with lagging receivers, "
            "leave this disabled and validate via external cluster probes."
        ),
    )
    parser.add_argument(
        "--weights-root",
        default="/tmp/tensorcast_weight_publisher_e2e",
        help="Root directory for generated version exports.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier used under weights-root.",
    )
    parser.add_argument(
        "--history-path",
        default=None,
        help="Optional explicit publisher history path.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeightPublisher E2E harness for single-host and distributed tests.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser(
        "single-host",
        help="Run publisher and receiver concurrently on one node.",
    )
    _add_runtime_args(
        single,
        default_init_mode="create",
        default_global_store_mode="start",
    )
    _add_common_stream_args(single, require_model_name=False)
    _add_publisher_args(single)
    single.add_argument(
        "--receiver-start-delay-s",
        type=float,
        default=1.0,
        help="Delay after receiver thread starts before publishing begins.",
    )

    pub = subparsers.add_parser(
        "publisher",
        help="Run publisher role only (for distributed test).",
    )
    _add_runtime_args(
        pub,
        default_init_mode="connect",
        default_global_store_mode="none",
    )
    _add_common_stream_args(pub, require_model_name=False)
    _add_publisher_args(pub)

    recv = subparsers.add_parser(
        "receiver",
        help="Run receiver role only (for distributed test).",
    )
    _add_runtime_args(
        recv,
        default_init_mode="connect",
        default_global_store_mode="none",
    )
    _add_common_stream_args(recv, require_model_name=True)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single-host":
        return _run_single_host(args)
    if args.mode == "publisher":
        return _run_publisher(args)
    if args.mode == "receiver":
        return _run_receiver(args)
    parser.error(f"unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
