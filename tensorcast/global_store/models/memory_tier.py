#  Copyright (c) 2025, TensorCast Team.

"""Memory tier telemetry and lease domain models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkRange:
    """Human-friendly chunk span for diagnostics."""

    start: int = 0
    count: int = 0


@dataclass
class MemoryTierSnapshot:
    """Represents a single telemetry sample for a node."""

    node_id: str
    epoch_ns: int
    stable_total_bytes: int
    stable_used_bytes: int
    preemptible_total_bytes: int
    preemptible_marked_bytes: int
    faults_per_sec: float
    rehydrate_p99_ns: int
    enable_preemptible: bool
    memory_tier_config_json: str = "{}"


@dataclass
class MemoryTierLease:
    """Stable/preemptible lease persisted for replay/audit."""

    lease_id: str
    node_id: str
    kind: str
    artifact_id: str
    chunk_range: ChunkRange
    chunk_ids: list[int]
    ledger_version: int
    bytes: int
    workload_id: str
    state: str
    request_id: str
    issued_at_ns: int
    ack_epoch_ns: Optional[int] = None
    expires_at_ns: Optional[int] = None
