#  Copyright (c) 2025-2026, TensorCast Team.

"""Worker domain artifact."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class WorkerMemoryTierState:
    """Latest memory tier snapshot materialized for a worker."""

    stable_total_bytes: int = 0
    stable_used_bytes: int = 0
    preemptible_total_bytes: int = 0
    preemptible_marked_bytes: int = 0
    faults_per_sec: float = 0.0
    rehydrate_p99_ns: int = 0
    enable_preemptible: bool = False
    memory_tier_config_json: str = "{}"
    snapshot_epoch_ns: int = 0


@dataclass
class Worker:
    """Represents a Store Daemon worker node in the cluster."""

    # Identity
    worker_id: str = ""
    daemon_id: str = ""
    node_id: str = ""

    # Network
    node_address: str = ""
    grpc_port: int = 0
    p2p_port: int = 0

    # Resources
    mem_pool_total_size: int = 0
    mem_pool_available_size: int = 0

    # Status
    accepting_new_requests: bool = True

    # Capabilities (bitset)
    capability_flags: int = 0

    # Timestamps
    registered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    inactive_at: Optional[datetime] = None

    # HA state tracking
    state_version: int = 1
    state_checksum: str = ""

    # Latest memory tier state (optional)
    memory_tier_state: Optional[WorkerMemoryTierState] = None

    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage."""
        if self.mem_pool_total_size == 0:
            return 0.0
        used = self.mem_pool_total_size - self.mem_pool_available_size
        return (used / self.mem_pool_total_size) * 100

    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy based on heartbeat."""
        if not self.last_heartbeat:
            return False
        # Consider healthy if heartbeat within last 30 seconds
        age = datetime.now() - self.last_heartbeat
        return age.total_seconds() < 30
