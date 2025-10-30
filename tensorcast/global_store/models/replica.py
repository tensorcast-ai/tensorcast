#  Copyright (c) 2025, TensorCast Team.

"""Artifact replica domain artifact."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List
from uuid import UUID, uuid4


class MemoryType(Enum):
    """Memory type enum matching proto definition."""

    GPU = "GPU"
    RAM = "RAM"
    DISK = "DISK"

    @property
    def priority(self) -> int:
        """Return priority value (lower = higher priority)."""
        priority_map = {
            MemoryType.GPU: 1,  # Highest priority
            MemoryType.RAM: 2,  # Medium priority
            MemoryType.DISK: 3,  # Lowest priority
        }
        return priority_map[self]


@dataclass
class Replica:
    """Represents a artifact replica in the distributed storage system."""

    # Identity
    replica_id: UUID = field(default_factory=uuid4)
    artifact_id: str = ""  # content-addressed artifact ID (mi2:...)
    disk_path: str | None = None  # physical path for disk-based replicas (optional)

    # Location
    node_id: str = ""
    node_address: str = ""
    node_port: int = 0

    # Storage details
    memory_size: int = 0
    memory_type: MemoryType = MemoryType.DISK
    device_id: int = 0

    # Concurrency control
    max_concurrency: int = 5
    current_requests: int = 0

    # Availability
    is_available: bool = True

    # RDMA/Transport info
    remote_memory_keys: List[str] = field(default_factory=list)
    buffer_sizes: List[int] = field(default_factory=list)
    # Optional verification metadata in JSON form (e.g., KEY_POINTS/SEGMENT_HASHES)
    verification_json: str | None = None

    # Worker association
    worker_id: str | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def load_ratio(self) -> float:
        """Calculate the current load ratio."""
        if self.max_concurrency == 0:
            # Special case: if both max_concurrency and current_requests are 0,
            # return 0.0 (no load). Otherwise, return 1.0 (maximum load).
            return 0.0 if self.current_requests == 0 else 1.0
        return self.current_requests / self.max_concurrency

    @property
    def has_capacity(self) -> bool:
        """Check if replica can accept more requests."""
        return self.current_requests < self.max_concurrency

    def increment_requests(self) -> bool:
        """Atomically increment request count if capacity available."""
        if self.has_capacity:
            self.current_requests += 1
            return True
        return False

    def decrement_requests(self) -> None:
        """Atomically decrement request count."""
        self.current_requests = max(0, self.current_requests - 1)
