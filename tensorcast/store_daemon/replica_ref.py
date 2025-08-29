#  Copyright (c) 2025, TensorCast Team.

"""Replica reference tracking data structures for lifecycle management."""

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReplicaKey:
    """Immutable key for identifying a unique replica replica.

    A replica is uniquely identified by the combination of artifact_id,
    and device_id.
    """

    artifact_id: str
    device_id: int

    def __str__(self) -> str:
        return f"{self.artifact_id}@device_{self.device_id}"


@dataclass
class ReplicaRefInfo:
    """Mutable information about a replica replica's references and state.

    Tracks which processes (PIDs) are using the replica, reference counts,
    and metadata for eviction decisions.
    """

    key: ReplicaKey
    size_bytes: int
    keep_for_global: bool = False
    last_access_ts: float = field(default_factory=time.time)
    pids: set[int] = field(default_factory=set)
    ref_count: int = 0
    # Identifier assigned by Global Store when the replica is registered.
    # Needed to later unregister the replica during eviction/unload.
    replica_id: str | None = None

    def add_pid(self, pid: int) -> bool:
        """Add a PID reference. Returns True if newly added."""
        if pid not in self.pids:
            self.pids.add(pid)
            self.ref_count += 1
            self.last_access_ts = time.time()
            return True
        return False

    def remove_pid(self, pid: int) -> bool:
        """Remove a PID reference. Returns True if removed."""
        if pid in self.pids:
            self.pids.discard(pid)
            self.ref_count = max(0, self.ref_count - 1)
            return True
        return False

    def is_evictable(self) -> bool:
        """Check if this replica can be evicted."""
        return self.ref_count == 0

    def touch(self) -> None:
        """Update last access timestamp."""
        self.last_access_ts = time.time()

    def __str__(self) -> str:
        return (
            f"ReplicaRefInfo(key={self.key}, refs={self.ref_count}, "
            f"pids={self.pids}, keep_global={self.keep_for_global})"
        )
