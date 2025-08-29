#  Copyright (c) 2025, TensorCast Team.

import asyncio
from collections import defaultdict
from typing import Any


class FakeP2PNetwork:
    """Very simple in-process "network" for pushing/pulling artifact bytes.

    Daemons register replicas via `register_replica` and peers request the data
    through `request`.  The network delivers a copy of the stored bytes.  This
    can be extended with delays / failures for fault-injection but keeps the
    API surface minimal for now.
    """

    def __init__(self):
        # maps replica_id -> bytes (artifact payload)
        self._storage: dict[str, bytes] = {}
        # simulate network delay/failure via hooks
        self._delays: dict[str, float] = defaultdict(lambda: 0.0)
        self._failures: set[str] = set()

    async def register_replica(self, replica_id: str, data: bytes):
        self._storage[replica_id] = data

    async def inject_failure(self, replica_id: str):
        """Mark the replica so that subsequent requests raise IOError."""
        self._failures.add(replica_id)

    async def set_delay(self, replica_id: str, delay_s: float):
        self._delays[replica_id] = delay_s

    async def request(self, replica_id: str) -> bytes:
        if replica_id not in self._storage:
            raise KeyError(f"Replica {replica_id} not found on network")

        if replica_id in self._failures:
            raise IOError("Injected transfer failure")

        delay = self._delays[replica_id]
        if delay > 0:
            await asyncio.sleep(delay)
        # Return a copy to avoid accidental shared mutability
        return bytes(self._storage[replica_id])