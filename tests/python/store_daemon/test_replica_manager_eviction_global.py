#  Copyright (c) 2025, StepCast Team.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scstore.store_daemon.replica_manager import ReplicaManager
from scstore.store_daemon.replica_ref import ReplicaKey


class MockServicer:
    """Minimal mock servicer object for ReplicaManager tests."""

    def __init__(self):
        self.checkpoint_store = Mock()
        # Default: one GPU with 10 GB, 5 GB free
        self.checkpoint_store.get_gpu_memory_stats.return_value = [
            (10 * 1024**3, 5 * 1024**3)
        ]
        self.global_store_stub = Mock()
        self.enable_p2p_engine = True
        self.global_store_enabled = True
        self.node_id = "test-node"
        self.node_address = "127.0.0.1"
        self.node_port = 9090
        self.worker_id = "test-worker"
        # Config will be injected per-test when required
        self.config = None


@pytest.mark.parametrize("fraction,expected_min_evict", [(0.1, 2), (0.5, 0)])
def test_global_cache_fraction_eviction(fraction: float, expected_min_evict: int):
    """Verify that ReplicaManager evicts global cache replicas when the usage
    exceeds `global_cache_fraction`.  When the limit is larger than current
    usage the eviction list should be empty (or satisfy bytes_needed only).
    """
    servicer = MockServicer()
    # Inject config with the given fraction
    servicer.config = SimpleNamespace(
        lifecycle=SimpleNamespace(global_cache_fraction=fraction)
    )

    # Create ReplicaManager with three 1 GB global cache replicas on device 0.
    manager = ReplicaManager(servicer)
    for i in range(3):
        manager.add_ref(
            model_path=f"model{i}",
            device_id=0,
            pid=1000 + i,
            size_bytes=1 * 1024**3,
            keep_for_global=True,
        )
        # Remove ref to make it evictable
        manager.remove_ref(f"model{i}", 0, 1000 + i)
        # Ensure deterministic ordering by making accesses equally old
        manager._replicas[ReplicaKey(f"model{i}", 0)].last_access_ts -= i + 1

    # bytes_needed=0 to test pure limit behaviour
    candidates = manager._select_eviction_candidates(bytes_needed=0, device_id=0)

    # When fraction is small (0.1), expect eviction of >=2 replicas to drop
    # global_bytes from 3 GB to <=1 GB.
    assert len(candidates) >= expected_min_evict

    # All selected replicas must be from the global cache (keep_for_global=True)
    assert all(rep.keep_for_global for rep in candidates)

    # Total bytes in candidates should satisfy requirement when over limit
    if expected_min_evict:
        freed_bytes = sum(rep.size_bytes for rep in candidates)
        assert freed_bytes >= 2 * 1024**3  # at least 2 GB freed

    # bytes_needed=0 to test pure limit behaviour
    candidates = manager._select_eviction_candidates(bytes_needed=0, device_id=0)

    # When fraction is small (0.1), expect eviction of >=2 replicas to drop
    # global_bytes from 3 GB to <=1 GB.
    assert len(candidates) >= expected_min_evict

    # All selected replicas must be from the global cache (keep_for_global=True)
    assert all(rep.keep_for_global for rep in candidates)

    # Total bytes in candidates should satisfy requirement when over limit
    if expected_min_evict:
        freed_bytes = sum(rep.size_bytes for rep in candidates)
        assert freed_bytes >= 2 * 1024**3  # at least 2 GB freed
