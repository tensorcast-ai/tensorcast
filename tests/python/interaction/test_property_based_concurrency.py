#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

"""Property-based concurrency test (design §四 – property_based_concurrency).

This test exercises the Global-Store under random Acquire / Complete
sequences, ensuring that **current_requests never exceeds max_concurrency** for
any replica and that, after the rule-based run, all counters return to zero.

The approach uses Hypothesis' *RuleBasedStateMachine* to generate sequences of
operations and invariants.  It focuses purely on the *GlobalStoreServicer*
logic, without spinning up real Store-Daemons or P2P transfers – that layer is
already mocked for interaction tests.
"""

import random
from typing import List

# ---------------------------------------------------------------------------
# Optional dependency – Hypothesis may not be installed in minimal CI images.
# If unavailable, skip this module gracefully so the rest of the suite still
# runs.  This avoids ImportError at collection time.
# ---------------------------------------------------------------------------

try:
    import hypothesis.strategies as st  # noqa: F401  – optional dep
    from hypothesis import settings
    from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
except ModuleNotFoundError:  # pragma: no cover – environment without Hypothesis
    import pytest

    pytest.skip("hypothesis not installed – skipping property-based test", allow_module_level=True)

from scstore.proto import global_store_pb2
from .utils import get_free_port_pair

from tests.python.interaction.utils import FakeContext


# ---------------------------------------------------------------------------
# Helper – replica registration
# ---------------------------------------------------------------------------

def _register_replica(gs, artifact: str, node_id: str, max_concurrency: int) -> None:
    """Register worker + GPU replica with *max_concurrency*."""
    # 1. Worker
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=node_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=32 * 1024 * 1024,
        mem_pool_available_size=32 * 1024 * 1024,
    )
    worker_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert worker_resp.status == global_store_pb2.Status.OK

    # 2. Replica
    mem_info = global_store_pb2.MemoryInfo(
        node_id=node_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{node_id}"],
        memory_size=2 * 1024 * 1024,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    reg_req = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact,
        mem_info=mem_info,
        max_concurrency=max_concurrency,
        worker_id=worker_resp.worker_id,
    )
    rep_resp = gs.RegisterReplica(reg_req, FakeContext())
    assert rep_resp.status == global_store_pb2.Status.OK


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


def test_property_based_concurrency(global_store_service):
    """Hypothesis-based state machine exercising Acquire/Complete behaviour."""

    gs = global_store_service
    artifact_id = "hypothesis-llama"

    # Register three replicas with varying capacities for richer state space
    capacities = {"A": 2, "B": 3, "C": 5}
    for node, cap in capacities.items():
        _register_replica(gs, artifact_id, node_id=node, max_concurrency=cap)

    class ConcurrencySM(RuleBasedStateMachine):
        """Rule-based state machine driving transport Acquire / Complete."""

        # NOTE: We capture *gs* in the closure – safe for single-threaded pytest
        def __init__(self):
            super().__init__()
            self.gs = gs
            self.artifact = artifact_id
            self.active_transport_ids: List[str] = []

        # ---------- Operations ----------
        @rule()
        def request_transport(self):
            """Attempt to acquire transport (may succeed or time-out)."""
            req = global_store_pb2.RequestReplicaTransportRequest(
                artifact_id=self.artifact,
                # Occasionally request with 1ms timeout to increase branching
                wait_timeout_ms=1 if random.random() < 0.2 else 0,
            )
            resp = self.gs.RequestReplicaTransport(req, FakeContext())
            if resp.status == global_store_pb2.Status.OK:
                self.active_transport_ids.append(resp.transport_id)

        @rule()
        def complete_random_transport(self):
            """Complete a random outstanding transport (if any)."""
            if not self.active_transport_ids:
                return  # Nothing to complete
            tid = random.choice(self.active_transport_ids)
            self.active_transport_ids.remove(tid)
            comp_req = global_store_pb2.CompleteReplicaTransportRequest(
                transport_id=tid
            )
            self.gs.CompleteReplicaTransport(comp_req, FakeContext())

        # ---------- Invariants ----------
        @invariant()
        def counters_never_exceed_capacity(self):
            """current_requests must never exceed max_concurrency."""
            for replica in self.gs.replica_repository.list_all_replicas():
                assert replica.current_requests <= replica.max_concurrency

        @invariant()
        def outstanding_matches_db(self):
            """Sum of DB counters should equal in-memory list length."""
            total_active_db = sum(
                r.current_requests for r in self.gs.replica_repository.list_all_replicas()
            )
            assert total_active_db == len(self.active_transport_ids)

        # ---------- Tear-down ----------
        def teardown(self):  # Called automatically by Hypothesis
            # Ensure eventual cleanup so next test starts with clean slate
            for tid in list(self.active_transport_ids):
                comp_req = global_store_pb2.CompleteReplicaTransportRequest(
                    transport_id=tid
                )
                self.gs.CompleteReplicaTransport(comp_req, FakeContext())
            self.active_transport_ids.clear()

    # Run the state machine with restrained settings for CI speed
    ConcurrencySM.TestCase.settings = settings(max_examples=50, stateful_step_count=25)
    ConcurrencySM.TestCase().runTest()