---
slug: control-plane-tree-broadcast-phase2
title: Control-Plane Tree Broadcast Phase 2 Implementation Plan
links:
  design: ../designs/0117-control-plane-tree-broadcast-phase2.md
---

# Control-Plane Tree Broadcast Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build end-to-end Global Store coordinated tree broadcast for model-weight materialization while reusing existing P2P data movement.

**Architecture:** Add durable broadcast session/target/edge state in Global Store, route broadcast-tagged `RequestReplicaTransport` calls through edge-assigned parents, and propagate broadcast hints from SDK through Store Daemon and C++ `MaterializeHints`. Broadcast success advances only after P2P materialization and local replica registration/export have completed.

**Tech Stack:** DuckDB schema, Python Global Store repositories/services/RPC handlers, protobuf/buf, Python SDK, C++ Store Daemon controllers, C++ StoreEngine materialization, pytest, Bazel/Catch2.

---

# Current State & Grounding

- Branch: `runze/broadcast-weight`.
- Design: `docs/designs/0117-control-plane-tree-broadcast-phase2.md`.
- Phase 1 design and implementation are already present in `docs/designs/0116-control-plane-coordinated-weight-broadcast.md`.
- `RequestReplicaTransportRequest` already has `request_id`, `requester_worker_id`, and `scheduling_group`.
- `artifact_transports` and `pending_transport_requests` already track transport request ids, requester worker ids, scheduling groups, and completion outcomes.
- `ReplicaRepository.find_available_for_transport()` already filters source replicas by export metadata, worker liveness, `accepting_new_requests`, capacity, and memory tier ordering.
- `MaterializeHints` already carries `transport_request_id`, `transport_requester_worker_id`, and `transport_scheduling_group`.
- Current gap: there is no broadcast session/edge state, no strict parent assignment in transport selection, and no SDK/daemon/core broadcast hint.
- Existing dirty worktree before this plan includes generated proto artifacts and `pyproject.toml`; implementation must not revert or stage unrelated pre-existing changes.

# File Structure

- Create: `tensorcast/global_store/models/broadcast.py` for session/target/edge dataclasses and state enums.
- Create: `tensorcast/global_store/repositories/broadcast_repository.py` for DuckDB CRUD and atomic edge state transitions.
- Create: `tensorcast/global_store/services/broadcast_service.py` for session creation, planning, transport-edge claim, completion advancement, retry, and cancellation.
- Create: `tensorcast/global_store/rpc/broadcast_rpc_handler.py` for Global Store broadcast RPC validation and protobuf mapping.
- Modify: `schema.sql` and add migration `tensorcast/global_store/migrations/0019_broadcast_sessions.py`.
- Modify: `tensorcast/global_store/models/__init__.py`, `repositories/__init__.py`, `services/__init__.py`, `grpc_service.py`, and `rpc_servicer_mixins.py` to wire the new domain.
- Modify: `tensorcast/global_store/repositories/transport_repository.py` to persist `broadcast_session_id` and `broadcast_edge_id`.
- Modify: `tensorcast/global_store/repositories/replica_repository.py` to claim a specific parent replica and to find a child replica by worker after registration.
- Modify: `tensorcast/global_store/services/transport_service.py` to delegate broadcast transport claims and completions to `BroadcastService`.
- Modify: `tensorcast/global_store/rpc/transport_rpc_handler.py` to parse `BroadcastTransportHint`.
- Modify: `proto/tensorcast/global_store/v1/global_store.proto` and `proto/tensorcast/daemon/v2/store_daemon.proto`.
- Modify: `tensorcast/api/context.py`, `tensorcast/api/__init__.py`, `tensorcast/__init__.py`, `tensorcast/api/store/artifact.py`, `tensorcast/api/_materialize.py`, and `tensorcast/daemon_ctl.py` for SDK/daemon hint propagation.
- Modify: `daemon/service/controllers/materialization_policy_utils.{h,cc}`, `daemon/service/controllers/replica_materialization_service.cc`, and daemon RPC controller files for broadcast session forwarding.
- Modify: `core/store/materialization/contracts/loading_spec.h`, `core/store/components/global_store_client.{h,cc}`, `core/store/testing/recording_global_store_client.h`, `core/store/materialization/control/materialize_orchestrator.cc`, and `core/store/runtime/ingestion/materialization_facade.cc`.
- Test: add Global Store repository/service/RPC tests under `tests/python/global_store/`.
- Test: update SDK tests under `tests/python/api/`.
- Test: update C++ daemon and core tests under existing Bazel targets.

# Phases & Milestones

- [ ] Phase 1: Add Global Store broadcast schema, domain models, and repository tests.
- [ ] Phase 2: Add Global Store broadcast service/RPC and first-layer tree planning.
- [ ] Phase 3: Route broadcast-tagged transport requests through assigned parent edges.
- [ ] Phase 4: Propagate broadcast hints through SDK, daemon proto, daemon controller, and C++ client.
- [ ] Phase 5: Enforce broadcast completion semantics after child replica registration.
- [ ] Phase 6: Add integration coverage, docs cross-links, and final verification.

### Task 1: Broadcast Schema, Models, And Repository

**Files:**
- Create: `tensorcast/global_store/models/broadcast.py`
- Create: `tensorcast/global_store/repositories/broadcast_repository.py`
- Modify: `tensorcast/global_store/models/__init__.py`
- Modify: `tensorcast/global_store/repositories/__init__.py`
- Modify: `schema.sql`
- Create: `tensorcast/global_store/migrations/0019_broadcast_sessions.py`
- Modify: `tests/python/global_store/conftest.py`
- Test: add `tests/python/global_store/test_broadcast_repository.py`

- [ ] **Step 1: Write failing repository tests**

Create `tests/python/global_store/test_broadcast_repository.py`:

```python
from __future__ import annotations

from uuid import UUID

from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
)
from tensorcast.global_store.repositories import BroadcastRepository


def test_broadcast_repository_creates_session_targets_and_edges(db_connection):
    repo = BroadcastRepository(db_connection)
    session = BroadcastSession(
        session_id="session-a",
        artifact_id="mi2:test",
        requested_view_id=None,
        epoch=42,
        fanout=2,
        max_attempts=3,
        strict_parent=True,
        state=BroadcastSessionState.ACTIVE,
        root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    repo.create_session(session)
    repo.upsert_target(
        BroadcastTarget(
            session_id="session-a",
            target_worker_id="worker-child-1",
            target_daemon_id="daemon-child-1",
            state=BroadcastTargetState.PENDING,
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-1",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child-1",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.PLANNED,
        )
    )

    loaded = repo.find_session("session-a")
    assert loaded is not None
    assert loaded.artifact_id == "mi2:test"
    assert loaded.epoch == 42
    assert loaded.state is BroadcastSessionState.ACTIVE

    target = repo.find_target("session-a", "worker-child-1")
    assert target is not None
    assert target.target_daemon_id == "daemon-child-1"
    assert target.state is BroadcastTargetState.PENDING

    edge = repo.find_active_edge_for_child("session-a", "worker-child-1")
    assert edge is not None
    assert edge.parent_worker_id == "worker-root"
    assert edge.state is BroadcastEdgeState.PLANNED


def test_broadcast_repository_prevents_two_active_edges_for_child(db_connection):
    repo = BroadcastRepository(db_connection)
    repo.create_session(
        BroadcastSession(
            session_id="session-a",
            artifact_id="mi2:test",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            max_attempts=3,
            strict_parent=True,
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    )
    first = BroadcastEdge(
        edge_id="edge-1",
        session_id="session-a",
        parent_worker_id="worker-root",
        parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        child_worker_id="worker-child",
        level=1,
        attempt=1,
        state=BroadcastEdgeState.PLANNED,
    )
    repo.create_edge(first)

    try:
        repo.create_edge(
            BroadcastEdge(
                edge_id="edge-2",
                session_id="session-a",
                parent_worker_id="worker-root",
                parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
                child_worker_id="worker-child",
                level=1,
                attempt=2,
                state=BroadcastEdgeState.ASSIGNED,
            )
        )
    except Exception as exc:  # noqa: BLE001
        assert "active" in str(exc).lower() or "constraint" in str(exc).lower()
    else:
        raise AssertionError("expected active edge uniqueness to reject duplicate child")


def test_broadcast_repository_marks_edge_completed_and_target_completed(db_connection):
    repo = BroadcastRepository(db_connection)
    repo.create_session(
        BroadcastSession(
            session_id="session-a",
            artifact_id="mi2:test",
            requested_view_id=None,
            epoch=1,
            fanout=1,
            max_attempts=3,
            strict_parent=True,
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
        )
    )
    repo.upsert_target(
        BroadcastTarget(
            session_id="session-a",
            target_worker_id="worker-child",
            target_daemon_id="daemon-child",
            state=BroadcastTargetState.MATERIALIZING,
            assigned_edge_id="edge-1",
        )
    )
    repo.create_edge(
        BroadcastEdge(
            edge_id="edge-1",
            session_id="session-a",
            parent_worker_id="worker-root",
            parent_replica_id=UUID("00000000-0000-0000-0000-000000000001"),
            child_worker_id="worker-child",
            level=1,
            attempt=1,
            state=BroadcastEdgeState.MATERIALIZING,
            transport_request_id="transport-request-1",
        )
    )

    completed_replica_id = UUID("00000000-0000-0000-0000-000000000002")
    assert repo.mark_edge_completed(
        edge_id="edge-1",
        completed_replica_id=completed_replica_id,
    )
    edge = repo.find_edge("edge-1")
    target = repo.find_target("session-a", "worker-child")
    assert edge is not None
    assert target is not None
    assert edge.state is BroadcastEdgeState.COMPLETED
    assert target.state is BroadcastTargetState.COMPLETED
    assert target.completed_replica_id == completed_replica_id
```

- [ ] **Step 2: Run repository tests and verify they fail**

Run:

```bash
pytest tests/python/global_store/test_broadcast_repository.py -v
```

Expected: FAIL with import errors for `BroadcastRepository` and broadcast model classes.

- [ ] **Step 3: Add schema and migration**

In `schema.sql`, add the `broadcast_sessions`, `broadcast_targets`, and `broadcast_edges` tables from [docs/designs/0117-control-plane-tree-broadcast-phase2.md](/data/tot/tensorcast/docs/designs/0117-control-plane-tree-broadcast-phase2.md), plus the `artifact_transports.broadcast_session_id` and `artifact_transports.broadcast_edge_id` columns.

Create `tensorcast/global_store/migrations/0019_broadcast_sessions.py`:

```python
#  Copyright (c) 2026, TensorCast Team.

"""Migration 0019: add broadcast session state."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

UP_QUERIES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS broadcast_sessions (
        session_id TEXT PRIMARY KEY,
        artifact_id TEXT NOT NULL,
        requested_view_id TEXT NULL,
        epoch BIGINT NOT NULL,
        fanout INTEGER NOT NULL,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        strict_parent BOOLEAN NOT NULL DEFAULT TRUE,
        state TEXT CHECK (state IN ('planning','active','completed','failed','cancelled')) NOT NULL,
        root_replica_id UUID NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broadcast_targets (
        session_id TEXT NOT NULL,
        target_worker_id TEXT NOT NULL,
        target_daemon_id TEXT NULL,
        state TEXT CHECK (state IN ('pending','assigned','materializing','completed','failed','cancelled')) NOT NULL,
        level INTEGER NULL,
        attempt INTEGER NOT NULL DEFAULT 0,
        assigned_edge_id TEXT NULL,
        completed_replica_id UUID NULL,
        failure_reason TEXT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
        PRIMARY KEY (session_id, target_worker_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS broadcast_edges (
        edge_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        parent_worker_id TEXT NOT NULL,
        parent_replica_id UUID NOT NULL,
        child_worker_id TEXT NOT NULL,
        level INTEGER NOT NULL,
        attempt INTEGER NOT NULL DEFAULT 1,
        state TEXT CHECK (state IN ('planned','assigned','materializing','completed','failed','cancelled')) NOT NULL,
        transport_request_id TEXT NULL,
        failure_reason TEXT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_broadcast_edges_one_active_child
        ON broadcast_edges(session_id, child_worker_id)
        WHERE state IN ('planned','assigned','materializing');
    """,
    "ALTER TABLE artifact_transports ADD COLUMN IF NOT EXISTS broadcast_session_id TEXT NULL;",
    "ALTER TABLE artifact_transports ADD COLUMN IF NOT EXISTS broadcast_edge_id TEXT NULL;",
    """
    CREATE INDEX IF NOT EXISTS idx_artifact_transports_broadcast
        ON artifact_transports(broadcast_session_id, broadcast_edge_id, status);
    """,
)

DOWN_QUERIES: tuple[str, ...] = (
    "DROP INDEX IF EXISTS idx_artifact_transports_broadcast;",
    "ALTER TABLE artifact_transports DROP COLUMN IF EXISTS broadcast_edge_id;",
    "ALTER TABLE artifact_transports DROP COLUMN IF EXISTS broadcast_session_id;",
    "DROP INDEX IF EXISTS idx_broadcast_edges_one_active_child;",
    "DROP TABLE IF EXISTS broadcast_edges;",
    "DROP TABLE IF EXISTS broadcast_targets;",
    "DROP TABLE IF EXISTS broadcast_sessions;",
)


def upgrade(conn: DuckDBPyConnection) -> None:
    """Apply migration 0019."""
    for query in UP_QUERIES:
        conn.execute(query)


def downgrade(conn: DuckDBPyConnection) -> None:
    """Rollback migration 0019."""
    for query in DOWN_QUERIES:
        conn.execute(query)
```

- [ ] **Step 4: Add broadcast models**

Create `tensorcast/global_store/models/broadcast.py`:

```python
#  Copyright (c) 2026, TensorCast Team.

"""Broadcast session domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class BroadcastSessionState(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BroadcastTargetState(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    MATERIALIZING = "materializing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BroadcastEdgeState(str, Enum):
    PLANNED = "planned"
    ASSIGNED = "assigned"
    MATERIALIZING = "materializing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BroadcastSession:
    session_id: str
    artifact_id: str
    requested_view_id: str | None
    epoch: int
    fanout: int
    max_attempts: int
    strict_parent: bool
    state: BroadcastSessionState = BroadcastSessionState.PLANNING
    root_replica_id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class BroadcastTarget:
    session_id: str
    target_worker_id: str
    target_daemon_id: str | None
    state: BroadcastTargetState = BroadcastTargetState.PENDING
    level: int | None = None
    attempt: int = 0
    assigned_edge_id: str | None = None
    completed_replica_id: UUID | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class BroadcastEdge:
    edge_id: str
    session_id: str
    parent_worker_id: str
    parent_replica_id: UUID
    child_worker_id: str
    level: int
    attempt: int = 1
    state: BroadcastEdgeState = BroadcastEdgeState.PLANNED
    transport_request_id: str | None = None
    failure_reason: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
```

Export these names from `tensorcast/global_store/models/__init__.py`.

- [ ] **Step 5: Add repository**

Create `tensorcast/global_store/repositories/broadcast_repository.py` and implement these public methods:

- `create_session(self, session: BroadcastSession, cursor: DuckDBPyConnection | None = None) -> BroadcastSession`
- `find_session(self, session_id: str, cursor: DuckDBPyConnection | None = None) -> BroadcastSession | None`
- `update_session_state(self, session_id: str, state: BroadcastSessionState, cursor: DuckDBPyConnection | None = None) -> bool`
- `upsert_target(self, target: BroadcastTarget, cursor: DuckDBPyConnection | None = None) -> BroadcastTarget`
- `find_target(self, session_id: str, target_worker_id: str, cursor: DuckDBPyConnection | None = None) -> BroadcastTarget | None`
- `list_targets(self, session_id: str, cursor: DuckDBPyConnection | None = None) -> list[BroadcastTarget]`
- `list_targets_by_state(self, session_id: str, state: BroadcastTargetState, limit: int, cursor: DuckDBPyConnection | None = None) -> list[BroadcastTarget]`
- `create_edge(self, edge: BroadcastEdge, cursor: DuckDBPyConnection | None = None) -> BroadcastEdge`
- `find_edge(self, edge_id: str, cursor: DuckDBPyConnection | None = None) -> BroadcastEdge | None`
- `find_active_edge_for_child(self, session_id: str, child_worker_id: str, cursor: DuckDBPyConnection | None = None) -> BroadcastEdge | None`
- `mark_edge_materializing(self, edge_id: str, transport_request_id: str, cursor: DuckDBPyConnection | None = None) -> bool`
- `mark_edge_failed(self, edge_id: str, reason: str, cursor: DuckDBPyConnection | None = None) -> bool`
- `mark_edge_completed(self, edge_id: str, completed_replica_id: UUID | None, cursor: DuckDBPyConnection | None = None) -> bool`
- `count_incomplete_targets(self, session_id: str, cursor: DuckDBPyConnection | None = None) -> int`

Use `_normalize_required_text()` and `_normalize_optional_text()` helpers patterned after `TransportRepository`. Convert state strings through the enum classes in `_row_to_session`, `_row_to_target`, and `_row_to_edge`.

Export `BroadcastRepository` from `tensorcast/global_store/repositories/__init__.py` and add it to the `repositories` fixture in `tests/python/global_store/conftest.py`.

- [ ] **Step 6: Run repository tests and commit**

Run:

```bash
pytest tests/python/global_store/test_broadcast_repository.py -v
```

Expected: PASS.

Commit:

```bash
git add schema.sql tensorcast/global_store/migrations/0019_broadcast_sessions.py tensorcast/global_store/models/__init__.py tensorcast/global_store/models/broadcast.py tensorcast/global_store/repositories/__init__.py tensorcast/global_store/repositories/broadcast_repository.py tests/python/global_store/conftest.py tests/python/global_store/test_broadcast_repository.py
git commit -m "feat(global-store): add broadcast state repository"
```

### Task 2: Broadcast Service And Global Store RPC

**Files:**
- Create: `tensorcast/global_store/services/broadcast_service.py`
- Create: `tensorcast/global_store/rpc/broadcast_rpc_handler.py`
- Modify: `tensorcast/global_store/services/__init__.py`
- Modify: `tensorcast/global_store/grpc_service.py`
- Modify: `tensorcast/global_store/rpc_servicer_mixins.py`
- Modify: `proto/tensorcast/global_store/v1/global_store.proto`
- Test: add `tests/python/global_store/test_broadcast_service.py`
- Test: add `tests/python/global_store/test_broadcast_rpc.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/python/global_store/test_broadcast_service.py`:

```python
from __future__ import annotations

from tensorcast.global_store.models import (
    BroadcastEdgeState,
    BroadcastSessionState,
    BroadcastTargetState,
    ExportState,
    MemoryType,
    Replica,
    Worker,
)
from tensorcast.global_store.services import BroadcastService


def _worker(worker_id: str, daemon_id: str, node_id: str) -> Worker:
    return Worker(
        worker_id=worker_id,
        daemon_id=daemon_id,
        node_id=node_id,
        node_address=f"10.0.0.{node_id[-1]}",
        grpc_port=5000 + int(node_id[-1]),
        p2p_port=6000 + int(node_id[-1]),
        mem_pool_total_size=4096,
        mem_pool_available_size=4096,
        accepting_new_requests=True,
    )


def _exportable_replica(artifact_id: str, worker: Worker) -> Replica:
    return Replica(
        artifact_id=artifact_id,
        node_id=worker.node_id,
        node_address=worker.node_address,
        node_port=worker.p2p_port,
        memory_size=1024,
        memory_type=MemoryType.GPU,
        device_id=0,
        max_concurrency=4,
        current_requests=0,
        is_available=True,
        remote_memory_keys=[f"rk-{worker.worker_id}"],
        buffer_sizes=[1024],
        export_state=ExportState.EXPORTABLE,
        worker_id=worker.worker_id,
    )


def test_create_session_plans_first_layer_by_fanout(repositories):
    worker_repo = repositories["worker"]
    replica_repo = repositories["replica"]
    broadcast_repo = repositories["broadcast"]
    service = BroadcastService(
        broadcast_repository=broadcast_repo,
        replica_repository=replica_repo,
        worker_repository=worker_repo,
    )
    root = _worker("worker-root", "daemon-root", "node1")
    child1 = _worker("worker-child-1", "daemon-child-1", "node2")
    child2 = _worker("worker-child-2", "daemon-child-2", "node3")
    child3 = _worker("worker-child-3", "daemon-child-3", "node4")
    for worker in (root, child1, child2, child3):
        worker_repo.create(worker)
        assert worker_repo.update_heartbeat(worker.worker_id, 4096, True)
    root_replica = replica_repo.create(_exportable_replica("mi2:model-a", root))

    session = service.create_session(
        session_id="session-a",
        artifact_id="mi2:model-a",
        requested_view_id=None,
        epoch=42,
        fanout=2,
        target_daemon_ids=["daemon-child-1", "daemon-child-2", "daemon-child-3"],
        root_replica_id=str(root_replica.replica_id),
        strict_parent=True,
        max_attempts=3,
    )

    assert session.state is BroadcastSessionState.ACTIVE
    targets = broadcast_repo.list_targets("session-a")
    assert len(targets) == 3
    assigned = [t for t in targets if t.state is BroadcastTargetState.ASSIGNED]
    pending = [t for t in targets if t.state is BroadcastTargetState.PENDING]
    assert len(assigned) == 2
    assert len(pending) == 1
    edges = [
        broadcast_repo.find_active_edge_for_child("session-a", t.target_worker_id)
        for t in assigned
    ]
    assert all(edge is not None for edge in edges)
    assert all(edge.state is BroadcastEdgeState.PLANNED for edge in edges if edge)
    assert all(edge.parent_replica_id == root_replica.replica_id for edge in edges if edge)
```

- [ ] **Step 2: Run service test and verify it fails**

Run:

```bash
pytest tests/python/global_store/test_broadcast_service.py::test_create_session_plans_first_layer_by_fanout -v
```

Expected: FAIL with `ImportError` for `BroadcastService`.

- [ ] **Step 3: Implement `BroadcastService.create_session()` and first-layer planning**

Create `tensorcast/global_store/services/broadcast_service.py`:

```python
#  Copyright (c) 2026, TensorCast Team.

"""Broadcast session planning and progress service."""

from __future__ import annotations

import uuid
from uuid import UUID

from tensorcast.global_store.exceptions import NotFoundError, ValidationError
from tensorcast.global_store.models import (
    BroadcastEdge,
    BroadcastEdgeState,
    BroadcastSession,
    BroadcastSessionState,
    BroadcastTarget,
    BroadcastTargetState,
)
from tensorcast.global_store.repositories.broadcast_repository import BroadcastRepository
from tensorcast.global_store.repositories.replica_repository import ReplicaRepository
from tensorcast.global_store.repositories.worker_repository import WorkerRepository


class BroadcastService:
    def __init__(
        self,
        *,
        broadcast_repository: BroadcastRepository,
        replica_repository: ReplicaRepository,
        worker_repository: WorkerRepository,
    ) -> None:
        self.broadcast_repository = broadcast_repository
        self.replica_repository = replica_repository
        self.worker_repository = worker_repository

    def create_session(
        self,
        *,
        session_id: str,
        artifact_id: str,
        requested_view_id: str | None,
        epoch: int,
        fanout: int,
        target_worker_ids: list[str] | None = None,
        target_daemon_ids: list[str] | None = None,
        root_replica_id: str | None = None,
        strict_parent: bool = True,
        max_attempts: int = 3,
    ) -> BroadcastSession:
        session_id = session_id.strip()
        artifact_id = artifact_id.strip()
        if not session_id:
            raise ValidationError("session_id is required")
        if not artifact_id:
            raise ValidationError("artifact_id is required")
        if int(epoch) < 0:
            raise ValidationError("epoch must be non-negative")
        if int(fanout) <= 0:
            raise ValidationError("fanout must be positive")
        if int(max_attempts) <= 0:
            raise ValidationError("max_attempts must be positive")

        targets = self._resolve_targets(
            target_worker_ids=target_worker_ids or [],
            target_daemon_ids=target_daemon_ids or [],
        )
        if not targets:
            raise ValidationError("at least one target is required")

        root_replica = self._resolve_root_replica(
            artifact_id=artifact_id,
            requested_view_id=requested_view_id,
            root_replica_id=root_replica_id,
        )
        session = BroadcastSession(
            session_id=session_id,
            artifact_id=artifact_id,
            requested_view_id=requested_view_id,
            epoch=int(epoch),
            fanout=int(fanout),
            max_attempts=int(max_attempts),
            strict_parent=bool(strict_parent),
            state=BroadcastSessionState.ACTIVE,
            root_replica_id=root_replica.replica_id,
        )
        with self.broadcast_repository.transaction() as tx:
            existing = self.broadcast_repository.find_session(session_id, cursor=tx)
            if existing is not None:
                return existing
            self.broadcast_repository.create_session(session, cursor=tx)
            for worker_id, daemon_id in targets:
                self.broadcast_repository.upsert_target(
                    BroadcastTarget(
                        session_id=session_id,
                        target_worker_id=worker_id,
                        target_daemon_id=daemon_id,
                        state=BroadcastTargetState.PENDING,
                    ),
                    cursor=tx,
                )
            self._plan_more_edges(session, cursor=tx)
        loaded = self.broadcast_repository.find_session(session_id)
        if loaded is None:
            raise RuntimeError(f"broadcast session missing after create: {session_id}")
        return loaded
```

Add `_resolve_targets()`, `_resolve_root_replica()`, and `_plan_more_edges()` in the same file. `_resolve_targets()` must call `WorkerRepository.find_by_id(worker_id, include_inactive=False)` and `WorkerRepository.find_by_daemon_id(daemon_id, include_inactive=False)`. `_resolve_root_replica()` must call `ReplicaRepository.find_by_id(UUID(root_replica_id))` when a root id is provided. For the empty root id case, select the root by calling `ReplicaRepository.find_available_for_transport()` with a short heartbeat timeout, store the returned replica id on the session, and immediately decrement that replica counter after planning because session creation only reserves topology, not an active byte transfer.

`_plan_more_edges()` should list pending targets, list completed targets, build the parent pool as `[root_replica_id] + completed_replica_id values`, and create at most `fanout - active_edges_count` new `BroadcastEdge` rows.

Export `BroadcastService` from `tensorcast/global_store/services/__init__.py`.

- [ ] **Step 4: Add Global Store proto RPCs and handler tests**

Append `tests/python/global_store/test_broadcast_rpc.py`:

```python
from __future__ import annotations

from tensorcast.proto.global_store.v1 import global_store_pb2


def test_create_broadcast_session_rpc_returns_edges(servicer, test_context, memory_info):
    root_worker = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-root",
            node_id="node-root",
            node_address="10.10.0.1",
            grpc_port=50101,
            p2p_port=50102,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    child_worker = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id="daemon-child",
            node_id="node-child",
            node_address="10.10.0.2",
            grpc_port=50201,
            p2p_port=50202,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        test_context,
    ).worker_id
    memory_info.node_id = "node-root"
    memory_info.node_address = "10.10.0.1"
    memory_info.node_port = 50102
    register_resp = servicer.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:model-rpc",
            worker_id=root_worker,
            memory_info=memory_info,
            max_concurrency=4,
        ),
        test_context,
    )
    assert register_resp.status == global_store_pb2.STATUS_OK

    response = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-rpc",
            artifact_id="mi2:model-rpc",
            epoch=7,
            fanout=1,
            strict_parent=True,
            max_attempts=3,
            root_replica_id=register_resp.replica_id,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(
                    worker_id=child_worker,
                    daemon_id="daemon-child",
                )
            ],
        ),
        test_context,
    )

    assert response.status == global_store_pb2.STATUS_OK
    assert response.session.session_id == "session-rpc"
    assert response.session.state == global_store_pb2.BROADCAST_SESSION_STATE_ACTIVE
    edge_resp = servicer.ListBroadcastEdges(
        global_store_pb2.ListBroadcastEdgesRequest(session_id="session-rpc"),
        test_context,
    )
    assert edge_resp.status == global_store_pb2.STATUS_OK
    assert len(edge_resp.edges) == 1
    assert edge_resp.edges[0].child_worker_id == child_worker
```

- [ ] **Step 5: Run RPC test and verify it fails**

Run:

```bash
pytest tests/python/global_store/test_broadcast_rpc.py::test_create_broadcast_session_rpc_returns_edges -v
```

Expected: FAIL because `CreateBroadcastSessionRequest` and related proto messages do not exist.

- [ ] **Step 6: Extend Global Store proto and regenerate Python stubs**

Modify `proto/tensorcast/global_store/v1/global_store.proto`:

```proto
service ClusterRuntimeService {
  // Insert these RPCs beside the existing transport and metadata RPCs.
  rpc CreateBroadcastSession(CreateBroadcastSessionRequest) returns (CreateBroadcastSessionResponse) {}
  rpc GetBroadcastSession(GetBroadcastSessionRequest) returns (GetBroadcastSessionResponse) {}
  rpc ListBroadcastEdges(ListBroadcastEdgesRequest) returns (ListBroadcastEdgesResponse) {}
  rpc CancelBroadcastSession(CancelBroadcastSessionRequest) returns (CancelBroadcastSessionResponse) {}
}

enum BroadcastSessionState {
  BROADCAST_SESSION_STATE_UNSPECIFIED = 0;
  BROADCAST_SESSION_STATE_PLANNING = 1;
  BROADCAST_SESSION_STATE_ACTIVE = 2;
  BROADCAST_SESSION_STATE_COMPLETED = 3;
  BROADCAST_SESSION_STATE_FAILED = 4;
  BROADCAST_SESSION_STATE_CANCELLED = 5;
}

enum BroadcastTargetState {
  BROADCAST_TARGET_STATE_UNSPECIFIED = 0;
  BROADCAST_TARGET_STATE_PENDING = 1;
  BROADCAST_TARGET_STATE_ASSIGNED = 2;
  BROADCAST_TARGET_STATE_MATERIALIZING = 3;
  BROADCAST_TARGET_STATE_COMPLETED = 4;
  BROADCAST_TARGET_STATE_FAILED = 5;
  BROADCAST_TARGET_STATE_CANCELLED = 6;
}

enum BroadcastEdgeState {
  BROADCAST_EDGE_STATE_UNSPECIFIED = 0;
  BROADCAST_EDGE_STATE_PLANNED = 1;
  BROADCAST_EDGE_STATE_ASSIGNED = 2;
  BROADCAST_EDGE_STATE_MATERIALIZING = 3;
  BROADCAST_EDGE_STATE_COMPLETED = 4;
  BROADCAST_EDGE_STATE_FAILED = 5;
  BROADCAST_EDGE_STATE_CANCELLED = 6;
}

message BroadcastTargetIdentity {
  string worker_id = 1;
  string daemon_id = 2;
}

message BroadcastSessionInfo {
  string session_id = 1;
  string artifact_id = 2;
  tensorcast.common.v1.ByteSpaceRef requested_byte_space = 3;
  uint64 epoch = 4;
  uint32 fanout = 5;
  uint32 max_attempts = 6;
  bool strict_parent = 7;
  BroadcastSessionState state = 8;
  string root_replica_id = 9;
}

message BroadcastTargetInfo {
  string session_id = 1;
  string target_worker_id = 2;
  string target_daemon_id = 3;
  BroadcastTargetState state = 4;
  uint32 level = 5;
  uint32 attempt = 6;
  string assigned_edge_id = 7;
  string completed_replica_id = 8;
  string failure_reason = 9;
}

message BroadcastEdgeInfo {
  string edge_id = 1;
  string session_id = 2;
  string parent_worker_id = 3;
  string parent_replica_id = 4;
  string child_worker_id = 5;
  uint32 level = 6;
  uint32 attempt = 7;
  BroadcastEdgeState state = 8;
  string transport_request_id = 9;
  string failure_reason = 10;
}

message CreateBroadcastSessionRequest {
  string session_id = 1;
  string artifact_id = 2;
  tensorcast.common.v1.ByteSpaceRef requested_byte_space = 3;
  uint64 epoch = 4;
  uint32 fanout = 5;
  repeated BroadcastTargetIdentity targets = 6;
  string root_replica_id = 7;
  bool strict_parent = 8;
  uint32 max_attempts = 9;
}

message CreateBroadcastSessionResponse {
  Status status = 1;
  BroadcastSessionInfo session = 2;
  repeated BroadcastTargetInfo targets = 3;
  repeated BroadcastEdgeInfo edges = 4;
}

message GetBroadcastSessionRequest {
  string session_id = 1;
}

message GetBroadcastSessionResponse {
  Status status = 1;
  BroadcastSessionInfo session = 2;
  repeated BroadcastTargetInfo targets = 3;
}

message ListBroadcastEdgesRequest {
  string session_id = 1;
}

message ListBroadcastEdgesResponse {
  Status status = 1;
  repeated BroadcastEdgeInfo edges = 2;
}

message CancelBroadcastSessionRequest {
  string session_id = 1;
  string reason = 2;
}

message CancelBroadcastSessionResponse {
  Status status = 1;
}
```

Regenerate stubs for local validation:

```bash
bash tools/build_proto_python.sh
```

Expected: command exits 0 and Python generated files contain the new broadcast messages. Do not stage unrelated generated files that were dirty before this task.

- [ ] **Step 7: Implement RPC handler and service wiring**

Create `tensorcast/global_store/rpc/broadcast_rpc_handler.py` with a `BroadcastRpcHandler` class exposing these methods: `__init__(self, *, broadcast_service: BroadcastService, logger) -> None`, `create_broadcast_session(self, request, context)`, `get_broadcast_session(self, request, context)`, `list_broadcast_edges(self, request, context)`, and `cancel_broadcast_session(self, request, context)`.

Use explicit mapping helpers:

```python
_SESSION_STATE_TO_PROTO = {
    BroadcastSessionState.PLANNING: global_store_pb2.BROADCAST_SESSION_STATE_PLANNING,
    BroadcastSessionState.ACTIVE: global_store_pb2.BROADCAST_SESSION_STATE_ACTIVE,
    BroadcastSessionState.COMPLETED: global_store_pb2.BROADCAST_SESSION_STATE_COMPLETED,
    BroadcastSessionState.FAILED: global_store_pb2.BROADCAST_SESSION_STATE_FAILED,
    BroadcastSessionState.CANCELLED: global_store_pb2.BROADCAST_SESSION_STATE_CANCELLED,
}
```

Wire `BroadcastRepository` in `GlobalStoreServicer._init_repositories()`, `BroadcastService` in the service initialization path, `BroadcastRpcHandler` in handler initialization, and add forwarding methods to `ClusterRuntimeRpcServicerMixin`.

- [ ] **Step 8: Run service/RPC tests and commit**

Run:

```bash
pytest tests/python/global_store/test_broadcast_service.py tests/python/global_store/test_broadcast_rpc.py -v
```

Expected: PASS.

Commit:

```bash
git add proto/tensorcast/global_store/v1/global_store.proto tensorcast/global_store/services/__init__.py tensorcast/global_store/services/broadcast_service.py tensorcast/global_store/rpc/broadcast_rpc_handler.py tensorcast/global_store/grpc_service.py tensorcast/global_store/rpc_servicer_mixins.py tests/python/global_store/test_broadcast_service.py tests/python/global_store/test_broadcast_rpc.py
git commit -m "feat(global-store): add broadcast session rpc"
```

### Task 3: Broadcast-Aware Transport Selection And Completion

**Files:**
- Modify: `tensorcast/global_store/models/transport.py`
- Modify: `tensorcast/global_store/repositories/transport_repository.py`
- Modify: `tensorcast/global_store/repositories/replica_repository.py`
- Modify: `tensorcast/global_store/services/broadcast_service.py`
- Modify: `tensorcast/global_store/services/transport_service.py`
- Modify: `tensorcast/global_store/rpc/transport_rpc_handler.py`
- Modify: `proto/tensorcast/global_store/v1/global_store.proto`
- Test: add `tests/python/global_store/test_broadcast_transport.py`
- Test: update `tests/python/global_store/test_services.py` only if existing group-dispatch assertions need broadcast columns in row projections.

- [ ] **Step 1: Write failing broadcast transport tests**

Create `tests/python/global_store/test_broadcast_transport.py`:

```python
from __future__ import annotations

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _register_worker(servicer, context, worker_id_suffix: str) -> str:
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id=f"daemon-{worker_id_suffix}",
            node_id=f"node-{worker_id_suffix}",
            node_address=f"10.20.0.{worker_id_suffix}",
            grpc_port=51000 + int(worker_id_suffix),
            p2p_port=52000 + int(worker_id_suffix),
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.worker_id


def _register_replica(servicer, context, worker_id: str, node_suffix: str, key: str) -> str:
    memory_info = global_store_pb2.RegisterReplicaRequest().memory_info
    memory_info.node_id = f"node-{node_suffix}"
    memory_info.node_address = f"10.20.0.{node_suffix}"
    memory_info.node_port = 52000 + int(node_suffix)
    memory_info.memory_size = 1024
    memory_info.memory_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
    memory_info.device_id = 0
    memory_info.transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    memory_info.transport.remote_memory_keys.append(key)
    memory_info.transport.buffer_sizes.append(1024)
    response = servicer.RegisterReplica(
        global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:model-transport",
            worker_id=worker_id,
            memory_info=memory_info,
            max_concurrency=4,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.replica_id


def test_broadcast_transport_uses_edge_parent(servicer, test_context):
    root_worker = _register_worker(servicer, test_context, "1")
    alternate_worker = _register_worker(servicer, test_context, "2")
    child_worker = _register_worker(servicer, test_context, "3")
    root_replica_id = _register_replica(servicer, test_context, root_worker, "1", "rk-root")
    _register_replica(servicer, test_context, alternate_worker, "2", "rk-alt")
    create = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-transport",
            artifact_id="mi2:model-transport",
            epoch=1,
            fanout=1,
            root_replica_id=root_replica_id,
            strict_parent=True,
            max_attempts=3,
            targets=[global_store_pb2.BroadcastTargetIdentity(worker_id=child_worker)],
        ),
        test_context,
    )
    assert create.status == global_store_pb2.STATUS_OK

    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id="mi2:model-transport",
        source_node_id="node-3",
        source_address="10.20.0.3",
        source_port=52003,
        requester_worker_id=child_worker,
        request_id="broadcast-request-1",
    )
    request.local_memory_info.memory_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
    request.local_memory_info.device_id = 0
    request.broadcast.session_id = "session-transport"
    request.broadcast.strict_parent = True
    response = servicer.RequestReplicaTransport(request, test_context)

    assert response.status == global_store_pb2.STATUS_OK
    assert response.remote_memory_info.transport.remote_memory_keys == ["rk-root"]
    assert response.remote_memory_info.node_id == "node-1"


def test_broadcast_failed_transport_requeues_target(servicer, test_context):
    root_worker = _register_worker(servicer, test_context, "1")
    child_worker = _register_worker(servicer, test_context, "3")
    root_replica_id = _register_replica(servicer, test_context, root_worker, "1", "rk-root")
    servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-failure",
            artifact_id="mi2:model-transport",
            epoch=1,
            fanout=1,
            root_replica_id=root_replica_id,
            strict_parent=True,
            max_attempts=3,
            targets=[global_store_pb2.BroadcastTargetIdentity(worker_id=child_worker)],
        ),
        test_context,
    )
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id="mi2:model-transport",
        source_node_id="node-3",
        source_address="10.20.0.3",
        source_port=52003,
        requester_worker_id=child_worker,
        request_id="broadcast-request-fail",
    )
    request.local_memory_info.memory_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
    request.local_memory_info.device_id = 0
    request.broadcast.session_id = "session-failure"
    transport = servicer.RequestReplicaTransport(request, test_context)
    assert transport.status == global_store_pb2.STATUS_OK

    complete = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
            outcome_detail="injected failure",
        ),
        test_context,
    )
    assert complete.status == global_store_pb2.STATUS_OK

    edges = servicer.ListBroadcastEdges(
        global_store_pb2.ListBroadcastEdgesRequest(session_id="session-failure"),
        test_context,
    )
    assert edges.status == global_store_pb2.STATUS_OK
    assert any(edge.state == global_store_pb2.BROADCAST_EDGE_STATE_FAILED for edge in edges.edges)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/python/global_store/test_broadcast_transport.py -v
```

Expected: FAIL because `RequestReplicaTransportRequest.broadcast` and broadcast-aware transport routing do not exist.

- [ ] **Step 3: Add transport broadcast fields**

Update `Transport` model with:

```python
broadcast_session_id: str | None = None
broadcast_edge_id: str | None = None
```

Update `TransportRepository._TRANSPORT_PROJECTION`, `create_with_cursor()`, `create_if_absent_with_cursor()`, `_row_to_model()`, and `list_rows_in_created_window()` to include the two fields. Preserve existing column order compatibility by appending the new fields to projections and mapping by column name.

- [ ] **Step 4: Add specific parent claim helpers to `ReplicaRepository`**

Add:

```python
def claim_replica_for_transport(
    self,
    *,
    replica_id: UUID,
    artifact_id: str,
    view_id: str | None,
    heartbeat_timeout_seconds: float,
    cursor=None,
) -> TransportSelectionResult:
    """Claim one exact replica if it is currently transport eligible."""
```

The method should load the same joined worker/liveness row used by `find_available_for_transport()`, call `_evaluate_transport_candidate()`, increment `replica_counters.current_requests` only when eligible, and return `TransportSelectionResult(replica=None, exportable_replicas=candidate_exportable_replicas)` when not eligible.

Add:

```python
def find_exportable_replica_for_worker(
    self,
    *,
    artifact_id: str,
    view_id: str | None,
    worker_id: str,
    heartbeat_timeout_seconds: float,
    cursor=None,
) -> Replica | None:
    """Return the best registered child replica after materialization completes."""
```

Use the same eligibility checks, but do not increment `current_requests`.

- [ ] **Step 5: Add `BroadcastService.claim_transport_edge()` and completion advancement**

In `BroadcastService`, add `claim_transport_edge(self, *, session_id: str, artifact_id: str, requested_view_id: str | None, requester_worker_id: str, request_id: str, heartbeat_timeout_seconds: float, cursor: DuckDBPyConnection) -> tuple[Replica, BroadcastEdge]`.

This method must verify session artifact/view/epoch, find or plan an active edge for `requester_worker_id`, claim `edge.parent_replica_id` with `ReplicaRepository.claim_replica_for_transport()`, mark the edge materializing with `request_id`, and return the claimed replica plus edge.

Add `complete_transport_edge(self, *, session_id: str, edge_id: str, transport_outcome: TransportCompletionOutcome, outcome_detail: str | None, cursor: DuckDBPyConnection) -> None`.

On success, call `ReplicaRepository.find_exportable_replica_for_worker()` for the child worker and record that replica id in `BroadcastRepository.mark_edge_completed()`. If no child replica is visible, mark the edge failed with reason `child_replica_not_exportable_after_success`. On failure, mark the edge failed and requeue the target until `max_attempts` is reached.

- [ ] **Step 6: Integrate broadcast claim into `TransportService`**

Add a small value object in `tensorcast/global_store/models/transport.py`:

```python
@dataclass(frozen=True)
class BroadcastTransportHint:
    session_id: str
    strict_parent: bool = True
```

Extend `TransportService.__init__()` with optional `broadcast_service: BroadcastService | None = None`.

Extend `request_transport()` and `_build_request_fingerprint()` with `broadcast_hint`. In `_dispatch_pending_requests()`, leave normal queued dispatch unchanged. For requests with a broadcast hint, bypass group-dispatch queueing and claim the edge synchronously inside one transaction:

When `broadcast_hint is not None`, call `_request_transport_broadcast()` with the same normalized artifact, requester, memory, request id, timeout, and scheduling parameters already available in `request_transport()`, plus the parsed `BroadcastTransportHint`.

`_request_transport_broadcast()` should call `broadcast_service.claim_transport_edge()`, build a `Transport` with `broadcast_session_id` and `broadcast_edge_id`, create it idempotently, and return the claimed parent replica.

In `complete_transport()`, after `transport_repository.complete_if_in_progress()`, if `transport.broadcast_session_id` and `transport.broadcast_edge_id` are set, call `broadcast_service.complete_transport_edge()`.

- [ ] **Step 7: Parse broadcast hint in transport RPC**

In `proto/tensorcast/global_store/v1/global_store.proto`, add:

```proto
message BroadcastTransportHint {
  string session_id = 1;
  bool strict_parent = 2;
}
```

Add field 11 to `RequestReplicaTransportRequest`:

```proto
BroadcastTransportHint broadcast = 11;
```

Run:

```bash
bash tools/build_proto_python.sh
```

In `TransportRpcHandler.request_replica_transport()`, parse:

```python
broadcast_hint: BroadcastTransportHint | None = None
if request.HasField("broadcast"):
    session_id = request.broadcast.session_id.strip()
    if not session_id:
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        context.set_details("broadcast.session_id is required")
        return global_store_pb2.RequestReplicaTransportResponse(
            status=global_store_pb2.Status.STATUS_ERROR
        )
    broadcast_hint = BroadcastTransportHint(
        session_id=session_id,
        strict_parent=bool(request.broadcast.strict_parent),
    )
```

Pass `broadcast_hint` into `TransportService.request_transport()`.

- [ ] **Step 8: Run transport tests and existing group-dispatch regression tests**

Run:

```bash
pytest tests/python/global_store/test_broadcast_transport.py tests/python/global_store/test_services.py::TestTransportService -v
```

Expected: PASS. Existing group-dispatch tests must still pass without broadcast hints.

Commit:

```bash
git add proto/tensorcast/global_store/v1/global_store.proto tensorcast/global_store/models/transport.py tensorcast/global_store/repositories/transport_repository.py tensorcast/global_store/repositories/replica_repository.py tensorcast/global_store/services/broadcast_service.py tensorcast/global_store/services/transport_service.py tensorcast/global_store/rpc/transport_rpc_handler.py tests/python/global_store/test_broadcast_transport.py
git commit -m "feat(global-store): route broadcast transports through tree edges"
```

### Task 4: SDK And Store Daemon Broadcast Hint Propagation

**Files:**
- Modify: `proto/tensorcast/daemon/v2/store_daemon.proto`
- Modify: `tensorcast/api/context.py`
- Modify: `tensorcast/api/__init__.py`
- Modify: `tensorcast/__init__.py`
- Modify: `tensorcast/api/store/artifact.py`
- Modify: `tensorcast/api/_materialize.py`
- Modify: `tensorcast/daemon_ctl.py`
- Modify: `daemon/service/controllers/materialization_policy_utils.h`
- Modify: `daemon/service/controllers/materialization_policy_utils.cc`
- Modify: `daemon/service/controllers/replica_materialization_service.cc`
- Test: update `tests/python/api/test_prefetch_operation.py`
- Test: add `tests/python/api/test_daemon_ctl_broadcast_hint.py`
- Test: update `daemon/service/materialization_policy_utils_test.cc`

- [ ] **Step 1: Write failing SDK prefetch test**

Append to `tests/python/api/test_prefetch_operation.py`:

```python
def test_prefetch_forwards_broadcast_context_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    ctx = tc.context(
        broadcast=tc.BroadcastContext(
            session_id="broadcast-session-1",
            strict_parent=True,
        )
    )

    artifact.prefetch(device="cuda:0", ctx=ctx)

    call = store._materialization.calls[0]
    assert call["broadcast_session_id"] == "broadcast-session-1"
    assert call["broadcast_strict_parent"] is True
```

- [ ] **Step 2: Run SDK test and verify it fails**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py::test_prefetch_forwards_broadcast_context_hint -v
```

Expected: FAIL because `BroadcastContext` and materialization kwargs do not exist.

- [ ] **Step 3: Add SDK context type and prefetch forwarding**

In `tensorcast/api/context.py`, add:

```python
@dataclass(frozen=True, slots=True)
class BroadcastContext:
    """Strict tree broadcast session hint for materialization."""

    session_id: str
    strict_parent: bool = True

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip()
        if not session_id:
            raise ValueError("BroadcastContext.session_id must be non-empty")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "strict_parent", bool(self.strict_parent))
```

Add `broadcast: BroadcastContext | None = None` to `CallContext` and `context()`, and export `BroadcastContext` from `tensorcast/api/context.py`, `tensorcast/api/__init__.py`, and `tensorcast/__init__.py`.

In `Artifact.prefetch()`, pass:

```python
broadcast_session_id=ctx.broadcast.session_id if ctx and ctx.broadcast else None,
broadcast_strict_parent=ctx.broadcast.strict_parent if ctx and ctx.broadcast else True,
```

to `pipeline.materialize_subset()`. Thread the same kwargs through `MaterializationPipeline.materialize_subset()` into `materialize_artifact_v2()`.

- [ ] **Step 4: Write failing DaemonCtl proto copy test**

Create `tests/python/api/test_daemon_ctl_broadcast_hint.py`:

```python
from __future__ import annotations

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _FakeUnary:
    def __init__(self) -> None:
        self.requests: list[store_daemon_pb2.MaterializeReplicaRequest] = []

    def __call__(self, request, timeout=None, metadata=None):  # noqa: ANN001
        del timeout, metadata
        self.requests.append(request)
        response = store_daemon_pb2.MaterializeReplicaResponse()
        response.status = store_daemon_pb2.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        return response


class _FakeStub:
    def __init__(self) -> None:
        self.MaterializeReplica = _FakeUnary()


def test_daemon_ctl_copies_broadcast_hint(monkeypatch) -> None:  # noqa: ANN001
    fake_stub = _FakeStub()
    ctl = DaemonCtl("127.0.0.1:1")
    ctl.stub_v2 = fake_stub
    monkeypatch.setattr(ctl, "_get_effective_pid", lambda: 123)

    selection = common_pb2.ArtifactSelection(artifact_id="mi2:model")
    ctl.materialize_by_artifact_id_v2(
        selection=selection,
        replica_uuid="replica-1",
        device_uuid="gpu-uuid",
        return_response=True,
        wait_for_completion=False,
        broadcast_session_id="session-a",
        broadcast_strict_parent=True,
    )

    request = fake_stub.MaterializeReplica.requests[0]
    assert request.broadcast.session_id == "session-a"
    assert request.broadcast.strict_parent is True
```

- [ ] **Step 5: Update daemon proto and DaemonCtl**

In `proto/tensorcast/daemon/v2/store_daemon.proto`, add:

```proto
message BroadcastMaterializationHint {
  string session_id = 1;
  bool strict_parent = 2;
}
```

Add field 23 to `MaterializeReplicaRequest`:

```proto
BroadcastMaterializationHint broadcast = 23;
```

Run:

```bash
bash tools/build_proto_python.sh
```

In `tensorcast/daemon_ctl.py`, add overload and implementation kwargs:

```python
broadcast_session_id: str | None = None,
broadcast_strict_parent: bool = True,
```

and copy:

```python
if broadcast_session_id:
    request.broadcast.session_id = str(broadcast_session_id)
    request.broadcast.strict_parent = bool(broadcast_strict_parent)
```

In `tensorcast/api/_materialize.py`, add the same kwargs and pass them to `client.materialize_by_artifact_id_v2()`.

- [ ] **Step 6: Add C++ daemon mapping test**

Append to `daemon/service/materialization_policy_utils_test.cc`:

```c++
using tensorcast::daemon::materialization_policy::resolve_broadcast_materialization_hint;

TEST_CASE("Broadcast materialization hint maps daemon proto", "[daemon][materialization][policy]") {
  v2::BroadcastMaterializationHint proto;
  proto.set_session_id("session-a");
  proto.set_strict_parent(true);

  auto hint = resolve_broadcast_materialization_hint(&proto);

  REQUIRE(hint.has_value());
  CHECK(hint->session_id == "session-a");
  CHECK(hint->strict_parent);
}
```

Run:

```bash
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: FAIL because `resolve_broadcast_materialization_hint` does not exist.

- [ ] **Step 7: Implement C++ daemon hint mapping**

In `core/store/materialization/contracts/loading_spec.h`, add:

```c++
struct BroadcastHint {
  std::string session_id;
  bool strict_parent{true};
};
```

Add to `MaterializeHints`:

```c++
std::optional<BroadcastHint> broadcast;
```

In `daemon/service/controllers/materialization_policy_utils.h`, declare:

```c++
std::optional<store::loading::BroadcastHint> resolve_broadcast_materialization_hint(
    const v2::BroadcastMaterializationHint* hint);
```

In `.cc`, define it to trim/validate non-empty `session_id` and return `std::nullopt` for null or empty hints.

In `ReplicaMaterializationService::materialize_replica()`, after transport group mapping:

```c++
if (req.has_broadcast()) {
  auto broadcast_hint = resolve_broadcast_materialization_hint(&req.broadcast());
  if (broadcast_hint.has_value()) {
    hints.broadcast = std::move(*broadcast_hint);
  }
}
```

- [ ] **Step 8: Run SDK and daemon hint tests and commit**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py::test_prefetch_forwards_broadcast_context_hint tests/python/api/test_daemon_ctl_broadcast_hint.py -v
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS.

Commit:

```bash
git add proto/tensorcast/daemon/v2/store_daemon.proto tensorcast/api/context.py tensorcast/api/__init__.py tensorcast/__init__.py tensorcast/api/store/artifact.py tensorcast/api/_materialize.py tensorcast/daemon_ctl.py daemon/service/controllers/materialization_policy_utils.h daemon/service/controllers/materialization_policy_utils.cc daemon/service/controllers/replica_materialization_service.cc core/store/materialization/contracts/loading_spec.h tests/python/api/test_prefetch_operation.py tests/python/api/test_daemon_ctl_broadcast_hint.py daemon/service/materialization_policy_utils_test.cc
git commit -m "feat(materialize): propagate broadcast session hints"
```

### Task 5: C++ Global Store Client And Materialization Completion Semantics

**Files:**
- Modify: `core/store/components/global_store_client.h`
- Modify: `core/store/components/global_store_client.cc`
- Modify: `core/store/testing/recording_global_store_client.h`
- Modify: `core/store/materialization/control/materialize_orchestrator.cc`
- Modify: `core/store/runtime/ingestion/materialization_facade.cc`
- Test: update `core/store/materialization/control/materialize_orchestrator_reselection_test.cc`

- [ ] **Step 1: Write failing C++ propagation test**

Append to `core/store/materialization/control/materialize_orchestrator_reselection_test.cc`:

```c++
TEST_CASE(
    "MaterializeOrchestrator propagates broadcast hint to transport request",
    "[store][materialize][broadcast]") {
  auto gs_client = std::make_shared<RecordingGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->allow_replica_transport = true;
  gs_client->push_scripted_transport_session(make_transport_session(
      "transport-broadcast", "node-remote", "10.9.9.2", 50042, common::memory::MemoryLocation::GPU, 0));

  FakeMaterializationBackend backend;
  MaterializeHints hints;
  hints.artifact_id = "artifact-broadcast";
  hints.allow_p2p = true;
  hints.allow_disk = false;
  hints.transport_request_id = "request-broadcast-1";
  hints.broadcast = loading::BroadcastHint{
      .session_id = "session-a",
      .strict_parent = true,
  };

  components::WorkerIdentity local_identity{
      .worker_id = "worker-local",
      .node_id = "node-local",
      .node_address = "10.9.9.1",
      .p2p_port = 50041,
  };
  MaterializeOrchestrator orchestrator(
      gsl::not_null<MaterializationBackend*>{&backend},
      gsl::not_null<components::IGlobalStoreClient*>{gs_client.get()},
      local_identity);

  auto result = orchestrator.run("artifact-broadcast", make_gpu_target(0), hints, std::nullopt);
  REQUIRE(result.ok());
  REQUIRE(gs_client->replica_request_broadcast_hints.size() == 1);
  REQUIRE(gs_client->replica_request_broadcast_hints.front().has_value());
  CHECK(gs_client->replica_request_broadcast_hints.front()->session_id == "session-a");
  CHECK(gs_client->replica_request_broadcast_hints.front()->strict_parent);
}
```

- [ ] **Step 2: Run C++ test and verify it fails**

Run:

```bash
bazel test //core/store/materialization/control:materialize_orchestrator_reselection_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: FAIL because `BroadcastHint` is not part of `IGlobalStoreClient` requests.

- [ ] **Step 3: Add C++ GlobalStoreClient broadcast hint type and proto mapping**

In `core/store/components/global_store_client.h`, add:

```c++
struct BroadcastTransportHint {
  std::string session_id;
  bool strict_parent{true};
};
```

Add `const std::optional<BroadcastTransportHint>& broadcast_hint = std::nullopt` to `request_replica_transport()` and `request_view_transport()` interface and implementation signatures, after `scheduling_group`.

In `core/store/components/global_store_client.cc`, add:

```c++
void apply_broadcast_transport_hint(
    const std::optional<BroadcastTransportHint>& broadcast_hint,
    global_store::RequestReplicaTransportRequest* request) {
  if (!broadcast_hint.has_value() || broadcast_hint->session_id.empty()) {
    return;
  }
  auto* out = request->mutable_broadcast();
  out->set_session_id(broadcast_hint->session_id);
  out->set_strict_parent(broadcast_hint->strict_parent);
}
```

Call this helper in both request methods.

- [ ] **Step 4: Thread hints from `MaterializeHints` to C++ client**

In `materialize_orchestrator.cc` and `materialization_facade.cc`, add a helper:

```c++
std::optional<components::BroadcastTransportHint> to_broadcast_transport_hint(
    const MaterializeHints& hints) {
  if (!hints.broadcast.has_value() || hints.broadcast->session_id.empty()) {
    return std::nullopt;
  }
  return components::BroadcastTransportHint{
      .session_id = hints.broadcast->session_id,
      .strict_parent = hints.broadcast->strict_parent,
  };
}
```

Pass `broadcast_hint` immediately after `scheduling_group_hint` in all `request_replica_transport()` and `request_view_transport()` calls. Update `core/store/testing/recording_global_store_client.h` to capture `replica_request_broadcast_hints` and `view_request_broadcast_hints`.

- [ ] **Step 5: Change broadcast success completion ordering**

In `MaterializeOrchestrator::run()`, change the P2P success block:

```c++
if (load_or.ok()) {
  const auto& handle = *load_or;
  absl::Status reg_status = backend_->register_replica_with_global_store(handle.key(), {});
  const bool broadcast_request = hints.broadcast.has_value() && !hints.broadcast->session_id.empty();
  if (!reg_status.ok()) {
    LOG(WARNING) << "register_replica_with_global_store returned error: " << reg_status;
    if (broadcast_request) {
      absl::Status comp_status = gs_client_->complete_replica_transport(
          session.transport_id,
          components::TransportCompletionOutcome::kFailed,
          reg_status.ToString());
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
      }
      last_p2p_status = reg_status;
      // Continue existing reselection/fallback handling.
    } else {
      absl::Status comp_status = gs_client_->complete_replica_transport(
          session.transport_id, components::TransportCompletionOutcome::kSuccess);
      if (!comp_status.ok()) {
        LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
      }
      return load_or;
    }
  } else {
    absl::Status comp_status = gs_client_->complete_replica_transport(
        session.transport_id, components::TransportCompletionOutcome::kSuccess);
    if (!comp_status.ok()) {
      LOG(WARNING) << "complete_replica_transport returned error: " << comp_status;
    }
    return load_or;
  }
}
```

Preserve the existing failed P2P completion path for `load_or` failures. Apply the same broadcast-specific ordering in `MaterializationFacade` paths that directly request P2P transport and register replicas.

- [ ] **Step 6: Add failure completion test**

Add a C++ test using `FakeMaterializationBackend` with `fail_register_replica` behavior. The assertion should verify:

```c++
REQUIRE(gs_client->completed_transport_ids.size() == 1);
CHECK(gs_client->completed_transport_ids.front() == "transport-broadcast");
CHECK(gs_client->completed_transport_outcomes.front() == components::TransportCompletionOutcome::kFailed);
```

Run:

```bash
bazel test //core/store/materialization/control:materialize_orchestrator_reselection_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS.

- [ ] **Step 7: Commit C++ propagation and completion semantics**

Commit:

```bash
git add core/store/components/global_store_client.h core/store/components/global_store_client.cc core/store/testing/recording_global_store_client.h core/store/materialization/control/materialize_orchestrator.cc core/store/runtime/ingestion/materialization_facade.cc core/store/materialization/control/materialize_orchestrator_reselection_test.cc
git commit -m "feat(core): enforce broadcast transport parent hints"
```

### Task 6: Daemon-Mediated Broadcast Session API

**Files:**
- Modify: `proto/tensorcast/daemon/v2/store_daemon.proto`
- Modify: `tensorcast/daemon_ctl.py`
- Modify: daemon service implementation files under `daemon/service/`
- Modify: `tensorcast/api/store/__init__.py`
- Test: add `tests/python/api/test_broadcast_session_api.py`
- Test: add daemon C++ controller test if a local controller seam exists; otherwise use Python DaemonCtl request construction test plus Global Store RPC tests.

- [ ] **Step 1: Add SDK API failing test**

Create `tests/python/api/test_broadcast_session_api.py`:

```python
from __future__ import annotations

import tensorcast as tc
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_store_create_broadcast_session_calls_daemon(monkeypatch) -> None:  # noqa: ANN001
    calls: list[dict[str, object]] = []

    class _Client:
        def create_broadcast_session(self, **kwargs):
            calls.append(kwargs)
            response = store_daemon_pb2.CreateBroadcastSessionResponse()
            response.status = store_daemon_pb2.BROADCAST_SESSION_STATUS_OK
            response.session_id = "session-a"
            return response

    class _Runtime:
        daemon_endpoint = "daemon"
        daemon_id = "daemon-local"
        closed = False

        def ensure_client(self):
            return _Client()

    from tensorcast.api.store import Store

    store = Store("daemon", runtime=_Runtime())
    session = store.create_broadcast_session(
        session_id="session-a",
        artifact_id="mi2:model",
        epoch=42,
        fanout=2,
        target_daemon_ids=["daemon-a", "daemon-b"],
        root_replica_id="00000000-0000-0000-0000-000000000001",
    )

    assert session.session_id == "session-a"
    assert calls[0]["artifact_id"] == "mi2:model"
    assert calls[0]["target_daemon_ids"] == ["daemon-a", "daemon-b"]
```

- [ ] **Step 2: Add daemon proto for session forwarding**

In `proto/tensorcast/daemon/v2/store_daemon.proto`, add daemon-facing messages:

```proto
enum BroadcastSessionStatus {
  BROADCAST_SESSION_STATUS_UNSPECIFIED = 0;
  BROADCAST_SESSION_STATUS_OK = 1;
  BROADCAST_SESSION_STATUS_ERROR = 2;
  BROADCAST_SESSION_STATUS_NOT_FOUND = 3;
}

message CreateBroadcastSessionRequest {
  string session_id = 1;
  string artifact_id = 2;
  string requested_view_id = 3;
  uint64 epoch = 4;
  uint32 fanout = 5;
  repeated string target_worker_ids = 6;
  repeated string target_daemon_ids = 7;
  string root_replica_id = 8;
  bool strict_parent = 9;
  uint32 max_attempts = 10;
}

message CreateBroadcastSessionResponse {
  BroadcastSessionStatus status = 1;
  string session_id = 2;
}
```

Add the Store Daemon service RPC:

```proto
rpc CreateBroadcastSession(CreateBroadcastSessionRequest) returns (CreateBroadcastSessionResponse);
```

Run:

```bash
bash tools/build_proto_python.sh
```

- [ ] **Step 3: Implement `DaemonCtl.create_broadcast_session()` and SDK store helper**

In `tensorcast/daemon_ctl.py`, add:

```python
def create_broadcast_session(
    self,
    *,
    session_id: str,
    artifact_id: str,
    epoch: int,
    fanout: int,
    target_worker_ids: list[str] | None = None,
    target_daemon_ids: list[str] | None = None,
    requested_view_id: str | None = None,
    root_replica_id: str | None = None,
    strict_parent: bool = True,
    max_attempts: int = 3,
    timeout_s: float = 30.0,
) -> store_daemon_pb2.CreateBroadcastSessionResponse:
    request = store_daemon_pb2.CreateBroadcastSessionRequest(
        session_id=session_id,
        artifact_id=artifact_id,
        requested_view_id=requested_view_id or "",
        epoch=int(epoch),
        fanout=int(fanout),
        root_replica_id=root_replica_id or "",
        strict_parent=bool(strict_parent),
        max_attempts=int(max_attempts),
    )
    request.target_worker_ids.extend(target_worker_ids or [])
    request.target_daemon_ids.extend(target_daemon_ids or [])
    return self._unary_call(
        self.stub_v2.CreateBroadcastSession,
        request,
        timeout=float(timeout_s),
        retries=1,
    )
```

In `tensorcast/api/store/__init__.py`, add `from dataclasses import dataclass` near the other standard-library imports, then add a small return dataclass:

```python
@dataclass(frozen=True, slots=True)
class BroadcastSessionHandle:
    session_id: str
```

and `Store.create_broadcast_session()` that calls `self._runtime.ensure_client().create_broadcast_session()` with the SDK method arguments and returns `BroadcastSessionHandle(session_id=response.session_id)`.

- [ ] **Step 4: Implement daemon forwarding**

Add a daemon service handler that maps daemon request fields to `global_store::CreateBroadcastSessionRequest` using the existing `GlobalStoreClient` stub. Return `BROADCAST_SESSION_STATUS_ERROR` when Global Store is disconnected or returns a non-OK status. Keep this handler thin; planning remains in Global Store.

- [ ] **Step 5: Run API tests and commit**

Run:

```bash
pytest tests/python/api/test_broadcast_session_api.py tests/python/api/test_prefetch_operation.py::test_prefetch_forwards_broadcast_context_hint -v
```

Expected: PASS.

Commit:

```bash
git add proto/tensorcast/daemon/v2/store_daemon.proto tensorcast/daemon_ctl.py tensorcast/api/store/__init__.py tests/python/api/test_broadcast_session_api.py daemon/service
git commit -m "feat(daemon): expose broadcast session creation"
```

### Task 7: End-To-End Regression And Documentation Updates

**Files:**
- Test: add `tests/python/global_store/test_broadcast_e2e.py`
- Modify: `tensorcast/global_store/README.md`
- Modify: `core/store/README.md` only if broadcast hint behavior changes public materialization semantics.
- Modify: `docs/designs/0117-control-plane-tree-broadcast-phase2.md` if implementation changes any accepted interface.

- [ ] **Step 1: Add lightweight Global Store E2E test**

Create `tests/python/global_store/test_broadcast_e2e.py`:

```python
from __future__ import annotations

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


def _register_worker(servicer, context, idx: int) -> str:
    response = servicer.RegisterWorker(
        global_store_pb2.RegisterWorkerRequest(
            daemon_id=f"daemon-e2e-{idx}",
            node_id=f"node-e2e-{idx}",
            node_address=f"10.30.0.{idx}",
            grpc_port=53000 + idx,
            p2p_port=54000 + idx,
            mem_pool_total_size=4096,
            mem_pool_available_size=4096,
        ),
        context,
    )
    assert response.status == global_store_pb2.STATUS_OK
    return response.worker_id


def _register_exportable_replica(servicer, context, worker_id: str, idx: int) -> str:
    request = global_store_pb2.RegisterReplicaRequest(
        artifact_id="mi2:model-e2e",
        worker_id=worker_id,
        max_concurrency=4,
    )
    request.memory_info.node_id = f"node-e2e-{idx}"
    request.memory_info.node_address = f"10.30.0.{idx}"
    request.memory_info.node_port = 54000 + idx
    request.memory_info.memory_size = 1024
    request.memory_info.memory_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
    request.memory_info.device_id = 0
    request.memory_info.transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    request.memory_info.transport.remote_memory_keys.append(f"rk-e2e-{idx}")
    request.memory_info.transport.buffer_sizes.append(1024)
    response = servicer.RegisterReplica(request, context)
    assert response.status == global_store_pb2.STATUS_OK
    return response.replica_id


def _request_broadcast_transport(servicer, context, session_id: str, worker_id: str, idx: int, request_id: str):
    request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id="mi2:model-e2e",
        source_node_id=f"node-e2e-{idx}",
        source_address=f"10.30.0.{idx}",
        source_port=54000 + idx,
        requester_worker_id=worker_id,
        request_id=request_id,
    )
    request.local_memory_info.memory_type = common_pb2.MemoryType.MEMORY_TYPE_GPU
    request.local_memory_info.device_id = 0
    request.broadcast.session_id = session_id
    request.broadcast.strict_parent = True
    response = servicer.RequestReplicaTransport(request, context)
    assert response.status == global_store_pb2.STATUS_OK
    return response


def test_tree_broadcast_promotes_first_child_to_second_layer_parent(servicer, test_context):
    root = _register_worker(servicer, test_context, 1)
    child1 = _register_worker(servicer, test_context, 2)
    child2 = _register_worker(servicer, test_context, 3)
    root_replica = _register_exportable_replica(servicer, test_context, root, 1)
    create = servicer.CreateBroadcastSession(
        global_store_pb2.CreateBroadcastSessionRequest(
            session_id="session-e2e",
            artifact_id="mi2:model-e2e",
            epoch=1,
            fanout=1,
            root_replica_id=root_replica,
            strict_parent=True,
            max_attempts=3,
            targets=[
                global_store_pb2.BroadcastTargetIdentity(worker_id=child1),
                global_store_pb2.BroadcastTargetIdentity(worker_id=child2),
            ],
        ),
        test_context,
    )
    assert create.status == global_store_pb2.STATUS_OK

    first = _request_broadcast_transport(servicer, test_context, "session-e2e", child1, 2, "request-child-1")
    assert first.remote_memory_info.node_id == "node-e2e-1"
    _register_exportable_replica(servicer, test_context, child1, 2)
    complete = servicer.CompleteReplicaTransport(
        global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=first.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        ),
        test_context,
    )
    assert complete.status == global_store_pb2.STATUS_OK

    second = _request_broadcast_transport(servicer, test_context, "session-e2e", child2, 3, "request-child-2")
    assert second.remote_memory_info.node_id in {"node-e2e-1", "node-e2e-2"}
```

- [ ] **Step 2: Run E2E and core regression suites**

Run:

```bash
pytest tests/python/global_store/test_broadcast_repository.py tests/python/global_store/test_broadcast_service.py tests/python/global_store/test_broadcast_rpc.py tests/python/global_store/test_broadcast_transport.py tests/python/global_store/test_broadcast_e2e.py -v
pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_daemon_ctl_broadcast_hint.py tests/python/api/test_broadcast_session_api.py -v
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
bazel test //core/store/materialization/control:materialize_orchestrator_reselection_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS.

- [ ] **Step 3: Update docs**

In `tensorcast/global_store/README.md`, add a short section under the transport scheduling area:

```markdown
### Broadcast Sessions

Broadcast sessions coordinate strict tree dissemination for model-weight prefetch. A session records the artifact/view/epoch, target workers, fanout, and parent-child edges. Broadcast-tagged `RequestReplicaTransport` calls resolve to the parent replica assigned by the active edge; untagged requests continue to use group dispatch or ordinary source selection.
```

Update `docs/designs/0117-control-plane-tree-broadcast-phase2.md` if implementation changes a proto field name, state name, or completion rule from the accepted design.

- [ ] **Step 4: Final verification and commit**

Run:

```bash
ruff check tensorcast/global_store tensorcast/api tests/python/global_store tests/python/api
ruff format tensorcast/global_store tensorcast/api tests/python/global_store tests/python/api
pytest tests/python/global_store/test_broadcast_repository.py tests/python/global_store/test_broadcast_service.py tests/python/global_store/test_broadcast_rpc.py tests/python/global_store/test_broadcast_transport.py tests/python/global_store/test_broadcast_e2e.py -v
pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_daemon_ctl_broadcast_hint.py tests/python/api/test_broadcast_session_api.py -v
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
bazel test //core/store/materialization/control:materialize_orchestrator_reselection_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS for all commands.

Commit:

```bash
git add tests/python/global_store/test_broadcast_e2e.py tensorcast/global_store/README.md docs/designs/0117-control-plane-tree-broadcast-phase2.md
git commit -m "test: cover broadcast tree dissemination"
```

# Self-Review Checklist

- [ ] Schema changes are represented in `schema.sql` and migration `0019_broadcast_sessions.py`.
- [ ] Global Store broadcast RPCs never move artifact bytes.
- [ ] SDK broadcast session creation goes through Store Daemon, not directly to Global Store.
- [ ] Broadcast source selection uses edge-assigned parents only in strict mode.
- [ ] Parent selection still enforces heartbeat, accepting-new-requests, capacity, export state, remote memory keys, and buffer sizes.
- [ ] `FAILED`, `EXPIRED`, and `CANCELLED` transport outcomes do not advance tree progress.
- [ ] Broadcast success waits for child replica visibility after registration.
- [ ] Existing unhinted materialization and Phase 1 group dispatch tests still pass.
