#  Copyright (c) 2025-2026, TensorCast Team.

"""Shared fixtures and utilities for Global Store tests."""

import os
import tempfile
import uuid

import duckdb
import pytest

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.db_utils import init_db
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.global_store.models import MemoryType, Replica, Worker
from tensorcast.global_store.repositories import (
    AssemblyAttemptRepository,
    AssemblyLayoutBindingRepository,
    AssemblyReadinessCutRepository,
    AssemblySlotOccupancyRepository,
    InstanceRepository,
    LayoutSpecRepository,
    LeafRepository,
    PendingTransportRequestRepository,
    ProofRepository,
    ReplicaRepository,
    TransportRepository,
    ViewCoverageRepository,
    ViewRepository,
    WorkerRepository,
)
from tensorcast.global_store.services import (
    ArtifactService,
    InstanceService,
    TransportService,
    ViewStateService,
    WorkerService,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

# =============================================================================
# Mock classes
# =============================================================================


class MockContext:
    """Enhanced mock gRPC ServicerContext for testing with full interface."""

    def __init__(self):
        self.code = None
        self.details = None
        self.invocation_metadata = []  # type: ignore
        self._cancelled = False
        self._time_remaining = float("inf")
        self._peer = "ipv4:127.0.0.1:50051"
        self._aborted = False
        self._abort_code = None
        self._abort_details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def abort(self, code, details):
        """Abort the RPC with given status code and details."""
        self._aborted = True
        self._abort_code = code
        self._abort_details = details
        self.code = code
        self.details = details
        # In real gRPC, this would raise an exception
        raise Exception(f"RPC aborted with {code}: {details}")

    def is_active(self):
        """Check if the RPC is still active."""
        return not self._cancelled and not self._aborted

    def time_remaining(self):
        """Get time remaining for the RPC."""
        return self._time_remaining

    def cancel(self):
        """Cancel the RPC."""
        self._cancelled = True

    def add_callback(self, callback):
        """Add callback for RPC completion (no-op in mock)."""
        pass

    def invocation_metadata(self):
        """Get invocation metadata."""
        return self.invocation_metadata

    def peer(self):
        """Get peer address."""
        return self._peer

    def peer_identities(self):
        """Get peer identities (empty for mock)."""
        return None

    def peer_identity_key(self):
        """Get peer identity key (empty for mock)."""
        return None

    def auth_context(self):
        """Get auth context (empty for mock)."""
        return {}

    def send_initial_metadata(self, initial_metadata):
        """Send initial metadata (no-op in mock)."""
        pass

    def set_trailing_metadata(self, trailing_metadata):
        """Set trailing metadata (no-op in mock)."""
        pass

    def disable_next_message_compression(self):
        """Disable compression for next message (no-op in mock)."""
        pass


# =============================================================================
# Fixtures for gRPC service testing
# =============================================================================


@pytest.fixture
def servicer():
    """Create an in-memory GlobalStoreServicer for testing"""
    try:
        get_config()
    except RuntimeError:
        set_config(GlobalStoreConfig())
    return GlobalStoreServicer()


@pytest.fixture
def test_context():
    """Create a mock gRPC ServicerContext"""
    return MockContext()


@pytest.fixture
def memory_info():
    """Create a sample memory info for testing"""
    info = common_pb2.MemoryInfo(
        node_id=str(uuid.uuid4()),
        node_address="192.168.1.1",
        node_port=8000,
        memory_size=1000000000,
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
        device_id=0,
    )
    transport = info.transport
    transport.export_state = common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    transport.export_generation = 1
    transport.remote_memory_keys.append("test_key")
    transport.buffer_sizes.append(info.memory_size)
    return info


@pytest.fixture
def registered_worker(servicer, test_context):
    """Register a worker and return the worker_id for testing"""
    worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id="test_node_1",
        node_address="192.168.1.1",
        grpc_port=8001,
        p2p_port=8002,
        mem_pool_total_size=10000000000,
        mem_pool_available_size=8000000000,
        daemon_id="daemon_test_node_1",
    )
    worker_response = servicer.RegisterWorker(worker_request, test_context)
    return worker_response.worker_id


# =============================================================================
# Fixtures for refactored components testing
# =============================================================================


@pytest.fixture
def db_connection():
    """Create in-memory DuckDB connection for testing."""
    conn = duckdb.connect()
    cursor = conn.cursor()
    init_db(cursor)
    return conn


@pytest.fixture
def repositories(db_connection):
    """Create repository instances."""
    return {
        "replica": ReplicaRepository(db_connection),
        "transport": TransportRepository(db_connection),
        "pending_transport_request": PendingTransportRequestRepository(db_connection),
        "view": ViewRepository(db_connection),
        "coverage": ViewCoverageRepository(db_connection),
        "leaf": LeafRepository(db_connection),
        "worker": WorkerRepository(db_connection),
        "instance": InstanceRepository(db_connection),
        "layout_spec": LayoutSpecRepository(db_connection),
        "assembly_attempt": AssemblyAttemptRepository(db_connection),
        "assembly_layout_binding": AssemblyLayoutBindingRepository(db_connection),
        "assembly_readiness_cut": AssemblyReadinessCutRepository(db_connection),
        "assembly_slot_occupancy": AssemblySlotOccupancyRepository(db_connection),
        "proof": ProofRepository(db_connection),
    }


@pytest.fixture
def services(tmp_path, repositories):
    """Create service instances with a file-based test config."""
    # Initialize from a minimal YAML config to avoid env reliance
    cfg_path = tmp_path / "global_store_test.yaml"
    cfg_path.write_text(
        """
database:
  db_file: ""  # use in-memory
server:
  listen:
    host: "127.0.0.1"
    port: 50051
  max_workers: 10
worker_policy:
  heartbeat_timeout: "30s"
  cleanup_interval: "60s"
  default_heartbeat_interval: "5s"
observability:
  logging:
    level: LOG_LEVEL_INFO
  otel:
    enabled: false
        """,
        encoding="utf-8",
    )
    set_config(GlobalStoreConfig.from_file(str(cfg_path)))
    return {
        "artifact": ArtifactService(repositories["replica"]),
        "transport": TransportService(
            repositories["replica"],
            repositories["transport"],
            repositories["pending_transport_request"],
        ),
        "worker": WorkerService(repositories["worker"], repositories["replica"]),
        "instance": InstanceService(repositories["instance"], repositories["worker"]),
        "view_state": ViewStateService(
            repositories["view"],
            repositories["leaf"],
            repositories["coverage"],
            repositories["layout_spec"],
            repositories["assembly_layout_binding"],
            repositories["proof"],
        ),
    }


# =============================================================================
# Utility fixtures
# =============================================================================


@pytest.fixture
def sample_worker():
    """Create a sample Worker instance for testing."""
    return Worker(
        worker_id="test_worker",
        daemon_id="daemon_test_worker",
        node_id="node1",
        node_address="192.168.1.1",
        grpc_port=50051,
        p2p_port=50052,
        mem_pool_total_size=1024,
        mem_pool_available_size=1024,
    )


@pytest.fixture
def sample_replica():
    """Create a sample Replica instance for testing."""
    return Replica(
        artifact_id="test_artifact",
        node_id="node1",
        node_address="192.168.1.1",
        node_port=8080,
        memory_size=1024,
        memory_type=MemoryType.GPU,
        device_id=0,
        worker_id="worker1",
    )


@pytest.fixture
def temp_db_file():
    """Create a temporary database file for persistence testing."""
    # Generate a temporary file path but don't create the file
    db_file = tempfile.gettempdir() + "/test_duckdb_" + str(uuid.uuid4()) + ".db"
    yield db_file
    # Clean up the temporary file
    if os.path.exists(db_file):
        os.unlink(db_file)


# =============================================================================
# Test utilities
# =============================================================================


def create_test_replicas(num_replicas, artifact_id="test_artifact", memory_types=None):
    """Create multiple test replicas with varied configurations."""
    if memory_types is None:
        memory_types = [MemoryType.GPU, MemoryType.RAM, MemoryType.DISK]

    return [
        Replica(
            artifact_id=artifact_id,
            node_id=f"node{i}",
            node_address=f"192.168.1.{i + 1}",
            node_port=8080 + i,
            memory_size=1024 * (i + 1),
            memory_type=memory_types[i % len(memory_types)],
            device_id=i if memory_types[i % len(memory_types)] == MemoryType.GPU else 0,
            worker_id=f"worker{i}",
            max_concurrency=5,
        )
        for i in range(num_replicas)
    ]


def create_test_workers(num_workers):
    """Create multiple test workers."""
    return [
        Worker(
            worker_id=f"worker{i}",
            daemon_id=f"daemon_worker{i}",
            node_id=f"node{i}",
            node_address=f"192.168.1.{i + 1}",
            grpc_port=50051 + i,
            p2p_port=50052 + i,
            mem_pool_total_size=10240 * (i + 1),
            mem_pool_available_size=8192 * (i + 1),
        )
        for i in range(num_workers)
    ]
