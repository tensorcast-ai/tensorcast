#  Copyright (c) 2025-2026, TensorCast Team.

"""
Tests for the artifact transport functionality of the GlobalStoreServicer.
"""

import concurrent.futures
import time
import uuid

import pytest
from google.protobuf import duration_pb2

from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.global_store.config.settings import get_config, set_config
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class _MockContext:
    """Mock gRPC ServicerContext for testing"""

    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


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
    return _MockContext()


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


def test_transport_concurrency(servicer, test_context, memory_info):
    """Test that the transport functionality respects the max_concurrency limit"""
    # First, register a worker
    worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=memory_info.node_id,
        node_address=memory_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{memory_info.node_id}",
    )
    worker_response = servicer.RegisterWorker(worker_request, test_context)
    assert worker_response.status == global_store_pb2.Status.STATUS_OK
    worker_id = worker_response.worker_id

    # Register a artifact replica with max_concurrency of 2
    artifact_id = "concurrency_test_artifact"
    max_concurrency = 2

    register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=memory_info,
        max_concurrency=max_concurrency,
        worker_id=worker_id,
    )
    register_response = servicer.RegisterReplica(register_request, test_context)
    assert register_response.status == global_store_pb2.Status.STATUS_OK
    assert register_response.replica_id

    # Request transports up to max_concurrency
    transports = []
    for i in range(max_concurrency):
        transport_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            local_memory_info=memory_info,
            source_node_id=f"source_node_{i}",
            source_address="192.168.1.2",
            source_port=9000,
            request_id=f"transport-concurrency-{i}",
        )
        response = servicer.RequestReplicaTransport(transport_request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_OK
        assert response.remote_memory_info.replica_id == register_response.replica_id
        transports.append(response.transport_id)

    # Try to request another transport, which should fail or time out
    transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=memory_info,
        wait_timeout_dur=duration_pb2.Duration(seconds=0, nanos=100_000_000),
        source_node_id="source_node_overflow",
        source_address="192.168.1.2",
        source_port=9000,
        request_id="transport-concurrency-overflow",
    )

    response = servicer.RequestReplicaTransport(transport_request, test_context)
    assert response.status == global_store_pb2.Status.STATUS_TIMED_OUT

    # Complete one transport
    complete_request = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=transports[0],
        outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
    )
    complete_response = servicer.CompleteReplicaTransport(
        complete_request, test_context
    )
    assert complete_response.status == global_store_pb2.Status.STATUS_OK

    # Now we should be able to request another transport
    transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=memory_info,
        source_node_id="source_node_new",
        source_address="192.168.1.2",
        source_port=9000,
        request_id="transport-concurrency-new",
    )

    response = servicer.RequestReplicaTransport(transport_request, test_context)
    assert response.status == global_store_pb2.Status.STATUS_OK


def test_transport_wait_timeout(servicer, test_context, memory_info):
    """Test that the transport request respects the wait_timeout parameter"""
    # First, register a worker
    worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=memory_info.node_id,
        node_address=memory_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{memory_info.node_id}",
    )
    worker_response = servicer.RegisterWorker(worker_request, test_context)
    assert worker_response.status == global_store_pb2.Status.STATUS_OK
    worker_id = worker_response.worker_id

    # Register a artifact replica with max_concurrency of 1
    artifact_id = "timeout_test_artifact"

    register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=memory_info,
        max_concurrency=1,
        worker_id=worker_id,
    )
    servicer.RegisterReplica(register_request, test_context)

    # Request the first transport
    transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=memory_info,
        source_node_id="source_node_1",
        source_address="192.168.1.2",
        source_port=9000,
        request_id="transport-timeout-first",
    )

    response = servicer.RequestReplicaTransport(transport_request, test_context)
    assert response.status == global_store_pb2.Status.STATUS_OK
    transport_id = response.transport_id

    # Try to request another transport with a very short timeout
    start_time = time.time()

    transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=memory_info,
        wait_timeout_dur=duration_pb2.Duration(nanos=100_000_000),  # 100ms timeout
        source_node_id="source_node_2",
        source_address="192.168.1.2",
        source_port=9000,
        request_id="transport-timeout-second",
    )

    response = servicer.RequestReplicaTransport(transport_request, test_context)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    assert response.status == global_store_pb2.Status.STATUS_TIMED_OUT
    assert elapsed_ms >= 100  # Should have waited at least the timeout period

    # Complete the first transport
    complete_request = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=transport_id,
        outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
    )
    servicer.CompleteReplicaTransport(complete_request, test_context)


def test_concurrent_transport_requests(servicer, test_context, memory_info):
    """Test handling multiple concurrent transport requests"""
    # First, register a worker
    worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=memory_info.node_id,
        node_address=memory_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{memory_info.node_id}",
    )
    worker_response = servicer.RegisterWorker(worker_request, test_context)
    assert worker_response.status == global_store_pb2.Status.STATUS_OK
    worker_id = worker_response.worker_id

    # Register a artifact replica with max_concurrency of 3
    artifact_id = "concurrent_test_artifact"
    max_concurrency = 3

    register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=memory_info,
        max_concurrency=max_concurrency,
        worker_id=worker_id,
    )
    servicer.RegisterReplica(register_request, test_context)

    # Function to request a transport
    def request_transport(i):
        local_context = _MockContext()
        transport_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=5),  # 5 second timeout
            source_node_id=f"source_node_{i}",
            source_address="192.168.1.2",
            source_port=9000,
            request_id=f"transport-concurrent-{i}",
        )

        response = servicer.RequestReplicaTransport(transport_request, local_context)
        return response

    # Request transports concurrently with a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(request_transport, i) for i in range(5)]
        responses = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # Verify that we got exactly max_concurrency successful responses
    successful = [r for r in responses if r.status == global_store_pb2.Status.STATUS_OK]
    timed_out = [
        r for r in responses if r.status == global_store_pb2.Status.STATUS_TIMED_OUT
    ]

    assert len(successful) == max_concurrency
    assert len(timed_out) == 5 - max_concurrency


def test_transport_memory_type_priority(servicer, test_context):
    """Test that transport requests prioritize memory types in order: GPU > RAM > DISK"""
    artifact_id = "priority_test_artifact"

    # Create memory infos with different types
    gpu_info = common_pb2.MemoryInfo(
        node_id=str(uuid.uuid4()),
        node_address="192.168.1.1",
        node_port=8000,
        memory_size=1000000000,
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
        device_id=0,
    )
    gpu_transport = gpu_info.transport
    gpu_transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    gpu_transport.export_generation = 1
    gpu_transport.remote_memory_keys.append("gpu_key")
    gpu_transport.buffer_sizes.append(gpu_info.memory_size)

    ram_info = common_pb2.MemoryInfo(
        node_id=str(uuid.uuid4()),
        node_address="192.168.1.2",
        node_port=8000,
        memory_size=1000000000,
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
        device_id=0,
    )
    ram_transport = ram_info.transport
    ram_transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    ram_transport.export_generation = 1
    ram_transport.remote_memory_keys.append("ram_key")
    ram_transport.buffer_sizes.append(ram_info.memory_size)

    disk_info = common_pb2.MemoryInfo(
        node_id=str(uuid.uuid4()),
        node_address="192.168.1.3",
        node_port=8000,
        memory_size=1000000000,
        memory_type=common_pb2.MemoryType.MEMORY_TYPE_DISK,
        device_id=0,
    )
    disk_transport = disk_info.transport
    disk_transport.export_state = (
        common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
    )
    disk_transport.export_generation = 1
    disk_transport.remote_memory_keys.append("disk_key")
    disk_transport.buffer_sizes.append(disk_info.memory_size)

    # Register workers first
    gpu_worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=gpu_info.node_id,
        node_address=gpu_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{gpu_info.node_id}",
    )
    gpu_worker_response = servicer.RegisterWorker(gpu_worker_request, test_context)
    assert gpu_worker_response.status == global_store_pb2.Status.STATUS_OK
    gpu_worker_id = gpu_worker_response.worker_id

    ram_worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=ram_info.node_id,
        node_address=ram_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{ram_info.node_id}",
    )
    ram_worker_response = servicer.RegisterWorker(ram_worker_request, test_context)
    assert ram_worker_response.status == global_store_pb2.Status.STATUS_OK
    ram_worker_id = ram_worker_response.worker_id

    disk_worker_request = global_store_pb2.RegisterWorkerRequest(
        node_id=disk_info.node_id,
        node_address=disk_info.node_address,
        grpc_port=8000,
        p2p_port=8001,
        mem_pool_total_size=10 * 1024 * 1024 * 1024,  # 10GB
        mem_pool_available_size=10 * 1024 * 1024 * 1024,  # 10GB
        daemon_id=f"daemon_{disk_info.node_id}",
    )
    disk_worker_response = servicer.RegisterWorker(disk_worker_request, test_context)
    assert disk_worker_response.status == global_store_pb2.Status.STATUS_OK
    disk_worker_id = disk_worker_response.worker_id

    # Register replicas with different memory types
    gpu_register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=gpu_info,
        max_concurrency=1,
        worker_id=gpu_worker_id,
    )
    servicer.RegisterReplica(gpu_register_request, test_context)

    ram_register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=ram_info,
        max_concurrency=1,
        worker_id=ram_worker_id,
    )
    servicer.RegisterReplica(ram_register_request, test_context)

    disk_register_request = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact_id,
        mem_info=disk_info,
        max_concurrency=1,
        worker_id=disk_worker_id,
    )
    servicer.RegisterReplica(disk_register_request, test_context)

    # Test 1: Request using DISK memory type should still get GPU (highest priority)
    disk_transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=disk_info,
        source_node_id="source_node_disk",
        source_address="192.168.1.4",
        source_port=9000,
        request_id="transport-priority-disk",
    )

    disk_response = servicer.RequestReplicaTransport(
        disk_transport_request, test_context
    )
    assert disk_response.status == global_store_pb2.Status.STATUS_OK
    assert list(disk_response.remote_memory_info.transport.remote_memory_keys) == [
        "gpu_key"
    ]

    # Complete the transport to release the GPU replica
    complete_request = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=disk_response.transport_id,
        outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
    )
    servicer.CompleteReplicaTransport(complete_request, test_context)

    # Test 2: Request using RAM memory type should also get GPU
    ram_transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=ram_info,
        source_node_id="source_node_ram",
        source_address="192.168.1.4",
        source_port=9000,
        request_id="transport-priority-ram",
    )

    ram_response = servicer.RequestReplicaTransport(ram_transport_request, test_context)
    assert ram_response.status == global_store_pb2.Status.STATUS_OK
    assert list(ram_response.remote_memory_info.transport.remote_memory_keys) == [
        "gpu_key"
    ]

    # Complete the transport to release the GPU replica
    complete_request = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=ram_response.transport_id,
        outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
    )
    servicer.CompleteReplicaTransport(complete_request, test_context)

    # Test 3: Let's make GPU unavailable and see if RAM is selected next
    # Update the GPU replica to be unavailable
    servicer.connection.execute(
        """
        UPDATE artifact_replicas
        SET is_available = FALSE
        WHERE memory_type = 'GPU' AND artifact_id = ?
        """,
        [artifact_id],
    )

    # Now try to request with any memory type, should get RAM (next priority)
    gpu_transport_request = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact_id,
        local_memory_info=gpu_info,
        source_node_id="source_node_gpu",
        source_address="192.168.1.4",
        source_port=9000,
        request_id="transport-priority-gpu",
    )

    response = servicer.RequestReplicaTransport(gpu_transport_request, test_context)
    assert response.status == global_store_pb2.Status.STATUS_OK
    assert list(response.remote_memory_info.transport.remote_memory_keys) == ["ram_key"]
