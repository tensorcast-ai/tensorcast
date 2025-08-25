#  Copyright (c) 2025, StepCast Team.

"""Test all Web UI endpoints with gRPC backend."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from scstore.global_store.webui_backend.api import (
    get_summary,
    list_workers,
    get_worker,
    list_replicas,
    get_replica,
    list_artifacts,
    get_artifact,
    list_nodes,
    list_transports,
)
from scstore.global_store.webui_backend.models import MemoryType
from scstore.proto import global_store_pb2
from scstore.global_store.webui_backend.grpc_client import WorkerInfoWrapper


def create_mock_client():
    """Create a mock gRPC client with test data."""
    client = AsyncMock()

    # Mock workers
    worker1 = WorkerInfoWrapper(global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
        worker_id="worker-1",
        node_id="node-1",
        node_address="192.168.1.1",
        grpc_port=50052,
        p2p_port=50053,
        mem_pool_total_size=10737418240,  # 10GB
        mem_pool_available_size=5368709120,  # 5GB
        accepting_new_requests=True,
        last_heartbeat_timestamp=1234567890,
    ))
    worker2 = WorkerInfoWrapper(global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
        worker_id="worker-2",
        node_id="node-2",
        node_address="192.168.1.2",
        grpc_port=50052,
        p2p_port=50053,
        mem_pool_total_size=10737418240,
        mem_pool_available_size=8589934592,  # 8GB
        accepting_new_requests=True,
        last_heartbeat_timestamp=1234567891,
    ))
    client.list_active_workers.return_value = [worker1, worker2]

    # Mock replicas
    replica1 = global_store_pb2.MemoryInfo(
        node_id="node-1",
        node_address="192.168.1.1",
        node_port=50052,
        memory_size=1073741824,  # 1GB
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    replica2 = global_store_pb2.MemoryInfo(
        node_id="node-2",
        node_address="192.168.1.2",
        node_port=50052,
        memory_size=2147483648,  # 2GB
        memory_type=global_store_pb2.MemoryType.RAM,
        device_id=0,
    )
    client.list_replicas.return_value = {
        "model1": [replica1],
        "model2": [replica2],
    }

    # Mock artifact info: WebUI client returns list[MemoryInfo] for a artifact_id
    client.get_artifact_info.return_value = [replica1]

    # Mock summary stats
    client.get_summary_stats.return_value = {
        "total_workers": 2,
        "active_workers": 2,
        "total_artifacts": 2,
        "total_replicas": 2,
        "gpu_replicas": 1,
        "ram_replicas": 1,
        "disk_replicas": 0,
        "active_transports": 0,
    }

    return client


@pytest.mark.asyncio
async def test_summary_endpoint():
    """Test /api/summary endpoint."""
    client = create_mock_client()
    response = await get_summary(client)

    assert response.data.total_workers == 2
    assert response.data.active_workers == 2
    assert response.data.total_artifacts == 2
    assert response.data.total_replicas == 2
    assert response.data.total_memory_bytes == 21474836480  # 20GB total


@pytest.mark.asyncio
async def test_workers_endpoints():
    """Test worker-related endpoints."""
    client = create_mock_client()

    # Test list workers
    response = await list_workers(client, include_unavailable=False, page=1, page_size=50)
    assert len(response.data) == 2
    assert response.data[0]["worker_id"] == "worker-1"
    assert response.meta is not None
    assert response.meta["total_count"] == 2

    # Test get specific worker
    response = await get_worker("worker-1", client)
    assert response.data["worker_id"] == "worker-1"
    assert response.data["node_id"] == "node-1"
    assert response.data["replica_count"] == 1


@pytest.mark.asyncio
async def test_replica_endpoints():
    """Test replica-related endpoints."""
    client = create_mock_client()

    # Test list replicas
    response = await list_replicas(
        client,
        artifact_id=None,
        node_id=None,
        memory_type=None,
        worker_id=None,
        page=1,
        page_size=100
    )
    assert len(response.data) == 2
    assert response.data[0]["memory_type"] == "GPU"
    assert response.data[1]["memory_type"] == "RAM"

    # Test list replicas with filters
    response = await list_replicas(
        client,
        artifact_id="model1",
        node_id=None,
        memory_type=MemoryType.GPU,
        worker_id=None,
        page=1,
        page_size=100
    )
    assert len(response.data) == 1
    assert response.data[0]["artifact_id"] == "model1"

    # Test get specific replica
    response = await get_replica("replica-0", client)
    assert response.data["replica_id"] == "replica-0"
    assert response.data["artifact_id"] == "model1"


@pytest.mark.asyncio
async def test_artifact_endpoints():
    """Test artifact-related endpoints."""
    client = create_mock_client()

    # Test list artifacts
    response = await list_artifacts(client)
    assert len(response.data) == 2
    assert response.data[0]["artifact_id"] == "model1"
    assert response.data[0]["gpu_replicas"] == 1
    assert response.data[1]["artifact_id"] == "model2"
    assert response.data[1]["ram_replicas"] == 1

    # Test get specific artifact
    response = await get_artifact("model1", client)
    assert response.data["artifact_id"] == "model1"
    assert response.data["replica_count"] == 1
    assert len(response.data["nodes"]) == 1


@pytest.mark.asyncio
async def test_nodes_endpoint():
    """Test /api/nodes endpoint."""
    client = create_mock_client()
    response = await list_nodes(client)

    assert len(response.data) == 2
    assert response.data[0]["node_id"] == "node-1"
    assert response.data[0]["replica_count"] == 1
    assert response.data[0]["gpu_memory"] == 1073741824
    assert response.data[1]["node_id"] == "node-2"
    assert response.data[1]["ram_memory"] == 2147483648


@pytest.mark.asyncio
async def test_transports_endpoint():
    """Test /api/transports endpoint."""
    client = create_mock_client()

    # This endpoint returns empty data due to proto limitations
    response = await list_transports(
        client,
        status=None,
        artifact_id=None,
        page=1,
        page_size=50
    )

    assert response.data == []
    assert response.meta is not None
    assert response.meta["total_count"] == 0


def test_all_endpoints_use_grpc():
    """Verify all endpoints are refactored to use gRPC."""
    import inspect
    from scstore.global_store.webui_backend import api

    # Get all functions in the api module
    functions = inspect.getmembers(api, inspect.isfunction)

    # Check that no function has 'db' or 'DbConnection' parameters
    for name, func in functions:
        if name.startswith("_"):
            continue

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Should not have 'db' parameter
        assert "db" not in param_names, f"Function {name} still has 'db' parameter"

        # Should have 'client' parameter for API endpoints
        if name in ["get_summary", "list_workers", "get_worker", "list_replicas",
                    "get_replica", "list_artifacts", "get_artifact", "list_nodes", "list_transports"]:
            assert "client" in param_names, f"Function {name} should have 'client' parameter"


if __name__ == "__main__":
    print("Testing all Web UI endpoints with gRPC...")

    # Run all tests
    asyncio.run(test_summary_endpoint())
    print("✓ Summary endpoint works")

    asyncio.run(test_workers_endpoints())
    print("✓ Worker endpoints work")

    asyncio.run(test_replica_endpoints())
    print("✓ Replica endpoints work")

    asyncio.run(test_artifact_endpoints())
    print("✓ Artifact endpoints work")

    asyncio.run(test_nodes_endpoint())
    print("✓ Nodes endpoint works")

    asyncio.run(test_transports_endpoint())
    print("✓ Transports endpoint works (returns empty due to proto limitations)")

    test_all_endpoints_use_grpc()
    print("✓ All endpoints use gRPC client")

    print("\nAll tests passed!")