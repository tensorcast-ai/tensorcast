#  Copyright (c) 2025, StepCast Team.

"""Comprehensive tests for the Web UI gRPC client.

This test module spawns a test gRPC server in a background thread
and tests all client functionality against it.
"""

import asyncio
import threading
import time
from concurrent import futures
from dataclasses import dataclass

import grpc
import pytest
import pytest_asyncio

from scstore.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
    close_global_store_client,
    get_global_store_client,
)
from scstore.proto import global_store_pb2, global_store_pb2_grpc


class MockGlobalModelStoreServicer(global_store_pb2_grpc.GlobalModelStoreServicer):
    """Test implementation of Global Store gRPC service."""

    def __init__(self):
        """Initialize test servicer with mock data."""
        # Mock workers
        self.workers = [
            global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                worker_id="worker-1",
                node_id="node-1",
                node_address="192.168.1.10",
                grpc_port=50052,
                p2p_port=60052,
                mem_pool_total_size=10737418240,  # 10GB
                mem_pool_available_size=5368709120,  # 5GB
                accepting_new_requests=True,
                last_heartbeat_timestamp=int(time.time()),
            ),
            global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                worker_id="worker-2",
                node_id="node-2",
                node_address="192.168.1.11",
                grpc_port=50053,
                p2p_port=60053,
                mem_pool_total_size=10737418240,  # 10GB
                mem_pool_available_size=8589934592,  # 8GB
                accepting_new_requests=False,
                last_heartbeat_timestamp=int(time.time()) - 10,
            ),
        ]

        # Mock model replicas
        self.replicas = {
            "model-1": [
                global_store_pb2.MemoryInfo(
                    memory_type=global_store_pb2.MemoryType.GPU,
                    memory_size=1073741824,  # 1GB
                    device_id=0,
                    node_id="node-1",
                    node_address="192.168.1.10",
                    node_port=60052,
                ),
                global_store_pb2.MemoryInfo(
                    memory_type=global_store_pb2.MemoryType.RAM,
                    memory_size=1073741824,  # 1GB
                    device_id=0,  # device_id must be non-negative
                    node_id="node-2",
                    node_address="192.168.1.11",
                    node_port=60053,
                ),
            ],
            "model-2": [
                global_store_pb2.MemoryInfo(
                    memory_type=global_store_pb2.MemoryType.DISK,
                    memory_size=2147483648,  # 2GB
                    device_id=0,  # device_id must be non-negative
                    node_id="node-1",
                    node_address="192.168.1.10",
                    node_port=60052,
                ),
            ],
        }

        # Track RPC calls for testing
        self.call_count = {}
        self.should_fail = False
        self.fail_count = 0

    def reset_stats(self):
        """Reset call statistics."""
        self.call_count = {}
        self.should_fail = False
        self.fail_count = 0

    def set_failure_mode(self, fail_count: int = 1):
        """Set the servicer to fail the next N requests."""
        self.should_fail = True
        self.fail_count = fail_count

    def _maybe_fail(self, context, method_name: str):
        """Conditionally fail based on failure mode."""
        self.call_count[method_name] = self.call_count.get(method_name, 0) + 1

        if self.should_fail and self.fail_count > 0:
            self.fail_count -= 1
            context.abort(grpc.StatusCode.UNAVAILABLE, "Simulated failure")

    def ListActiveWorkers(self, request, context):
        """List active workers."""
        self._maybe_fail(context, "ListActiveWorkers")

        workers = self.workers
        if not request.include_unavailable:
            workers = [w for w in workers if w.accepting_new_requests]

        return global_store_pb2.ListActiveWorkersResponse(workers=workers)

    def ListModelReplicas(self, request, context):
        """List model replicas with optional filters."""
        self._maybe_fail(context, "ListModelReplicas")

        # Apply filters
        result = {}
        for model_id, replicas in self.replicas.items():
            # Filter by content-addressed model_id (tests use simple names)
            if request.HasField("model_id") and request.model_id and request.model_id != model_id:
                continue

            filtered_replicas = []
            for replica in replicas:
                # Filter by node_id
                if request.node_id and replica.node_id != request.node_id:
                    continue

                # Filter by memory_type
                if request.HasField("memory_type") and replica.memory_type != request.memory_type:
                    continue

                # Filter by device_id
                if request.HasField("device_id") and replica.device_id != request.device_id:
                    continue

                filtered_replicas.append(replica)

            if filtered_replicas:
                result[model_id] = global_store_pb2.MemoryInfoList(
                    list=filtered_replicas
                )

        return global_store_pb2.ListModelReplicasResponse(model_replicas=result)

    def GetModelInfoById(self, request, context):
        """Get model information by content-addressed model_id."""
        self._maybe_fail(context, "GetModelInfoById")

        if request.model_id in self.replicas:
            return global_store_pb2.GetModelInfoByIdResponse(
                status=global_store_pb2.Status.OK,
                replicas=self.replicas[request.model_id],
            )
        else:
            return global_store_pb2.GetModelInfoByIdResponse(
                status=global_store_pb2.Status.NOT_FOUND,
            )

    # Add other required RPCs with minimal implementation
    def RegisterModelReplica(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def UpdateModelReplica(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def UnregisterModelReplica(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def RequestModelReplicaTransport(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def CompleteModelReplicaTransport(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def RegisterWorker(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def WorkerHeartbeat(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def UnregisterWorker(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def SynchronizeWorkerState(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def RequestFullStateSync(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Not implemented in test")

    def HealthCheck(self, request, context):
        """Health check endpoint for testing."""
        self._maybe_fail(context, "HealthCheck")
        return global_store_pb2.HealthCheckResponse(
            status=global_store_pb2.Status.OK
        )


@dataclass
class MockServer:
    """Test gRPC server wrapper."""
    server: grpc.Server
    port: int
    servicer: MockGlobalModelStoreServicer
    thread: threading.Thread


def start_test_server(port: int = 0) -> MockServer:
    """Start a test gRPC server in a background thread.

    Parameters
    ----------
    port : int
        Port to bind to (0 for automatic)

    Returns
    -------
    MockServer
        Server information including actual port
    """
    servicer = MockGlobalModelStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    global_store_pb2_grpc.add_GlobalModelStoreServicer_to_server(servicer, server)

    # Bind to port (0 means choose automatically)
    actual_port = server.add_insecure_port(f'127.0.0.1:{port}')

    # Start server
    server.start()

    # Create a dummy thread since we're not blocking
    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()

    # Wait for server to be ready
    time.sleep(0.5)

    # Verify server is running
    channel = grpc.insecure_channel(f'127.0.0.1:{actual_port}')
    try:
        grpc.channel_ready_future(channel).result(timeout=2)
    except Exception as e:
        server.stop(grace=0)
        raise RuntimeError(f"Failed to start test server: {e}") from e
    finally:
        channel.close()

    return MockServer(
        server=server,
        port=actual_port,
        servicer=servicer,
        thread=thread,
    )


def stop_test_server(test_server: MockServer):
    """Stop the test gRPC server."""
    test_server.server.stop(grace=1)
    test_server.server.wait_for_termination(timeout=2)


@pytest.fixture(scope="function")
def test_server():
    """Fixture that provides a test gRPC server."""
    server = start_test_server()
    yield server
    stop_test_server(server)


@pytest_asyncio.fixture
async def client(test_server):
    """Fixture that provides a connected gRPC client."""
    config = GlobalStoreClientConfig(
        host="127.0.0.1",
        port=test_server.port,
        max_retries=3,
        retry_delay=0.1,
        timeout=5.0,
    )
    client = GlobalStoreClient(config)
    await client.connect()
    yield client
    await client.close()


class TestGlobalStoreClient:
    """Test cases for GlobalStoreClient."""

    @pytest.mark.asyncio
    async def test_connection(self, test_server):
        """Test client connection to server."""
        config = GlobalStoreClientConfig(host="127.0.0.1", port=test_server.port)
        client = GlobalStoreClient(config)

        # Test connection
        await client.connect()
        assert client._channel is not None
        assert client._stub is not None

        # Test close
        await client.close()
        assert client._channel is None
        assert client._stub is None

    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test client behavior with connection failure."""
        # Use a port that's not listening
        config = GlobalStoreClientConfig(host="127.0.0.1", port=59999, timeout=0.5)
        client = GlobalStoreClient(config)

        # Set a timeout for the connection attempt
        with pytest.raises(grpc.RpcError):
            await asyncio.wait_for(client.connect(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_list_active_workers(self, client, test_server):
        """Test listing active workers."""
        # Test without unavailable workers
        workers = await client.list_active_workers(include_unavailable=False)
        assert len(workers) == 1
        assert workers[0].worker_id == "worker-1"
        assert workers[0].accepting_new_requests is True

        # Test with unavailable workers
        workers = await client.list_active_workers(include_unavailable=True)
        assert len(workers) == 2
        assert any(w.worker_id == "worker-2" for w in workers)

    @pytest.mark.asyncio
    async def test_list_model_replicas(self, client, test_server):
        """Test listing model replicas with filters."""
        # Test without filters
        replicas = await client.list_model_replicas()
        assert len(replicas) == 2
        assert "model-1" in replicas
        assert "model-2" in replicas
        assert len(replicas["model-1"]) == 2
        assert len(replicas["model-2"]) == 1

        # Test with model id filter
        replicas = await client.list_model_replicas(model_id="model-1")
        assert len(replicas) == 1
        assert "model-1" in replicas

        # Test with node_id filter
        replicas = await client.list_model_replicas(node_id="node-1")
        assert all(
            any(r.node_id == "node-1" for r in replica_list)
            for replica_list in replicas.values()
        )

        # Test with memory_type filter
        replicas = await client.list_model_replicas(
            memory_type=global_store_pb2.MemoryType.GPU
        )
        assert all(
            any(r.memory_type == global_store_pb2.MemoryType.GPU for r in replica_list)
            for replica_list in replicas.values()
        )

        # Test with device_id filter
        replicas = await client.list_model_replicas(device_id=0)
        assert all(
            any(r.device_id == 0 for r in replica_list)
            for replica_list in replicas.values()
        )

    @pytest.mark.asyncio
    async def test_get_model_info(self, client, test_server):
        """Test getting model information."""
        # Test existing model
        model_info = await client.get_model_info("model-1")
        assert model_info is not None
        assert len(model_info) == 2

        # Test non-existing model
        model_info = await client.get_model_info("model-nonexistent")
        assert model_info is None

    @pytest.mark.asyncio
    async def test_get_summary_stats(self, client, test_server):
        """Test getting summary statistics."""
        stats = await client.get_summary_stats()

        assert stats["total_workers"] == 2
        assert stats["active_workers"] == 1
        assert stats["total_models"] == 2
        assert stats["total_replicas"] == 3
        assert stats["gpu_replicas"] == 1
        assert stats["ram_replicas"] == 1
        assert stats["disk_replicas"] == 1
        assert stats["active_transports"] == 0

    @pytest.mark.asyncio
    async def test_retry_logic(self, client, test_server):
        """Test client retry logic on failures."""
        # Reset stats to exclude the initial ListActiveWorkers call that occurs during
        # the `client.connect()` handshake.
        test_server.servicer.reset_stats()

        # Set servicer to fail the first 2 requests
        test_server.servicer.set_failure_mode(fail_count=2)

        # This should succeed after 2 retries
        workers = await client.list_active_workers()
        assert len(workers) > 0

        # Verify it took 3 attempts (1 initial + 2 retries)
        assert test_server.servicer.call_count["ListActiveWorkers"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, client, test_server):
        """Test client behavior when retries are exhausted."""
        # Reset stats to exclude the initial handshake RPC.
        test_server.servicer.reset_stats()

        # Set servicer to fail more times than max retries
        test_server.servicer.set_failure_mode(fail_count=5)

        # This should fail after exhausting retries
        with pytest.raises(grpc.RpcError) as exc_info:
            await client.list_active_workers()

        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE

        # Verify it attempted max_retries times
        assert test_server.servicer.call_count["ListActiveWorkers"] == client.config.max_retries

    @pytest.mark.asyncio
    async def test_singleton_client(self, test_server):
        """Test singleton client functionality."""
        config = GlobalStoreClientConfig(host="127.0.0.1", port=test_server.port)

        # Get singleton client
        client1 = await get_global_store_client(config)
        client2 = await get_global_store_client(config)

        # Should be the same instance
        assert client1 is client2

        # Test functionality
        workers = await client1.list_active_workers()
        assert len(workers) > 0

        # Close singleton
        await close_global_store_client()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, test_server):
        """Test concurrent requests through the client."""
        # Reset stats
        test_server.servicer.reset_stats()

        # Make concurrent requests
        tasks = [
            client.list_active_workers(),
            client.list_model_replicas(),
            client.get_model_info("model-1"),
            client.get_summary_stats(),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all requests succeeded
        assert len(results[0]) > 0  # workers
        assert len(results[1]) > 0  # replicas
        assert results[2] is not None  # model info
        assert results[3]["total_workers"] > 0  # stats

        # Verify concurrent execution
        assert test_server.servicer.call_count["ListActiveWorkers"] >= 1  # Called by get_summary_stats too
        assert test_server.servicer.call_count["ListModelReplicas"] >= 1  # Called by get_summary_stats too
        assert test_server.servicer.call_count["GetModelInfoById"] >= 1

    @pytest.mark.asyncio
    async def test_ensure_connected(self, test_server):
        """Test ensure_connected functionality."""
        config = GlobalStoreClientConfig(host="127.0.0.1", port=test_server.port)
        client = GlobalStoreClient(config)

        # Should connect automatically
        await client.ensure_connected()
        assert client._channel is not None

        # Should not reconnect if already connected
        channel1 = client._channel
        await client.ensure_connected()
        assert client._channel is channel1

        await client.close()

    @pytest.mark.asyncio
    async def test_get_active_transports(self, client):
        """Test get_active_transports placeholder."""
        # This should return empty list and log warning
        transports = await client.get_active_transports()
        assert transports == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, test_server):
        """Test handling of empty responses."""
        # Modify servicer to return empty data
        test_server.servicer.workers = []
        test_server.servicer.replicas = {}

        config = GlobalStoreClientConfig(host="127.0.0.1", port=test_server.port)
        client = GlobalStoreClient(config)
        await client.connect()

        # Test empty workers
        workers = await client.list_active_workers()
        assert workers == []

        # Test empty replicas
        replicas = await client.list_model_replicas()
        assert replicas == {}

        # Test summary with empty data
        stats = await client.get_summary_stats()
        assert stats["total_workers"] == 0
        assert stats["total_replicas"] == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test request timeout handling."""
        # Create a special servicer that always delays
        class SlowServicer(MockGlobalModelStoreServicer):
            def ListActiveWorkers(self, request, context):
                # This will block the server thread
                time.sleep(0.5)  # 500ms delay
                return super().ListActiveWorkers(request, context)

        # Start a new server with the slow servicer
        servicer = SlowServicer()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        global_store_pb2_grpc.add_GlobalModelStoreServicer_to_server(servicer, server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()

        try:
            # Create client with short timeout
            config = GlobalStoreClientConfig(
                host="127.0.0.1",
                port=port,
                timeout=0.1,  # 100ms timeout (less than server delay)
                max_retries=1,  # Don't retry on timeout
            )
            client = GlobalStoreClient(config)
            await client.connect()

            # This should timeout
            with pytest.raises(grpc.RpcError) as exc_info:
                await client.list_active_workers()

            assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED

            await client.close()
        finally:
            server.stop(grace=0)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

