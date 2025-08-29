#  Copyright (c) 2025, TensorCast Team.

"""Simple test to verify gRPC server starts correctly."""

from concurrent import futures

import grpc

from scstore.proto import global_store_pb2, global_store_pb2_grpc
from tests.python.global_store.test_grpc_client import MockGlobalStoreServicer


def test_server_startup():
    """Test that we can start a gRPC server."""
    print("Creating servicer...")
    servicer = MockGlobalStoreServicer()

    print("Creating server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    print("Adding servicer to server...")
    global_store_pb2_grpc.add_GlobalStoreServicer_to_server(servicer, server)

    print("Binding to port...")
    port = server.add_insecure_port("127.0.0.1:0")
    print(f"Bound to port: {port}")

    print("Starting server...")
    server.start()

    print("Testing connection...")
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        # Test connection
        future = grpc.channel_ready_future(channel)
        future.result(timeout=2)
        print("✓ Server is ready!")

        # Test a simple RPC
        stub = global_store_pb2_grpc.GlobalStoreStub(channel)
        request = global_store_pb2.ListActiveWorkersRequest(include_unavailable=True)
        response = stub.ListActiveWorkers(request, timeout=5)
        print(f"✓ RPC successful! Got {len(response.workers)} workers")

    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    finally:
        channel.close()
        print("Stopping server...")
        server.stop(grace=1)
        server.wait_for_termination(timeout=2)
        print("✓ Server stopped")


if __name__ == "__main__":
    test_server_startup()
    print("\nAll tests passed!")
