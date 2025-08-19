#!/usr/bin/env python3
#  Copyright (c) 2025, StepCast Team.

"""Demo script showing how to use the gRPC client for testing.

This demonstrates how to:
1. Start a test gRPC server in a thread
2. Create and use the gRPC client
3. Test various operations
4. Clean up properly
"""

import asyncio
import time
from concurrent import futures

import grpc

from scstore.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
)
from scstore.proto import global_store_pb2, global_store_pb2_grpc
from tests.python.global_store.test_grpc_client import MockGlobalModelStoreServicer


async def demo_basic_usage():
    """Demonstrate basic gRPC client usage."""
    print("=== Basic gRPC Client Usage Demo ===\n")

    # 1. Start test server
    print("1. Starting test gRPC server...")
    servicer = MockGlobalModelStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    global_store_pb2_grpc.add_GlobalModelStoreServicer_to_server(servicer, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    print(f"   Server started on port {port}\n")

    # 2. Create client
    print("2. Creating gRPC client...")
    config = GlobalStoreClientConfig(
        host="127.0.0.1",
        port=port,
        max_retries=3,
        retry_delay=0.5,
    )
    client = GlobalStoreClient(config)
    await client.connect()
    print("   Client connected!\n")

    # 3. Test operations
    print("3. Testing operations:\n")

    # List workers
    print("   a) List workers:")
    workers = await client.list_active_workers(include_unavailable=True)
    for w in workers:
        status = "Active" if w.accepting_new_requests else "Inactive"
        print(f"      - {w.worker_id}: {status} (Memory: {w.mem_pool_available_size / 1024**3:.1f}GB)")

    # List model replicas
    print("\n   b) List model replicas:")
    replicas_by_model = await client.list_model_replicas()
    for model_id, replica_list in replicas_by_model.items():
        print(f"      - {model_id}: {len(replica_list)} replicas")
        for r in replica_list:
            mem_type = ["GPU", "RAM", "DISK"][r.memory_type]
            print(f"        • {mem_type} on {r.node_id}")

    # Get model info
    print("\n   c) Get model info:")
    model_replicas = await client.get_model_info("model-1")
    if model_replicas is not None:
        print(f"      - model-1: {len(model_replicas)} available replicas")

    # Get summary
    print("\n   d) Get summary statistics:")
    stats = await client.get_summary_stats()
    print(f"      - Workers: {stats['active_workers']}/{stats['total_workers']} active")
    print(f"      - Models: {stats['total_models']}")
    print(f"      - Replicas: {stats['total_replicas']} (GPU:{stats['gpu_replicas']}, RAM:{stats['ram_replicas']}, Disk:{stats['disk_replicas']})")

    # 4. Cleanup
    print("\n4. Cleaning up...")
    await client.close()
    server.stop(grace=1)
    print("   Done!\n")


async def demo_error_handling():
    """Demonstrate error handling and retry logic."""
    print("=== Error Handling Demo ===\n")

    # Start server
    servicer = MockGlobalModelStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    global_store_pb2_grpc.add_GlobalModelStoreServicer_to_server(servicer, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()

    # Create client
    config = GlobalStoreClientConfig(host="127.0.0.1", port=port, max_retries=3, retry_delay=0.1)
    client = GlobalStoreClient(config)
    await client.connect()

    # Test retry logic
    print("1. Testing retry logic:")
    servicer.set_failure_mode(fail_count=2)  # Fail first 2 attempts

    print("   Making request (will retry automatically)...")
    start = time.time()
    workers = await client.list_active_workers()
    elapsed = time.time() - start
    print(f"   Success after {servicer.call_count.get('ListActiveWorkers', 0)} attempts ({elapsed:.2f}s)")

    # Test connection failure
    print("\n2. Testing connection failure:")
    bad_config = GlobalStoreClientConfig(host="127.0.0.1", port=99999, timeout=1.0)
    bad_client = GlobalStoreClient(bad_config)

    try:
        await asyncio.wait_for(bad_client.connect(), timeout=2.0)
    except Exception as e:
        print(f"   Expected error: {type(e).__name__}: {e}")

    # Cleanup
    await client.close()
    server.stop(grace=1)
    print("\nDone!\n")


async def demo_concurrent_usage():
    """Demonstrate concurrent request handling."""
    print("=== Concurrent Usage Demo ===\n")

    # Start server
    servicer = MockGlobalModelStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    global_store_pb2_grpc.add_GlobalModelStoreServicer_to_server(servicer, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()

    # Create client
    config = GlobalStoreClientConfig(host="127.0.0.1", port=port)
    client = GlobalStoreClient(config)
    await client.connect()

    print("Making 10 concurrent requests...")
    start = time.time()

    # Create concurrent tasks
    tasks = []
    for i in range(10):
        if i % 4 == 0:
            tasks.append(client.list_active_workers())
        elif i % 4 == 1:
            tasks.append(client.list_model_replicas())
        elif i % 4 == 2:
            tasks.append(client.get_model_info("model-1"))
        else:
            tasks.append(client.get_summary_stats())

    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start
    success_count = sum(1 for r in results if not isinstance(r, Exception))

    print(f"Completed in {elapsed:.2f}s")
    print(f"Success: {success_count}/{len(tasks)} requests")

    # Cleanup
    await client.close()
    server.stop(grace=1)
    print("\nDone!\n")


async def main():
    """Run all demos."""
    await demo_basic_usage()
    await demo_error_handling()
    await demo_concurrent_usage()

    print("All demos completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())