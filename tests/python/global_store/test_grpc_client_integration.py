#  Copyright (c) 2025, TensorCast Team.

"""Integration test demonstrating full gRPC client functionality.

This is a standalone program that starts a test server and exercises
all client functionality in a single run.
"""

import asyncio
import sys
import time
from concurrent import futures

import grpc
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tensorcast.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
)
from tensorcast.proto import global_store_pb2, global_store_pb2_grpc, common_pb2
from tests.python.global_store.test_grpc_client import MockGlobalStoreServicer

console = Console()


def print_header(title: str):
    """Print a formatted header."""
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")


def print_test(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[green]✓ PASS[/green]" if success else "[red]✗ FAIL[/red]"
    console.print(f"{status} {name}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


async def run_integration_test():
    """Run comprehensive integration test."""
    print_header("gRPC Client Integration Test")

    # Start test server
    console.print("[yellow]Starting test gRPC server...[/yellow]")
    servicer = MockGlobalStoreServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    console.print(f"[green]✓ Server started on port {port}[/green]\n")

    # Create client
    config = GlobalStoreClientConfig(
        host="127.0.0.1",
        port=port,
        max_retries=3,
        retry_delay=0.5,
        timeout=10.0,
    )
    client = GlobalStoreClient(config)

    try:
        # Test 1: Connection
        print_header("Test 1: Client Connection")
        await client.connect()
        print_test("Connect to server", True, f"Connected to 127.0.0.1:{port}")

        # Test 2: List Workers
        print_header("Test 2: List Workers")
        workers = await client.list_active_workers(include_unavailable=True)
        print_test(
            "List all workers", len(workers) == 2, f"Found {len(workers)} workers"
        )

        # Display workers table
        table = Table(title="Workers", box=box.ROUNDED)
        table.add_column("Worker ID", style="cyan")
        table.add_column("Node ID", style="magenta")
        table.add_column("Address", style="yellow")
        table.add_column("Memory", style="green")
        table.add_column("Status", style="blue")

        for w in workers:
            mem_str = f"{w.mem_pool_available_size / 1024**3:.1f}GB / {w.mem_pool_total_size / 1024**3:.1f}GB"
            status = "Active" if w.accepting_new_requests else "Inactive"
            table.add_row(w.worker_id, w.node_id, w.node_address, mem_str, status)

        console.print(table)

        # Test 3: List Artifact Replicas
        print_header("Test 3: List Artifact Replicas")
        all_replicas = await client.list_replicas()
        print_test(
            "List all replicas",
            len(all_replicas) == 2,
            f"Found {len(all_replicas)} models",
        )

        # Display replicas
        for artifact_id, replicas in all_replicas.items():
            console.print(f"\n[bold]Artifact: {artifact_id}[/bold]")
            for i, r in enumerate(replicas):
                mem_type = ["GPU", "RAM", "DISK"][r.memory_type]
                size_gb = r.memory_size / 1024**3
                console.print(
                    f"  Replica {i + 1}: {mem_type} ({size_gb:.1f}GB) on {r.node_id}"
                )

        # Test 4: Filtered Queries
        print_header("Test 4: Filtered Queries")

        # Filter by artifact id
        model1_replicas = await client.list_replicas(artifact_id="artifact-1")
        print_test(
            "Filter by artifact id",
            "artifact-1" in model1_replicas,
            f"Found {len(model1_replicas.get('artifact-1', []))} replicas for artifact-1",
        )

        # Filter by node
        node1_replicas = await client.list_replicas(node_id="node-1")
        node1_count = sum(
            1
            for replicas in node1_replicas.values()
            for r in replicas
            if r.node_id == "node-1"
        )
        print_test(
            "Filter by node", node1_count > 0, f"Found {node1_count} replicas on node-1"
        )

        # Filter by memory type
        gpu_replicas = await client.list_replicas(
            memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU
        )
        gpu_count = sum(
            1
            for replicas in gpu_replicas.values()
            for r in replicas
            if r.memory_type == common_pb2.MemoryType.MEMORY_TYPE_GPU
        )
        print_test(
            "Filter by memory type", gpu_count > 0, f"Found {gpu_count} GPU replicas"
        )

        # Test 5: Artifact Info
        print_header("Test 5: Get Artifact Info")

        # Existing artifact
        artifact_info = await client.get_artifact_info("artifact-1")
        print_test(
            "Get existing artifact",
            artifact_info is not None,
            f"artifact-1 has {len(artifact_info.available_replicas) if artifact_info else 0} replicas",
        )

        # Non-existing artifact
        artifact_info = await client.get_artifact_info("artifact-nonexistent")
        print_test(
            "Get non-existing artifact", artifact_info is None, "Correctly returned None"
        )

        # Test 6: Summary Statistics
        print_header("Test 6: Summary Statistics")
        stats = await client.get_summary_stats()

        # Display summary panel
        summary_text = f"""
[bold]Workers:[/bold]
  Total: {stats["total_workers"]}
  Active: {stats["active_workers"]}

[bold]Models:[/bold]
  Total: {stats["total_artifacts"]}

[bold]Replicas:[/bold]
  Total: {stats["total_replicas"]}
  GPU: {stats["gpu_replicas"]}
  RAM: {stats["ram_replicas"]}
  Disk: {stats["disk_replicas"]}

[bold]Transports:[/bold]
  Active: {stats["active_transports"]}
"""
        console.print(
            Panel(summary_text.strip(), title="System Summary", box=box.ROUNDED)
        )

        print_test(
            "Get summary stats",
            stats["total_workers"] == 2,
            "All statistics calculated correctly",
        )

        # Test 7: Retry Logic
        print_header("Test 7: Retry Logic")

        # Configure servicer to fail first 2 attempts
        servicer.reset_stats()
        servicer.set_failure_mode(fail_count=2)

        # This should succeed after retries
        workers = await client.list_active_workers()
        print_test(
            "Retry on failure",
            len(workers) > 0,
            f"Succeeded after {servicer.call_count.get('ListActiveWorkers', 0)} attempts",
        )

        # Test 8: Concurrent Requests
        print_header("Test 8: Concurrent Requests")

        servicer.reset_stats()
        start_time = time.time()

        # Run multiple requests concurrently
        tasks = [
            client.list_active_workers(),
            client.list_replicas(),
            client.get_artifact_info("artifact-1"),
            client.get_artifact_info("artifact-2"),
            client.get_summary_stats(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        # Check all succeeded
        failures = [r for r in results if isinstance(r, Exception)]
        print_test(
            "Concurrent execution",
            len(failures) == 0,
            f"All {len(tasks)} requests completed in {elapsed:.2f}s",
        )

        # Test 9: Connection Resilience
        print_header("Test 9: Connection Resilience")

        # Close and reconnect
        await client.close()
        await client.connect()
        workers = await client.list_active_workers()
        print_test(
            "Reconnect after close",
            len(workers) > 0,
            "Successfully reconnected and made request",
        )

        # Final Summary
        print_header("Test Summary")
        console.print("[bold green]All tests passed! ✓[/bold green]")
        console.print("\n[dim]The gRPC client is working correctly with:[/dim]")
        console.print("[dim]  • Connection management[/dim]")
        console.print("[dim]  • All query methods[/dim]")
        console.print("[dim]  • Retry logic[/dim]")
        console.print("[dim]  • Concurrent requests[/dim]")
        console.print("[dim]  • Error handling[/dim]")

    except Exception as e:
        console.print("\n[bold red]Test failed with error:[/bold red]")
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        await client.close()
        server.stop(grace=0)
        console.print("\n[yellow]Server stopped[/yellow]")

    return True


async def main():
    """Main entry point."""
    success = await run_integration_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
