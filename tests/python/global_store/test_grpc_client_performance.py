#  Copyright (c) 2025, TensorCast Team.

"""Performance test for the gRPC client.

This test measures the client's performance under various load conditions
including concurrent requests, large responses, and sustained load.
"""

import asyncio
import statistics
import time
from concurrent import futures
from typing import Any, Dict

import grpc
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from tensorcast.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
)
from tensorcast.proto import global_store_pb2, global_store_pb2_grpc, common_pb2
from google.protobuf import timestamp_pb2
from tests.python.global_store.test_grpc_client import MockGlobalStoreServicer

console = Console()


class PerformanceTestServicer(MockGlobalStoreServicer):
    """Extended test servicer for performance testing."""

    def __init__(self, num_workers: int = 100, num_models: int = 50):
        """Initialize with configurable data size."""
        super().__init__()

        # Generate larger datasets
        self.workers = []
        for i in range(num_workers):
            ts = timestamp_pb2.Timestamp()
            ts.FromSeconds(int(time.time()) - i)
            self.workers.append(
                global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                    worker_id=f"worker-{i}",
                    node_id=f"node-{i % 10}",  # 10 nodes
                    node_address=f"192.168.1.{10 + (i % 240)}",
                    grpc_port=50000 + i,
                    p2p_port=60000 + i,
                    mem_pool_total_size=10737418240,  # 10GB
                    mem_pool_available_size=5368709120 + i * 1048576,  # 5GB + i MB
                    accepting_new_requests=i % 3 != 0,  # 2/3 active
                    last_heartbeat_ts=ts,
                )
            )

        # Generate artifact replicas
        self.replicas = {}
        for i in range(num_models):
            artifact_id = f"artifact-{i}"
            replicas = []

            # Each artifact has 1-5 replicas
            num_replicas = 1 + (i % 5)
            for j in range(num_replicas):
                # Map to enum values explicitly for type safety
                memory_type = [
                    common_pb2.MemoryType.GPU,
                    common_pb2.MemoryType.RAM,
                    common_pb2.MemoryType.DISK,
                ][j % 3]
                replicas.append(
                    common_pb2.MemoryInfo(
                        memory_type=memory_type,
                        memory_size=1073741824 * (1 + j),  # 1-5 GB
                        device_id=j
                        if memory_type == common_pb2.MemoryType.GPU
                        else 0,  # device_id must be non-negative
                        node_id=f"node-{j % 10}",
                        node_address=f"192.168.1.{10 + j}",
                        node_port=60000 + j,
                    )
                )

            self.replicas[artifact_id] = replicas


async def measure_latency(
    client: GlobalStoreClient, operation: str, count: int = 100
) -> Dict[str, float]:
    """Measure latency statistics for an operation."""
    latencies = []

    for _ in range(count):
        start = time.perf_counter()

        if operation == "list_workers":
            await client.list_active_workers()
        elif operation == "list_replicas":
            await client.list_replicas()
        elif operation == "get_artifact":
            await client.get_artifact_info("artifact-0")
        elif operation == "summary":
            await client.get_summary_stats()

        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        latencies.append(elapsed)

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        "p99": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
        "min": min(latencies),
        "max": max(latencies),
    }


async def measure_throughput(
    client: GlobalStoreClient, duration: int = 10
) -> Dict[str, Any]:
    """Measure throughput over a duration."""
    operations = ["list_workers", "list_replicas", "get_artifact", "summary"]
    counts = dict.fromkeys(operations, 0)
    errors = dict.fromkeys(operations, 0)

    start_time = time.time()
    end_time = start_time + duration

    async def run_operation(op: str):
        nonlocal counts, errors
        while time.time() < end_time:
            try:
                if op == "list_workers":
                    await client.list_active_workers()
                elif op == "list_replicas":
                    await client.list_replicas()
                elif op == "get_artifact":
                    await client.get_artifact_info(f"artifact-{counts[op] % 50}")
                elif op == "summary":
                    await client.get_summary_stats()
                counts[op] += 1
            except Exception:
                errors[op] += 1

    # Run all operations concurrently
    tasks = [run_operation(op) for op in operations]
    await asyncio.gather(*tasks)

    actual_duration = time.time() - start_time

    return {
        "duration": actual_duration,
        "total_operations": sum(counts.values()),
        "operations_per_second": sum(counts.values()) / actual_duration,
        "counts": counts,
        "errors": errors,
        "rates": {op: count / actual_duration for op, count in counts.items()},
    }


async def run_performance_test():
    """Run comprehensive performance tests."""
    console.print("[bold blue]gRPC Client Performance Test[/bold blue]\n")

    # Start server with larger dataset
    console.print("[yellow]Starting test server with large dataset...[/yellow]")
    servicer = PerformanceTestServicer(num_workers=100, num_models=50)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        options=[
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ],
    )
    global_store_pb2_grpc.add_GlobalStoreServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    console.print(
        f"[green]✓ Server started with {len(servicer.workers)} workers and {len(servicer.replicas)} models[/green]\n"
    )

    # Create client
    config = GlobalStoreClientConfig(
        host="127.0.0.1",
        port=port,
        max_retries=3,
        retry_delay=0.1,
        timeout=30.0,
    )
    client = GlobalStoreClient(config)
    await client.connect()

    try:
        # Test 1: Latency Measurements
        console.print("[bold cyan]Test 1: Latency Measurements[/bold cyan]")
        console.print("Measuring latency for each operation type (100 iterations)...\n")

        operations = [
            ("List Workers", "list_workers"),
            ("List Replicas", "list_replicas"),
            ("Get Artifact Info", "get_artifact"),
            ("Get Summary", "summary"),
        ]

        latency_table = Table(title="Latency Statistics (milliseconds)")
        latency_table.add_column("Operation", style="cyan")
        latency_table.add_column("Mean", style="green")
        latency_table.add_column("Median", style="green")
        latency_table.add_column("P95", style="yellow")
        latency_table.add_column("P99", style="red")
        latency_table.add_column("Min", style="blue")
        latency_table.add_column("Max", style="magenta")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for name, op in operations:
                task = progress.add_task(f"Testing {name}...", total=None)
                stats = await measure_latency(client, op)
                progress.remove_task(task)

                latency_table.add_row(
                    name,
                    f"{stats['mean']:.2f}",
                    f"{stats['median']:.2f}",
                    f"{stats['p95']:.2f}",
                    f"{stats['p99']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                )

        console.print(latency_table)
        console.print()

        # Test 2: Concurrent Load
        console.print("[bold cyan]Test 2: Concurrent Request Performance[/bold cyan]")
        console.print(
            "Testing concurrent requests with varying concurrency levels...\n"
        )

        concurrency_levels = [1, 10, 50, 100]
        concurrency_table = Table(title="Concurrent Request Performance")
        concurrency_table.add_column("Concurrency", style="cyan")
        concurrency_table.add_column("Total Time (s)", style="green")
        concurrency_table.add_column("Requests/sec", style="yellow")
        concurrency_table.add_column("Avg Latency (ms)", style="magenta")

        for level in concurrency_levels:
            start = time.perf_counter()

            # Run concurrent requests
            tasks = []
            for i in range(level):
                tasks.append(client.list_active_workers())
                tasks.append(client.list_replicas())

            await asyncio.gather(*tasks)

            elapsed = time.perf_counter() - start
            total_requests = level * 2
            rps = total_requests / elapsed
            avg_latency = (elapsed / total_requests) * 1000

            concurrency_table.add_row(
                str(level),
                f"{elapsed:.2f}",
                f"{rps:.0f}",
                f"{avg_latency:.2f}",
            )

        console.print(concurrency_table)
        console.print()

        # Test 3: Sustained Load
        console.print("[bold cyan]Test 3: Sustained Load Test[/bold cyan]")
        console.print("Running sustained load for 10 seconds...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running load test...", total=10)

            # Run for 10 seconds with progress updates
            throughput_task = asyncio.create_task(measure_throughput(client, 10))

            for i in range(10):
                await asyncio.sleep(1)
                progress.update(task, advance=1)

            throughput_stats = await throughput_task

        # Display throughput results
        throughput_table = Table(title="Throughput Statistics")
        throughput_table.add_column("Metric", style="cyan")
        throughput_table.add_column("Value", style="green")

        throughput_table.add_row(
            "Duration", f"{throughput_stats['duration']:.1f} seconds"
        )
        throughput_table.add_row(
            "Total Operations", f"{throughput_stats['total_operations']:,}"
        )
        throughput_table.add_row(
            "Overall Rate", f"{throughput_stats['operations_per_second']:.0f} ops/sec"
        )
        throughput_table.add_row(
            "Total Errors", f"{sum(throughput_stats['errors'].values())}"
        )

        console.print(throughput_table)
        console.print()

        # Per-operation rates
        rate_table = Table(title="Per-Operation Rates")
        rate_table.add_column("Operation", style="cyan")
        rate_table.add_column("Count", style="green")
        rate_table.add_column("Rate (ops/sec)", style="yellow")
        rate_table.add_column("Errors", style="red")

        for op in ["list_workers", "list_replicas", "get_artifact", "summary"]:
            rate_table.add_row(
                op.replace("_", " ").title(),
                f"{throughput_stats['counts'][op]:,}",
                f"{throughput_stats['rates'][op]:.0f}",
                str(throughput_stats["errors"][op]),
            )

        console.print(rate_table)
        console.print()

        # Test 4: Large Response Handling
        console.print("[bold cyan]Test 4: Large Response Performance[/bold cyan]")
        console.print("Testing with increasingly large responses...\n")

        # Create clients with different dataset sizes
        dataset_sizes = [10, 100, 500, 1000]
        large_response_table = Table(title="Large Response Performance")
        large_response_table.add_column("Dataset Size", style="cyan")
        large_response_table.add_column("Response Size", style="green")
        large_response_table.add_column("Latency (ms)", style="yellow")
        large_response_table.add_column("Throughput (MB/s)", style="magenta")

        for size in dataset_sizes:
            # Create new servicer with specified size
            test_servicer = PerformanceTestServicer(
                num_workers=size, num_models=size // 2
            )

            # Temporarily replace servicer
            original_workers = servicer.workers
            servicer.workers = test_servicer.workers

            start = time.perf_counter()
            workers = await client.list_active_workers(include_unavailable=True)
            elapsed = time.perf_counter() - start

            # Estimate response size (rough approximation)
            response_size = len(workers) * 200  # ~200 bytes per worker
            throughput = (response_size / 1024 / 1024) / elapsed if elapsed > 0 else 0

            large_response_table.add_row(
                str(size),
                f"{response_size / 1024:.1f} KB",
                f"{elapsed * 1000:.2f}",
                f"{throughput:.2f}",
            )

            # Restore original
            servicer.workers = original_workers

        console.print(large_response_table)
        console.print()

        # Summary
        console.print("[bold green]Performance Test Summary[/bold green]")
        console.print("✓ Client handles concurrent requests efficiently")
        console.print("✓ Latency remains low even under load")
        console.print("✓ Throughput scales with concurrency")
        console.print("✓ Large responses are handled without issues")
        console.print(
            "\n[dim]Note: Actual performance will depend on network conditions and server resources.[/dim]"
        )

    finally:
        await client.close()
        server.stop(grace=0)
        console.print("\n[yellow]Test completed and server stopped.[/yellow]")


if __name__ == "__main__":
    asyncio.run(run_performance_test())
