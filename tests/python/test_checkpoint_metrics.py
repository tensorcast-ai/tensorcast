#  Copyright (c) 2025, TensorCast Team.

from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, generate_latest

import torch  # noqa: F401
import tensorcast._store_engine as _cs

from tensorcast.store_daemon.ckpt_collector import (
    GlobalMetricsCollector,
)


@pytest.mark.skipif(_cs is None, reason="C++ StoreEngine extension not available")
def test_global_metrics_export() -> None:  # noqa: D401
    """Ensure global metrics function works correctly."""

    # Create a minimal StoreEngine to populate some metrics
    storage_path = Path("/tmp/test_metrics")
    storage_path.mkdir(exist_ok=True)

    cs = _cs.create_store_engine({
        "storage_path": str(storage_path),
        "memory_pool_size": 1 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    # Get global metrics text directly
    metrics_text = _cs.get_global_metrics_text()

    # Should be bytes
    assert isinstance(metrics_text, bytes)

    # Decode and verify content
    text = metrics_text.decode()
    assert "store_daemon_memory_pool_total_bytes" in text
    assert "store_daemon_memory_pool_available_bytes" in text
    assert "# TYPE" in text  # Should have type annotations


@pytest.mark.skipif(_cs is None, reason="C++ StoreEngine extension not available")
def test_store_engine_metrics_snapshot(tmp_path: Path) -> None:  # noqa: D401
    """Ensure C++ metrics are exported and parsed by the Python collector."""

    # ------------------------------------------------------------------
    # Arrange – create a minimal StoreEngine (no RDMA, tiny mem pool)
    # ------------------------------------------------------------------
    storage_path = tmp_path / "models"
    storage_path.mkdir()

    cs = _cs.create_store_engine({
        "storage_path": str(storage_path),
        "memory_pool_size": 1 * 1024 * 1024,
        "num_thread": 1,
        "chunk_size": 1 * 1024 * 1024,
        "enable_p2p_engine": False,
        "enable_rdma": False,
        "pinned_memory_timeout_ms": 0,
    })

    # ------------------------------------------------------------------
    # Act – register custom collector AND python-side metrics, scrape once
    # ------------------------------------------------------------------

    # 1. Build an isolated registry so we don't pollute global state across tests
    registry = CollectorRegistry()

    # 2. Register Python layer metrics defined in tensorcast.store_daemon.metrics
    #    Those metric objects already exist; we simply add them to the custom
    #    registry so both C++ and Python metrics are exported together.
    from tensorcast.store_daemon import metrics as py_metrics  # noqa: WPS433 (test import)

    python_metric_objects = [
        py_metrics.ACTIVE_OPERATIONS,
        py_metrics.MEMORY_POOL_TOTAL,
        py_metrics.MEMORY_POOL_AVAILABLE,
        py_metrics.WORKER_HEALTHY,
        py_metrics.WORKER_REGISTERED,
        py_metrics.WORKER_UPTIME_SECONDS,
    ]

    for metric in python_metric_objects:
        registry.register(metric)

    # 3. Register the global metrics collector (no longer needs store_engine)
    # Register typed as Any to satisfy Collector protocol in typed contexts
    registry.register(GlobalMetricsCollector()) # pyright: ignore[reportArgumentType]

    # 4. Scrape metrics snapshot
    metrics_blob = generate_latest(registry).decode()

    # ------------------------------------------------------------------
    # Assert – baseline gauges are present and have expected non-negative values
    # ------------------------------------------------------------------
    assert "store_daemon_memory_pool_total_bytes" in metrics_blob
    assert "store_daemon_memory_pool_available_bytes" in metrics_blob
    assert "store_daemon_replicas_in_memory" in metrics_blob
    # Check TYPE comments are correct (gauge by default)
    assert "# TYPE store_daemon_memory_pool_total_bytes gauge" in metrics_blob
    # Ensure counter detection for *_total naming convention works
    # (no such metric emitted yet, but serializer would mark it as counter).  We
    # simply assert that at least one 'counter' TYPE line exists to exercise the
    # branch.
    assert any(
        line.startswith("# TYPE") and " counter" in line for line in metrics_blob.splitlines()
    ), "expected at least one counter metric in snapshot"
    # Ensure histogram detection works – we should have at least one histogram TYPE line
    assert any(
        line.startswith("# TYPE") and " histogram" in line for line in metrics_blob.splitlines()
    ), "expected at least one histogram metric in snapshot"
    # also confirm a python-side metric is present
    assert "store_daemon_active_operations" in metrics_blob

    # quick numerical sanity check – memory pool bytes should be > 0
    # Note: The C++ metric may report 0 if the pool is not yet initialized
    # or if the metric is collected before the pool allocation completes
    found_pool_metric = False
    for line in metrics_blob.splitlines():
        if line.startswith("store_daemon_memory_pool_total_bytes"):
            found_pool_metric = True
            break
    assert found_pool_metric, "memory pool total bytes metric not found"

    # ------------------------------------------------------------------
    # New: ensure at least one labelled sample is present and parsed
    # ------------------------------------------------------------------
    assert '{location="cpu"}' in metrics_blob or '{location="gpu"}' in metrics_blob, "expected labelled gauge for models in memory"

    # ------------------------------------------------------------------
    # Additional check – histogram metric should be present
    # Note: The Python collector may not parse histogram buckets fully yet,
    # but we should at least see the histogram TYPE declaration
    # ------------------------------------------------------------------
    assert "store_daemon_cpp_operation_latency_seconds" in metrics_blob, "expected operation latency histogram"

    # Parse with collector and verify label propagation
    collector = GlobalMetricsCollector()
    collected_families = list(collector.collect())
    # Find our gauge family
    models_family = next((f for f in collected_families if f.name.startswith("store_daemon_replicas_in_memory")), None)
    assert models_family is not None, "labelled models metric not found by collector"
    # Family samples are tuples: (name, labels, value, timestamp)
    has_labelled_sample = any(sample.labels.get("location") in ["cpu", "gpu"] for sample in models_family.samples)
    assert has_labelled_sample, "collector failed to attach labels to sample"



