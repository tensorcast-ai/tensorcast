#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import prometheus_client
import pytest

from tensorcast.proto import global_store_pb2
from tensorcast.global_store import metrics as gs_metrics

from tests.python.interaction.utils import FakeContext


# -----------------------------------------------------------------------------
# Helper to read metric value safely
# -----------------------------------------------------------------------------

def _counter_value(counter, **labels):
    fams = list(counter.collect())
    sample = next(
        s
        for s in fams[0].samples
        if all(s.labels[k] == v for k, v in labels.items())
    )
    return sample.value


def _gauge_value(gauge):
    fams = list(gauge.collect())
    return fams[0].samples[0].value


# -----------------------------------------------------------------------------
# Test – correlation between RPC events and Prometheus metrics
# -----------------------------------------------------------------------------


def _register_replica(gs, *, artifact: str, replica_id: str, capacity: int = 1):
    worker_req = global_store_pb2.RegisterWorkerRequest(
        node_id=replica_id,
        node_address="127.0.0.1",
        grpc_port=9000,
        p2p_port=9001,
        mem_pool_total_size=16 * 1024 * 1024,
        mem_pool_available_size=16 * 1024 * 1024,
    )
    w_resp = gs.RegisterWorker(worker_req, FakeContext())
    assert w_resp.status == global_store_pb2.Status.OK

    mem_info = global_store_pb2.MemoryInfo(
        node_id=replica_id,
        node_address="127.0.0.1",
        node_port=9000,
        remote_memory_keys=[f"key_{replica_id}"],
        memory_size=1 * 1024,
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    rep_req = global_store_pb2.RegisterReplicaRequest(
        artifact_id=artifact,
        mem_info=mem_info,
        max_concurrency=capacity,
        worker_id=w_resp.worker_id,
    )
    gs.RegisterReplica(rep_req, FakeContext())


@pytest.mark.integration
def test_metrics_consistency(global_store_service):
    """Ensure Prometheus counters/gauges reflect transport lifecycle (Scenario 11)."""

    gs = global_store_service
    artifact = "metrics-test-artifact"

    # Clean registry (best-effort) to avoid metric carry-over across tests.
    for metric in (
        gs_metrics.TRANSPORT_REQUEST_COUNTER,
        gs_metrics.TRANSPORT_WAIT_SECONDS,
        gs_metrics.ACTIVE_TRANSPORTS_GAUGE,
    ):
        try:
            prometheus_client.REGISTRY.unregister(metric)
        except KeyError:
            # May already be unregistered – ignore
            pass
        prometheus_client.REGISTRY.register(metric)

    # Register 1-capacity replica so active transports gauge increments to 1 only.
    _register_replica(gs, artifact=artifact, replica_id="METRIC", capacity=1)

    # ---- Acquire first transport ----
    req = global_store_pb2.RequestReplicaTransportRequest(artifact_id=artifact)
    resp = gs.RequestReplicaTransport(req, FakeContext())
    assert resp.status == global_store_pb2.Status.OK

    # Gauge should be 1 (one in-flight transport)
    assert _gauge_value(gs_metrics.ACTIVE_TRANSPORTS_GAUGE) == 1

    # ---- Acquire second request with wait_timeout so that it times out ----
    to_req = global_store_pb2.RequestReplicaTransportRequest(
        artifact_id=artifact,
        wait_timeout_ms=5,
    )
    to_resp = gs.RequestReplicaTransport(to_req, FakeContext())
    assert to_resp.status == global_store_pb2.Status.TIMED_OUT

    # Counter check
    assert _counter_value(gs_metrics.TRANSPORT_REQUEST_COUNTER, artifact_id=artifact, status="success") == 1
    assert _counter_value(gs_metrics.TRANSPORT_REQUEST_COUNTER, artifact_id=artifact, status="timeout") == 1

    # ---- Complete first transport ----
    comp_req = global_store_pb2.CompleteReplicaTransportRequest(
        transport_id=resp.transport_id
    )
    comp_resp = gs.CompleteReplicaTransport(comp_req, FakeContext())
    assert comp_resp.status == global_store_pb2.Status.OK

    # Gauge should be back to 0
    assert _gauge_value(gs_metrics.ACTIVE_TRANSPORTS_GAUGE) == 0

    # Histogram – at least *two* observations (success + timeout)
    # The Prometheus histogram exposes multiple sample rows; the entry with
    # the suffix *_count holds the observation count for each label set.
    metric_families = list(gs_metrics.TRANSPORT_WAIT_SECONDS.collect())
    samples = metric_families[0].samples
    hist_count = next(
        s.value
        for s in samples
        if s.name.endswith("_count") and s.labels.get("artifact_id") == artifact
    )
    assert hist_count >= 2