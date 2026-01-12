#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from tensorcast.cli_utils.health import GlobalStoreHealth, ping_global_store
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def test_ping_global_store_prefers_service_health(monkeypatch):
    class DummyStub:
        def __init__(self, _channel):
            pass

        def HealthCheck(self, _request, timeout=None):
            return global_store_pb2.HealthCheckResponse(
                status=global_store_pb2.Status.STATUS_OK,
                cluster_token="cluster-abc",
            )

        def GetServerInfo(self, _request, timeout=None):
            return global_store_pb2.GetServerInfoResponse(
                status=global_store_pb2.Status.STATUS_OK,
                listen_host="127.0.0.1",
                listen_port=6100,
                advertise_host="127.0.0.1",
                advertise_port=6100,
                metrics_port=7100,
                db_file="/tmp/db",
                version="v1",
            )

    monkeypatch.setattr(global_store_pb2_grpc, "GlobalStoreServiceStub", DummyStub)
    monkeypatch.setattr(health_pb2_grpc, "HealthStub", lambda _channel: None)

    health = ping_global_store("127.0.0.1:6100", timeout=0.1)
    assert isinstance(health, GlobalStoreHealth)
    assert health.cluster_token == "cluster-abc"
    assert health.listen_port == 6100
    assert health.advertise_host == "127.0.0.1"
    assert health.advertise_port == 6100
    assert health.metrics_port == 7100
    assert health.db_file == "/tmp/db"


def test_ping_global_store_fallback_to_grpc_health(monkeypatch):
    class FailingStub:
        def __init__(self, _channel):
            pass

        def HealthCheck(self, _request, timeout=None):
            raise RuntimeError("unavailable")

    class DummyHealthStub:
        def __init__(self, _channel):
            pass

        def Check(self, _request, timeout=None):
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )

    monkeypatch.setattr(global_store_pb2_grpc, "GlobalStoreServiceStub", FailingStub)
    monkeypatch.setattr(health_pb2_grpc, "HealthStub", DummyHealthStub)

    health = ping_global_store("127.0.0.1:6200", timeout=0.1)
    assert isinstance(health, GlobalStoreHealth)
    assert health.metrics_port is None
