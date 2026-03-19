#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass

import pytest

import tensorcast
from tensorcast.api.context import CallContext, GovernanceContext
from tensorcast.api.plan import PlanResult
from tensorcast.api.runtime import connect
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.node_agent.v1 import node_agent_pb2


@dataclass
class _FakeDaemonClient:
    address: str
    last_plan = None
    last_execution_class: str | None = None
    last_dry_run: bool | None = None

    def execute_plan(
        self,
        *,
        plan,
        execution_class: str = "terminal_only",
        dry_run: bool = False,
    ) -> node_agent_pb2.ExecutePlanResponse:
        self.last_plan = plan
        self.last_execution_class = execution_class
        self.last_dry_run = dry_run
        step = node_agent_pb2.StepResult(
            step_id="step-0001",
            target_id="daemon-a",
            action="prefetch_set",
            status=node_agent_pb2.OperationStatus(
                state=node_agent_pb2.OPERATION_STATE_SUCCESS,
                message="ok",
            ),
        )
        return node_agent_pb2.ExecutePlanResponse(
            request_id=plan.context.request_id,
            ok=True,
            steps=[step],
        )


def test_connect_registers_active_runtime_and_plan_uses_ingress() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda address: client,
    )
    ctx = CallContext(
        request_id="req-runtime",
        governance=GovernanceContext(
            lane="runtime-lane",
            policy_version=3,
            staleness_budget_ms=50,
        ),
    )

    plan = tensorcast.plan(ctx)
    result = plan.run()

    assert runtime.daemon_address == "127.0.0.1:50051"
    assert isinstance(result, PlanResult)
    assert client.last_execution_class == "terminal_only"
    assert client.last_dry_run is False
    assert client.last_plan is not None
    assert client.last_plan.context.request_id == "req-runtime"
    assert client.last_plan.governance.lane == "runtime-lane"
    runtime.close()


def test_runtime_rejects_non_terminal_ingress_class_before_rpc() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda address: client,
    )

    with pytest.raises(RuntimeError, match="terminal_only"):
        runtime.plan(
            CallContext(request_id="req-runtime-async"),
            execution_class="public_continuation_required",
        ).run()

    assert client.last_plan is None
    runtime.close()


def test_daemon_execute_plan_request_has_execution_class_enum() -> None:
    request = store_daemon_pb2.ExecutePlanRequest(
        execution_class=store_daemon_pb2.PLAN_EXECUTION_CLASS_TERMINAL_ONLY,
        dry_run=True,
    )

    assert (
        request.execution_class == store_daemon_pb2.PLAN_EXECUTION_CLASS_TERMINAL_ONLY
    )
    assert request.dry_run is True


def test_runtime_signals_reads_connected_worker_status() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    client.get_worker_status = lambda: store_daemon_pb2.GetWorkerStatusResponse(  # type: ignore[attr-defined]
        is_registered=True,
        is_healthy=True,
        is_shutting_down=False,
        mem_pool_total_size=4096,
        mem_pool_available_size=1024,
        uptime_seconds=17,
        worker_id="worker-a",
        daemon_id="daemon-a",
        as_of_ms=123456,
        staleness_ms=7,
        cache_epoch=9,
        freshness_state="current",
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda address: client,
    )

    snapshot = runtime.signals().get_worker_status()

    assert snapshot.value.worker_id == "worker-a"
    assert snapshot.value.daemon_id == "daemon-a"
    assert snapshot.value.mem_pool_total_size == 4096
    assert snapshot.as_of_ms == 123456
    assert snapshot.staleness_ms == 7
    assert snapshot.cache_epoch == 9
    assert snapshot.freshness_state == "current"
    runtime.close()


def test_runtime_signals_falls_back_when_daemon_omits_freshness_fields() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    client.get_worker_status = lambda: store_daemon_pb2.GetWorkerStatusResponse(  # type: ignore[attr-defined]
        is_registered=True,
        is_healthy=True,
        is_shutting_down=False,
        mem_pool_total_size=4096,
        mem_pool_available_size=1024,
        uptime_seconds=17,
        worker_id="worker-a",
        daemon_id="daemon-a",
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda address: client,
    )

    snapshot = runtime.signals().get_worker_status()

    assert snapshot.as_of_ms > 0
    assert snapshot.staleness_ms == 0
    assert snapshot.cache_epoch is None
    assert snapshot.freshness_state == "current"
    runtime.close()


def test_runtime_directory_reads_daemon_served_routes() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    client.list_directory_workers = (
        lambda **_kwargs: store_daemon_pb2.ListDirectoryWorkersResponse(  # type: ignore[attr-defined]
            workers=[
                store_daemon_pb2.WorkerDirectoryRoute(
                    daemon_id="daemon-a",
                    worker_id="worker-a",
                    daemon_address="10.0.0.1:50051",
                    capability_flags=7,
                )
            ],
            as_of_ms=123000,
            staleness_ms=11,
            cache_epoch=5,
            freshness_state="current",
            authority_mode="GLOBAL_STORE_BACKED",
        )
    )
    client.list_directory_instances = (
        lambda **_kwargs: store_daemon_pb2.ListDirectoryInstancesResponse(  # type: ignore[attr-defined]
            instances=[
                store_daemon_pb2.InstanceExecutionRoute(
                    instance_id="inst-a",
                    daemon_id="daemon-a",
                    execution_host_kind="node_agent_grpc",
                    execution_endpoint="10.0.0.1:7001",
                    engine="test",
                    capability_flags=9,
                )
            ],
            as_of_ms=123100,
            staleness_ms=13,
            cache_epoch=6,
            freshness_state="current",
            authority_mode="GLOBAL_STORE_BACKED",
        )
    )
    client.resolve_instance_execution = (
        lambda **_kwargs: store_daemon_pb2.ResolveInstanceExecutionResponse(  # type: ignore[attr-defined]
            route=store_daemon_pb2.InstanceExecutionRoute(
                instance_id="inst-a",
                daemon_id="daemon-a",
                execution_host_kind="node_agent_grpc",
                execution_endpoint="10.0.0.1:7001",
                engine="test",
                capability_flags=9,
            ),
            as_of_ms=123200,
            staleness_ms=17,
            cache_epoch=7,
            freshness_state="current",
            authority_mode="GLOBAL_STORE_BACKED",
        )
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda address: client,
    )

    worker_snapshot = runtime.directory().list_workers()
    instance_snapshot = runtime.directory().list_instances()
    route_snapshot = runtime.directory().resolve_instance_execution("inst-a")
    compat_worker_snapshot = runtime.signals().list_workers()

    assert worker_snapshot.authority_mode == "GLOBAL_STORE_BACKED"
    assert worker_snapshot.value[0].daemon_id == "daemon-a"
    assert worker_snapshot.value[0].daemon_address == "10.0.0.1:50051"
    assert instance_snapshot.value[0].instance_id == "inst-a"
    assert instance_snapshot.value[0].execution_endpoint == "10.0.0.1:7001"
    assert route_snapshot.value.execution_host_kind == "node_agent_grpc"
    assert route_snapshot.cache_epoch == 7
    assert compat_worker_snapshot.value[0].daemon_id == "daemon-a"
    runtime.close()
