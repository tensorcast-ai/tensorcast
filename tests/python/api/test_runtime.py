#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import pytest

import tensorcast
from tensorcast.api.context import CallContext, GovernanceContext
from tensorcast.api.plan import Instance, PlanFailedError, PlanResult
from tensorcast.api.runtime import connect
from tensorcast.engine_adapter.artifact_api import (
    EngineOwnedManifest,
    ManifestResult,
    PublishManifest,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.node_agent.v1 import node_agent_pb2


def _sample_publish_manifest(
    *, rid: str = "rid-123", payload: bytes | None = None
) -> PublishManifest:
    artifact_manifest = ManifestResult.from_artifact_ids(
        engine_request_id=rid,
        layout_id="layout-v1",
        artifact_ids=(
            "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",
        ),
    )
    return PublishManifest(
        artifact_manifest=artifact_manifest,
        engine_owned_manifest=EngineOwnedManifest(
            engine="sglang",
            schema="sglang.engine_owned_manifest.v1",
            version=1,
            encoding="json",
            created_at_ms=1774223000123,
            expires_at_ms=1774223060123,
            artifact_manifest_digest=artifact_manifest.key_set_digest_hex,
            payload_sha256="f" * 64,
            payload=payload or f'{{"logical_request_id":"{rid}"}}'.encode("utf-8"),
        ),
    )


@dataclass
class _FakeDaemonClient:
    address: str
    response_factory: Callable[[object], node_agent_pb2.ExecutePlanResponse] | None = (
        None
    )
    last_plan = None
    last_execution_class: str | None = None
    last_dry_run: bool | None = None
    last_timeout_s: float | None = None

    def execute_plan(
        self,
        *,
        plan,
        execution_class: str = "terminal_only",
        dry_run: bool = False,
        timeout_s: float = 30.0,
    ) -> node_agent_pb2.ExecutePlanResponse:
        self.last_plan = plan
        self.last_execution_class = execution_class
        self.last_dry_run = dry_run
        self.last_timeout_s = timeout_s
        if self.response_factory is not None:
            return self.response_factory(plan)
        response = node_agent_pb2.ExecutePlanResponse(
            request_id=plan.context.request_id,
            ok=True,
        )
        for step in plan.steps:
            response.steps.add(
                step_id=str(step.step_id),
                target_id=str(step.target.target_id),
                action=str(step.action.WhichOneof("kind") or "unknown"),
                status=node_agent_pb2.OperationStatus(
                    state=node_agent_pb2.OPERATION_STATE_SUCCESS,
                    message="ok",
                ),
            )
        return response


def test_connect_registers_active_runtime_and_plan_uses_ingress() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
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
    assert client.last_timeout_s == 30.0
    assert client.last_plan is not None
    assert client.last_plan.context.request_id == "req-runtime"
    assert client.last_plan.governance.lane == "runtime-lane"
    runtime.close()


def test_runtime_execute_plan_propagates_call_deadline_to_daemon_timeout() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
    )
    plan = runtime.plan(
        CallContext(request_id="req-runtime-deadline", deadline_ms=120_000)
    )

    result = plan.run()

    assert result.ok is True
    assert client.last_timeout_s == 120.0
    runtime.close()


def test_runtime_hydrate_engine_request_id_goes_to_daemon_unrewritten() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
    )

    plan = runtime.plan(CallContext(request_id="req-runtime-hydrate"))
    plan.on_instance(
        Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")
    ).hydrate(engine_request_id="rid-unified")

    result = plan.run()

    assert result.ok is True
    assert client.last_plan is not None
    hydrate = client.last_plan.steps[0].action.hydrate
    assert hydrate.WhichOneof("request_source") == "engine_request_id"
    assert hydrate.engine_request_id == "rid-unified"
    runtime.close()


def test_runtime_publish_result_does_not_install_hydrate_rewrite_cache() -> None:
    publish_manifest = _sample_publish_manifest(rid="rid-publish-cache")

    def _response_factory(plan) -> node_agent_pb2.ExecutePlanResponse:  # noqa: ANN001
        response = node_agent_pb2.ExecutePlanResponse(
            request_id=plan.context.request_id,
            ok=True,
        )
        step = response.steps.add(
            step_id=str(plan.steps[0].step_id),
            target_id=str(plan.steps[0].target.target_id),
            action="publish",
            status=node_agent_pb2.OperationStatus(
                state=node_agent_pb2.OPERATION_STATE_SUCCESS,
                message="ok",
            ),
        )
        step.artifact_result.publish.manifest.CopyFrom(
            publish_manifest.artifact_manifest.to_proto()
        )
        step.artifact_result.publish.publish_manifest.CopyFrom(
            publish_manifest.to_proto()
        )
        return response

    client = _FakeDaemonClient(
        "127.0.0.1:50051",
        response_factory=_response_factory,
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
    )
    publish_plan = runtime.plan(CallContext(request_id="req-runtime-publish"))
    publish_plan.on_instance(
        Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")
    ).publish(engine_request_id="rid-publish-cache")

    publish_result = publish_plan.run()

    assert publish_result.ok is True
    assert not hasattr(runtime, "resolve_publish_manifest")
    assert not hasattr(runtime, "remember_publish_manifest")
    runtime.close()


def test_runtime_publish_failure_surfaces_to_controller() -> None:
    def _response_factory(plan) -> node_agent_pb2.ExecutePlanResponse:  # noqa: ANN001
        response = node_agent_pb2.ExecutePlanResponse(
            request_id=plan.context.request_id,
            ok=False,
        )
        response.steps.add(
            step_id=str(plan.steps[0].step_id),
            target_id=str(plan.steps[0].target.target_id),
            action="publish",
            status=node_agent_pb2.OperationStatus(
                state=node_agent_pb2.OPERATION_STATE_FAILED,
                message="publish failed closed",
            ),
        )
        return response

    client = _FakeDaemonClient(
        "127.0.0.1:50051",
        response_factory=_response_factory,
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
    )
    plan = runtime.plan(CallContext(request_id="req-runtime-publish-fail"))
    plan.on_instance(
        Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")
    ).publish(engine_request_id="rid-publish-fail")

    with pytest.raises(PlanFailedError) as exc_info:
        plan.run()

    result = exc_info.value.result
    assert result.ok is False
    assert result.steps["step-0001"].action == "publish"
    assert result.steps["step-0001"].status.state == "failed"
    assert result.steps["step-0001"].status.message == "publish failed closed"
    runtime.close()


def test_runtime_hydrate_failure_surfaces_to_controller() -> None:
    publish_manifest = _sample_publish_manifest(rid="rid-hydrate-fail")

    def _response_factory(plan) -> node_agent_pb2.ExecutePlanResponse:  # noqa: ANN001
        response = node_agent_pb2.ExecutePlanResponse(
            request_id=plan.context.request_id,
            ok=False,
        )
        response.steps.add(
            step_id=str(plan.steps[0].step_id),
            target_id=str(plan.steps[0].target.target_id),
            action="hydrate",
            status=node_agent_pb2.OperationStatus(
                state=node_agent_pb2.OPERATION_STATE_FAILED,
                message="hydrate failed closed",
            ),
        )
        return response

    client = _FakeDaemonClient(
        "127.0.0.1:50051",
        response_factory=_response_factory,
    )
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
    )
    plan = runtime.plan(CallContext(request_id="req-runtime-hydrate-fail"))
    plan.on_instance(
        Instance(instance_id="inst-a", worker_id="worker-a", engine="sglang")
    ).hydrate(publish_manifest=publish_manifest)

    with pytest.raises(PlanFailedError) as exc_info:
        plan.run()

    result = exc_info.value.result
    assert result.ok is False
    assert result.steps["step-0001"].action == "hydrate"
    assert result.steps["step-0001"].status.state == "failed"
    assert result.steps["step-0001"].status.message == "hydrate failed closed"
    runtime.close()


def test_runtime_rejects_non_terminal_ingress_class_before_rpc() -> None:
    client = _FakeDaemonClient("127.0.0.1:50051")
    runtime = connect(
        daemon_address="127.0.0.1:50051",
        client_factory=lambda _address: cast(Any, client),
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
        client_factory=lambda _address: cast(Any, client),
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
        client_factory=lambda _address: cast(Any, client),
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
        client_factory=lambda _address: cast(Any, client),
    )

    worker_snapshot = runtime.directory().list_workers()
    instance_snapshot = runtime.directory().list_instances()
    route_snapshot = runtime.directory().resolve_instance_execution("inst-a")

    assert worker_snapshot.authority_mode == "GLOBAL_STORE_BACKED"
    assert worker_snapshot.value[0].daemon_id == "daemon-a"
    assert worker_snapshot.value[0].daemon_address == "10.0.0.1:50051"
    assert instance_snapshot.value[0].instance_id == "inst-a"
    assert instance_snapshot.value[0].execution_endpoint == "10.0.0.1:7001"
    assert route_snapshot.value.execution_host_kind == "node_agent_grpc"
    assert route_snapshot.cache_epoch == 7
    assert not hasattr(runtime.signals(), "list_workers")
    assert not hasattr(runtime.signals(), "list_instances")
    runtime.close()
