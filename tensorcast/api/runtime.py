#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from tensorcast.api.context import CallContext
from tensorcast.api.directory import TensorCastDirectory
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan.plan import (
    _PUBLIC_CONTINUATION_REQUIRED_EXECUTION_CLASS,
    _TERMINAL_ONLY_EXECUTION_CLASS,
    Plan,
    PlanExecutionClass,
    PlanResult,
)
from tensorcast.api.signals import TensorCastSignals
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.engine_adapter.artifact_api import PublishManifest, PublishResult

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.proto.plan.v1 import plan_pb2


_RUNTIME_LOCK = threading.RLock()
_ACTIVE_RUNTIME: "Runtime | None" = None


@dataclass(slots=True)
class Runtime:
    daemon_address: str
    _client: DaemonCtl
    _publish_manifest_cache: dict[str, dict[bytes, PublishManifest]] = field(
        default_factory=dict,
        repr=False,
    )
    _publish_manifest_cache_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @property
    def client(self) -> DaemonCtl:
        return self._client

    def store(self) -> "Store":
        from tensorcast.api.store import Store

        return Store(self.daemon_address)

    def signals(self) -> TensorCastSignals:
        return TensorCastSignals(self._client)

    def directory(self) -> TensorCastDirectory:
        return TensorCastDirectory(self._client)

    def plan(
        self,
        ctx: CallContext,
        *,
        execution_class: PlanExecutionClass = _TERMINAL_ONLY_EXECUTION_CLASS,
    ) -> Plan:
        return Plan(ctx, runtime=self, execution_class=execution_class)

    def execute_plan(
        self,
        plan_spec: "plan_pb2.PlanSpec",
        *,
        execution_class: PlanExecutionClass = _TERMINAL_ONLY_EXECUTION_CLASS,
        dry_run: bool = False,
    ) -> PlanResult:
        if execution_class == _PUBLIC_CONTINUATION_REQUIRED_EXECUTION_CLASS:
            raise RuntimeError(
                "Daemon ingress currently supports terminal_only execution only"
            )
        resolved_plan_spec = self._rewrite_compat_hydrate_actions(plan_spec)
        timeout_s: float | None = None
        if resolved_plan_spec.context.HasField("deadline_ms"):
            timeout_s = max(
                0.001,
                float(resolved_plan_spec.context.deadline_ms) / 1000.0,
            )
        if timeout_s is None:
            response = self._client.execute_plan(
                plan=resolved_plan_spec,
                execution_class=execution_class,
                dry_run=dry_run,
            )
        else:
            response = self._client.execute_plan(
                plan=resolved_plan_spec,
                execution_class=execution_class,
                dry_run=dry_run,
                timeout_s=timeout_s,
            )
        result = PlanResult.from_node_agent_response(response)
        self._remember_publish_manifests_from_result(result)
        return result

    def remember_publish_manifest(self, publish_manifest: PublishManifest) -> None:
        if not isinstance(publish_manifest, PublishManifest):
            raise ArtifactError(
                "publish_manifest must be a PublishManifest",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        engine_request_id = str(
            publish_manifest.artifact_manifest.engine_request_id
        ).strip()
        if not engine_request_id:
            raise ArtifactError(
                "publish_manifest.artifact_manifest.engine_request_id is required for compatibility caching",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        serialized = publish_manifest.to_proto().SerializeToString(deterministic=True)
        with self._publish_manifest_cache_lock:
            bucket = self._publish_manifest_cache.setdefault(engine_request_id, {})
            bucket[serialized] = publish_manifest

    def resolve_publish_manifest(self, *, engine_request_id: str) -> PublishManifest:
        normalized_request_id = str(engine_request_id).strip()
        if not normalized_request_id:
            raise ArtifactError(
                "engine_request_id is required for compatibility hydrate resolution",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        with self._publish_manifest_cache_lock:
            matches: tuple[PublishManifest, ...] = tuple(
                self._publish_manifest_cache.get(normalized_request_id, {}).values()
            )
        if not matches:
            raise ArtifactError(
                "compatibility hydrate(engine_request_id=...) failed closed: no cached PublishManifest found; explicit publish_manifest is required",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if len(matches) > 1:
            raise ArtifactError(
                "compatibility hydrate(engine_request_id=...) failed closed: multiple cached PublishManifest generations found; explicit publish_manifest is required",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return next(iter(matches))

    def clear_publish_manifest_cache(
        self, *, engine_request_id: str | None = None
    ) -> None:
        if engine_request_id is None:
            with self._publish_manifest_cache_lock:
                self._publish_manifest_cache.clear()
            return
        normalized_request_id = str(engine_request_id).strip()
        if not normalized_request_id:
            raise ArtifactError(
                "engine_request_id must be non-empty when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        with self._publish_manifest_cache_lock:
            self._publish_manifest_cache.pop(normalized_request_id, None)

    def _rewrite_compat_hydrate_actions(
        self, plan_spec: "plan_pb2.PlanSpec"
    ) -> "plan_pb2.PlanSpec":
        needs_rewrite = False
        for step in plan_spec.steps:
            if step.action.WhichOneof("kind") != "hydrate":
                continue
            if step.action.hydrate.WhichOneof("request_source") != "engine_request_id":
                continue
            needs_rewrite = True
            break
        if not needs_rewrite:
            return plan_spec

        rewritten = type(plan_spec)()
        rewritten.CopyFrom(plan_spec)
        for step in rewritten.steps:
            if step.action.WhichOneof("kind") != "hydrate":
                continue
            if step.action.hydrate.WhichOneof("request_source") != "engine_request_id":
                continue
            publish_manifest = self.resolve_publish_manifest(
                engine_request_id=str(step.action.hydrate.engine_request_id)
            )
            step.action.hydrate.ClearField("engine_request_id")
            step.action.hydrate.publish_manifest.CopyFrom(publish_manifest.to_proto())
        return rewritten

    def _remember_publish_manifests_from_result(self, result: PlanResult) -> None:
        for step in result.steps.values():
            if step.status.state != "success":
                continue
            artifact_result = step.artifact_result
            if not isinstance(artifact_result, PublishResult):
                continue
            if artifact_result.publish_manifest is None:
                continue
            self.remember_publish_manifest(artifact_result.publish_manifest)

    def close(self) -> None:
        global _ACTIVE_RUNTIME
        self.clear_publish_manifest_cache()
        with _RUNTIME_LOCK:
            if _ACTIVE_RUNTIME is self:
                _ACTIVE_RUNTIME = None


def connect(
    *,
    daemon_address: str,
    client_factory: Callable[[str], DaemonCtl] = DaemonCtl,
) -> Runtime:
    runtime = Runtime(
        daemon_address=str(daemon_address),
        _client=client_factory(str(daemon_address)),
    )
    global _ACTIVE_RUNTIME
    with _RUNTIME_LOCK:
        _ACTIVE_RUNTIME = runtime
    return runtime


def get_active_runtime() -> Runtime | None:
    with _RUNTIME_LOCK:
        return _ACTIVE_RUNTIME


def runtime() -> Runtime:
    active = get_active_runtime()
    if active is None:
        raise RuntimeError("No active TensorCast runtime; call tensorcast.connect()")
    return active


__all__ = [
    "Runtime",
    "connect",
    "get_active_runtime",
    "runtime",
]
