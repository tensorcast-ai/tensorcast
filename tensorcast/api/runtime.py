#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from tensorcast.api.context import CallContext
from tensorcast.api.plan.plan import (
    _PUBLIC_CONTINUATION_REQUIRED_EXECUTION_CLASS,
    _TERMINAL_ONLY_EXECUTION_CLASS,
    Plan,
    PlanExecutionClass,
    PlanResult,
)
from tensorcast.api.signals import TensorCastSignals
from tensorcast.daemon_ctl import DaemonCtl

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.proto.plan.v1 import plan_pb2


_RUNTIME_LOCK = threading.RLock()
_ACTIVE_RUNTIME: "Runtime | None" = None


@dataclass(slots=True)
class Runtime:
    daemon_address: str
    _client: DaemonCtl

    @property
    def client(self) -> DaemonCtl:
        return self._client

    def store(self) -> "Store":
        from tensorcast.api.store import Store

        return Store(self.daemon_address)

    def signals(self) -> TensorCastSignals:
        return TensorCastSignals(self._client)

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
        response = self._client.execute_plan(
            plan=plan_spec,
            execution_class=execution_class,
            dry_run=dry_run,
        )
        return PlanResult.from_node_agent_response(response)

    def close(self) -> None:
        global _ACTIVE_RUNTIME
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
