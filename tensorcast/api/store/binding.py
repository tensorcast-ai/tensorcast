#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from tensorcast.api.context import CallContext
from tensorcast.api.store.inplace_slot import InplaceSlot, _ctx_timeout_s
from tensorcast.api.store.types import ArtifactError

if TYPE_CHECKING:
    from tensorcast.api.store.artifact import Artifact
    from tensorcast.api.store.runtime import StoreRuntimeContext
    from tensorcast.daemon_ctl import DaemonCtl
    from tensorcast.proto.common.v1 import common_pb2


class _PublishedLeaseKeepalive:
    def __init__(self, client: "DaemonCtl", ttl_ms: int) -> None:
        self._client = client
        self._ttl_ms = int(ttl_ms)
        self._lease_id: str | None = None
        self._epoch = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def start(self, lease_id: str) -> None:
        if not lease_id or self._ttl_ms <= 0:
            return
        with self._lock:
            if self._lease_id == lease_id and self._thread and self._thread.is_alive():
                return
            self._stop_locked()
            self._epoch = 0
            self._lease_id = lease_id
            self._stop.clear()

            interval = max(1.0, self._ttl_ms / 2000.0)

            def _run() -> None:
                import random

                while not self._stop.wait(interval * (0.9 + 0.2 * random.random())):
                    try:
                        self._client.keep_alive_registered_artifact(
                            lease_id, self._ttl_ms, self._epoch
                        )
                        self._epoch += 1
                    except Exception:  # noqa: BLE001
                        continue

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            self._thread = t

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        thread = self._thread
        self._thread = None
        self._lease_id = None
        self._epoch = 0
        self._stop.set()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)


class Binding:
    """Stable, client-owned CUDA layout that can be refilled in-place."""

    def __init__(
        self,
        slot: InplaceSlot,
        *,
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> None:
        self._slot = slot
        runtime: StoreRuntimeContext = slot._runtime
        self._runtime = runtime
        self._publish_ttl_ms = 0
        self._keepalive = _PublishedLeaseKeepalive(
            runtime.ensure_client(), self._publish_ttl_ms
        )
        if publish:
            self.publish_replica(ctx=ctx)

    def __enter__(self) -> "Binding":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._slot.tensors

    @property
    def artifact_id(self) -> str:
        return self._slot.artifact_id

    @property
    def selection(self) -> "common_pb2.ArtifactSelection":
        return self._slot.selection

    def swap(
        self,
        artifact: "Artifact | str",
        *,
        publish: bool = False,
        activate_key: str | None = None,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        operation_id = uuid.uuid4().hex
        self._stop_keepalive()
        try:
            self._slot.swap(
                artifact,
                publish=publish,
                wait=wait,
                drain_timeout_s=drain_timeout_s,
                ctx=ctx,
                operation_id=operation_id,
                publish_ttl_ms=self._publish_ttl_ms if publish else None,
            )
        except Exception:
            if self._slot.published_lease_id:
                self._start_keepalive()
            raise
        if publish:
            self._start_keepalive()
        if activate_key:
            self._activate_key(
                activate_key,
                expected_active_artifact_id=expected_active_artifact_id,
                expected_active_generation=expected_active_generation,
                operation_id=operation_id,
                ctx=ctx,
            )

    def close(self) -> None:
        self._stop_keepalive()
        self._slot.close()

    def publish_replica(self, *, ctx: CallContext | None = None) -> None:
        self._slot.publish_replica(ttl_ms=self._publish_ttl_ms, ctx=ctx)
        self._start_keepalive()

    def _start_keepalive(self) -> None:
        lease_id = self._slot.published_lease_id
        if lease_id:
            self._keepalive.start(lease_id)

    def _stop_keepalive(self) -> None:
        self._keepalive.stop()

    def _activate_key(
        self,
        key: str,
        *,
        expected_active_artifact_id: str | None,
        expected_active_generation: int | None,
        operation_id: str,
        ctx: CallContext | None,
    ) -> None:
        if not key:
            raise ArtifactError(
                "activate_key must be non-empty",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        timeout_s = _ctx_timeout_s(ctx)
        result = self._runtime.ensure_client().swap_key_mapping(
            key=key,
            new_artifact_id=self._slot.artifact_id,
            expected_artifact_id=expected_active_artifact_id,
            expected_generation=expected_active_generation,
            operation_id=operation_id,
            timeout_s=timeout_s if timeout_s is not None else 10.0,
        )
        if not result.ok:
            message = (
                f"Activation key '{key}' conflict: current artifact_id="
                f"{result.artifact_id or 'unknown'} generation={result.generation}"
            )
            raise ArtifactError(
                message,
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        self._runtime.invalidate_artifact(None, key=key, reason="activate_key")


__all__ = ["Binding"]
