#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from tensorcast.api._config import StorePolicy
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan.targets import TargetSpec
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store import Artifact, Store


def _encode_token(token: bytes) -> str:
    import base64

    raw = base64.urlsafe_b64encode(token).decode("ascii")
    return raw.rstrip("=")


def _spec_payload_bytes(
    *,
    instance_id: str,
    name: str,
    tensors: Mapping[str, str],
    layout_hash: str | None,
    expires_at_ms: int,
) -> bytes:
    payload = {
        "instance_id": instance_id,
        "name": name,
        "layout_hash": layout_hash or "",
        "expires_at_ms": int(expires_at_ms),
        "tensors": [(k, tensors[k]) for k in sorted(tensors)],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TransformContext:
    spec: TransformSpec
    source: Artifact
    store: Store
    targets: Mapping[str, torch.Tensor] | None
    out_key: str | None
    policy: StorePolicy | None
    ctx: CallContext | None
    tensor_names: tuple[str, ...] | None = None


TransformIntoFn = Callable[[TransformContext], None]
TransformRegisterFn = Callable[[TransformContext], object | None]


@dataclass(frozen=True, slots=True)
class TransformPlugin:
    name: str
    into: TransformIntoFn | None = None
    register: TransformRegisterFn | None = None


@dataclass(frozen=True, slots=True)
class _BufferEntry:
    tensor: torch.Tensor
    expires_at_ms: int


class TargetRegistry:
    def __init__(self) -> None:
        self._buffers: dict[str, _BufferEntry] = {}

    def register_handle(
        self, handle_id: str, tensor: torch.Tensor, *, expires_at_ms: int
    ) -> None:
        self._buffers[str(handle_id)] = _BufferEntry(
            tensor=tensor, expires_at_ms=int(expires_at_ms)
        )

    def resolve_handle(self, handle_id: str, *, now_ms: int) -> torch.Tensor | None:
        entry = self._buffers.get(handle_id)
        if entry is None:
            return None
        if int(entry.expires_at_ms) <= int(now_ms):
            self._buffers.pop(handle_id, None)
            return None
        return entry.tensor

    def prune_expired(self, *, now_ms: int) -> None:
        expired = [
            handle
            for handle, entry in self._buffers.items()
            if int(entry.expires_at_ms) <= int(now_ms)
        ]
        for handle in expired:
            self._buffers.pop(handle, None)


class TransformRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, TransformPlugin] = {}

    def register(self, plugin: TransformPlugin) -> None:
        if not plugin.name:
            raise ArtifactError(
                "Transform plugin name is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._plugins[plugin.name] = plugin

    def resolve(self, name: str) -> TransformPlugin | None:
        return self._plugins.get(name)

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins.keys()))


class EngineAdapter:
    def __init__(
        self,
        *,
        instance_id: str,
        engine: str,
        capability_secret: bytes | None = None,
        default_target_ttl_ms: int = 30_000,
        register_identity_transform: bool = True,
    ) -> None:
        if not instance_id:
            raise ArtifactError(
                "EngineAdapter requires a non-empty instance_id",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not engine:
            raise ArtifactError(
                "EngineAdapter requires a non-empty engine name",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._instance_id = instance_id
        self._engine = engine
        self._secret = capability_secret or os.urandom(32)
        self._default_ttl_ms = max(1, int(default_target_ttl_ms))
        self._targets = TargetRegistry()
        self._transforms = TransformRegistry()
        if register_identity_transform:
            self._register_identity_transform()

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def transform_registry(self) -> TransformRegistry:
        return self._transforms

    def register_transform(self, plugin: TransformPlugin) -> None:
        self._transforms.register(plugin)

    def register_transform_fn(
        self,
        name: str,
        *,
        into: TransformIntoFn | None = None,
        register: TransformRegisterFn | None = None,
    ) -> TransformPlugin:
        plugin = TransformPlugin(name=name, into=into, register=register)
        self.register_transform(plugin)
        return plugin

    def mint_target(
        self,
        name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        layout_hash: str | None = None,
        ttl_ms: int | None = None,
        now_ms: int | None = None,
    ) -> TargetSpec:
        if not name:
            raise ArtifactError(
                "Target name is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if not tensors:
            raise ArtifactError(
                "Target tensors are required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        now_value = int(now_ms) if now_ms is not None else int(time.time() * 1000)
        ttl_value = (
            int(ttl_ms)
            if ttl_ms is not None and int(ttl_ms) > 0
            else self._default_ttl_ms
        )
        expires_at_ms = now_value + ttl_value
        handles: dict[str, str] = {}
        for tensor_name, tensor in tensors.items():
            handle_id = uuid.uuid4().hex
            handles[str(tensor_name)] = handle_id
            self._targets.register_handle(
                handle_id, tensor, expires_at_ms=expires_at_ms
            )
        token = self._token_for_spec(
            instance_id=self._instance_id,
            name=name,
            tensors=handles,
            layout_hash=layout_hash,
            expires_at_ms=expires_at_ms,
        )
        return TargetSpec(
            instance_id=self._instance_id,
            name=name,
            tensors=handles,
            layout_hash=layout_hash,
            expires_at_ms=expires_at_ms,
            capability_token=token,
        )

    def resolve_targets(self, spec: TargetSpec) -> dict[str, torch.Tensor]:
        spec.validate()
        if spec.instance_id != self._instance_id:
            raise ArtifactError(
                "TargetSpec instance_id does not match engine adapter",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        expected = self._token_for_spec(
            instance_id=spec.instance_id,
            name=spec.name,
            tensors=spec.tensors,
            layout_hash=spec.layout_hash,
            expires_at_ms=int(spec.expires_at_ms),
        )
        if not hmac.compare_digest(expected, spec.capability_token):
            raise ArtifactError(
                "TargetSpec capability token invalid",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        now_ms = int(time.time() * 1000)
        resolved: dict[str, torch.Tensor] = {}
        for name, handle_id in spec.tensors.items():
            tensor = self._targets.resolve_handle(handle_id, now_ms=now_ms)
            if tensor is None:
                raise ArtifactError(
                    f"Target handle '{name}' is unavailable",
                    status_code="NOT_FOUND",
                    retryable=True,
                )
            resolved[name] = tensor
        return resolved

    def execute_transform_into(
        self,
        *,
        spec: TransformSpec,
        source: Artifact,
        targets: TargetSpec,
        store: Store,
        ctx: CallContext | None = None,
        tensor_names: tuple[str, ...] | None = None,
    ) -> None:
        plugin = self._require_transform(spec.name)
        if plugin.into is None:
            raise ArtifactError(
                f"Transform '{spec.name}' does not support into",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        resolved_targets = self.resolve_targets(targets)
        plugin.into(
            TransformContext(
                spec=spec,
                source=source,
                store=store,
                targets=resolved_targets,
                out_key=None,
                policy=None,
                ctx=ctx,
                tensor_names=tensor_names,
            )
        )

    def execute_transform_register(
        self,
        *,
        spec: TransformSpec,
        source: Artifact,
        out_key: str,
        store: Store,
        policy: StorePolicy | None = None,
        ctx: CallContext | None = None,
        tensor_names: tuple[str, ...] | None = None,
    ) -> object | None:
        plugin = self._require_transform(spec.name)
        if plugin.register is None:
            raise ArtifactError(
                f"Transform '{spec.name}' does not support register",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return plugin.register(
            TransformContext(
                spec=spec,
                source=source,
                store=store,
                targets=None,
                out_key=out_key,
                policy=policy,
                ctx=ctx,
                tensor_names=tensor_names,
            )
        )

    def _require_transform(self, name: str) -> TransformPlugin:
        plugin = self._transforms.resolve(name)
        if plugin is None:
            raise ArtifactError(
                f"Unknown transform '{name}'",
                status_code="NOT_FOUND",
                retryable=False,
            )
        return plugin

    def _token_for_spec(
        self,
        *,
        instance_id: str,
        name: str,
        tensors: Mapping[str, str],
        layout_hash: str | None,
        expires_at_ms: int,
    ) -> str:
        payload = _spec_payload_bytes(
            instance_id=instance_id,
            name=name,
            tensors=tensors,
            layout_hash=layout_hash,
            expires_at_ms=expires_at_ms,
        )
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return _encode_token(digest)

    def _register_identity_transform(self) -> None:
        def _into(ctx: TransformContext) -> None:
            if ctx.targets is None:
                raise ArtifactError(
                    "Identity transform requires targets",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            ctx.source.tensor_dict_into(
                dict(ctx.targets),
                ctx=ctx.ctx,
            )

        def _register(ctx: TransformContext) -> object | None:
            if not ctx.out_key:
                raise ArtifactError(
                    "Identity transform requires out_key",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            tensors = ctx.source.tensor_dict(
                device="cpu",
                names=list(ctx.tensor_names) if ctx.tensor_names else None,
                ctx=ctx.ctx,
            )
            return ctx.store.register(tensors, key=ctx.out_key, policy=ctx.policy)

        self.register_transform_fn("identity.v1", into=_into, register=_register)


__all__ = [
    "EngineAdapter",
    "TargetRegistry",
    "TransformContext",
    "TransformPlugin",
    "TransformRegistry",
]
