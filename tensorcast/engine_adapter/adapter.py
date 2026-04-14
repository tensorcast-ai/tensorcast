#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from tensorcast.api._config import StorePolicy
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan.targets import TargetSpec
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store import Artifact, Store
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.serving_builder import (
    PURE_TRANSFORM_SERVING_ARG_ABI_VERSION,
    PURE_TRANSFORM_SERVING_ARG_ADAPTER_VERSION,
    PURE_TRANSFORM_SERVING_ARG_BUILD_PIPELINE_VERSION,
    PURE_TRANSFORM_SERVING_ARG_CONTRACT_FAMILY,
    PURE_TRANSFORM_SERVING_ARG_ENABLE,
    PURE_TRANSFORM_SERVING_ARG_FRAMEWORK_NAME,
    PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON,
    PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF,
    PURE_TRANSFORM_SERVING_ARG_REPRESENTATION_CONTRACT_HASH,
    PURE_TRANSFORM_SERVING_ARG_SERVING_VERSION_KEY,
    PURE_TRANSFORM_SERVING_ARG_SOURCE_VERSION_KEY,
    RepresentationPublishSpec,
    build_pure_transform_publication_bundle_from_registered_artifact,
    prepare_pure_transform_serving_registration,
)
from tensorcast.engine_adapter.artifact_api import (
    BatchResult,
    HydrateResult,
    ManifestResult,
    PublishResult,
    SealedByteArtifact,
)
from tensorcast.types import BuilderMode, ServingBuildIntent


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


def _serving_bool_arg(args: Mapping[str, str | int], key: str) -> bool:
    value = args.get(key)
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ArtifactError(
        f"TransformSpec.args['{key}'] must be a boolean-like string or int",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _serving_optional_arg(
    args: Mapping[str, str | int],
    key: str,
) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _serving_required_arg(
    args: Mapping[str, str | int],
    key: str,
) -> str:
    value = _serving_optional_arg(args, key)
    if value is None:
        raise ArtifactError(
            f"TransformSpec.args['{key}'] is required when tc_serving_enable=1",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return value


def _pure_transform_build_intent(
    ctx: "TransformContext",
    *,
    source_artifact: Artifact,
) -> ServingBuildIntent | None:
    publication_spec = ctx.spec.publication_spec
    if publication_spec is not None:
        return publication_spec.build_intent.model_copy(
            update={"source_artifact_ref": source_artifact._ensure_identified()}
        )
    if not _serving_bool_arg(ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_ENABLE):
        return None
    return ServingBuildIntent(
        representation_contract_hash=_serving_optional_arg(
            ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_REPRESENTATION_CONTRACT_HASH
        ),
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name=_serving_required_arg(
            ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_FRAMEWORK_NAME
        ),
        adapter_version=_serving_required_arg(
            ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_ADAPTER_VERSION
        ),
        serving_abi_version=_serving_required_arg(
            ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_ABI_VERSION
        ),
        build_pipeline_version=_serving_required_arg(
            ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_BUILD_PIPELINE_VERSION
        ),
        source_artifact_ref=source_artifact._ensure_identified(),
    )


def _maybe_build_pure_transform_publication_bundle(
    ctx: "TransformContext",
    registered_artifact: object,
    *,
    source_artifact: Artifact,
    build_intent: ServingBuildIntent | None,
) -> RepresentationPublishSpec | None:
    if build_intent is None:
        return None
    if not isinstance(registered_artifact, RegisteredArtifact):
        raise ArtifactError(
            "PURE_TRANSFORM serving publication requires transform_register to return RegisteredArtifact",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    publication_spec = ctx.spec.publication_spec
    return build_pure_transform_publication_bundle_from_registered_artifact(
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=(
            publication_spec.contract_family
            if publication_spec is not None
            else _serving_optional_arg(
                ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_CONTRACT_FAMILY
            )
        ),
        serving_artifact=registered_artifact,
        source_version_key=(
            publication_spec.source_version_key
            if publication_spec is not None
            else _serving_optional_arg(
                ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_SOURCE_VERSION_KEY
            )
        ),
        serving_version_key=(
            publication_spec.serving_version_key
            if publication_spec is not None
            else _serving_optional_arg(
                ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_SERVING_VERSION_KEY
            )
        ),
        logical_topology_json=(
            publication_spec.logical_topology_json
            if publication_spec is not None
            else _serving_optional_arg(
                ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON
            )
        ),
        serving_manifest_ref=(
            publication_spec.serving_manifest_ref
            if publication_spec is not None
            else _serving_optional_arg(
                ctx.spec.args, PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF
            )
        ),
        layout_id=(
            publication_spec.layout_id if publication_spec is not None else None
        ),
        requirements=(
            publication_spec.requirements if publication_spec is not None else None
        ),
        readiness_policy=(
            publication_spec.readiness_policy if publication_spec is not None else None
        ),
        structural_view_ids=(
            publication_spec.structural_view_ids if publication_spec is not None else ()
        ),
    )


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
ManifestFn = Callable[[str, CallContext | None], ManifestResult]
PublishFn = Callable[
    [str, int | None, tuple[SealedByteArtifact, ...], CallContext | None], PublishResult
]
HydrateFn = Callable[[str, CallContext | None], HydrateResult]
EvictLocalFn = Callable[[str | None, CallContext | None], BatchResult]


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
        self._manifest_fn: ManifestFn | None = None
        self._publish_fn: PublishFn | None = None
        self._hydrate_fn: HydrateFn | None = None
        self._evict_local_fn: EvictLocalFn | None = None
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

    def register_artifact_fns(
        self,
        *,
        manifest: ManifestFn | None = None,
        publish: PublishFn | None = None,
        hydrate: HydrateFn | None = None,
        evict_local: EvictLocalFn | None = None,
    ) -> None:
        if manifest is not None:
            self._manifest_fn = manifest
        if publish is not None:
            self._publish_fn = publish
        if hydrate is not None:
            self._hydrate_fn = hydrate
        if evict_local is not None:
            self._evict_local_fn = evict_local

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

    def execute_manifest(
        self,
        *,
        engine_request_id: str,
        ctx: CallContext | None = None,
    ) -> ManifestResult:
        engine_request_id = str(engine_request_id).strip()
        if not engine_request_id:
            raise ArtifactError(
                "engine_request_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if self._manifest_fn is None:
            raise ArtifactError(
                "manifest is not configured on this engine adapter",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        return self._manifest_fn(engine_request_id, ctx)

    def execute_publish(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None = None,
        sealed_artifacts: Sequence[SealedByteArtifact] = (),
        ctx: CallContext | None = None,
    ) -> PublishResult:
        engine_request_id = str(engine_request_id).strip()
        if not engine_request_id:
            raise ArtifactError(
                "engine_request_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if ttl_ms is not None and int(ttl_ms) <= 0:
            raise ArtifactError(
                "ttl_ms must be positive when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if self._publish_fn is None:
            raise ArtifactError(
                "publish is not configured on this engine adapter",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        normalized_artifacts = tuple(sealed_artifacts)
        for artifact in normalized_artifacts:
            if not isinstance(artifact, SealedByteArtifact):
                raise ArtifactError(
                    "publish requires sealed byte artifacts",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        return self._publish_fn(
            engine_request_id,
            int(ttl_ms) if ttl_ms is not None else None,
            normalized_artifacts,
            ctx,
        )

    def execute_hydrate(
        self,
        *,
        engine_request_id: str,
        ctx: CallContext | None = None,
    ) -> HydrateResult:
        engine_request_id = str(engine_request_id).strip()
        if not engine_request_id:
            raise ArtifactError(
                "engine_request_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if self._hydrate_fn is None:
            raise ArtifactError(
                "hydrate is not configured on this engine adapter",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        return self._hydrate_fn(engine_request_id, ctx)

    def execute_evict_local(
        self,
        *,
        engine_request_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> BatchResult:
        if self._evict_local_fn is None:
            raise ArtifactError(
                "evict_local is not configured on this engine adapter",
                status_code="UNIMPLEMENTED",
                retryable=False,
            )
        normalized_request_id = (
            str(engine_request_id).strip() if engine_request_id is not None else None
        )
        if normalized_request_id == "":
            raise ArtifactError(
                "engine_request_id must be non-empty when provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return self._evict_local_fn(normalized_request_id, ctx)

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
            selected_source = (
                ctx.source.subset(list(ctx.tensor_names))
                if ctx.tensor_names
                else ctx.source
            )
            build_intent = _pure_transform_build_intent(
                ctx,
                source_artifact=selected_source,
            )
            publication_spec = ctx.spec.publication_spec
            tensors = selected_source.tensor_dict(
                device="cpu",
                ctx=ctx.ctx,
            )
            registration_tensors = dict(tensors)
            if build_intent is not None:
                prepared = prepare_pure_transform_serving_registration(
                    build_intent=build_intent,
                    source_artifact=selected_source,
                    tensors=registration_tensors,
                    logical_topology_json=(
                        publication_spec.logical_topology_json
                        if publication_spec is not None
                        else _serving_optional_arg(
                            ctx.spec.args,
                            PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON,
                        )
                    ),
                    serving_manifest_ref=(
                        publication_spec.serving_manifest_ref
                        if publication_spec is not None
                        else _serving_optional_arg(
                            ctx.spec.args,
                            PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF,
                        )
                    ),
                )
                registration_tensors = prepared.tensors
            registered_artifact = ctx.store.register(
                registration_tensors, key=ctx.out_key, policy=ctx.policy
            )
            return (
                _maybe_build_pure_transform_publication_bundle(
                    ctx,
                    registered_artifact,
                    source_artifact=selected_source,
                    build_intent=build_intent,
                )
                or registered_artifact
            )

        self.register_transform_fn("identity.v1", into=_into, register=_register)


__all__ = [
    "EngineAdapter",
    "EvictLocalFn",
    "HydrateFn",
    "ManifestFn",
    "PublishFn",
    "TargetRegistry",
    "TransformContext",
    "TransformPlugin",
    "TransformRegistry",
]
