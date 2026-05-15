#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import grpc
import torch

from tensorcast._c_ext import (
    get_cuda_memory_handle,
    get_cuda_memory_handle_with_offset,
)
from tensorcast.api._config import RegisterArtifactOptions, StorePolicy
from tensorcast.api._device import (
    device_uuid_for,
    protocol_device_id_for,
    resolve_device,
)
from tensorcast.api._materialize import (
    MaterializationPayload,
    materialize_artifact_v2,
)
from tensorcast.api._region_cache import (
    register_region as _cache_register_region,
)
from tensorcast.api._region_cache import (
    unregister_region as _cache_unregister_region,
)
from tensorcast.api._register import RegistrationResult, _register_artifact_core
from tensorcast.api._runtime import require_runtime
from tensorcast.api.context import CallContext
from tensorcast.api.operation import (
    DaemonGlobalStoreOperation,
    Operation,
    OperationError,
    OperationStatus,
    OperationTimeoutError,
    PollingOperation,
)
from tensorcast.api.store.artifact import (
    Artifact,
    ArtifactDescriptor,
    MaterializationDiagnostics,
    PlacementPin,
    PrefetchedReplica,
    TensorDictMaterializationResult,
    TensorMeta,
)
from tensorcast.api.store.async_ops import ArtifactFuture
from tensorcast.api.store.batch_context import (
    BatchContext,
    MaterializationBatcher,
)
from tensorcast.api.store.binding import (
    Binding,
    BindingUpdateEpoch,
    SealedBindingValue,
    StagedBindingValue,
)
from tensorcast.api.store.binding_state import parse_binding_value_or_raise
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.inplace_slot import InplaceSlot, _ctx_timeout_s
from tensorcast.api.store.mapped_binding import (
    CopyPlan,
    CopyPlanEntry,
    Range,
    TargetTensors,
    normalize_copy_plan,
)
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.owned_binding_layout import (
    BindingLayout,
    build_binding_layout,
    build_owned_layout,
)
from tensorcast.api.store.owned_binding_slot import (
    OwnedBindingSlot,
    restore_owned_binding_tensors,
)
from tensorcast.api.store.realization_plan import (
    BindingRealizationEntry,
    BindingRealizationPlan,
    binding_realization_plan_to_proto,
    normalize_binding_realization_plan,
)
from tensorcast.api.store.region_utils import collect_storage_bases
from tensorcast.api.store.registration import RegistrationPipeline
from tensorcast.api.store.runtime import (
    StoreRuntimeContext,
    shutdown_context,
)
from tensorcast.api.store.runtime import (
    get_context as get_runtime_context,
)
from tensorcast.api.store.serving_binding_reference_consumer import (
    REFERENCE_RUNTIME,
    ReferenceServingAcquireResult,
    ReferenceServingResolvedSpec,
    ReferenceServingTensorSpec,
    acquire_reference_binding,
    build_reference_resolved_spec,
    build_reference_target_layout,
    build_reference_tensor_index_bytes,
    prefetch_reference_binding,
    release_reference_acquire,
    target_from_reference_cache_record,
    unpack_prefetched_serving_binding,
    write_reference_resolved_spec_cache_entry,
)
from tensorcast.api.store.serving_binding_spec_cache import (
    ServingBindingSpecCacheGroupIndex,
    ServingBindingSpecCacheRecord,
    canonical_json_bytes,
    read_matching_resolved_spec_cache_entry,
    read_resolved_spec_cache_entry,
    read_resolved_spec_cache_group_index,
    serving_binding_spec_cache_root,
    write_resolved_spec_cache_entry,
    write_resolved_spec_cache_group_index,
)
from tensorcast.api.store.serving_builder import (
    PreparedServingRegistration,
    RegisteredServingPublication,
    build_binding_finalize_admission_facts,
    build_binding_finalize_publication_bundle,
    build_pure_transform_publication_bundle,
    build_pure_transform_publication_bundle_from_registered_artifact,
    build_pure_transform_publication_spec,
    build_pure_transform_serving_args,
    build_pure_transform_transform_spec,
    build_serving_publication_bundle,
    build_serving_publication_bundle_from_registered_artifact,
    compute_pure_transform_representation_contract_hash,
    compute_serving_tensor_schema_hash,
    count_canonical_serving_tensors,
    prepare_binding_finalize_serving_registration,
    prepare_pure_transform_serving_registration,
    prepare_serving_registration,
)
from tensorcast.api.store.types import (
    ArtifactError,
    ArtifactStatusCode,
    CanonicalIndex,
    CanonicalIndexEntry,
    LeaseHandle,
    PersistenceShardStatus,
    PersistenceStatusResult,
    ReplicaInfo,
    RetryPolicy,
    StoreCapabilities,
    StoreOptions,
    TensorDict,
)
from tensorcast.api.store.view_composer import compute_index_multihash
from tensorcast.api.store.views import TransformPlacement, ViewOrchestrator
from tensorcast.common.identity import ArtifactIdKind, infer_artifact_id_kind
from tensorcast.daemon_ctl import get_daemon_client
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.profile_utils import (
    emit_tensorcast_profile_event,
    tensorcast_profile_stage,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.layout.v1 import layout_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    SERVING_BUILD_DIGEST_VERSION,
    SERVING_MANIFEST_TENSOR_NAME,
    AssemblyAttemptRef,
    AssemblyCloseoutContract,
    AssemblyContractFamily,
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    BindingPromotionStatusState,
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    BlobRef,
    BuilderMode,
    DeregisterArtifactOutcome,
    ExecutionDiagnostics,
    FinalizeClass,
    GroupRealizationAcquireRef,
    HashBackend,
    HashLocation,
    HostSharedRegionAttachment,
    HostSharedRegionClass,
    IdentityMintStrategy,
    LocalRegionHandle,
    PartialSealResult,
    PrefetchedServingBinding,
    PrefetchedServingBindingSet,
    PrefetchRetentionPolicy,
    PublicDiskSourceHandle,
    PublishedModelVersion,
    RegionMemoryKind,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    SealAssemblyResult,
    ServingAdmissionFacts,
    ServingArtifactManifest,
    ServingBindingMemberRef,
    ServingBindingReadiness,
    ServingBindingResolvedLayout,
    ServingBindingResolvedSpecCacheEntry,
    ServingBindingSetTarget,
    ServingBindingSourceKind,
    ServingBindingSourceMemberRef,
    ServingBindingSourceRef,
    ServingBindingSourceReuseDecision,
    ServingBindingSourceReuseMode,
    ServingBindingTarget,
    ServingBuildIntent,
    ServingPublicationSubject,
    ServingRuntimePolicy,
    ServingRuntimePolicyInput,
    ServingSupportLevel,
    SourceBoundCapability,
    SourceBoundPlanDiagnostics,
    VramRegionHandle,
    build_serving_manifest_ref,
    coerce_serving_runtime_policy,
    parse_serving_manifest_ref,
)
from tensorcast.types import (
    ArtifactDescriptor as TypedArtifactDescriptor,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tensorcast.api.plan import PlanResult, PlanStepRef


def _coerce_representation_publish_closeout(
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
) -> AssemblyCloseoutContract:
    if isinstance(publication, RepresentationPublishSpec):
        closeout_contract = publication.closeout_contract
    elif isinstance(publication, AssemblyCloseoutContract):
        closeout_contract = publication
    else:
        raise ArtifactError(
            "representation publication requires RepresentationPublishSpec or AssemblyCloseoutContract",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if closeout_contract.kind != "representation_publish":
        raise ArtifactError(
            "representation publication helpers require AssemblyCloseoutContract(kind='representation_publish')",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if closeout_contract.representation_publish_contract is None:
        raise ArtifactError(
            "representation publication helpers require representation_publish_contract",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return closeout_contract


def _coerce_representation_publish_spec(
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
) -> RepresentationPublishSpec | None:
    if isinstance(publication, RepresentationPublishSpec):
        return publication
    return None


def _resolve_runtime_global_store_address() -> str | None:
    with contextlib.suppress(Exception):
        import tensorcast.runtime as tensorcast_runtime

        session = tensorcast_runtime.reconcile()
        if session is not None and session.global_store_address:
            return str(session.global_store_address)
    for env_name in ("TENSORCAST_GLOBAL_STORE_ADDRESS", "TENSORCAST_GLOBAL_STORE"):
        raw = os.getenv(env_name)
        if raw:
            resolved = str(raw).strip()
            if resolved:
                return resolved
    return None


def _artifact_index_multihash_for_layout_provision(
    store: "Store",
    *,
    artifact_id: str,
) -> str:
    index_multihash, _ = _split_mi2_artifact_id(artifact_id)
    if index_multihash:
        return str(index_multihash)
    index_bytes = store._runtime.ensure_client().get_artifact_index_by_id(artifact_id)
    if not index_bytes:
        raise ArtifactError(
            f"canonical index bytes missing for artifact '{artifact_id}'",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return compute_index_multihash(index_bytes)


def _ensure_canonical_layout_for_index(
    *,
    canonical_index: CanonicalIndex,
) -> str:
    global_store_address = _resolve_runtime_global_store_address()
    if not global_store_address:
        raise ArtifactError(
            "Global Store address unavailable while provisioning canonical layout "
            "for publication canonical index",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    canonical_index_data = canonical_index_to_bytes(canonical_index)
    index_multihash = compute_index_multihash(canonical_index_data)
    channel = grpc.insecure_channel(global_store_address)
    try:
        stub = GlobalStoreCompositeStub(channel)
        layout = layout_pb2.LayoutSpec(
            layout_schema_version=1,
            index_multihash=index_multihash,
        )
        put_resp = stub.PutLayoutSpec(
            global_store_pb2.PutLayoutSpecRequest(
                layout=layout,
                canonical_index_data=canonical_index_data,
            ),
            timeout=10.0,
        )
        if (
            put_resp.status != global_store_pb2.Status.STATUS_OK
            or not put_resp.layout_id
        ):
            raise ArtifactError(
                "failed to register canonical layout spec for representation publication",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return str(put_resp.layout_id)
    except grpc.RpcError as exc:
        message = exc.details() or str(exc)
        raise ArtifactError(
            "failed to provision canonical layout for representation publication: "
            f"{message}",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc
    finally:
        with contextlib.suppress(Exception):
            channel.close()


def _ensure_canonical_layout_for_artifact(
    store: "Store",
    *,
    artifact_id: str,
) -> str:
    attached_layout_ids = tuple(store.list_artifact_layouts(artifact_id))
    if len(attached_layout_ids) == 1:
        return str(attached_layout_ids[0])
    if len(attached_layout_ids) > 1:
        raise ArtifactError(
            "representation publication layout inference is ambiguous for "
            f"{artifact_id}: {', '.join(attached_layout_ids)}; pass layout_id explicitly",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    global_store_address = _resolve_runtime_global_store_address()
    if not global_store_address:
        raise ArtifactError(
            "Global Store address unavailable while provisioning canonical layout "
            f"for '{artifact_id}'",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    channel = grpc.insecure_channel(global_store_address)
    try:
        stub = GlobalStoreCompositeStub(channel)
        layout = layout_pb2.LayoutSpec(
            layout_schema_version=1,
            index_multihash=_artifact_index_multihash_for_layout_provision(
                store,
                artifact_id=artifact_id,
            ),
        )
        put_resp = stub.PutLayoutSpec(
            global_store_pb2.PutLayoutSpecRequest(layout=layout),
            timeout=10.0,
        )
        if (
            put_resp.status != global_store_pb2.Status.STATUS_OK
            or not put_resp.layout_id
        ):
            raise ArtifactError(
                f"failed to register canonical layout spec for '{artifact_id}'",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        attach_resp = stub.AttachLayoutToArtifact(
            global_store_pb2.AttachLayoutToArtifactRequest(
                mi2_id=artifact_id,
                layout_id=str(put_resp.layout_id),
            ),
            timeout=10.0,
        )
        if attach_resp.status != global_store_pb2.Status.STATUS_OK:
            raise ArtifactError(
                f"failed to attach canonical layout spec to '{artifact_id}'",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return str(put_resp.layout_id)
    except grpc.RpcError as exc:
        message = exc.details() or str(exc)
        raise ArtifactError(
            "failed to provision canonical layout for representation publication: "
            f"{message}",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc
    finally:
        with contextlib.suppress(Exception):
            channel.close()


def _resolve_representation_publish_layout_id(
    store: "Store",
    *,
    layout_id: str | None,
    layout_artifact_id: str | None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
) -> str:
    if layout_id:
        return str(layout_id)
    if (
        isinstance(publication, RepresentationPublishSpec)
        and publication.layout_id is not None
    ):
        return str(publication.layout_id)
    if (
        isinstance(publication, RepresentationPublishSpec)
        and publication.contract_family == "canonical_full"
        and publication.representation_publish_contract.binding_value_ref is not None
        and publication.canonical_index is not None
    ):
        return _ensure_canonical_layout_for_index(
            canonical_index=cast(CanonicalIndex, publication.canonical_index),
        )
    if (
        isinstance(publication, RepresentationPublishSpec)
        and publication.contract_family == "canonical_full"
    ):
        auto_artifact_id = (
            str(layout_artifact_id).strip()
            if layout_artifact_id
            else str(
                publication.source_artifact_ref
                or publication.serving_manifest.source_artifact_ref
                or ""
            ).strip()
        )
        if auto_artifact_id:
            return _ensure_canonical_layout_for_artifact(
                store,
                artifact_id=auto_artifact_id,
            )

    candidate_artifact_ids: list[str] = []
    if layout_artifact_id:
        candidate_artifact_ids.append(str(layout_artifact_id))
    if isinstance(publication, RepresentationPublishSpec):
        if publication.serving_manifest.source_artifact_ref:
            candidate_artifact_ids.append(
                str(publication.serving_manifest.source_artifact_ref)
            )
        if publication.serving_artifact_id is not None:
            candidate_artifact_ids.append(str(publication.serving_artifact_id))

    seen_artifact_ids: set[str] = set()
    for artifact_id in candidate_artifact_ids:
        normalized_artifact_id = str(artifact_id).strip()
        if not normalized_artifact_id or normalized_artifact_id in seen_artifact_ids:
            continue
        seen_artifact_ids.add(normalized_artifact_id)
        layout_ids = tuple(store.list_artifact_layouts(normalized_artifact_id))
        if len(layout_ids) == 1:
            return str(layout_ids[0])
        if len(layout_ids) > 1:
            raise ArtifactError(
                "representation publication layout inference is ambiguous for "
                f"{normalized_artifact_id}: {', '.join(layout_ids)}; pass layout_id explicitly",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    raise ArtifactError(
        "layout_id is required for representation publication when no unique artifact-attached layout can be inferred",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _resolve_representation_publish_requirements(
    *,
    requirements: AssemblyRequirementSetRef | None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
) -> AssemblyRequirementSetRef | None:
    if requirements is not None:
        return requirements
    if isinstance(publication, RepresentationPublishSpec):
        return publication.requirements
    return None


def _resolve_representation_publish_readiness_policy(
    *,
    readiness_policy: AssemblyReadinessPolicy | None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
) -> AssemblyReadinessPolicy | None:
    if readiness_policy is not None:
        return readiness_policy
    if isinstance(publication, RepresentationPublishSpec):
        return publication.readiness_policy
    return None


def _resolve_representation_publish_structural_view_ids(
    *,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract | None = None,
    source_artifact: Artifact | None,
    structural_view_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    explicit_view_ids = tuple(
        str(view_id).strip()
        for view_id in (structural_view_ids or ())
        if str(view_id).strip()
    )
    if explicit_view_ids:
        return explicit_view_ids
    if isinstance(publication, RepresentationPublishSpec):
        bundled_view_ids = tuple(
            str(view_id).strip()
            for view_id in publication.structural_view_ids
            if str(view_id).strip()
        )
        if bundled_view_ids:
            return bundled_view_ids
    if source_artifact is None:
        raise ArtifactError(
            "structural representation publication requires source_artifact or structural_view_ids",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    view_metadata = source_artifact._ensure_view_metadata_cache(require_view_id=True)
    view_id = (
        str(view_metadata.view_id).strip()
        if view_metadata and view_metadata.view_id
        else ""
    )
    if not view_id:
        raise ArtifactError(
            "source_artifact does not carry a structural view_id; pass structural_view_ids explicitly or use the canonical helper",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    if view_id.startswith("mapped:v1:"):
        raise ArtifactError(
            "mapped view ids are not admitted as structural assembly requirements; pass structural_view_ids explicitly",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return (view_id,)


def build_representation_publish_requirements(
    *,
    contract_family: AssemblyContractFamily,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract | None = None,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
) -> AssemblyRequirementSetRef:
    if contract_family == "canonical_full":
        return AssemblyRequirementSetRef.canonical_full()
    return AssemblyRequirementSetRef.from_contract_family(
        family=contract_family,
        structural_view_ids=_resolve_representation_publish_structural_view_ids(
            publication=publication,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
        ),
    )


def _resolve_repo_owned_contract_family(
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    contract_family: AssemblyContractFamily | str | None,
) -> AssemblyContractFamily:
    explicit_family = (
        str(contract_family).strip() if contract_family is not None else ""
    )
    if explicit_family:
        if explicit_family not in {"pp", "ep", "canonical_full"}:
            raise ArtifactError(
                "contract_family must be one of: pp, ep, canonical_full",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return explicit_family  # type: ignore[return-value]
    if (
        isinstance(publication, RepresentationPublishSpec)
        and publication.contract_family
    ):
        bundled_family = str(publication.contract_family).strip()
        if bundled_family in {"pp", "ep", "canonical_full"}:
            return bundled_family  # type: ignore[return-value]
    raise ArtifactError(
        "repo-owned representation publication requires contract_family on the bundle or as an explicit argument",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _resolve_plan_publication_bundle(
    *,
    plan_result: "PlanResult",
    publication_step: "PlanStepRef[RepresentationPublishSpec]",
) -> RepresentationPublishSpec:
    return plan_result.require_representation_publish_spec(publication_step)


def _resolve_source_contribution_view_ids(
    artifacts: Sequence[Artifact] | None,
) -> tuple[str, ...]:
    if not artifacts:
        return ()
    resolved: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        view_cache = artifact._ensure_view_metadata_cache(require_view_id=True)
        if view_cache is None or not view_cache.view_id:
            raise ArtifactError(
                "source contribution artifacts must expose deterministic view_id metadata",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        view_id = str(view_cache.view_id).strip()
        if not view_id or view_id.startswith("mapped:v1:"):
            raise ArtifactError(
                "source contribution artifacts must expose structural view_id metadata",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if view_id in seen:
            continue
        seen.add(view_id)
        resolved.append(view_id)
    return tuple(resolved)


def _copy_plan_to_proto(
    mapping: CopyPlan,
) -> store_daemon_pb2.CopyPlan:
    normalized = normalize_copy_plan(mapping)
    proto = store_daemon_pb2.CopyPlan(version=1)
    for entry in normalized:
        entry_proto = proto.entries.add()
        entry_proto.ckpt_name = str(entry.ckpt_name)
        entry_proto.dst_name = str(entry.dst_name)
        if entry.ckpt_range is not None:
            entry_proto.ckpt_range.dim = int(entry.ckpt_range.dim)
            entry_proto.ckpt_range.start = int(entry.ckpt_range.start)
            entry_proto.ckpt_range.end = int(entry.ckpt_range.end)
        if entry.dst_range is not None:
            entry_proto.dst_range.dim = int(entry.dst_range.dim)
            entry_proto.dst_range.start = int(entry.dst_range.start)
            entry_proto.dst_range.end = int(entry.dst_range.end)
    return proto


def _normalize_target_layout_contract(
    target_layout: store_daemon_pb2.TargetLayout,
) -> tuple[
    int,
    int,
    int,
    str,
    bytes,
    tuple[tuple[int, int], ...],
    tuple[tuple[str, int, int, int], ...],
]:
    storage_order = {
        str(storage.storage_id): idx
        for idx, storage in enumerate(target_layout.storages)
    }
    storages = tuple(
        (int(storage.device_id), int(storage.storage_length))
        for storage in target_layout.storages
    )
    offsets = tuple(
        sorted(
            (
                str(offset.name),
                int(storage_order[str(offset.storage_id)]),
                int(offset.storage_offset),
                int(offset.logical_length),
            )
            for offset in target_layout.offsets
        )
    )
    return (
        int(target_layout.layout_kind),
        int(target_layout.index_kind),
        int(target_layout.tensor_spec_kind),
        str(target_layout.view_id or ""),
        bytes(target_layout.logical_layout_hash),
        storages,
        offsets,
    )


def _build_bound_publication_canonical_index(layout: BindingLayout) -> CanonicalIndex:
    base_index = canonical_index_from_bytes(layout.target_index_bytes)
    entry_by_name = {str(entry.name): entry for entry in base_index.entries}
    storage_offsets: dict[str, int] = {}
    storage_lengths: dict[str, int] = {}
    cursor = 0
    for storage in layout.target_layout.storages:
        storage_id = str(storage.storage_id)
        storage_offsets[storage_id] = int(cursor)
        storage_lengths[storage_id] = int(storage.storage_length)
        cursor += int(storage.storage_length)
    entries: list[CanonicalIndexEntry] = []
    for offset in sorted(layout.target_layout.offsets, key=lambda item: str(item.name)):
        name = str(offset.name)
        meta = entry_by_name.get(name)
        if meta is None:
            raise ArtifactError(
                f"binding target_index_bytes missing tensor entry for {name}",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        storage_id = str(offset.storage_id)
        if storage_id not in storage_offsets or storage_id not in storage_lengths:
            raise ArtifactError(
                f"binding target_layout missing storage entry for {storage_id}",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        entries.append(
            CanonicalIndexEntry(
                name=name,
                dtype=meta.dtype,
                shape=tuple(int(v) for v in meta.shape),
                stride=tuple(int(v) for v in meta.stride),
                storage_offset=int(offset.storage_offset),
                segment_offset=int(storage_offsets[storage_id]),
                size_bytes=int(storage_lengths[storage_id]),
            )
        )
    return CanonicalIndex(
        entries=tuple(entries),
        total_size_bytes=int(cursor),
        avbs_hash=str(base_index.avbs_hash or ""),
    )


def _validate_client_binding_targets(
    *,
    layout: BindingLayout,
    target_tensors: Mapping[str, torch.Tensor],
    device_id: int,
    pipeline: MaterializationPipeline,
    expected_index: CanonicalIndex,
) -> object:
    if layout.target_layout.index_kind == store_daemon_pb2.TargetLayout.INDEX_KIND_VIEW:
        selection_order = tuple(
            str(offset.name) for offset in layout.target_layout.offsets
        )
        derived_layout = pipeline._build_region_backed_layout(
            canonical_index=expected_index,
            canonical_index_bytes=layout.target_index_bytes,
            target=target_tensors,
            device_id=device_id,
            tensor_names=selection_order,
            view_spec=None,
            view_id=str(layout.target_layout.view_id or "") or None,
            view_index_hint=layout.target_index_bytes,
            selection_order=selection_order,
        )
        if bytes(derived_layout.view_index_bytes or b"") != bytes(
            layout.target_index_bytes
        ):
            raise ArtifactError(
                "target_tensors do not match BindingLayout index bytes",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
    else:
        derived_layout = pipeline._build_region_backed_layout(
            canonical_index=expected_index,
            canonical_index_bytes=layout.target_index_bytes,
            target=target_tensors,
            device_id=device_id,
            tensor_names=None,
            view_spec=None,
            view_id=None,
            view_index_hint=None,
            selection_order=None,
        )
    if _normalize_target_layout_contract(
        derived_layout.layout
    ) != _normalize_target_layout_contract(layout.target_layout):
        raise ArtifactError(
            "target_tensors do not match the BindingLayout storage contract",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return derived_layout


def _artifact_id_kind_from_proto(kind: int, artifact_id: str) -> ArtifactIdKind:
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2:
        return ArtifactIdKind.MI2
    if kind == common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID:
        return ArtifactIdKind.CGID
    inferred = infer_artifact_id_kind(artifact_id)
    return inferred or ArtifactIdKind.MI2


def _target_layout_with_protocol_device_id(
    layout: BindingLayout,
    *,
    device_id: int,
) -> store_daemon_pb2.TargetLayout:
    target_layout = store_daemon_pb2.TargetLayout()
    target_layout.CopyFrom(layout.target_layout)
    for storage in target_layout.storages:
        storage.device_id = int(device_id)
    return target_layout


def _decode_published_model_version_from_response(
    resp: operation_pb2.GetOperationResponse, *, assembly_id: str
) -> PublishedModelVersion:
    if resp.status.state != operation_pb2.OPERATION_STATE_SUCCESS:
        message = (
            resp.status.message or "Assembly attempt did not complete successfully"
        )
        if resp.status.HasField("error"):
            message = resp.status.error.message or message
        raise ArtifactError(
            message,
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    payload = store_daemon_pb2.SealAssemblyResult()
    if not resp.status.result.Unpack(payload):
        raise ArtifactError(
            "Unexpected assembly attempt result type",
            status_code="INTERNAL",
            retryable=False,
        )
    artifact = payload.artifact
    descriptor = TypedArtifactDescriptor(
        artifact_id=str(artifact.artifact_id),
        index_multihash=str(artifact.index_multihash or "") or None,
        data_multihash=str(artifact.data_multihash or "") or None,
        schema_version=str(artifact.schema_version or "") or None,
        encoding=str(artifact.encoding or "") or None,
        total_size=int(artifact.total_size),
        id_kind=_artifact_id_kind_from_proto(artifact.id_kind, artifact.artifact_id),
    )
    return PublishedModelVersion(
        assembly_id=assembly_id,
        source_artifact_id=descriptor.artifact_id,
        source_descriptor=descriptor,
        serving_artifact_id=(
            str(payload.serving_artifact.artifact_id)
            if payload.HasField("serving_artifact")
            and payload.serving_artifact.artifact_id
            else None
        ),
        serving_descriptor=(
            TypedArtifactDescriptor(
                artifact_id=str(payload.serving_artifact.artifact_id),
                index_multihash=str(payload.serving_artifact.index_multihash or "")
                or None,
                data_multihash=str(payload.serving_artifact.data_multihash or "")
                or None,
                schema_version=str(payload.serving_artifact.schema_version or "")
                or None,
                encoding=str(payload.serving_artifact.encoding or "") or None,
                total_size=int(payload.serving_artifact.total_size),
                id_kind=_artifact_id_kind_from_proto(
                    payload.serving_artifact.id_kind,
                    payload.serving_artifact.artifact_id,
                ),
            )
            if payload.HasField("serving_artifact")
            and payload.serving_artifact.artifact_id
            else None
        ),
        source_version_key=str(payload.source_version_key or "") or None,
        serving_version_key=str(payload.serving_version_key or "") or None,
        representation_contract_hash=(
            str(payload.representation_contract_hash or "") or None
        ),
        serving_build_digest=str(payload.serving_build_digest or "") or None,
        serving_manifest_ref=str(payload.serving_manifest_ref or "") or None,
        serving_execution_diagnostics=(
            ExecutionDiagnostics.from_proto(payload.serving_execution_diagnostics)
            if payload.HasField("serving_execution_diagnostics")
            else None
        ),
    )


def _split_mi2_artifact_id(artifact_id: str) -> tuple[str | None, str | None]:
    if not artifact_id.startswith("mi2:"):
        return None, None
    remainder = artifact_id[len("mi2:") :]
    parts = remainder.split(":", 1)
    if len(parts) != 2:
        return None, None
    index_multihash = parts[0].strip()
    data_multihash = parts[1].strip()
    if not index_multihash or not data_multihash:
        return None, None
    return index_multihash, data_multihash


def _parse_artifact_ref(
    ref: str | None,
    *,
    artifact_id: str | None,
    key: str | None,
) -> tuple[str | None, str | None]:
    if ref is None:
        return artifact_id, key
    if artifact_id or key:
        raise ValueError("ref cannot be combined with artifact_id or key")
    ref_value = str(ref)
    if not ref_value:
        raise ValueError("ref must be non-empty")
    if ref_value.startswith(("mi2:", "cgid:", "msa1:")):
        return ref_value, None
    if ref_value.startswith("disk:"):
        raise ValueError(
            "disk: ref is no longer supported; use Store.from_disk(...) for the "
            "mounted-source fast path or Store.import_from_disk(...) for explicit "
            "mi2 import, then reference the artifact by id or key."
        )
    return None, ref_value


def _should_show_from_disk_progress(show_progress: bool | None) -> bool:
    return bool(show_progress) if show_progress is not None else False


_IMPORT_STREAM_ERROR_STATUS: dict[int, ArtifactStatusCode] = {
    int(store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_NOT_FOUND): "NOT_FOUND",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_PERMISSION_DENIED
    ): "PERMISSION_DENIED",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_FORMAT_INVALID
    ): "INVALID_ARGUMENT",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_SOURCE_MUTATED
    ): "FAILED_PRECONDITION",
    int(store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_REGISTRY_IO_FAILURE): "UNAVAILABLE",
    int(
        store_daemon_pb2.IMPORT_ARTIFACT_ERROR_CODE_POLICY_DENIED_NON_LOCAL_PEER
    ): "PERMISSION_DENIED",
}
_IMPORT_STREAM_DEFAULT_STATUS: ArtifactStatusCode = "INTERNAL"


def _stream_error_from_import_event(
    event: store_daemon_pb2.ImportArtifactFromPathStreamEvent,
) -> ArtifactError:
    message = (
        str(getattr(event, "message", "") or "")
        or "ImportArtifactFromPathStream reported an error"
    )
    status_code = _IMPORT_STREAM_ERROR_STATUS.get(
        int(getattr(event, "error_code", 0) or 0), _IMPORT_STREAM_DEFAULT_STATUS
    )
    return ArtifactError(
        message,
        status_code=status_code,
        retryable=(status_code == "UNAVAILABLE"),
    )


def _consume_import_artifact_stream_with_tqdm(
    stream,
    *,
    disk_path: str,
) -> store_daemon_pb2.ImportArtifactFromPathResponse:
    from tqdm.auto import tqdm

    desc = f"resolve:{os.path.basename(disk_path) or disk_path}"
    bar = tqdm(total=None, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
    final_response: store_daemon_pb2.ImportArtifactFromPathResponse | None = None
    try:
        for event in stream:
            total_bytes = int(getattr(event, "total_bytes", 0) or 0)
            processed_bytes = int(getattr(event, "processed_bytes", 0) or 0)
            if total_bytes > 0 and bar.total != total_bytes:
                bar.total = total_bytes
            if processed_bytes > bar.n:
                bar.update(processed_bytes - bar.n)

            message = str(getattr(event, "message", "") or "")
            if message:
                bar.set_postfix_str(message, refresh=False)

            if bool(getattr(event, "done", False)):
                if total_bytes > 0 and bar.n < total_bytes:
                    bar.update(total_bytes - bar.n)
                if bool(getattr(event, "error", False)):
                    raise _stream_error_from_import_event(event)
                if not event.HasField("result"):
                    raise ArtifactError(
                        "ImportArtifactFromPathStream done event missing result",
                        status_code="DATA_LOSS",
                        retryable=False,
                    )
                final_response = event.result
                break
    finally:
        bar.close()

    if final_response is None:
        raise ArtifactError(
            "ImportArtifactFromPathStream ended without terminal result",
            status_code="DATA_LOSS",
            retryable=False,
        )
    return final_response


def _consume_import_artifact_stream(
    stream,
) -> store_daemon_pb2.ImportArtifactFromPathResponse:
    final_response: store_daemon_pb2.ImportArtifactFromPathResponse | None = None
    for event in stream:
        if bool(getattr(event, "done", False)):
            if bool(getattr(event, "error", False)):
                raise _stream_error_from_import_event(event)
            if not event.HasField("result"):
                raise ArtifactError(
                    "ImportArtifactFromPathStream done event missing result",
                    status_code="DATA_LOSS",
                    retryable=False,
                )
            final_response = event.result
            break
    if final_response is None:
        raise ArtifactError(
            "ImportArtifactFromPathStream ended without terminal result",
            status_code="DATA_LOSS",
            retryable=False,
        )
    return final_response


def _is_import_startup_in_progress_error(exc: BaseException) -> bool:
    return "startup still in progress" in str(exc).lower()


class Store:
    """Store façade delegating to runtime, registration, and materialization pipelines."""

    _PERSISTENCE_STATE_FROM_PROTO = {
        store_daemon_pb2.PERSISTENCE_STATE_PENDING: "pending",
        store_daemon_pb2.PERSISTENCE_STATE_RUNNING: "running",
        store_daemon_pb2.PERSISTENCE_STATE_DEGRADED: "degraded",
        store_daemon_pb2.PERSISTENCE_STATE_SUCCESS: "success",
        store_daemon_pb2.PERSISTENCE_STATE_FAILED: "failed",
    }

    def __init__(
        self,
        daemon_endpoint: str,
        *,
        opts: StoreOptions | None = None,
        runtime: StoreRuntimeContext | None = None,
        register_fn: Callable[..., RegistrationResult] | None = None,
        materialize_fn: Callable[..., MaterializationPayload] | None = None,
    ) -> None:
        self._runtime = runtime or StoreRuntimeContext(
            daemon_endpoint, opts=opts, client_factory=get_daemon_client
        )
        self._views = ViewOrchestrator(self._runtime)
        self._registration = RegistrationPipeline(
            self._runtime,
            self._views,
            register_fn=register_fn or _register_artifact_core,
        )
        self._materialization = MaterializationPipeline(
            self._runtime,
            self._views,
            materialize_fn=materialize_fn or materialize_artifact_v2,
        )
        self._enable_batcher = os.getenv(
            "TENSORCAST_STORE_ENABLE_BATCHER", "1"
        ).lower() not in ("0", "false", "no")
        self._enable_prefetch = os.getenv(
            "TENSORCAST_STORE_ENABLE_PREFETCH", "1"
        ).lower() not in ("0", "false", "no")
        self._batcher: MaterializationBatcher | None = (
            MaterializationBatcher(
                self._runtime,
                self._materialization,
            )
            if self._enable_batcher
            else None
        )
        _LIVE_STORES.add(self)

    def set_register_fn(self, register_fn: Callable[..., RegistrationResult]) -> None:
        self._registration.set_register_fn(register_fn)

    def set_materialize_fn(
        self, materialize_fn: Callable[..., MaterializationPayload]
    ) -> None:
        self._materialization.set_materialize_fn(materialize_fn)

    # ------------------------------------------------------------------
    # Registration APIs
    # ------------------------------------------------------------------
    def register(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        ttl_ms: int | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.register_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            ttl_ms=ttl_ms,
        )

    def register_view(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        view_id: str | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        options: RegisterArtifactOptions | None = None,
        canonical_index_bytes: bytes | None = None,
        registration_kind: str | int | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register_view(
            tensors,
            artifact_id=artifact_id,
            key=key,
            slices=slices,
            transpose=transpose,
            view_id=view_id,
            placement=placement,
            ttl_ms=ttl_ms,
            options=options,
            canonical_index_bytes=canonical_index_bytes,
            registration_kind=registration_kind,
            resolver=self._views.resolve_view_inputs,
        )

    def register_piece(
        self,
        tensors: TensorDict,
        *,
        assembly_id: str,
        key: str | None = None,
        slices: Mapping[str, Sequence[object]] | None = None,
        canonical_index_bytes: bytes | None = None,
        placement: str | None = None,
        ttl_ms: int | None = None,
        options: RegisterArtifactOptions | None = None,
    ) -> RegisteredArtifact:
        return self._registration.register_piece(
            tensors,
            assembly_id=assembly_id,
            key=key,
            slices=slices,
            canonical_index_bytes=canonical_index_bytes,
            placement=placement,
            ttl_ms=ttl_ms,
            options=options,
            resolver=self._views.resolve_view_inputs,
        )

    def put(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        return self._registration.put(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def put_async(
        self,
        tensors: TensorDict,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        options: RegisterArtifactOptions | None = None,
        device: int | torch.device | None = None,
    ) -> ArtifactFuture[RegisteredArtifact]:
        return self._registration.put_async(
            tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            options=options,
            device=device,
        )

    def create_binding(
        self,
        layout: BindingLayout,
        *,
        ownership: str = "daemon",
        device: torch.device | str | None = None,
        target_tensors: Mapping[str, torch.Tensor] | None = None,
        mapping: CopyPlan | None = None,
        ctx: CallContext | None = None,
        restore_tensors_async: bool = False,
    ) -> Binding:
        if not isinstance(layout, BindingLayout):
            raise ArtifactError(
                "layout must be a BindingLayout",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        normalized_mapping: tuple[CopyPlanEntry, ...] | None = None
        copy_plan_proto: store_daemon_pb2.CopyPlan | None = None
        if mapping is not None:
            normalized_mapping = normalize_copy_plan(mapping)
            if not layout.dst_specs:
                raise ArtifactError(
                    "mapping requires a mapped BindingLayout with dst_specs",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            copy_plan_proto = _copy_plan_to_proto(normalized_mapping)
        runtime = self._runtime
        profile_start = time.perf_counter()
        profile_last = profile_start
        logger.info(
            "tc_profile_py store.create_binding enter ownership=%s device=%s "
            "layout_id=%s tensor_count=%d target_index_bytes=%d mapped=%s",
            ownership,
            device,
            layout.binding_layout_id,
            len(layout.target_layout.offsets),
            len(layout.target_index_bytes),
            bool(layout.dst_specs or normalized_mapping),
        )
        client = runtime.ensure_client()
        now = time.perf_counter()
        logger.info(
            "tc_profile_py store.create_binding ensure_client step_sec=%.6f total_sec=%.6f",
            now - profile_last,
            now - profile_start,
        )
        profile_last = now
        timeout_s = _ctx_timeout_s(ctx)
        mode = str(ownership).strip().lower()
        if mode == "daemon":
            if device is None:
                raise ArtifactError(
                    "device is required for daemon-owned bindings",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if target_tensors is not None:
                raise ArtifactError(
                    "target_tensors must be omitted for daemon-owned bindings",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            device_obj = torch.device(device)
            device_id = resolve_device(device_obj, allow_cpu=False)
            protocol_device_id = protocol_device_id_for(device_id)
            now = time.perf_counter()
            logger.info(
                "tc_profile_py store.create_binding resolved_device device_id=%d "
                "protocol_device_id=%d step_sec=%.6f total_sec=%.6f",
                device_id,
                protocol_device_id,
                now - profile_last,
                now - profile_start,
            )
            profile_last = now
            target_layout_proto = _target_layout_with_protocol_device_id(
                layout,
                device_id=protocol_device_id,
            )
            now = time.perf_counter()
            logger.info(
                "tc_profile_py store.create_binding built_target_layout "
                "step_sec=%.6f total_sec=%.6f",
                now - profile_last,
                now - profile_start,
            )
            profile_last = now
            device_uuid = device_uuid_for(device_id)
            now = time.perf_counter()
            logger.info(
                "tc_profile_py store.create_binding resolved_device_uuid "
                "device_uuid=%s step_sec=%.6f total_sec=%.6f",
                device_uuid,
                now - profile_last,
                now - profile_start,
            )
            profile_last = now
            logger.info("tc_profile_py store.create_binding rpc_start")
            response = client.create_binding(
                ownership=store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_DAEMON,
                target_layout=target_layout_proto,
                target_index_bytes=layout.target_index_bytes,
                device_uuid=device_uuid,
                binding_layout_id=layout.binding_layout_id,
                copy_plan=copy_plan_proto,
                dst_specs=layout.dst_specs if layout.dst_specs else None,
                timeout_s=timeout_s if timeout_s is not None else 600.0,
            )
            now = time.perf_counter()
            logger.info(
                "tc_profile_py store.create_binding rpc_done binding_id=%s "
                "payloads=%d step_sec=%.6f total_sec=%.6f",
                getattr(response, "binding_id", None),
                len(getattr(response, "payloads", ())),
                now - profile_last,
                now - profile_start,
            )
            profile_last = now
            tensors = None
            try:
                current_value_metadata = parse_binding_value_or_raise(
                    response.current_value
                    if hasattr(response, "current_value")
                    else None,
                    rpc_name="CreateBinding",
                    expected_binding_id=str(response.binding_id),
                    expected_binding_layout_id=layout.binding_layout_id,
                )
                now = time.perf_counter()
                logger.info(
                    "tc_profile_py store.create_binding parse_current_done "
                    "step_sec=%.6f total_sec=%.6f",
                    now - profile_last,
                    now - profile_start,
                )
                profile_last = now
                if restore_tensors_async:
                    logger.info(
                        "tc_profile_py store.create_binding restore_tensors_deferred "
                        "binding_id=%s total_sec=%.6f",
                        getattr(response, "binding_id", None),
                        now - profile_start,
                    )
                else:
                    logger.info(
                        "tc_profile_py store.create_binding restore_tensors_start"
                    )
                    tensors = restore_owned_binding_tensors(
                        response=response,
                        runtime=runtime,
                        device_id=device_id,
                    )
                    now = time.perf_counter()
                    logger.info(
                        "tc_profile_py store.create_binding restore_tensors_done "
                        "tensor_count=%d step_sec=%.6f total_sec=%.6f",
                        len(tensors),
                        now - profile_last,
                        now - profile_start,
                    )
                    profile_last = now
            except Exception:
                with contextlib.suppress(Exception):
                    client.close_owned_binding(binding_id=str(response.binding_id))
                raise
            slot = OwnedBindingSlot(
                store=self,
                runtime=runtime,
                tensors=tensors,
                layout=layout,
                binding_id=str(response.binding_id),
                current_value_metadata=current_value_metadata,
                device=device_obj,
                device_id=device_id,
                binding_current_value_publication_token=None,
                restore_response=response if restore_tensors_async else None,
                start_restore=restore_tensors_async,
            )
            now = time.perf_counter()
            logger.info(
                "tc_profile_py store.create_binding return step_sec=%.6f total_sec=%.6f",
                now - profile_last,
                now - profile_start,
            )
            return Binding(slot)

        if mode != "client":
            raise ArtifactError(
                "ownership must be 'daemon' or 'client'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if layout.dst_specs and copy_plan_proto is None:
            raise ArtifactError(
                "mapped BindingLayout requires mapping for client-owned bindings",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not target_tensors:
            raise ArtifactError(
                "target_tensors are required for client-owned bindings",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        expected_index = canonical_index_from_bytes(layout.target_index_bytes)
        expected_names = {entry.name for entry in expected_index.entries}
        if {str(name) for name in target_tensors} != expected_names:
            raise ArtifactError(
                "target_tensors must match the BindingLayout tensor set",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if device is not None:
            requested_device_id = resolve_device(torch.device(device), allow_cpu=False)
        else:
            requested_device_id = None
        first_tensor = next(iter(target_tensors.values()))
        if not first_tensor.is_cuda:
            raise ArtifactError(
                "client-owned bindings require CUDA target tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        device_id = resolve_device(first_tensor.device, allow_cpu=False)
        if requested_device_id is not None and requested_device_id != device_id:
            raise ArtifactError(
                "device does not match target_tensors",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        for name, tensor in target_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise ArtifactError(
                    f"target tensor '{name}' must be a torch.Tensor",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if not tensor.is_cuda:
                raise ArtifactError(
                    f"target tensor '{name}' must be CUDA",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if resolve_device(tensor.device, allow_cpu=False) != device_id:
                raise ArtifactError(
                    "target_tensors must share the same CUDA device",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
        for entry in expected_index.entries:
            tensor = target_tensors[entry.name]
            if tensor.dtype != entry.dtype:
                raise ArtifactError(
                    f"target tensor '{entry.name}' dtype does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if tuple(int(v) for v in tensor.shape) != tuple(
                int(v) for v in entry.shape
            ):
                raise ArtifactError(
                    f"target tensor '{entry.name}' shape does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            if tuple(int(v) for v in tensor.stride()) != tuple(
                int(v) for v in entry.stride
            ):
                raise ArtifactError(
                    f"target tensor '{entry.name}' stride does not match BindingLayout",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )

        region_ids: list[str] = []
        try:
            for base_ptr, nbytes in sorted(
                collect_storage_bases(target_tensors).items()
            ):
                handle = self.register_vram_region(
                    device_id=device_id,
                    base_ptr=base_ptr,
                    size_bytes=nbytes,
                    ttl_ms=0,
                )
                region_ids.append(handle.region_id)
            readable_layout = _validate_client_binding_targets(
                layout=layout,
                target_tensors=target_tensors,
                device_id=device_id,
                pipeline=self._materialization,
                expected_index=expected_index,
            )
            protocol_device_id = protocol_device_id_for(device_id)
            response = client.create_binding(
                ownership=store_daemon_pb2.BindingOwnership.BINDING_OWNERSHIP_CLIENT,
                # Client-owned bindings need a readable source layout on the
                # daemon side so later SubmitBindingContribution calls can lower
                # directly from the live tensors without fabricating an
                # allocation-backed handle path.
                target_layout=_target_layout_with_protocol_device_id(
                    build_binding_layout(
                        target_layout=readable_layout.layout,
                        target_index_bytes=layout.target_index_bytes,
                        dst_specs=layout.dst_specs
                        if copy_plan_proto is not None
                        else None,
                    ),
                    device_id=protocol_device_id,
                ),
                target_index_bytes=layout.target_index_bytes,
                device_uuid=device_uuid_for(device_id),
                binding_layout_id=layout.binding_layout_id,
                copy_plan=copy_plan_proto,
                dst_specs=layout.dst_specs if copy_plan_proto is not None else None,
                timeout_s=timeout_s if timeout_s is not None else 600.0,
            )
        except Exception:
            with contextlib.suppress(Exception):
                for region_id in region_ids:
                    self.unregister_vram_region(region_id)
            raise
        try:
            current_value_metadata = parse_binding_value_or_raise(
                response.current_value if hasattr(response, "current_value") else None,
                rpc_name="CreateBinding",
                expected_binding_id=str(response.binding_id),
                expected_binding_layout_id=layout.binding_layout_id,
            )
        except Exception:
            with contextlib.suppress(Exception):
                client.close_owned_binding(binding_id=str(response.binding_id))
            with contextlib.suppress(Exception):
                for region_id in region_ids:
                    self.unregister_vram_region(region_id)
            raise
        inplace_slot = InplaceSlot(
            store=self,
            runtime=runtime,
            pipeline=self._materialization,
            tensors=target_tensors,
            device=first_tensor.device,
            device_id=device_id,
            layout=layout,
            binding_id=str(response.binding_id),
            region_ids=tuple(region_ids),
            selection_names=tuple(
                offset.name for offset in layout.target_layout.offsets
            ),
            view_id=str(layout.target_layout.view_id or "") or None,
            view_subset_hash=None,
            view_spec=None,
            current_value_metadata=current_value_metadata,
            binding_current_value_publication_token=None,
            copy_plan=normalized_mapping,
        )
        return Binding(inplace_slot)

    def query_persistence_status(
        self, *, task_id: str | None = None, artifact_id: str | None = None
    ) -> PersistenceStatusResult:
        """Query persistence task state via the local daemon."""
        if not task_id and not artifact_id:
            raise ArtifactError(
                "task_id or artifact_id must be provided",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resp = self._runtime.ensure_client().query_persistence_status(
            task_id=task_id, artifact_id=artifact_id
        )
        return self._persistence_status_from_proto(resp)

    def seal_assembly(
        self,
        assembly_id: str,
        *,
        publish_canonical: bool = True,
        wait: bool = True,
        layout_id: str | None = None,
        timeout_s: float = 120.0,
        ctx: CallContext | None = None,
    ) -> SealAssemblyResult | Operation[SealAssemblyResult]:
        if publish_canonical is False:
            # Operation-based sealing always publishes canonical; use the direct
            # RPC when the caller explicitly requests a non-canonical closeout.
            return self._runtime.ensure_client().seal_assembly(
                assembly_id,
                publish_canonical=False,
                timeout_s=timeout_s,
            )

        op = self.seal_assembly_operation(
            assembly_id,
            layout_id=layout_id,
            ctx=ctx,
        )
        if wait:
            return op.wait(timeout_s=timeout_s)
        return op

    def seal_assembly_operation(
        self,
        assembly_id: str,
        *,
        layout_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> Operation[SealAssemblyResult]:
        start_resp = self._runtime.ensure_client().start_seal_assembly(
            assembly_id=assembly_id,
            layout_id=layout_id,
        )
        operation_id = start_resp.operation.operation_id
        context: dict[str, str] = {"assembly_id": assembly_id}
        if layout_id:
            context["layout_id"] = str(layout_id)

        def _decode(resp) -> SealAssemblyResult:
            payload = store_daemon_pb2.SealAssemblyResult()
            if not resp.status.result.Unpack(payload):
                raise ArtifactError(
                    f"Unexpected seal assembly result type (assembly_id={assembly_id})",
                    status_code="INTERNAL",
                    retryable=False,
                )
            artifact = payload.artifact
            descriptor = TypedArtifactDescriptor(
                artifact_id=str(artifact.artifact_id),
                index_multihash=str(artifact.index_multihash or "") or None,
                data_multihash=str(artifact.data_multihash or "") or None,
                schema_version=str(artifact.schema_version or "") or None,
                encoding=str(artifact.encoding or "") or None,
                total_size=int(artifact.total_size),
                id_kind=_artifact_id_kind_from_proto(
                    artifact.id_kind, artifact.artifact_id
                ),
            )
            return SealAssemblyResult(
                sealed_artifact_id=descriptor.artifact_id,
                descriptor=descriptor,
                already_sealed=False,
            )

        return DaemonGlobalStoreOperation(
            operation_id=operation_id,
            runtime_ref=weakref.ref(self._runtime),
            ctx=ctx,
            context=context,
            result_factory=_decode,
        )

    def _persistence_status_from_proto(
        self, resp: store_daemon_pb2.QueryPersistenceStatusResponse
    ) -> PersistenceStatusResult:
        shards: list[PersistenceShardStatus] = []
        for shard in resp.shards:
            state = self._PERSISTENCE_STATE_FROM_PROTO.get(shard.state, "unknown")
            shards.append(
                PersistenceShardStatus(
                    shard_id=shard.shard_id,
                    shard_idx=int(shard.shard_idx),
                    state=state,
                    progress=float(shard.progress),
                    degraded_reason=shard.degraded_reason or None,
                    last_error=shard.last_error or None,
                    target_nodes=tuple(shard.target_nodes),
                    lease_ids=tuple(shard.lease_ids),
                )
            )
        state = self._PERSISTENCE_STATE_FROM_PROTO.get(resp.state, "unknown")
        return PersistenceStatusResult(
            task_id=resp.task_id,
            artifact_id=resp.artifact_id,
            plan_id=resp.plan_id,
            state=state,
            progress=float(resp.progress),
            degraded_reason=resp.degraded_reason or None,
            last_error=resp.last_error or None,
            shards=tuple(shards),
        )

    def register_pure_transform_publication(
        self,
        tensors: TensorDict,
        *,
        build_intent: ServingBuildIntent,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        contract_family: AssemblyContractFamily | str | None = None,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        device: int | torch.device | None = None,
        source_version_key: str | None = None,
        serving_version_key: str | None = None,
        logical_topology_json: str | None = None,
        serving_manifest_ref: str | None = None,
    ) -> RegisteredServingPublication:
        prepared = prepare_pure_transform_serving_registration(
            build_intent=build_intent,
            source_artifact=source_artifact,
            tensors=tensors,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=serving_manifest_ref,
        )
        registered_artifact = self.put(
            prepared.tensors,
            artifact_id=artifact_id,
            key=key,
            policy=policy,
            device=device,
        )
        publication = build_pure_transform_publication_bundle_from_registered_artifact(
            build_intent=build_intent,
            source_artifact=source_artifact,
            contract_family=contract_family,
            serving_artifact=registered_artifact,
            source_version_key=source_version_key,
            serving_version_key=serving_version_key,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=prepared.serving_manifest_ref,
        )
        return RegisteredServingPublication(
            registered_artifact=registered_artifact,
            prepared_registration=prepared,
            publication=publication,
        )

    def complete_pure_transform_publication(
        self,
        tensors: TensorDict,
        *,
        build_intent: ServingBuildIntent,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        contract_family: AssemblyContractFamily | str | None = None,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: StorePolicy | str | None = None,
        device: int | torch.device | None = None,
        source_version_key: str | None = None,
        serving_version_key: str | None = None,
        logical_topology_json: str | None = None,
        serving_manifest_ref: str | None = None,
        structural_view_ids: Sequence[str] | None = None,
        source_contribution_device: str | int | None = None,
        source_contribution_artifacts: Sequence[Artifact] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.register_pure_transform_publication",
            logger=logger,
            extra={
                "tensor_count": len(tensors),
                "device": None if device is None else str(device),
                "contract_family": contract_family,
            },
        ) as profile:
            publication = self.register_pure_transform_publication(
                tensors,
                build_intent=build_intent,
                source_artifact=source_artifact,
                contract_family=contract_family,
                artifact_id=artifact_id,
                key=key,
                policy=policy if policy is not None else "pinned",
                device=device,
                source_version_key=source_version_key,
                serving_version_key=serving_version_key,
                logical_topology_json=logical_topology_json,
                serving_manifest_ref=serving_manifest_ref,
            )
            if profile is not None:
                profile["serving_artifact_id"] = getattr(
                    publication.publication, "serving_artifact_id", None
                )
        return self._complete_registered_representation_publication(
            publication=publication,
            contract_family=contract_family,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
            source_contribution_device=source_contribution_device,
            source_contribution_artifacts=source_contribution_artifacts,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def complete_pure_transform_publication_from_binding(
        self,
        binding: Binding | SealedBindingValue,
        *,
        build_intent: ServingBuildIntent,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        contract_family: AssemblyContractFamily | str | None = None,
        source_version_key: str | None = None,
        serving_version_key: str | None = None,
        logical_topology_json: str | None = None,
        serving_manifest_ref: str | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        resolved_binding, current_value = self._resolve_bound_publication_subject(
            binding
        )
        publication_subject = current_value.to_binding_value_ref()
        authoritative_canonical_index = _build_bound_publication_canonical_index(
            resolved_binding.layout
        )
        prepared = prepare_pure_transform_serving_registration(
            build_intent=build_intent,
            source_artifact=source_artifact,
            tensors=dict(resolved_binding.tensors),
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=serving_manifest_ref,
        )
        publication = build_pure_transform_publication_bundle(
            build_intent=build_intent,
            source_artifact=source_artifact,
            contract_family=contract_family,
            publication_subject=publication_subject,
            canonical_index=authoritative_canonical_index,
            source_version_key=source_version_key,
            serving_version_key=serving_version_key,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=prepared.serving_manifest_ref,
            layout_id=layout_id,
            readiness_policy=readiness_policy,
        )
        return self._complete_bound_representation_publication(
            binding=resolved_binding,
            current_value=current_value,
            publication=publication,
            contract_family=contract_family,
            source_artifact=source_artifact,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def _complete_registered_representation_publication(
        self,
        *,
        publication: RegisteredServingPublication,
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        structural_view_ids: Sequence[str] | None = None,
        source_contribution_device: str | int | None = None,
        source_contribution_artifacts: Sequence[Artifact] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        resolved_contract_family = _resolve_repo_owned_contract_family(
            publication.publication,
            contract_family,
        )
        resolved_structural_view_ids = tuple(
            str(view_id).strip() for view_id in (structural_view_ids or ())
        )
        if (
            not resolved_structural_view_ids
            and resolved_contract_family != "canonical_full"
        ):
            resolved_structural_view_ids = _resolve_source_contribution_view_ids(
                source_contribution_artifacts
            )
        if source_contribution_device is None:
            with tensorcast_profile_stage(
                "tensorcast",
                "publication.complete_repo_owned_representation_publish_attempt",
                logger=logger,
                extra={
                    "contract_family": resolved_contract_family,
                    "has_source_contribution_device": False,
                    "structural_view_count": len(resolved_structural_view_ids),
                },
            ) as profile:
                result = self.complete_repo_owned_representation_publish_attempt(
                    publication=publication.publication,
                    contract_family=contract_family,
                    source_artifact=(
                        source_artifact
                        if isinstance(source_artifact, Artifact)
                        else None
                    ),
                    structural_view_ids=resolved_structural_view_ids or None,
                    layout_id=layout_id,
                    layout_artifact_id=layout_artifact_id,
                    readiness_policy=readiness_policy,
                    timeout_s=timeout_s,
                    ctx=ctx,
                )
                if profile is not None:
                    profile["serving_artifact_id"] = getattr(
                        result, "serving_artifact_id", None
                    )
                    profile["serving_manifest_ref"] = getattr(
                        result, "serving_manifest_ref", None
                    )
                return result
        resolved_contribution_artifacts = tuple(source_contribution_artifacts or ())
        if resolved_contribution_artifacts:
            contribution_artifacts = resolved_contribution_artifacts
        elif isinstance(source_artifact, Artifact):
            contribution_artifacts = (source_artifact,)
        else:
            contribution_artifacts = ()
        if not contribution_artifacts:
            raise ArtifactError(
                "source_contribution_device requires source_artifact or source_contribution_artifacts",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.start_repo_owned_representation_publish_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
                "has_source_contribution_device": True,
                "source_contribution_device": source_contribution_device,
                "structural_view_count": len(resolved_structural_view_ids),
                "contribution_artifact_count": len(contribution_artifacts),
            },
        ) as profile:
            source_contribution_artifact = (
                source_artifact if isinstance(source_artifact, Artifact) else None
            )
            attempt = self.start_repo_owned_representation_publish_attempt(
                publication=publication.publication,
                contract_family=contract_family,
                source_artifact=source_contribution_artifact,
                structural_view_ids=resolved_structural_view_ids or None,
                layout_id=layout_id,
                layout_artifact_id=layout_artifact_id,
                readiness_policy=readiness_policy,
                ctx=ctx,
            )
            if profile is not None:
                profile["attempt_id"] = getattr(attempt, "operation_id", None)
        if resolved_contract_family == "canonical_full":
            with tensorcast_profile_stage(
                "tensorcast",
                "publication.contribute_source_current_values_to_attempt_and_keep_bindings",
                logger=logger,
                extra={
                    "contract_family": resolved_contract_family,
                    "source_contribution_device": source_contribution_device,
                    "contribution_artifact_count": len(contribution_artifacts),
                },
            ) as profile:
                live_bindings = (
                    self._contribute_source_current_values_to_attempt_and_keep_bindings(
                        source_artifacts=contribution_artifacts,
                        attempt=attempt,
                        device=source_contribution_device,
                        ctx=ctx,
                    )
                )
                if profile is not None:
                    profile["live_binding_count"] = len(live_bindings)
            try:
                with tensorcast_profile_stage(
                    "tensorcast",
                    "publication.seal_assembly_attempt",
                    logger=logger,
                    extra={
                        "contract_family": resolved_contract_family,
                    },
                ) as profile:
                    operation = self.seal_assembly_attempt(attempt, ctx=ctx)
                    if profile is not None:
                        profile["operation_id"] = getattr(
                            operation, "operation_id", None
                        )
                with tensorcast_profile_stage(
                    "tensorcast",
                    "publication.wait_assembly_attempt",
                    logger=logger,
                    extra={
                        "contract_family": resolved_contract_family,
                        "timeout_s": timeout_s,
                    },
                ) as profile:
                    result = self.wait_assembly_attempt(
                        operation,
                        timeout_s=timeout_s,
                        ctx=ctx,
                    )
                    if profile is not None:
                        profile["serving_artifact_id"] = getattr(
                            result, "serving_artifact_id", None
                        )
                        profile["serving_manifest_ref"] = getattr(
                            result, "serving_manifest_ref", None
                        )
                    return result
            finally:
                for binding in live_bindings:
                    binding.close()
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.contribute_source_artifacts_to_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
                "source_contribution_device": source_contribution_device,
                "contribution_artifact_count": len(contribution_artifacts),
            },
        ):
            self._contribute_source_artifacts_to_attempt(
                source_artifacts=contribution_artifacts,
                attempt=attempt,
                device=source_contribution_device,
                ctx=ctx,
            )
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.seal_assembly_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
            },
        ) as profile:
            operation = self.seal_assembly_attempt(attempt, ctx=ctx)
            if profile is not None:
                profile["operation_id"] = getattr(operation, "operation_id", None)
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.wait_assembly_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
                "timeout_s": timeout_s,
            },
        ) as profile:
            result = self.wait_assembly_attempt(operation, timeout_s=timeout_s, ctx=ctx)
            if profile is not None:
                profile["serving_artifact_id"] = getattr(
                    result, "serving_artifact_id", None
                )
                profile["serving_manifest_ref"] = getattr(
                    result, "serving_manifest_ref", None
                )
            return result

    def complete_binding_finalize_publication_from_binding(
        self,
        binding: Binding | SealedBindingValue,
        *,
        build_intent: ServingBuildIntent,
        admission_facts: ServingAdmissionFacts,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        contract_family: AssemblyContractFamily | str | None = None,
        representation_contract_hash: str | None = None,
        source_version_key: str | None = None,
        serving_version_key: str | None = None,
        logical_topology_json: str | None = None,
        serving_manifest_ref: str | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        resolved_binding, current_value = self._resolve_bound_publication_subject(
            binding
        )
        publication_subject = current_value.to_binding_value_ref()
        authoritative_canonical_index = _build_bound_publication_canonical_index(
            resolved_binding.layout
        )
        prepared = prepare_binding_finalize_serving_registration(
            build_intent=build_intent,
            tensors=dict(resolved_binding.tensors),
            representation_contract_hash=representation_contract_hash,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=serving_manifest_ref,
        )
        publication = build_binding_finalize_publication_bundle(
            build_intent=build_intent,
            source_artifact=source_artifact,
            contract_family=contract_family,
            publication_subject=publication_subject,
            canonical_index=authoritative_canonical_index,
            representation_contract_hash=representation_contract_hash,
            source_version_key=source_version_key,
            serving_version_key=serving_version_key,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=prepared.serving_manifest_ref,
            layout_id=layout_id,
            readiness_policy=readiness_policy,
            admission_facts=admission_facts,
        )
        return self._complete_bound_representation_publication(
            binding=resolved_binding,
            current_value=current_value,
            publication=publication,
            contract_family=contract_family,
            source_artifact=source_artifact,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def _contribute_source_artifacts_to_attempt(
        self,
        *,
        source_artifacts: Sequence[Artifact],
        attempt: AssemblyAttemptRef,
        device: str | int,
        ctx: CallContext | None = None,
    ) -> tuple[PartialSealResult, ...]:
        results: list[PartialSealResult] = []
        binding_device = cast(torch.device | str, device)
        for source_artifact in source_artifacts:
            binding = source_artifact.bind(device=binding_device, packing="byte_space")
            try:
                sealed = binding.seal_current(
                    update_epoch=binding.begin_update(ctx=ctx),
                    ctx=ctx,
                )
                results.append(sealed.contribute_to_assembly(attempt=attempt, ctx=ctx))
            finally:
                with contextlib.suppress(Exception):
                    binding.close()
        return tuple(results)

    def _contribute_source_current_values_to_attempt_and_keep_bindings(
        self,
        *,
        source_artifacts: Sequence[Artifact],
        attempt: AssemblyAttemptRef,
        device: str | int,
        ctx: CallContext | None = None,
    ) -> tuple[Binding, ...]:
        live_bindings: list[Binding] = []
        binding_device = cast(torch.device | str, device)
        try:
            for source_artifact in source_artifacts:
                binding = source_artifact.bind(
                    device=binding_device,
                    packing="byte_space",
                )
                try:
                    current_value = binding.current_value
                    if current_value is None:
                        raise ArtifactError(
                            "source contribution binding is missing current_value",
                            status_code="FAILED_PRECONDITION",
                            retryable=False,
                        )
                    current_value.contribute_to_assembly(attempt=attempt, ctx=ctx)
                    live_bindings.append(binding)
                except Exception:
                    with contextlib.suppress(Exception):
                        binding.close()
                    raise
            return tuple(live_bindings)
        except Exception:
            for binding in live_bindings:
                with contextlib.suppress(Exception):
                    binding.close()
            raise

    def _resolve_bound_publication_subject(
        self,
        binding: Binding | SealedBindingValue,
    ) -> tuple[Binding, SealedBindingValue]:
        if isinstance(binding, Binding):
            current_value = binding.current_value
            if current_value is None:
                raise ArtifactError(
                    "binding publication requires a current sealed value",
                    status_code="FAILED_PRECONDITION",
                    retryable=False,
                )
            return binding, current_value
        if isinstance(binding, SealedBindingValue):
            return binding._require_current_binding(), binding
        raise ArtifactError(
            "binding publication requires a Binding or SealedBindingValue",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    def _complete_bound_representation_publication(
        self,
        *,
        binding: Binding,
        current_value: SealedBindingValue,
        publication: RepresentationPublishSpec,
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact
        | RegisteredArtifact
        | CanonicalIndex
        | object
        | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        del binding
        resolved_contract_family = _resolve_repo_owned_contract_family(
            publication,
            contract_family,
        )
        if resolved_contract_family != "canonical_full":
            raise ArtifactError(
                "binding-native publication currently requires contract_family='canonical_full'",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.start_bound_representation_publish_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
                "binding_id": current_value.binding_id,
                "binding_value_id": current_value.binding_value_id,
            },
        ) as profile:
            attempt = self.start_repo_owned_representation_publish_attempt(
                publication=publication,
                contract_family=contract_family,
                source_artifact=(
                    source_artifact if isinstance(source_artifact, Artifact) else None
                ),
                structural_view_ids=None,
                layout_id=layout_id,
                layout_artifact_id=layout_artifact_id,
                readiness_policy=readiness_policy,
                ctx=ctx,
            )
            if profile is not None:
                profile["attempt_id"] = getattr(attempt, "attempt_id", None)
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.contribute_binding_current_value_to_attempt",
            logger=logger,
            extra={
                "binding_id": current_value.binding_id,
                "binding_value_id": current_value.binding_value_id,
            },
        ):
            current_value.contribute_to_assembly(attempt=attempt, ctx=ctx)
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.seal_assembly_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
            },
        ) as profile:
            operation = self.seal_assembly_attempt(attempt, ctx=ctx)
            if profile is not None:
                profile["operation_id"] = getattr(operation, "operation_id", None)
        with tensorcast_profile_stage(
            "tensorcast",
            "publication.wait_assembly_attempt",
            logger=logger,
            extra={
                "contract_family": resolved_contract_family,
                "timeout_s": timeout_s,
            },
        ) as profile:
            result = self.wait_assembly_attempt(
                operation,
                timeout_s=timeout_s,
                ctx=ctx,
            )
            if profile is not None:
                profile["serving_artifact_id"] = getattr(
                    result, "serving_artifact_id", None
                )
                profile["serving_manifest_ref"] = getattr(
                    result, "serving_manifest_ref", None
                )
            return result

    def start_assembly_attempt(
        self,
        *,
        layout_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        closeout_contract: AssemblyCloseoutContract | None = None,
        representation_publish_spec: RepresentationPublishSpec | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        del ctx
        client = self._runtime.ensure_client()
        if representation_publish_spec is not None:
            if representation_publish_spec.readiness_policy is None:
                if representation_publish_spec.closeout_contract is None:
                    return client.start_assembly_attempt(
                        layout_id=representation_publish_spec.layout_id,
                        requirements=representation_publish_spec.requirements,
                        representation_publish_spec=representation_publish_spec,
                    )
                return client.start_assembly_attempt(
                    layout_id=representation_publish_spec.layout_id,
                    requirements=representation_publish_spec.requirements,
                    closeout_contract=representation_publish_spec.closeout_contract,
                    representation_publish_spec=representation_publish_spec,
                )
            if representation_publish_spec.closeout_contract is None:
                return client.start_assembly_attempt(
                    layout_id=representation_publish_spec.layout_id,
                    requirements=representation_publish_spec.requirements,
                    readiness_policy=representation_publish_spec.readiness_policy,
                    representation_publish_spec=representation_publish_spec,
                )
            return client.start_assembly_attempt(
                layout_id=representation_publish_spec.layout_id,
                requirements=representation_publish_spec.requirements,
                readiness_policy=representation_publish_spec.readiness_policy,
                closeout_contract=representation_publish_spec.closeout_contract,
                representation_publish_spec=representation_publish_spec,
            )
        if not layout_id:
            raise ValueError("layout_id is required")
        if requirements is None:
            raise ValueError(
                "requirements are required; construct them explicitly with "
                "AssemblyRequirementSetRef.pp_from_structural_views(...), "
                "AssemblyRequirementSetRef.ep_from_structural_views(...), "
                "or AssemblyRequirementSetRef.canonical_full()"
            )
        if readiness_policy is None:
            if closeout_contract is None:
                return client.start_assembly_attempt(
                    layout_id=layout_id,
                    requirements=requirements,
                )
            return client.start_assembly_attempt(
                layout_id=layout_id,
                requirements=requirements,
                closeout_contract=closeout_contract,
            )
        if closeout_contract is None:
            return client.start_assembly_attempt(
                layout_id=layout_id,
                requirements=requirements,
                readiness_policy=readiness_policy,
            )
        return client.start_assembly_attempt(
            layout_id=layout_id,
            requirements=requirements,
            readiness_policy=readiness_policy,
            closeout_contract=closeout_contract,
        )

    def list_artifact_layouts(
        self,
        artifact_id: str,
        *,
        ctx: CallContext | None = None,
    ) -> tuple[str, ...]:
        del ctx
        if not artifact_id:
            raise ArtifactError(
                "artifact_id is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return tuple(
            str(layout_id)
            for layout_id in self._runtime.ensure_client().list_artifact_layouts(
                artifact_id
            )
        )

    def start_representation_publish_attempt(
        self,
        *,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        resolved_layout_id = _resolve_representation_publish_layout_id(
            self,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            publication=publication,
        )
        resolved_requirements = _resolve_representation_publish_requirements(
            requirements=requirements,
            publication=publication,
        )
        resolved_readiness_policy = _resolve_representation_publish_readiness_policy(
            readiness_policy=readiness_policy,
            publication=publication,
        )
        publication_spec = _coerce_representation_publish_spec(publication)
        if publication_spec is not None:
            if resolved_requirements is None:
                raise ArtifactError(
                    "representation publication requires explicit requirements or a RepresentationPublishSpec carrying requirements",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            return self.start_assembly_attempt(
                representation_publish_spec=publication_spec.with_attempt_inputs(
                    layout_id=resolved_layout_id,
                    requirements=resolved_requirements,
                    readiness_policy=resolved_readiness_policy,
                ),
                ctx=ctx,
            )
        closeout_contract = _coerce_representation_publish_closeout(publication)
        return self.start_assembly_attempt(
            layout_id=resolved_layout_id,
            requirements=resolved_requirements,
            readiness_policy=resolved_readiness_policy,
            closeout_contract=closeout_contract,
            ctx=ctx,
        )

    def complete_representation_publish_attempt(
        self,
        *,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        attempt = self.start_representation_publish_attempt(
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            requirements=requirements,
            publication=publication,
            readiness_policy=readiness_policy,
            ctx=ctx,
        )
        operation = self.seal_assembly_attempt(attempt, ctx=ctx)
        return self.wait_assembly_attempt(operation, timeout_s=timeout_s, ctx=ctx)

    def start_canonical_representation_publish_attempt(
        self,
        *,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        return self.start_representation_publish_attempt(
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            requirements=AssemblyRequirementSetRef.canonical_full(),
            publication=publication,
            readiness_policy=readiness_policy,
            ctx=ctx,
        )

    def complete_canonical_representation_publish_attempt(
        self,
        *,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        return self.complete_representation_publish_attempt(
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            requirements=AssemblyRequirementSetRef.canonical_full(),
            publication=publication,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def start_structural_representation_publish_attempt(
        self,
        *,
        contract_family: AssemblyContractFamily,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        requirements = build_representation_publish_requirements(
            contract_family=contract_family,
            publication=publication,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
        )
        return self.start_representation_publish_attempt(
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            requirements=requirements,
            publication=publication,
            readiness_policy=readiness_policy,
            ctx=ctx,
        )

    def complete_structural_representation_publish_attempt(
        self,
        *,
        contract_family: AssemblyContractFamily,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        requirements = build_representation_publish_requirements(
            contract_family=contract_family,
            publication=publication,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
        )
        return self.complete_representation_publish_attempt(
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            requirements=requirements,
            publication=publication,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def start_repo_owned_representation_publish_attempt(
        self,
        *,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        resolved_contract_family = _resolve_repo_owned_contract_family(
            publication,
            contract_family,
        )
        if resolved_contract_family == "canonical_full":
            return self.start_canonical_representation_publish_attempt(
                layout_id=layout_id,
                layout_artifact_id=layout_artifact_id,
                publication=publication,
                readiness_policy=readiness_policy,
                ctx=ctx,
            )
        return self.start_structural_representation_publish_attempt(
            contract_family=resolved_contract_family,
            publication=publication,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            ctx=ctx,
        )

    def complete_repo_owned_representation_publish_attempt(
        self,
        *,
        publication: RepresentationPublishSpec | AssemblyCloseoutContract,
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        resolved_contract_family = _resolve_repo_owned_contract_family(
            publication,
            contract_family,
        )
        if resolved_contract_family == "canonical_full":
            return self.complete_canonical_representation_publish_attempt(
                layout_id=layout_id,
                layout_artifact_id=layout_artifact_id,
                publication=publication,
                readiness_policy=readiness_policy,
                timeout_s=timeout_s,
                ctx=ctx,
            )
        return self.complete_structural_representation_publish_attempt(
            contract_family=resolved_contract_family,
            publication=publication,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def start_plan_repo_owned_representation_publish_attempt(
        self,
        *,
        plan_result: "PlanResult",
        publication_step: "PlanStepRef[RepresentationPublishSpec]",
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        ctx: CallContext | None = None,
    ) -> AssemblyAttemptRef:
        publication = _resolve_plan_publication_bundle(
            plan_result=plan_result,
            publication_step=publication_step,
        )
        return self.start_repo_owned_representation_publish_attempt(
            publication=publication,
            contract_family=contract_family,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            ctx=ctx,
        )

    def complete_plan_repo_owned_representation_publish_attempt(
        self,
        *,
        plan_result: "PlanResult",
        publication_step: "PlanStepRef[RepresentationPublishSpec]",
        contract_family: AssemblyContractFamily | str | None = None,
        source_artifact: Artifact | None = None,
        structural_view_ids: Sequence[str] | None = None,
        layout_id: str | None = None,
        layout_artifact_id: str | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        publication = _resolve_plan_publication_bundle(
            plan_result=plan_result,
            publication_step=publication_step,
        )
        return self.complete_repo_owned_representation_publish_attempt(
            publication=publication,
            contract_family=contract_family,
            source_artifact=source_artifact,
            structural_view_ids=structural_view_ids,
            layout_id=layout_id,
            layout_artifact_id=layout_artifact_id,
            readiness_policy=readiness_policy,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    def seal_assembly_attempt(
        self,
        attempt: AssemblyAttemptRef,
        *,
        ctx: CallContext | None = None,
    ) -> Operation[PublishedModelVersion]:
        if not attempt.coordinator_operation_id:
            raise ArtifactError(
                "AssemblyAttemptRef is missing coordinator operation metadata",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        resp = self._runtime.ensure_client().seal_assembly_attempt(
            attempt_id=attempt.attempt_id,
            timeout_s=10.0,
        )
        operation_ref = operation_pb2.OperationRef()
        operation_ref.CopyFrom(resp.operation)
        context = {
            "attempt_id": attempt.attempt_id,
            "workspace_assembly_id": attempt.workspace_assembly_id,
            "layout_id": attempt.layout_id,
        }

        def _decode(
            op_resp: operation_pb2.GetOperationResponse,
        ) -> PublishedModelVersion:
            return _decode_published_model_version_from_response(
                op_resp, assembly_id=attempt.workspace_assembly_id
            )

        return DaemonGlobalStoreOperation(
            operation_id=str(operation_ref.operation_id),
            runtime_ref=weakref.ref(self._runtime),
            ctx=ctx,
            context=context,
            result_factory=_decode,
            operation_ref=operation_ref,
        )

    def wait_assembly_attempt(
        self,
        attempt: AssemblyAttemptRef | Operation[PublishedModelVersion],
        *,
        timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> PublishedModelVersion:
        del ctx
        if isinstance(attempt, Operation):
            return attempt.wait(timeout_s=timeout_s)

        client = self._runtime.ensure_client()
        operation_id: str
        assembly_id: str
        operation_ref: operation_pb2.OperationRef | None = None
        if isinstance(attempt, AssemblyAttemptRef):
            if not attempt.coordinator_operation_id:
                raise ArtifactError(
                    "AssemblyAttemptRef is missing coordinator operation metadata",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            operation_id = attempt.coordinator_operation_id
            assembly_id = attempt.workspace_assembly_id
            operation_ref = operation_pb2.OperationRef()
            operation_ref.CopyFrom(attempt.coordinator_operation)
        wait_timeout_s = 120.0 if timeout_s is None else float(timeout_s)
        resp = client.wait_operation(
            operation_id,
            operation_ref=operation_ref,
            timeout_ms=max(1, int(wait_timeout_s * 1000)),
            timeout_s=wait_timeout_s + 5.0,
        )
        return _decode_published_model_version_from_response(
            resp, assembly_id=assembly_id
        )

    def persistence_operation(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> Operation[PersistenceStatusResult]:
        op_id = task_id or f"persist:{artifact_id or ''}"
        created_at = time.monotonic()

        def _persistence_error_context(
            *,
            task_id_value: str,
            artifact_id_value: str,
        ) -> dict[str, str]:
            return {
                "operation_kind": "persistence_task",
                "task_id": task_id_value,
                "artifact_id": artifact_id_value,
                "target_artifact_id": artifact_id_value,
            }

        def _ctx_remaining_timeout_s() -> float | None:
            if ctx is None or ctx.deadline_ms is None:
                return None
            remaining = (float(ctx.deadline_ms) / 1000.0) - (
                time.monotonic() - created_at
            )
            return max(0.0, remaining)

        def _query() -> PersistenceStatusResult:
            timeout_s = _ctx_remaining_timeout_s()
            if timeout_s is not None and timeout_s <= 0:
                raise OperationTimeoutError(
                    "Operation deadline exceeded (ctx.deadline_ms)",
                    retryable=True,
                )
            resp = self._runtime.ensure_client().query_persistence_status(
                task_id=task_id,
                artifact_id=artifact_id,
                timeout_s=timeout_s if timeout_s is not None else 10.0,
            )
            return self._persistence_status_from_proto(resp)

        def _status() -> OperationStatus:
            timeout_s = _ctx_remaining_timeout_s()
            if timeout_s is not None and timeout_s <= 0:
                return OperationStatus(
                    state="degraded",
                    message="CallContext deadline exceeded",
                    progress=0.0,
                    as_of_ms=int(time.time() * 1000),
                    error=OperationError(
                        status_code="DEADLINE_EXCEEDED",
                        message="CallContext deadline exceeded",
                        retryable=True,
                        context=_persistence_error_context(
                            task_id_value=task_id or "",
                            artifact_id_value=artifact_id or "",
                        ),
                    ),
                )
            try:
                result = _query()
            except OperationTimeoutError as exc:
                return OperationStatus(
                    state="degraded",
                    message=str(exc),
                    progress=0.0,
                    as_of_ms=int(time.time() * 1000),
                    error=OperationError(
                        status_code="DEADLINE_EXCEEDED",
                        message=str(exc),
                        retryable=True,
                        context=_persistence_error_context(
                            task_id_value=task_id or "",
                            artifact_id_value=artifact_id or "",
                        ),
                    ),
                )
            state: str
            if result.state in {"pending", "running", "unknown"}:
                state = "running"
            elif result.state == "success":
                state = "success"
            elif result.state == "failed":
                state = "failed"
            elif result.state == "degraded":
                state = "degraded"
            else:
                state = "running"
            error: OperationError | None = None
            if state in {"failed", "degraded"}:
                message = (
                    result.last_error
                    or result.degraded_reason
                    or "persistence degraded"
                )
                error = OperationError(
                    status_code="INTERNAL"
                    if state == "failed"
                    else "DEADLINE_EXCEEDED",
                    message=message,
                    retryable=True,
                    context=_persistence_error_context(
                        task_id_value=result.task_id,
                        artifact_id_value=result.artifact_id,
                    ),
                )
            return OperationStatus(
                state=state,  # type: ignore[arg-type]
                message=result.degraded_reason or None,
                progress=float(result.progress),
                as_of_ms=int(time.time() * 1000),
                error=error,
            )

        def _result() -> PersistenceStatusResult:
            return _query()

        return PollingOperation(
            operation_id=op_id,
            status_fn=_status,
            result_fn=_result,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------
    def artifact(
        self,
        ref: str | None = None,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
    ) -> Artifact:
        artifact_id, key = _parse_artifact_ref(
            ref,
            artifact_id=artifact_id,
            key=key,
        )
        return Artifact(
            store_ref=weakref.ref(self),
            artifact_id=artifact_id,
            key=key,
        )

    async def artifact_async(
        self,
        ref: str | None = None,
        *,
        artifact_id: str | None = None,
        key: str | None = None,
    ) -> Artifact:
        return self.artifact(
            ref,
            artifact_id=artifact_id,
            key=key,
        )

    def _artifact_from_disk_metadata(
        self,
        *,
        disk_path: str,
        artifact_id: str,
        canonical_index_bytes: bytes,
        generation: int | None,
        key: str | None,
        event_name: str,
        resolution_mode: str,
        trusted_content_artifact_id: str | None = None,
    ) -> Artifact:
        if not artifact_id:
            raise ArtifactError(
                "disk resolution returned empty artifact_id",
                status_code="DATA_LOSS",
                retryable=False,
            )
        if not canonical_index_bytes:
            raise ArtifactError(
                "disk resolution returned empty canonical index bytes",
                status_code="DATA_LOSS",
                retryable=False,
            )
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        entry = ArtifactCacheEntry(
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            parsed_index=canonical_index,
            generation=generation,
            expires_at=time.monotonic(),
        )
        self._runtime.cache_artifact_index(entry)
        artifact_id_kind = infer_artifact_id_kind(artifact_id)
        trusted_hint_kind = (
            infer_artifact_id_kind(trusted_content_artifact_id)
            if trusted_content_artifact_id
            else None
        )
        emit_tensorcast_profile_event(
            "tensorcast",
            event_name,
            logger=logger,
            payload={
                "path": disk_path,
                "artifact_id": artifact_id,
                "artifact_id_kind": (
                    artifact_id_kind.value.lower() if artifact_id_kind else None
                ),
                "trusted_content_artifact_id": trusted_content_artifact_id,
                "trusted_content_hint_kind": (
                    trusted_hint_kind.value.lower() if trusted_hint_kind else None
                ),
                "tensor_count": len(canonical_index.entries),
                "total_size_bytes": canonical_index.total_size_bytes,
                "daemon_endpoint": self._runtime.daemon_endpoint,
                "resolution_mode": resolution_mode,
            },
        )
        return Artifact(
            store_ref=weakref.ref(self),
            artifact_id=artifact_id,
            key=key,
            canonical_index_bytes=canonical_index_bytes or None,
            canonical_index=canonical_index,
            generation=generation,
        )

    def import_from_disk(
        self,
        path: str,
        *,
        key: str | None = None,
        verify_checksums: bool = True,
        show_progress: bool | None = None,
    ) -> Artifact:
        disk_path = os.fspath(path)
        if not disk_path:
            raise ValueError("path is required")
        client = self._runtime.ensure_client()
        use_progress = _should_show_from_disk_progress(show_progress)
        response = None
        deadline = time.monotonic() + 30.0
        with tensorcast_profile_stage(
            "tensorcast",
            "store.import_from_disk",
            logger=logger,
            extra={
                "path": disk_path,
                "verify_checksums": bool(verify_checksums),
                "show_progress": use_progress,
                "daemon_endpoint": self._runtime.daemon_endpoint,
            },
        ) as profile:
            startup_retry_count = 0
            while response is None:
                try:
                    stream = client.import_artifact_from_path_stream_v2(
                        path=disk_path,
                        verify_checksums=bool(verify_checksums),
                    )

                    event_count = 0
                    processed_bytes = 0
                    total_bytes = 0
                    messages: list[str] = []
                    if profile is not None:

                        def _profiled_stream(
                            *,
                            bound_stream=stream,
                            bound_messages=messages,
                        ):
                            nonlocal event_count, processed_bytes, total_bytes
                            for event in bound_stream:
                                event_count += 1
                                total_bytes = int(
                                    getattr(event, "total_bytes", 0) or total_bytes
                                )
                                processed_bytes = int(
                                    getattr(event, "processed_bytes", 0)
                                    or processed_bytes
                                )
                                message = str(getattr(event, "message", "") or "")
                                if message and (
                                    not bound_messages or bound_messages[-1] != message
                                ):
                                    bound_messages.append(message)
                                yield event

                        stream_to_consume = _profiled_stream()
                    else:
                        stream_to_consume = stream

                    if use_progress:
                        response = _consume_import_artifact_stream_with_tqdm(
                            stream_to_consume,
                            disk_path=disk_path,
                        )
                    else:
                        response = _consume_import_artifact_stream(stream_to_consume)
                    if profile is not None:
                        profile["stream_event_count"] = event_count
                        profile["processed_bytes"] = processed_bytes
                        profile["total_bytes"] = total_bytes
                        profile["message_samples"] = messages[:20]
                        profile["resolution_mode"] = "import"
                except RuntimeError as exc:
                    remaining = deadline - time.monotonic()
                    if not _is_import_startup_in_progress_error(exc) or remaining <= 0:
                        raise
                    startup_retry_count += 1
                    if profile is not None:
                        profile["startup_retry_count"] = startup_retry_count
                    logger.info(
                        "store.import_from_disk_retry_startup",
                        extra={
                            "tc.store.daemon": self._runtime.daemon_endpoint,
                            "path": disk_path,
                            "remaining_s": round(remaining, 3),
                        },
                        exc_info=exc,
                    )
                    time.sleep(min(1.0, remaining))
        artifact_id = response.artifact_id or ""
        canonical_index_bytes = bytes(response.canonical_index_bytes)
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        generation_value: int | None = (
            int(response.generation) if response.generation else None
        )
        artifact = self._artifact_from_disk_metadata(
            disk_path=disk_path,
            artifact_id=artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            generation=generation_value,
            key=key,
            event_name="store.import_from_disk.summary",
            resolution_mode="import",
        )
        if key:
            index_multihash, data_multihash = _split_mi2_artifact_id(artifact_id)
            id_kind = infer_artifact_id_kind(artifact_id) or ArtifactIdKind.MI2
            descriptor = TypedArtifactDescriptor(
                artifact_id=artifact_id,
                index_multihash=index_multihash,
                data_multihash=data_multihash,
                schema_version=None,
                encoding=None,
                total_size=canonical_index.total_size_bytes,
                id_kind=id_kind,
            )
            try:
                ok = client.publish_replica_key(key=key, descriptor=descriptor)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to publish key %s via daemon", key)
            else:
                if not ok:
                    logger.warning(
                        "Key mapping for %s already exists; keeping existing mapping",
                        key,
                    )
                else:
                    self._runtime.cache_key_mapping(key, artifact_id=artifact_id)
        return artifact

    def from_disk(
        self,
        path: str,
        *,
        key: str | None = None,
        verify_checksums: bool = True,
        show_progress: bool | None = None,
    ) -> Artifact:
        disk_path = os.fspath(path)
        if not disk_path:
            raise ValueError("path is required")
        use_progress = _should_show_from_disk_progress(show_progress)
        if key is not None or use_progress:
            return self.import_from_disk(
                disk_path,
                key=key,
                verify_checksums=verify_checksums,
                show_progress=show_progress,
            )
        client = self._runtime.ensure_client()
        response = None
        deadline = time.monotonic() + 30.0
        with tensorcast_profile_stage(
            "tensorcast",
            "store.from_disk",
            logger=logger,
            extra={
                "path": disk_path,
                "verify_checksums": bool(verify_checksums),
                "show_progress": use_progress,
                "daemon_endpoint": self._runtime.daemon_endpoint,
            },
        ) as profile:
            startup_retry_count = 0
            while response is None:
                try:
                    response = client.resolve_public_disk_source(
                        path=disk_path,
                        verify_checksums=bool(verify_checksums),
                    )
                    if profile is not None:
                        profile["resolution_mode"] = "attested_mounted_source"
                except RuntimeError as exc:
                    remaining = deadline - time.monotonic()
                    if not _is_import_startup_in_progress_error(exc) or remaining <= 0:
                        raise
                    startup_retry_count += 1
                    if profile is not None:
                        profile["startup_retry_count"] = startup_retry_count
                    logger.info(
                        "store.from_disk_retry_startup",
                        extra={
                            "tc.store.daemon": self._runtime.daemon_endpoint,
                            "path": disk_path,
                            "remaining_s": round(remaining, 3),
                        },
                        exc_info=exc,
                    )
                    time.sleep(min(1.0, remaining))
        has_source = (
            response.HasField("source")
            if hasattr(response, "HasField")
            else getattr(response, "source", None) is not None
        )
        if not has_source:
            raise ArtifactError(
                "ResolvePublicDiskSource returned empty source",
                status_code="DATA_LOSS",
                retryable=False,
            )
        source = PublicDiskSourceHandle.from_proto(response.source)
        if profile is not None:
            profile["artifact_id"] = source.artifact_id
            profile["trusted_content_artifact_id"] = (
                source.trusted_content_artifact_id or ""
            )
            profile["metadata_capability"] = (
                str(source.metadata_capability)
                if source.metadata_capability is not None
                else ""
            )
        generation_value: int | None = (
            int(source.generation) if source.generation else None
        )
        return self._artifact_from_disk_metadata(
            disk_path=disk_path,
            artifact_id=source.artifact_id,
            canonical_index_bytes=bytes(source.canonical_index_bytes),
            generation=generation_value,
            key=None,
            event_name="store.from_disk.summary",
            resolution_mode="attested_mounted_source",
            trusted_content_artifact_id=source.trusted_content_artifact_id,
        )

    def resolve_public_disk_source(
        self,
        path: str,
        *,
        verify_checksums: bool = True,
    ) -> PublicDiskSourceHandle:
        disk_path = os.fspath(path)
        if not disk_path:
            raise ValueError("path is required")
        try:
            response = self._runtime.ensure_client().resolve_public_disk_source(
                path=disk_path,
                verify_checksums=bool(verify_checksums),
            )
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"ResolvePublicDiskSource RPC failed for {disk_path}",
                status_code="UNAVAILABLE",
                retryable=True,
            ) from exc
        has_source = (
            response.HasField("source")
            if hasattr(response, "HasField")
            else getattr(response, "source", None) is not None
        )
        if not has_source:
            raise ArtifactError(
                "ResolvePublicDiskSource returned empty source",
                status_code="DATA_LOSS",
                retryable=False,
            )
        return PublicDiskSourceHandle.from_proto(response.source)

    def promote_mounted_source(
        self,
        artifact: Artifact | str,
        *,
        verify_checksums: bool = True,
        timeout_s: float | None = None,
    ) -> Artifact:
        source_artifact_id = (
            artifact.artifact_id if isinstance(artifact, Artifact) else str(artifact)
        )
        if not source_artifact_id:
            raise ValueError("artifact is required")
        artifact_kind = infer_artifact_id_kind(source_artifact_id)
        if artifact_kind is ArtifactIdKind.MI2:
            return self.artifact(source_artifact_id)
        if artifact_kind is not ArtifactIdKind.MSA1:
            raise ValueError("artifact must be an msa1 mounted-source artifact id")
        try:
            response = self._runtime.ensure_client().promote_mounted_source_artifact(
                artifact_id=source_artifact_id,
                verify_checksums=bool(verify_checksums),
                timeout_s=float(timeout_s) if timeout_s is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"PromoteMountedSourceArtifact RPC failed for {source_artifact_id}",
                status_code="UNAVAILABLE",
                retryable=True,
            ) from exc
        promoted_artifact_id = str(response.artifact_id or "")
        canonical_index_bytes = bytes(response.canonical_index_bytes)
        generation_value: int | None = (
            int(response.generation) if response.generation else None
        )
        return self._artifact_from_disk_metadata(
            disk_path=source_artifact_id,
            artifact_id=promoted_artifact_id,
            canonical_index_bytes=canonical_index_bytes,
            generation=generation_value,
            key=None,
            event_name="store.promote_mounted_source.summary",
            resolution_mode="mounted_source_promotion",
        )

    def realize_into_binding(
        self,
        binding: Binding,
        source: Artifact | PublicDiskSourceHandle | str,
        *,
        realization_plan: object,
        ctx: CallContext | None = None,
    ) -> BindingUpdateEpoch:
        if not isinstance(binding, Binding):
            raise ArtifactError(
                "realize_into_binding() requires a Binding",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        return binding.realize_from(
            source,
            realization_plan=realization_plan,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Region-backed registration
    # ------------------------------------------------------------------
    def register_region(
        self,
        *,
        memory_kind: RegionMemoryKind,
        size_bytes: int,
        ttl_ms: int,
        device_id: int | None = None,
        cuda_ipc_handle: bytes | None = None,
        host_shared_attach_token: bytes | None = None,
        daemon_managed: bool = False,
        host_shared_region_class: HostSharedRegionClass | None = None,
        session_id: str | None = None,
        name: str | None = None,
        timeout_s: float = 10.0,
    ) -> LocalRegionHandle:
        client = self._runtime.ensure_client()
        return client.register_region(
            memory_kind=memory_kind,
            size_bytes=int(size_bytes),
            ttl_ms=int(ttl_ms),
            device_id=device_id,
            cuda_ipc_handle=cuda_ipc_handle,
            host_shared_attach_token=host_shared_attach_token,
            daemon_managed=bool(daemon_managed),
            host_shared_region_class=host_shared_region_class,
            session_id=session_id,
            region_name=name,
            timeout_s=float(timeout_s),
        )

    def unregister_region(
        self,
        region_id: str,
        *,
        session_id: str | None = None,
        force: bool | None = None,
        timeout_s: float = 10.0,
    ) -> bool:
        client = self._runtime.ensure_client()
        released = client.unregister_region(
            region_id,
            session_id=session_id,
            force=force,
            timeout_s=float(timeout_s),
        )
        if released:
            with contextlib.suppress(Exception):
                _cache_unregister_region(region_id)
        return released

    def activate_stable_local_backing(
        self,
        region_id: str,
        *,
        slot_bytes: int,
        session_id: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        client = self._runtime.ensure_client()
        client.activate_stable_local_backing(
            region_id,
            slot_bytes=int(slot_bytes),
            session_id=session_id,
            timeout_s=float(timeout_s),
        )

    def attach_host_shared_region(
        self,
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> HostSharedRegionAttachment:
        client = self._runtime.ensure_client()
        return client.attach_host_shared_region(handle, timeout_s=float(timeout_s))

    def release_host_shared_region(
        self,
        handle: LocalRegionHandle,
        *,
        timeout_s: float = 5.0,
    ) -> bool:
        client = self._runtime.ensure_client()
        return client.release_host_shared_region(handle, timeout_s=float(timeout_s))

    def register_vram_region(
        self,
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
        name: str | None = None,
    ) -> VramRegionHandle:
        client = self._runtime.ensure_client()
        base_ptr_value = int(base_ptr)
        size_value = int(size_bytes)
        base_offset = 0
        try:
            handle_bytes, base_offset = get_cuda_memory_handle_with_offset(
                int(device_id), base_ptr_value
            )
        except Exception:  # noqa: BLE001
            handle_bytes = get_cuda_memory_handle(int(device_id), base_ptr_value)
            base_offset = 0
        if base_offset:
            base_ptr_value -= int(base_offset)
            size_value += int(base_offset)
        handle = client.register_vram_region(
            device_id=int(device_id),
            size_bytes=int(size_value),
            ttl_ms=int(ttl_ms),
            cuda_ipc_handle=handle_bytes,
            region_name=name,
        )
        with contextlib.suppress(Exception):
            _cache_register_region(
                region_id=handle.region_id,
                device_id=int(device_id),
                base_ptr=int(base_ptr_value),
                size_bytes=int(size_value),
                ttl_ms=int(ttl_ms),
            )
        return handle

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        client = self._runtime.ensure_client()
        released = client.unregister_vram_region(region_id, force=force)
        if released:
            with contextlib.suppress(Exception):
                _cache_unregister_region(region_id)
        return released

    def deregister_artifact(
        self,
        artifact_id: str,
        *,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        extend_ttl_ms: int | None = None,
        device_id: int | None = None,
        byte_space: common_pb2.ByteSpaceRef | None = None,
        keep_shared_disk_copy: bool = False,
        operation_id: str | None = None,
    ) -> DeregisterArtifactOutcome:
        client = self._runtime.ensure_client()
        drain_ms = int(drain_timeout_s * 1000) if drain_timeout_s is not None else None
        outcome = client.deregister_artifact(
            artifact_id,
            wait_for_drain=bool(wait),
            drain_timeout_ms=drain_ms,
            extend_ttl_ms=extend_ttl_ms,
            device_id=device_id,
            byte_space=byte_space,
            keep_shared_disk_copy=keep_shared_disk_copy,
            operation_id=operation_id,
        )
        self._runtime.invalidate_artifact(artifact_id, key=None, reason="deregister")
        return outcome

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        return self._runtime.capabilities

    @property
    def daemon_endpoint(self) -> str:
        return self._runtime.daemon_endpoint

    @property
    def closed(self) -> bool:
        return self._runtime.closed

    @property
    def batcher(self) -> MaterializationBatcher:
        if self._batcher is None:
            raise RuntimeError("Materialization batcher is disabled")
        return self._batcher

    def close(self) -> None:
        with contextlib.suppress(Exception):
            _LIVE_STORES.discard(self)
        with contextlib.suppress(Exception):
            if self._batcher is not None:
                self._batcher.close()
        self._runtime.close()


_PROCESS_STORE_LOCK = threading.RLock()
_PROCESS_STORE: Store | None = None
_PROCESS_STORE_OPTS: StoreOptions | None = None
_LIVE_STORES: "weakref.WeakSet[Store]" = weakref.WeakSet()


def _ensure_process_store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    context = get_runtime_context(
        opts=opts,
        force_recreate=force_recreate,
        client_factory=get_daemon_client,
        runtime_provider=require_runtime,
    )

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        current_closed = current.closed if current is not None else False
        if current is None or force_recreate or current_closed:
            prior = _PROCESS_STORE
            opts_marker: StoreOptions | None
            if opts is not None:
                opts_marker = opts
            elif current_closed and not force_recreate:
                opts_marker = _PROCESS_STORE_OPTS
            else:
                opts_marker = None
            effective_opts: StoreOptions = opts_marker or context.opts
            _PROCESS_STORE = Store(
                context.daemon_endpoint, opts=effective_opts, runtime=context
            )
            _PROCESS_STORE_OPTS = opts_marker
            if prior is not None:
                with contextlib.suppress(Exception):
                    prior.close()
        elif (
            opts is not None
            and _PROCESS_STORE_OPTS is not None
            and opts != _PROCESS_STORE_OPTS
        ):
            raise RuntimeError(
                "Store already initialized with different options. Pass force_recreate=True to replace."
            )
        result = _PROCESS_STORE
        if result is None:
            raise RuntimeError("Failed to initialize process Store")
        return result


def store(
    *,
    opts: StoreOptions | None = None,
    force_recreate: bool = False,
) -> Store:
    """Return the process-wide Store session, creating it lazily as needed."""

    return _ensure_process_store(opts=opts, force_recreate=force_recreate)


def shutdown_process_store() -> None:
    """Close and clear the process-wide Store if it exists."""

    global _PROCESS_STORE, _PROCESS_STORE_OPTS

    with _PROCESS_STORE_LOCK:
        current = _PROCESS_STORE
        _PROCESS_STORE = None
        _PROCESS_STORE_OPTS = None
    if current is not None:
        with contextlib.suppress(Exception):
            current.close()
    shutdown_context()


def _shutdown_live_stores() -> None:
    for current in list(_LIVE_STORES):
        with contextlib.suppress(Exception):
            current.close()
    shutdown_context()


atexit.register(_shutdown_live_stores)


def _coerce_store() -> Store:
    current = _PROCESS_STORE
    if current is not None and not current.closed:
        return current
    return store()


def register(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    ttl_ms: int | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().register_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        ttl_ms=ttl_ms,
    )


def register_view(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    slices: Mapping[str, Sequence[object]] | None = None,
    transpose: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    view_id: str | None = None,
    placement: str | None = None,
    ttl_ms: int | None = None,
    options: RegisterArtifactOptions | None = None,
    canonical_index_bytes: bytes | None = None,
    registration_kind: str | int | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register_view(
        tensors,
        artifact_id=artifact_id,
        key=key,
        slices=slices,
        transpose=transpose,
        view_id=view_id,
        placement=placement,
        ttl_ms=ttl_ms,
        options=options,
        canonical_index_bytes=canonical_index_bytes,
        registration_kind=registration_kind,
    )


def register_piece(
    tensors: TensorDict,
    *,
    assembly_id: str,
    key: str | None = None,
    slices: Mapping[str, Sequence[object]] | None = None,
    canonical_index_bytes: bytes | None = None,
    placement: str | None = None,
    ttl_ms: int | None = None,
    options: RegisterArtifactOptions | None = None,
) -> RegisteredArtifact:
    return _coerce_store().register_piece(
        tensors,
        assembly_id=assembly_id,
        key=key,
        slices=slices,
        canonical_index_bytes=canonical_index_bytes,
        placement=placement,
        ttl_ms=ttl_ms,
        options=options,
    )


def register_pure_transform_publication(
    tensors: TensorDict,
    *,
    build_intent: ServingBuildIntent,
    source_artifact: Artifact
    | RegisteredArtifact
    | CanonicalIndex
    | object
    | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    device: int | torch.device | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
) -> RegisteredServingPublication:
    return _coerce_store().register_pure_transform_publication(
        tensors,
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=contract_family,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        device=device,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
    )


def complete_pure_transform_publication(
    tensors: TensorDict,
    *,
    build_intent: ServingBuildIntent,
    source_artifact: Artifact
    | RegisteredArtifact
    | CanonicalIndex
    | object
    | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    device: int | torch.device | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    structural_view_ids: Sequence[str] | None = None,
    source_contribution_device: str | int | None = None,
    source_contribution_artifacts: Sequence[Artifact] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_pure_transform_publication(
        tensors,
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=contract_family,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        device=device,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        structural_view_ids=structural_view_ids,
        source_contribution_device=source_contribution_device,
        source_contribution_artifacts=source_contribution_artifacts,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def complete_pure_transform_publication_from_binding(
    binding: Binding | SealedBindingValue,
    *,
    build_intent: ServingBuildIntent,
    source_artifact: Artifact
    | RegisteredArtifact
    | CanonicalIndex
    | object
    | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_pure_transform_publication_from_binding(
        binding,
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=contract_family,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def complete_binding_finalize_publication_from_binding(
    binding: Binding | SealedBindingValue,
    *,
    build_intent: ServingBuildIntent,
    admission_facts: ServingAdmissionFacts,
    source_artifact: Artifact
    | RegisteredArtifact
    | CanonicalIndex
    | object
    | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    representation_contract_hash: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_binding_finalize_publication_from_binding(
        binding,
        build_intent=build_intent,
        admission_facts=admission_facts,
        source_artifact=source_artifact,
        contract_family=contract_family,
        representation_contract_hash=representation_contract_hash,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def register_vram_region(
    *,
    device_id: int,
    base_ptr: int,
    size_bytes: int,
    ttl_ms: int,
    name: str | None = None,
) -> VramRegionHandle:
    return _coerce_store().register_vram_region(
        device_id=device_id,
        base_ptr=base_ptr,
        size_bytes=size_bytes,
        ttl_ms=ttl_ms,
        name=name,
    )


def unregister_vram_region(region_id: str, *, force: bool | None = None) -> bool:
    return _coerce_store().unregister_vram_region(region_id, force=force)


def deregister_artifact(
    artifact_id: str,
    *,
    wait: bool = True,
    drain_timeout_s: float | None = None,
    extend_ttl_ms: int | None = None,
    device_id: int | None = None,
    byte_space: common_pb2.ByteSpaceRef | None = None,
    keep_shared_disk_copy: bool = False,
    operation_id: str | None = None,
) -> DeregisterArtifactOutcome:
    return _coerce_store().deregister_artifact(
        artifact_id,
        wait=wait,
        drain_timeout_s=drain_timeout_s,
        extend_ttl_ms=extend_ttl_ms,
        device_id=device_id,
        byte_space=byte_space,
        keep_shared_disk_copy=keep_shared_disk_copy,
        operation_id=operation_id,
    )


def put(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> RegisteredArtifact:
    return _coerce_store().put(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def put_async(
    tensors: TensorDict,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
    policy: StorePolicy | str | None = None,
    options: RegisterArtifactOptions | None = None,
    device: int | torch.device | None = None,
) -> ArtifactFuture[RegisteredArtifact]:
    return _coerce_store().put_async(
        tensors,
        artifact_id=artifact_id,
        key=key,
        policy=policy,
        options=options,
        device=device,
    )


def create_binding(
    layout: BindingLayout,
    *,
    ownership: str = "daemon",
    device: torch.device | str | None = None,
    target_tensors: Mapping[str, torch.Tensor] | None = None,
    mapping: CopyPlan | None = None,
    ctx: CallContext | None = None,
    restore_tensors_async: bool = False,
) -> Binding:
    return _coerce_store().create_binding(
        layout,
        ownership=ownership,
        device=device,
        target_tensors=target_tensors,
        mapping=mapping,
        ctx=ctx,
        restore_tensors_async=restore_tensors_async,
    )


def query_persistence_status(
    *, task_id: str | None = None, artifact_id: str | None = None
) -> PersistenceStatusResult:
    return _coerce_store().query_persistence_status(
        task_id=task_id, artifact_id=artifact_id
    )


def persistence_operation(
    *,
    task_id: str | None = None,
    artifact_id: str | None = None,
    ctx: CallContext | None = None,
) -> Operation[PersistenceStatusResult]:
    return _coerce_store().persistence_operation(
        task_id=task_id, artifact_id=artifact_id, ctx=ctx
    )


def start_assembly_attempt(
    *,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    closeout_contract: AssemblyCloseoutContract | None = None,
    representation_publish_spec: RepresentationPublishSpec | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_assembly_attempt(
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        closeout_contract=closeout_contract,
        representation_publish_spec=representation_publish_spec,
        ctx=ctx,
    )


def start_representation_publish_attempt(
    *,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_representation_publish_attempt(
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        requirements=requirements,
        publication=publication,
        readiness_policy=readiness_policy,
        ctx=ctx,
    )


def seal_assembly_attempt(
    attempt: AssemblyAttemptRef,
    *,
    ctx: CallContext | None = None,
) -> Operation[PublishedModelVersion]:
    return _coerce_store().seal_assembly_attempt(attempt, ctx=ctx)


def wait_assembly_attempt(
    attempt: AssemblyAttemptRef | Operation[PublishedModelVersion],
    *,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().wait_assembly_attempt(
        attempt,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def complete_representation_publish_attempt(
    *,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_representation_publish_attempt(
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        requirements=requirements,
        publication=publication,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def start_canonical_representation_publish_attempt(
    *,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_canonical_representation_publish_attempt(
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        publication=publication,
        readiness_policy=readiness_policy,
        ctx=ctx,
    )


def complete_canonical_representation_publish_attempt(
    *,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_canonical_representation_publish_attempt(
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        publication=publication,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def start_structural_representation_publish_attempt(
    *,
    contract_family: AssemblyContractFamily,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_structural_representation_publish_attempt(
        contract_family=contract_family,
        publication=publication,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        ctx=ctx,
    )


def complete_structural_representation_publish_attempt(
    *,
    contract_family: AssemblyContractFamily,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_structural_representation_publish_attempt(
        contract_family=contract_family,
        publication=publication,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def start_repo_owned_representation_publish_attempt(
    *,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    contract_family: AssemblyContractFamily | str | None = None,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_repo_owned_representation_publish_attempt(
        publication=publication,
        contract_family=contract_family,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        ctx=ctx,
    )


def complete_repo_owned_representation_publish_attempt(
    *,
    publication: RepresentationPublishSpec | AssemblyCloseoutContract,
    contract_family: AssemblyContractFamily | str | None = None,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_repo_owned_representation_publish_attempt(
        publication=publication,
        contract_family=contract_family,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def start_plan_repo_owned_representation_publish_attempt(
    *,
    plan_result: "PlanResult",
    publication_step: "PlanStepRef[RepresentationPublishSpec]",
    contract_family: AssemblyContractFamily | str | None = None,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    ctx: CallContext | None = None,
) -> AssemblyAttemptRef:
    return _coerce_store().start_plan_repo_owned_representation_publish_attempt(
        plan_result=plan_result,
        publication_step=publication_step,
        contract_family=contract_family,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        ctx=ctx,
    )


def complete_plan_repo_owned_representation_publish_attempt(
    *,
    plan_result: "PlanResult",
    publication_step: "PlanStepRef[RepresentationPublishSpec]",
    contract_family: AssemblyContractFamily | str | None = None,
    source_artifact: Artifact | None = None,
    structural_view_ids: Sequence[str] | None = None,
    layout_id: str | None = None,
    layout_artifact_id: str | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> PublishedModelVersion:
    return _coerce_store().complete_plan_repo_owned_representation_publish_attempt(
        plan_result=plan_result,
        publication_step=publication_step,
        contract_family=contract_family,
        source_artifact=source_artifact,
        structural_view_ids=structural_view_ids,
        layout_id=layout_id,
        layout_artifact_id=layout_artifact_id,
        readiness_policy=readiness_policy,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def list_artifact_layouts(
    artifact_id: str,
    *,
    ctx: CallContext | None = None,
) -> tuple[str, ...]:
    return _coerce_store().list_artifact_layouts(artifact_id, ctx=ctx)


def seal_assembly(
    assembly_id: str,
    *,
    publish_canonical: bool = True,
    wait: bool = True,
    layout_id: str | None = None,
    timeout_s: float = 120.0,
    ctx: CallContext | None = None,
) -> SealAssemblyResult | Operation[SealAssemblyResult]:
    return _coerce_store().seal_assembly(
        assembly_id,
        publish_canonical=publish_canonical,
        wait=wait,
        layout_id=layout_id,
        timeout_s=timeout_s,
        ctx=ctx,
    )


def artifact(
    ref: str | None = None,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
) -> Artifact:
    store = _coerce_store()
    if ref is None:
        return store.artifact(
            artifact_id=artifact_id,
            key=key,
        )
    return store.artifact(
        ref=ref,
        artifact_id=artifact_id,
        key=key,
    )


async def artifact_async(
    ref: str | None = None,
    *,
    artifact_id: str | None = None,
    key: str | None = None,
) -> Artifact:
    store = _coerce_store()
    if ref is None:
        return await store.artifact_async(
            artifact_id=artifact_id,
            key=key,
        )
    return await store.artifact_async(
        ref=ref,
        artifact_id=artifact_id,
        key=key,
    )


def from_disk(
    path: str,
    *,
    key: str | None = None,
    verify_checksums: bool = True,
    show_progress: bool | None = None,
) -> Artifact:
    return _coerce_store().from_disk(
        path,
        key=key,
        verify_checksums=verify_checksums,
        show_progress=show_progress,
    )


def import_from_disk(
    path: str,
    *,
    key: str | None = None,
    verify_checksums: bool = True,
    show_progress: bool | None = None,
) -> Artifact:
    return _coerce_store().import_from_disk(
        path,
        key=key,
        verify_checksums=verify_checksums,
        show_progress=show_progress,
    )


def promote_mounted_source(
    artifact: Artifact | str,
    *,
    verify_checksums: bool = True,
    timeout_s: float | None = None,
) -> Artifact:
    return _coerce_store().promote_mounted_source(
        artifact,
        verify_checksums=verify_checksums,
        timeout_s=timeout_s,
    )


def resolve_public_disk_source(
    path: str,
    *,
    verify_checksums: bool = True,
) -> PublicDiskSourceHandle:
    return _coerce_store().resolve_public_disk_source(
        path,
        verify_checksums=verify_checksums,
    )


def realize_into_binding(
    binding: Binding,
    source: Artifact | PublicDiskSourceHandle | str,
    *,
    realization_plan: object,
    ctx: CallContext | None = None,
) -> BindingUpdateEpoch:
    return _coerce_store().realize_into_binding(
        binding,
        source,
        realization_plan=realization_plan,
        ctx=ctx,
    )


__all__ = [
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactError",
    "ArtifactFuture",
    "ArtifactStatusCode",
    "AssemblyAttemptRef",
    "AssemblyCloseoutContract",
    "AssemblyReadinessPolicy",
    "AssemblyRequirementSetRef",
    "BindingReservationCapability",
    "BindingPromotionStatusState",
    "Binding",
    "BindingValueRef",
    "GroupRealizationAcquireRef",
    "BindingValueVerificationState",
    "BindingLayout",
    "BindingUpdateEpoch",
    "BlobRef",
    "BuilderMode",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "CopyPlan",
    "CopyPlanEntry",
    "FinalizeClass",
    "HostSharedRegionAttachment",
    "HostSharedRegionClass",
    "LeaseHandle",
    "LocalRegionHandle",
    "MaterializationPayload",
    "MaterializationDiagnostics",
    "MaterializationBatcher",
    "PlacementPin",
    "PrefetchRetentionPolicy",
    "PrefetchedReplica",
    "PrefetchedServingBinding",
    "PrefetchedServingBindingSet",
    "PartialSealResult",
    "PublicDiskSourceHandle",
    "PreparedServingRegistration",
    "PublishedModelVersion",
    "RegionMemoryKind",
    "ExecutionDiagnostics",
    "SourceBoundPlanDiagnostics",
    "HashBackend",
    "HashLocation",
    "IdentityMintStrategy",
    "RegisteredServingPublication",
    "RegisteredArtifact",
    "RepresentationPublishContract",
    "RepresentationPublishSpec",
    "SourceBoundCapability",
    "ServingPublicationSubject",
    "ReplicaInfo",
    "RetryPolicy",
    "SERVING_MANIFEST_TENSOR_NAME",
    "SealedBindingValue",
    "StagedBindingValue",
    "ServingArtifactManifest",
    "ServingAdmissionFacts",
    "ServingBindingMemberRef",
    "ServingBindingReadiness",
    "ServingBindingResolvedLayout",
    "ServingBindingResolvedSpecCacheEntry",
    "ServingBindingSetTarget",
    "ServingBindingSourceKind",
    "ServingBindingSourceMemberRef",
    "ServingBindingSourceRef",
    "ServingBindingSourceReuseDecision",
    "ServingBindingSourceReuseMode",
    "ServingBindingTarget",
    "ServingBindingSpecCacheRecord",
    "REFERENCE_RUNTIME",
    "ReferenceServingAcquireResult",
    "ReferenceServingResolvedSpec",
    "ReferenceServingTensorSpec",
    "acquire_reference_binding",
    "build_reference_resolved_spec",
    "build_reference_target_layout",
    "build_reference_tensor_index_bytes",
    "canonical_json_bytes",
    "prefetch_reference_binding",
    "read_matching_resolved_spec_cache_entry",
    "read_resolved_spec_cache_entry",
    "release_reference_acquire",
    "serving_binding_spec_cache_root",
    "target_from_reference_cache_record",
    "unpack_prefetched_serving_binding",
    "write_resolved_spec_cache_entry",
    "write_reference_resolved_spec_cache_entry",
    "ServingBuildIntent",
    "SERVING_BUILD_DIGEST_VERSION",
    "ServingRuntimePolicy",
    "ServingRuntimePolicyInput",
    "ServingSupportLevel",
    "StoreCapabilities",
    "Store",
    "StoreOptions",
    "TensorMeta",
    "TensorDictMaterializationResult",
    "TensorDict",
    "TransformPlacement",
    "Range",
    "BindingRealizationEntry",
    "BindingRealizationPlan",
    "binding_realization_plan_to_proto",
    "normalize_binding_realization_plan",
    "build_owned_layout",
    "build_binding_finalize_admission_facts",
    "build_binding_finalize_publication_bundle",
    "build_owned_layout",
    "build_serving_publication_bundle",
    "build_serving_publication_bundle_from_registered_artifact",
    "build_pure_transform_publication_bundle",
    "build_pure_transform_publication_bundle_from_registered_artifact",
    "build_pure_transform_publication_spec",
    "build_pure_transform_serving_args",
    "build_pure_transform_transform_spec",
    "build_representation_publish_requirements",
    "complete_binding_finalize_publication_from_binding",
    "complete_pure_transform_publication",
    "complete_pure_transform_publication_from_binding",
    "complete_canonical_representation_publish_attempt",
    "complete_plan_repo_owned_representation_publish_attempt",
    "complete_representation_publish_attempt",
    "complete_repo_owned_representation_publish_attempt",
    "complete_structural_representation_publish_attempt",
    "compute_pure_transform_representation_contract_hash",
    "build_serving_manifest_ref",
    "coerce_serving_runtime_policy",
    "compute_serving_tensor_schema_hash",
    "count_canonical_serving_tensors",
    "prepare_binding_finalize_serving_registration",
    "prepare_serving_registration",
    "prepare_pure_transform_serving_registration",
    "parse_serving_manifest_ref",
    "TargetTensors",
    "PersistenceStatusResult",
    "PersistenceShardStatus",
    "ServingBindingSpecCacheGroupIndex",
    "ServingBindingSpecCacheRecord",
    "REFERENCE_RUNTIME",
    "ReferenceServingAcquireResult",
    "ReferenceServingResolvedSpec",
    "ReferenceServingTensorSpec",
    "acquire_reference_binding",
    "build_reference_resolved_spec",
    "build_reference_target_layout",
    "build_reference_tensor_index_bytes",
    "canonical_json_bytes",
    "prefetch_reference_binding",
    "read_matching_resolved_spec_cache_entry",
    "read_resolved_spec_cache_entry",
    "read_resolved_spec_cache_group_index",
    "release_reference_acquire",
    "serving_binding_spec_cache_root",
    "target_from_reference_cache_record",
    "unpack_prefetched_serving_binding",
    "write_resolved_spec_cache_entry",
    "write_resolved_spec_cache_group_index",
    "write_reference_resolved_spec_cache_entry",
    "artifact",
    "artifact_async",
    "create_binding",
    "from_disk",
    "import_from_disk",
    "promote_mounted_source",
    "realize_into_binding",
    "resolve_public_disk_source",
    "list_artifact_layouts",
    "store",
    "shutdown_process_store",
    "BatchContext",
    "register",
    "register_async",
    "put",
    "put_async",
    "query_persistence_status",
    "persistence_operation",
    "register_pure_transform_publication",
    "start_canonical_representation_publish_attempt",
    "start_assembly_attempt",
    "start_plan_repo_owned_representation_publish_attempt",
    "start_representation_publish_attempt",
    "start_repo_owned_representation_publish_attempt",
    "start_structural_representation_publish_attempt",
    "register_view",
    "register_piece",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "seal_assembly",
    "seal_assembly_attempt",
    "wait_assembly_attempt",
    "get_daemon_client",
    "require_runtime",
    "_register_artifact_core",
    "materialize_artifact_v2",
]
