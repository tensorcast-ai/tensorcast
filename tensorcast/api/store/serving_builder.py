#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch

from tensorcast.api._config import DEFAULT_ALIGN
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.common.selection_contract import compute_selected_index_bytes
from tensorcast.types import (
    SERVING_MANIFEST_TENSOR_NAME,
    ArtifactDescriptor,
    AssemblyCloseoutContract,
    AssemblyContractFamily,
    AssemblyReadinessPolicy,
    AssemblyRequirementSetRef,
    BuilderMode,
    PureTransformPublicationSpec,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    ServingArtifactManifest,
    ServingBuildIntent,
    build_serving_manifest_ref,
)

if TYPE_CHECKING:
    from tensorcast.api.store.artifact import (
        ArtifactDescriptor as StoreArtifactDescriptor,
    )

ServingPublicationBundle = RepresentationPublishSpec
PureTransformPublicationBundle = ServingPublicationBundle


@dataclass(frozen=True, slots=True)
class PreparedServingRegistration:
    tensors: dict[str, torch.Tensor]
    serving_manifest_ref: str
    manifest_tensor_name: str
    serving_manifest: ServingArtifactManifest
    serving_manifest_bytes: bytes
    representation_contract_hash: str
    canonical_index: CanonicalIndex


@dataclass(frozen=True, slots=True)
class RegisteredServingPublication:
    registered_artifact: RegisteredArtifact
    prepared_registration: PreparedServingRegistration
    publication: RepresentationPublishSpec


PreparedPureTransformServingRegistration = PreparedServingRegistration
RegisteredPureTransformPublication = RegisteredServingPublication


PURE_TRANSFORM_SERVING_ARG_ENABLE = "tc_serving_enable"
PURE_TRANSFORM_SERVING_ARG_REPRESENTATION_CONTRACT_HASH = (
    "tc_serving_representation_contract_hash"
)
PURE_TRANSFORM_SERVING_ARG_FRAMEWORK_NAME = "tc_serving_framework_name"
PURE_TRANSFORM_SERVING_ARG_ADAPTER_VERSION = "tc_serving_adapter_version"
PURE_TRANSFORM_SERVING_ARG_ABI_VERSION = "tc_serving_abi_version"
PURE_TRANSFORM_SERVING_ARG_BUILD_PIPELINE_VERSION = "tc_serving_build_pipeline_version"
PURE_TRANSFORM_SERVING_ARG_SOURCE_VERSION_KEY = "tc_serving_source_version_key"
PURE_TRANSFORM_SERVING_ARG_SERVING_VERSION_KEY = "tc_serving_serving_version_key"
PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON = "tc_serving_logical_topology_json"
PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF = "tc_serving_manifest_ref"
PURE_TRANSFORM_SERVING_ARG_CONTRACT_FAMILY = "tc_serving_contract_family"
PURE_TRANSFORM_TARGET_REPRESENTATION_FAMILY = "runtime_serving"
_PURE_TRANSFORM_SOURCE_BYTE_SPACE_KIND_CANONICAL = 1
_PURE_TRANSFORM_TARGET_REALIZATION_KIND = "artifact_publishable"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pad_manifest_carrier_bytes(payload: bytes) -> bytes:
    align = int(DEFAULT_ALIGN)
    if align <= 1:
        return payload
    remainder = len(payload) % align
    if remainder == 0:
        return payload
    # JSON permits trailing whitespace, so padding here keeps the manifest
    # parseable while avoiding PAD gaps in aligned stable registration layouts.
    return payload + (b" " * (align - remainder))


def _multibase_multihash_sha256(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("SHA256 digest must be 32 bytes")
    import base64

    multihash = b"\x12\x20" + digest
    encoded = base64.b32encode(multihash).decode("ascii").lower().rstrip("=")
    return f"b{encoded}"


def _hash_payload_to_multihash(payload: object) -> str:
    import hashlib

    return _multibase_multihash_sha256(
        hashlib.sha256(_canonical_json_bytes(payload)).digest()
    )


def _hash_versioned_payload_to_multihash(version: str, payload: object) -> str:
    import hashlib

    serialized = _canonical_json_bytes(payload)
    versioned_payload = version.encode("utf-8") + b"\n" + serialized
    return _multibase_multihash_sha256(hashlib.sha256(versioned_payload).digest())


def _dtype_to_string(dtype: torch.dtype) -> str:
    return str(dtype)


def _canonical_index_bytes_from_index(canonical_index: CanonicalIndex) -> bytes:
    return canonical_index_to_bytes(canonical_index)


def _normalize_contract_family(
    contract_family: AssemblyContractFamily | str | None,
) -> str | None:
    if contract_family is None:
        return None
    normalized = str(contract_family).strip()
    if not normalized:
        return None
    if normalized not in {"pp", "ep", "canonical_full"}:
        raise ArtifactError(
            "contract_family must be one of: pp, ep, canonical_full",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return normalized


def _resolve_manifest_tensor_name(
    serving_manifest_ref: str | None,
    *,
    helper_name: str,
) -> tuple[str, str]:
    resolved_manifest_ref = (
        build_serving_manifest_ref()
        if serving_manifest_ref is None
        else str(serving_manifest_ref)
    )
    if not resolved_manifest_ref.startswith("tensor:"):
        raise ArtifactError(
            f"{helper_name} currently supports only tensor:<name> manifest carriers",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    manifest_tensor_name = resolved_manifest_ref.split(":", 1)[1]
    if not manifest_tensor_name:
        raise ArtifactError(
            "serving_manifest_ref tensor carrier requires a tensor name",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return resolved_manifest_ref, manifest_tensor_name


def _resolve_explicit_representation_contract_hash(
    *,
    build_intent: ServingBuildIntent,
    representation_contract_hash: str | None,
    helper_name: str,
) -> str:
    resolved_representation_contract_hash = (
        representation_contract_hash or build_intent.representation_contract_hash
    )
    if not resolved_representation_contract_hash:
        raise ArtifactError(
            f"{helper_name} requires an explicit representation_contract_hash in "
            "the argument or ServingBuildIntent",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return resolved_representation_contract_hash


def _canonical_serving_entries(
    canonical_index: CanonicalIndex,
    *,
    manifest_tensor_name: str,
) -> tuple[CanonicalIndexEntry, ...]:
    return tuple(
        entry
        for entry in canonical_index.entries
        if str(entry.name) != str(manifest_tensor_name)
    )


def _repack_canonical_index(
    canonical_index: CanonicalIndex,
    *,
    manifest_tensor_name: str,
) -> CanonicalIndex:
    repacked_entries: list[CanonicalIndexEntry] = []
    offset = 0
    for entry in sorted(
        _canonical_serving_entries(
            canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        ),
        key=lambda item: (int(item.segment_offset), str(item.name)),
    ):
        repacked_entries.append(
            CanonicalIndexEntry(
                name=str(entry.name),
                dtype=entry.dtype,
                shape=tuple(int(dim) for dim in entry.shape),
                stride=tuple(int(dim) for dim in entry.stride),
                storage_offset=int(entry.storage_offset),
                segment_offset=int(offset),
                size_bytes=int(entry.size_bytes),
            )
        )
        offset += int(entry.size_bytes)
    return CanonicalIndex(
        entries=tuple(repacked_entries),
        total_size_bytes=int(offset),
        avbs_hash=str(canonical_index.avbs_hash),
    )


def compute_serving_tensor_schema_hash(
    canonical_index: CanonicalIndex,
    *,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> str:
    tensors = [
        {
            "name": str(entry.name),
            "dtype": _dtype_to_string(entry.dtype),
            "shape": [int(dim) for dim in entry.shape],
            "stride": [int(dim) for dim in entry.stride],
            "element_size": int(entry.dtype.itemsize),
        }
        for entry in sorted(
            _canonical_serving_entries(
                canonical_index,
                manifest_tensor_name=manifest_tensor_name,
            ),
            key=lambda entry: str(entry.name),
        )
    ]
    return _hash_versioned_payload_to_multihash(
        "tensorcast.representation.tensor_schema.v1",
        {"tensors": tensors},
    )


def count_canonical_serving_tensors(
    canonical_index: CanonicalIndex,
    *,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> int:
    return len(
        _canonical_serving_entries(
            canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        )
    )


def _artifact_id_from_input(
    serving_artifact: RegisteredArtifact
    | ArtifactDescriptor
    | "StoreArtifactDescriptor"
    | str,
) -> str:
    if isinstance(serving_artifact, RegisteredArtifact):
        return str(serving_artifact.artifact_id)
    if isinstance(serving_artifact, ArtifactDescriptor):
        return str(serving_artifact.artifact_id)
    if isinstance(serving_artifact, str):
        return str(serving_artifact)
    artifact_id = getattr(serving_artifact, "artifact_id", None)
    if artifact_id:
        return str(artifact_id)
    raise ArtifactError(
        "serving_artifact must be a RegisteredArtifact, ArtifactDescriptor, store descriptor, or artifact_id string",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _canonical_index_from_input(
    serving_artifact: RegisteredArtifact | CanonicalIndex | Mapping[str, torch.Tensor],
) -> CanonicalIndex:
    if isinstance(serving_artifact, Mapping):
        return _canonical_index_from_tensors(serving_artifact)
    if isinstance(serving_artifact, CanonicalIndex):
        return serving_artifact
    if isinstance(serving_artifact, RegisteredArtifact):
        return serving_artifact.canonical_index
    raise ArtifactError(
        "PURE_TRANSFORM publication bundle requires a RegisteredArtifact, explicit CanonicalIndex, or tensor mapping",
        status_code="INVALID_ARGUMENT",
        retryable=False,
    )


def _selected_canonical_index_from_source_artifact(
    source_artifact: RegisteredArtifact | CanonicalIndex | object,
) -> CanonicalIndex:
    if isinstance(source_artifact, CanonicalIndex):
        return source_artifact
    if isinstance(source_artifact, RegisteredArtifact):
        return source_artifact.canonical_index

    ensure_metadata = getattr(source_artifact, "_ensure_metadata", None)
    if callable(ensure_metadata):
        canonical_index = ensure_metadata()
    else:
        canonical_index = getattr(source_artifact, "_canonical_index", None)
    if canonical_index is None or not isinstance(canonical_index, CanonicalIndex):
        raise ArtifactError(
            "PURE_TRANSFORM representation hash derivation requires a source artifact with canonical index metadata",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    view_cache = None
    ensure_view_cache = getattr(source_artifact, "_ensure_view_metadata_cache", None)
    if callable(ensure_view_cache):
        view_cache = ensure_view_cache(require_selected_index=True)
    if view_cache is not None:
        selected_index = getattr(view_cache, "selected_index", None)
        if isinstance(selected_index, CanonicalIndex):
            return selected_index

    canonical_index_bytes = getattr(source_artifact, "_canonical_index_bytes", None)
    if canonical_index_bytes is None:
        canonical_index_bytes = _canonical_index_bytes_from_index(canonical_index)
    view_spec_build = getattr(source_artifact, "_view_spec", None)
    view_spec = getattr(view_spec_build, "proto", None)
    tensor_names = tuple(getattr(view_cache, "tensor_names", ()) or ())
    if view_spec is None and not tensor_names:
        return canonical_index
    selected_index_bytes = compute_selected_index_bytes(
        canonical_index_bytes=canonical_index_bytes,
        view_spec=view_spec,
        tensor_names=tensor_names or None,
    )
    return canonical_index_from_bytes(selected_index_bytes)


def _structural_view_ids_from_source_artifact(
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None,
) -> tuple[str, ...]:
    if source_artifact is None or isinstance(source_artifact, CanonicalIndex):
        return ()
    ensure_view_cache = getattr(source_artifact, "_ensure_view_metadata_cache", None)
    if not callable(ensure_view_cache):
        return ()
    view_cache = ensure_view_cache(require_view_id=True)
    if view_cache is None or not getattr(view_cache, "view_id", None):
        return ()
    view_id = str(view_cache.view_id).strip()
    if not view_id or view_id.startswith("mapped:v1:"):
        return ()
    return (view_id,)


def _canonical_index_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> CanonicalIndex:
    from tensorcast.api._register import (
        BuildContext,
        CoalescedLayout,
        build_canonical_index_bytes,
    )

    context = BuildContext.from_artifact(dict(tensors), device_id=None)
    layout = CoalescedLayout.compute(context.tensor_source_index, context.device_id)
    return canonical_index_from_bytes(
        build_canonical_index_bytes(
            context.tensor_meta_index,
            context.tensor_source_index,
            layout.offsets,
            context.device_id,
        )
    )


def _normalize_logical_topology_payload(
    logical_topology_json: str | None,
) -> dict[str, object] | None:
    if logical_topology_json is None:
        return None
    try:
        payload = json.loads(logical_topology_json)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            "logical_topology_json must be valid JSON",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactError(
            "logical_topology_json must encode an object",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    family = str(payload.get("family", "")).strip()
    version = str(payload.get("version", "")).strip()
    raw_dimensions = payload.get("dimensions", [])
    if not family:
        raise ArtifactError(
            "logical_topology_json.family must not be empty",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if not version:
        raise ArtifactError(
            "logical_topology_json.version must not be empty",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if not isinstance(raw_dimensions, list):
        raise ArtifactError(
            "logical_topology_json.dimensions must be a list",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    dimensions: list[dict[str, int | str]] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            raise ArtifactError(
                "logical_topology_json.dimensions items must be objects",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        name = str(raw_dimension.get("name", "")).strip()
        if not name:
            raise ArtifactError(
                "logical_topology_json dimensions require non-empty name",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        size = raw_dimension.get("size", None)
        if not isinstance(size, int) or size <= 0:
            raise ArtifactError(
                "logical_topology_json dimensions require positive integer size",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        dimensions.append({"name": name, "size": int(size)})
    dimensions.sort(key=lambda item: (str(item["name"]), int(item["size"])))
    return {
        "family": family,
        "version": version,
        "dimensions": dimensions,
    }


def _tensor_spec_payload(
    entry: CanonicalIndexEntry,
    *,
    include_offsets: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": str(entry.name),
        "dtype": _dtype_to_string(entry.dtype),
        "shape": [int(dim) for dim in entry.shape],
        "stride": [int(dim) for dim in entry.stride],
        "element_size": int(entry.dtype.itemsize),
    }
    if include_offsets:
        payload["logical_offset"] = int(entry.segment_offset)
        payload["logical_length"] = int(entry.size_bytes)
        payload["storage_offset"] = int(entry.storage_offset)
    return payload


def _full_coordinate_spec_payload() -> dict[str, object]:
    return {"selects_scalar": False, "axes": []}


def _validate_exact_copy_compatible(
    source_entry: CanonicalIndexEntry,
    target_entry: CanonicalIndexEntry,
    *,
    tensor_name: str,
) -> None:
    if (
        _dtype_to_string(source_entry.dtype) != _dtype_to_string(target_entry.dtype)
        or tuple(int(dim) for dim in source_entry.shape)
        != tuple(int(dim) for dim in target_entry.shape)
        or tuple(int(dim) for dim in source_entry.stride)
        != tuple(int(dim) for dim in target_entry.stride)
        or int(source_entry.size_bytes) != int(target_entry.size_bytes)
    ):
        raise ArtifactError(
            f"PURE_TRANSFORM representation hash derivation currently requires exact-copy tensor compatibility for '{tensor_name}'",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )


def compute_pure_transform_representation_contract_hash(
    *,
    source_artifact: RegisteredArtifact | CanonicalIndex | object,
    serving_artifact: RegisteredArtifact | CanonicalIndex | Mapping[str, torch.Tensor],
    logical_topology_json: str | None = None,
    manifest_tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
    target_representation_family: str = PURE_TRANSFORM_TARGET_REPRESENTATION_FAMILY,
) -> str:
    source_canonical_index = _repack_canonical_index(
        _selected_canonical_index_from_source_artifact(source_artifact),
        manifest_tensor_name=manifest_tensor_name,
    )
    target_canonical_index = _repack_canonical_index(
        _canonical_index_from_input(serving_artifact),
        manifest_tensor_name=manifest_tensor_name,
    )
    source_entries = {
        str(entry.name): entry
        for entry in _canonical_serving_entries(
            source_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        )
    }
    target_entries = {
        str(entry.name): entry
        for entry in _canonical_serving_entries(
            target_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        )
    }
    if set(source_entries) != set(target_entries):
        missing_in_target = sorted(set(source_entries) - set(target_entries))
        missing_in_source = sorted(set(target_entries) - set(source_entries))
        raise ArtifactError(
            "PURE_TRANSFORM representation hash derivation requires source/target tensor sets to match "
            f"(missing_in_target={missing_in_target}, missing_in_source={missing_in_source})",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    tensor_schema_hash = compute_serving_tensor_schema_hash(
        target_canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    ordered_names = sorted(
        source_entries,
        key=lambda name: (
            int(source_entries[name].segment_offset),
            str(name),
        ),
    )
    tensor_bindings: list[dict[str, object]] = []
    for name in ordered_names:
        source_entry = source_entries[name]
        target_entry = target_entries[name]
        _validate_exact_copy_compatible(
            source_entry,
            target_entry,
            tensor_name=name,
        )
        tensor_bindings.append(
            {
                "dst_name": str(name),
                "dst_spec": _tensor_spec_payload(
                    target_entry,
                    include_offsets=True,
                ),
                "op_kind": "exact_copy",
                "coverage_kind": "exact",
                "sources": [
                    {
                        "source_spec": _tensor_spec_payload(
                            source_entry,
                            include_offsets=True,
                        ),
                        "source_range": _full_coordinate_spec_payload(),
                        "destination_range": _full_coordinate_spec_payload(),
                        "role": "default",
                    }
                ],
                "fill_rule": None,
            }
        )

    payload = {
        "source_byte_space": {
            "kind": _PURE_TRANSFORM_SOURCE_BYTE_SPACE_KIND_CANONICAL,
            "id": "",
        },
        "target_representation": {
            "family": str(target_representation_family),
            "realization_kind": _PURE_TRANSFORM_TARGET_REALIZATION_KIND,
            "logical_topology": _normalize_logical_topology_payload(
                logical_topology_json
            ),
        },
        "tensor_schema_hash": tensor_schema_hash,
        "tensor_bindings": tensor_bindings,
        "residual_fallback_map": {
            "total_bytes": 0,
            "num_sources": 0,
            "segments": [],
        },
    }
    return _hash_versioned_payload_to_multihash(
        "tensorcast.representation.contract.v1",
        payload,
    )


def prepare_pure_transform_serving_registration(
    *,
    build_intent: ServingBuildIntent,
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None = None,
    tensors: Mapping[str, torch.Tensor],
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
) -> PreparedPureTransformServingRegistration:
    if build_intent.builder_mode is not BuilderMode.PURE_TRANSFORM:
        raise ArtifactError(
            "prepare_pure_transform_serving_registration requires ServingBuildIntent.builder_mode=PURE_TRANSFORM",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    normalized_logical_topology = _normalize_logical_topology_payload(
        logical_topology_json
    )
    resolved_manifest_ref, manifest_tensor_name = _resolve_manifest_tensor_name(
        serving_manifest_ref,
        helper_name="PURE_TRANSFORM serving registration",
    )
    if manifest_tensor_name in tensors:
        raise ArtifactError(
            f"PURE_TRANSFORM serving registration cannot reuse reserved manifest tensor name '{manifest_tensor_name}'",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    prepared_tensors = {str(name): tensor for name, tensor in dict(tensors).items()}
    base_canonical_index = _canonical_index_from_tensors(prepared_tensors)
    resolved_representation_contract_hash = (
        build_intent.representation_contract_hash
        or compute_pure_transform_representation_contract_hash(
            source_artifact=source_artifact,
            serving_artifact=base_canonical_index,
            logical_topology_json=logical_topology_json,
            manifest_tensor_name=manifest_tensor_name,
        )
    )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=build_intent,
        representation_contract_hash=resolved_representation_contract_hash,
        tensor_schema_hash=compute_serving_tensor_schema_hash(
            base_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        ),
        canonical_tensor_count=count_canonical_serving_tensors(
            base_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        ),
        serving_manifest_ref=resolved_manifest_ref,
        logical_topology_json=(
            json.dumps(
                normalized_logical_topology, sort_keys=True, separators=(",", ":")
            )
            if normalized_logical_topology is not None
            else None
        ),
    )
    manifest_bytes = _pad_manifest_carrier_bytes(manifest.to_bytes())
    prepared_tensors[manifest_tensor_name] = torch.tensor(
        list(manifest_bytes),
        dtype=torch.uint8,
    )
    final_canonical_index = _canonical_index_from_tensors(prepared_tensors)
    return PreparedServingRegistration(
        tensors=prepared_tensors,
        serving_manifest_ref=resolved_manifest_ref,
        manifest_tensor_name=manifest_tensor_name,
        serving_manifest=manifest,
        serving_manifest_bytes=manifest_bytes,
        representation_contract_hash=resolved_representation_contract_hash,
        canonical_index=final_canonical_index,
    )


def prepare_serving_registration(
    *,
    build_intent: ServingBuildIntent,
    tensors: Mapping[str, torch.Tensor],
    representation_contract_hash: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
) -> PreparedServingRegistration:
    normalized_logical_topology = _normalize_logical_topology_payload(
        logical_topology_json
    )
    resolved_manifest_ref, manifest_tensor_name = _resolve_manifest_tensor_name(
        serving_manifest_ref,
        helper_name="serving registration",
    )
    if manifest_tensor_name in tensors:
        raise ArtifactError(
            f"serving registration cannot reuse reserved manifest tensor name '{manifest_tensor_name}'",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    prepared_tensors = {str(name): tensor for name, tensor in dict(tensors).items()}
    base_canonical_index = _canonical_index_from_tensors(prepared_tensors)
    resolved_representation_contract_hash = (
        _resolve_explicit_representation_contract_hash(
            build_intent=build_intent,
            representation_contract_hash=representation_contract_hash,
            helper_name="serving registration",
        )
    )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=build_intent,
        representation_contract_hash=resolved_representation_contract_hash,
        tensor_schema_hash=compute_serving_tensor_schema_hash(
            base_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        ),
        canonical_tensor_count=count_canonical_serving_tensors(
            base_canonical_index,
            manifest_tensor_name=manifest_tensor_name,
        ),
        serving_manifest_ref=resolved_manifest_ref,
        logical_topology_json=(
            json.dumps(
                normalized_logical_topology, sort_keys=True, separators=(",", ":")
            )
            if normalized_logical_topology is not None
            else None
        ),
    )
    manifest_bytes = _pad_manifest_carrier_bytes(manifest.to_bytes())
    prepared_tensors[manifest_tensor_name] = torch.tensor(
        list(manifest_bytes),
        dtype=torch.uint8,
    )
    final_canonical_index = _canonical_index_from_tensors(prepared_tensors)
    return PreparedServingRegistration(
        tensors=prepared_tensors,
        serving_manifest_ref=resolved_manifest_ref,
        manifest_tensor_name=manifest_tensor_name,
        serving_manifest=manifest,
        serving_manifest_bytes=manifest_bytes,
        representation_contract_hash=resolved_representation_contract_hash,
        canonical_index=final_canonical_index,
    )


def build_pure_transform_serving_args(
    *,
    build_intent: ServingBuildIntent,
    contract_family: AssemblyContractFamily | str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    extra_args: dict[str, str | int] | None = None,
) -> dict[str, str | int]:
    args: dict[str, str | int] = dict(extra_args or {})
    args[PURE_TRANSFORM_SERVING_ARG_ENABLE] = 1
    if build_intent.representation_contract_hash:
        args[PURE_TRANSFORM_SERVING_ARG_REPRESENTATION_CONTRACT_HASH] = (
            build_intent.representation_contract_hash
        )
    args[PURE_TRANSFORM_SERVING_ARG_FRAMEWORK_NAME] = build_intent.framework_name
    args[PURE_TRANSFORM_SERVING_ARG_ADAPTER_VERSION] = build_intent.adapter_version
    args[PURE_TRANSFORM_SERVING_ARG_ABI_VERSION] = build_intent.serving_abi_version
    args[PURE_TRANSFORM_SERVING_ARG_BUILD_PIPELINE_VERSION] = (
        build_intent.build_pipeline_version
    )
    normalized_contract_family = _normalize_contract_family(contract_family)
    if normalized_contract_family is not None:
        args[PURE_TRANSFORM_SERVING_ARG_CONTRACT_FAMILY] = normalized_contract_family
    if source_version_key:
        args[PURE_TRANSFORM_SERVING_ARG_SOURCE_VERSION_KEY] = str(source_version_key)
    if serving_version_key:
        args[PURE_TRANSFORM_SERVING_ARG_SERVING_VERSION_KEY] = str(serving_version_key)
    if logical_topology_json:
        args[PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON] = str(
            logical_topology_json
        )
    if serving_manifest_ref:
        args[PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF] = str(serving_manifest_ref)
    return args


def build_pure_transform_publication_spec(
    *,
    build_intent: ServingBuildIntent,
    contract_family: AssemblyContractFamily | str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
) -> PureTransformPublicationSpec:
    return PureTransformPublicationSpec(
        build_intent=build_intent,
        contract_family=_normalize_contract_family(contract_family),
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=(
            None if serving_manifest_ref is None else str(serving_manifest_ref)
        ),
        layout_id=None if layout_id is None else str(layout_id),
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=tuple(
            str(view_id).strip()
            for view_id in structural_view_ids
            if str(view_id).strip()
        ),
    )


def build_pure_transform_transform_spec(
    *,
    transform_name: str,
    build_intent: ServingBuildIntent,
    contract_family: AssemblyContractFamily | str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
    transform_args: dict[str, str | int] | None = None,
    layout_hash: str | None = None,
) -> TransformSpec:
    if not transform_name:
        raise ArtifactError(
            "transform_name is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return TransformSpec(
        name=str(transform_name),
        args=dict(transform_args or {}),
        layout_hash=layout_hash,
        publication_spec=build_pure_transform_publication_spec(
            build_intent=build_intent,
            contract_family=contract_family,
            source_version_key=source_version_key,
            serving_version_key=serving_version_key,
            logical_topology_json=logical_topology_json,
            serving_manifest_ref=serving_manifest_ref,
            layout_id=layout_id,
            requirements=requirements,
            readiness_policy=readiness_policy,
            structural_view_ids=structural_view_ids,
        ),
    )


def build_pure_transform_publication_bundle(
    *,
    build_intent: ServingBuildIntent,
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    serving_artifact: RegisteredArtifact
    | ArtifactDescriptor
    | "StoreArtifactDescriptor"
    | str,
    canonical_index: CanonicalIndex,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
) -> RepresentationPublishSpec:
    if build_intent.builder_mode is not BuilderMode.PURE_TRANSFORM:
        raise ArtifactError(
            "build_pure_transform_publication_bundle requires ServingBuildIntent.builder_mode=PURE_TRANSFORM",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    normalized_logical_topology = _normalize_logical_topology_payload(
        logical_topology_json
    )
    resolved_manifest_ref, manifest_tensor_name = _resolve_manifest_tensor_name(
        serving_manifest_ref,
        helper_name="PURE_TRANSFORM publication bundle",
    )

    tensor_schema_hash = compute_serving_tensor_schema_hash(
        canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    canonical_tensor_count = count_canonical_serving_tensors(
        canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    resolved_representation_contract_hash = (
        build_intent.representation_contract_hash
        or (
            compute_pure_transform_representation_contract_hash(
                source_artifact=source_artifact,
                serving_artifact=canonical_index,
                logical_topology_json=logical_topology_json,
                manifest_tensor_name=manifest_tensor_name,
            )
            if source_artifact is not None
            else None
        )
    )
    if not resolved_representation_contract_hash:
        raise ArtifactError(
            "PURE_TRANSFORM publication bundle requires representation_contract_hash or source_artifact for auto-derivation",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=build_intent,
        representation_contract_hash=resolved_representation_contract_hash,
        tensor_schema_hash=tensor_schema_hash,
        canonical_tensor_count=canonical_tensor_count,
        serving_manifest_ref=resolved_manifest_ref,
        logical_topology_json=(
            json.dumps(
                normalized_logical_topology, sort_keys=True, separators=(",", ":")
            )
            if normalized_logical_topology is not None
            else None
        ),
    )
    serving_manifest_bytes = _pad_manifest_carrier_bytes(manifest.to_bytes())

    representation_publish_contract = RepresentationPublishContract(
        serving_artifact_id=_artifact_id_from_input(serving_artifact),
        serving_manifest_ref=resolved_manifest_ref,
        representation_contract_hash=manifest.representation_contract_hash,
        serving_build_digest=manifest.serving_build_digest,
    )
    closeout_contract = AssemblyCloseoutContract(
        kind="representation_publish",
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        serving_artifact_id=representation_publish_contract.serving_artifact_id,
        serving_manifest_ref=representation_publish_contract.serving_manifest_ref,
        representation_publish_contract=representation_publish_contract,
    )
    resolved_structural_view_ids = tuple(
        str(view_id).strip() for view_id in structural_view_ids if str(view_id).strip()
    ) or _structural_view_ids_from_source_artifact(source_artifact)
    return RepresentationPublishSpec(
        serving_artifact_id=representation_publish_contract.serving_artifact_id,
        serving_manifest_ref=resolved_manifest_ref,
        serving_manifest=manifest,
        serving_manifest_bytes=serving_manifest_bytes,
        source_artifact_ref=build_intent.source_artifact_ref,
        contract_family=_normalize_contract_family(contract_family),
        structural_view_ids=resolved_structural_view_ids,
        representation_publish_contract=representation_publish_contract,
        closeout_contract=closeout_contract,
        layout_id=None if layout_id is None else str(layout_id),
        requirements=requirements,
        readiness_policy=readiness_policy,
    )


def build_serving_publication_bundle(
    *,
    build_intent: ServingBuildIntent,
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    serving_artifact: RegisteredArtifact
    | ArtifactDescriptor
    | "StoreArtifactDescriptor"
    | str,
    canonical_index: CanonicalIndex,
    representation_contract_hash: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
) -> RepresentationPublishSpec:
    normalized_logical_topology = _normalize_logical_topology_payload(
        logical_topology_json
    )
    resolved_manifest_ref, manifest_tensor_name = _resolve_manifest_tensor_name(
        serving_manifest_ref,
        helper_name="serving publication bundle",
    )

    tensor_schema_hash = compute_serving_tensor_schema_hash(
        canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    canonical_tensor_count = count_canonical_serving_tensors(
        canonical_index,
        manifest_tensor_name=manifest_tensor_name,
    )
    resolved_representation_contract_hash = (
        _resolve_explicit_representation_contract_hash(
            build_intent=build_intent,
            representation_contract_hash=representation_contract_hash,
            helper_name="serving publication bundle",
        )
    )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=build_intent,
        representation_contract_hash=resolved_representation_contract_hash,
        tensor_schema_hash=tensor_schema_hash,
        canonical_tensor_count=canonical_tensor_count,
        serving_manifest_ref=resolved_manifest_ref,
        logical_topology_json=(
            json.dumps(
                normalized_logical_topology, sort_keys=True, separators=(",", ":")
            )
            if normalized_logical_topology is not None
            else None
        ),
    )
    serving_manifest_bytes = _pad_manifest_carrier_bytes(manifest.to_bytes())

    representation_publish_contract = RepresentationPublishContract(
        serving_artifact_id=_artifact_id_from_input(serving_artifact),
        serving_manifest_ref=resolved_manifest_ref,
        representation_contract_hash=manifest.representation_contract_hash,
        serving_build_digest=manifest.serving_build_digest,
    )
    closeout_contract = AssemblyCloseoutContract(
        kind="representation_publish",
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        serving_artifact_id=representation_publish_contract.serving_artifact_id,
        serving_manifest_ref=representation_publish_contract.serving_manifest_ref,
        representation_publish_contract=representation_publish_contract,
    )
    resolved_structural_view_ids = tuple(
        str(view_id).strip() for view_id in structural_view_ids if str(view_id).strip()
    ) or _structural_view_ids_from_source_artifact(source_artifact)
    return RepresentationPublishSpec(
        serving_artifact_id=representation_publish_contract.serving_artifact_id,
        serving_manifest_ref=resolved_manifest_ref,
        serving_manifest=manifest,
        serving_manifest_bytes=serving_manifest_bytes,
        source_artifact_ref=build_intent.source_artifact_ref,
        contract_family=_normalize_contract_family(contract_family),
        structural_view_ids=resolved_structural_view_ids,
        representation_publish_contract=representation_publish_contract,
        closeout_contract=closeout_contract,
        layout_id=None if layout_id is None else str(layout_id),
        requirements=requirements,
        readiness_policy=readiness_policy,
    )


def build_pure_transform_publication_bundle_from_registered_artifact(
    *,
    build_intent: ServingBuildIntent,
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    serving_artifact: RegisteredArtifact,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
) -> RepresentationPublishSpec:
    return build_pure_transform_publication_bundle(
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=contract_family,
        serving_artifact=serving_artifact,
        canonical_index=_canonical_index_from_input(serving_artifact),
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=structural_view_ids,
    )


def build_serving_publication_bundle_from_registered_artifact(
    *,
    build_intent: ServingBuildIntent,
    source_artifact: RegisteredArtifact | CanonicalIndex | object | None = None,
    contract_family: AssemblyContractFamily | str | None = None,
    serving_artifact: RegisteredArtifact,
    representation_contract_hash: str | None = None,
    source_version_key: str | None = None,
    serving_version_key: str | None = None,
    logical_topology_json: str | None = None,
    serving_manifest_ref: str | None = None,
    layout_id: str | None = None,
    requirements: AssemblyRequirementSetRef | None = None,
    readiness_policy: AssemblyReadinessPolicy | None = None,
    structural_view_ids: tuple[str, ...] = (),
) -> RepresentationPublishSpec:
    return build_serving_publication_bundle(
        build_intent=build_intent,
        source_artifact=source_artifact,
        contract_family=contract_family,
        serving_artifact=serving_artifact,
        canonical_index=_canonical_index_from_input(serving_artifact),
        representation_contract_hash=representation_contract_hash,
        source_version_key=source_version_key,
        serving_version_key=serving_version_key,
        logical_topology_json=logical_topology_json,
        serving_manifest_ref=serving_manifest_ref,
        layout_id=layout_id,
        requirements=requirements,
        readiness_policy=readiness_policy,
        structural_view_ids=structural_view_ids,
    )


__all__ = [
    "PreparedServingRegistration",
    "PreparedPureTransformServingRegistration",
    "PureTransformPublicationSpec",
    "PureTransformPublicationBundle",
    "RegisteredServingPublication",
    "RegisteredPureTransformPublication",
    "ServingPublicationBundle",
    "PURE_TRANSFORM_SERVING_ARG_ABI_VERSION",
    "PURE_TRANSFORM_SERVING_ARG_ADAPTER_VERSION",
    "PURE_TRANSFORM_SERVING_ARG_BUILD_PIPELINE_VERSION",
    "PURE_TRANSFORM_SERVING_ARG_CONTRACT_FAMILY",
    "PURE_TRANSFORM_SERVING_ARG_ENABLE",
    "PURE_TRANSFORM_SERVING_ARG_FRAMEWORK_NAME",
    "PURE_TRANSFORM_SERVING_ARG_LOGICAL_TOPOLOGY_JSON",
    "PURE_TRANSFORM_SERVING_ARG_MANIFEST_REF",
    "PURE_TRANSFORM_SERVING_ARG_REPRESENTATION_CONTRACT_HASH",
    "PURE_TRANSFORM_SERVING_ARG_SERVING_VERSION_KEY",
    "PURE_TRANSFORM_SERVING_ARG_SOURCE_VERSION_KEY",
    "build_serving_publication_bundle",
    "build_serving_publication_bundle_from_registered_artifact",
    "build_pure_transform_serving_args",
    "build_pure_transform_publication_spec",
    "build_pure_transform_transform_spec",
    "build_pure_transform_publication_bundle",
    "build_pure_transform_publication_bundle_from_registered_artifact",
    "compute_serving_tensor_schema_hash",
    "compute_pure_transform_representation_contract_hash",
    "count_canonical_serving_tensors",
    "prepare_serving_registration",
    "prepare_pure_transform_serving_registration",
]
