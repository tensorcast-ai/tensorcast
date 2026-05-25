#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from enum import Enum, IntFlag
from typing import Iterable, Literal, Mapping, Union, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tensorcast.common.identity import ArtifactIdKind
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.proto.publication.v1 import publication_pb2

# ---------------------------------------------------------------------------
# Canonical typed models used across the Python SDK in place of raw dicts
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    """Daemon server runtime configuration (client-facing subset)."""

    model_config = ConfigDict(frozen=True)

    # Canonical transfer slice size
    tx_slice_bytes: int
    mem_pool_size: int
    artifact_chunk_bytes: int = 0
    local_handle_socket_path: str = ""
    cpu_shared_memory_enabled: bool = True
    source_bound_capability_flags: int = 0
    source_bound_contract_version: int = 0

    def has_source_bound_capability(self, capability: "SourceBoundCapability") -> bool:
        return bool(int(self.source_bound_capability_flags) & int(capability))


class SourceBoundCapability(IntFlag):
    FIRST_CLASS_COLLECTIVE_INGRESS = int(
        store_daemon_pb2.SOURCE_BOUND_CAPABILITY_FLAG_FIRST_CLASS_COLLECTIVE_INGRESS
    )
    TYPED_EXECUTION_DIAGNOSTICS = int(
        store_daemon_pb2.SOURCE_BOUND_CAPABILITY_FLAG_TYPED_EXECUTION_DIAGNOSTICS
    )
    SINGLE_MINT_BINDING_CLOSEOUT = int(
        store_daemon_pb2.SOURCE_BOUND_CAPABILITY_FLAG_SINGLE_MINT_BINDING_CLOSEOUT
    )


class BindingValueVerificationState(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    LOCAL_ONLY = "local_only"

    @classmethod
    def from_proto(cls, value: int) -> "BindingValueVerificationState | None":
        if value == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_PENDING:
            return cls.PENDING
        if value == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_VERIFIED:
            return cls.VERIFIED
        if value == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_FAILED:
            return cls.FAILED
        if value == store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY:
            return cls.LOCAL_ONLY
        return None

    def to_proto(self) -> int:
        if self is BindingValueVerificationState.PENDING:
            return store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_PENDING
        if self is BindingValueVerificationState.VERIFIED:
            return store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_VERIFIED
        if self is BindingValueVerificationState.FAILED:
            return store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_FAILED
        return store_daemon_pb2.BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY


class BindingPromotionStatusState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @classmethod
    def from_proto(cls, value: int) -> "BindingPromotionStatusState | None":
        if value == store_daemon_pb2.BINDING_PROMOTION_JOB_STATE_PENDING:
            return cls.PENDING
        if value == store_daemon_pb2.BINDING_PROMOTION_JOB_STATE_RUNNING:
            return cls.RUNNING
        if value == store_daemon_pb2.BINDING_PROMOTION_JOB_STATE_SUCCEEDED:
            return cls.SUCCEEDED
        if value == store_daemon_pb2.BINDING_PROMOTION_JOB_STATE_FAILED:
            return cls.FAILED
        if value == store_daemon_pb2.BINDING_PROMOTION_JOB_STATE_CANCELED:
            return cls.CANCELED
        return None


class MountedSourceFormatKind(str, Enum):
    PARTITIONED = "partitioned"
    SAFETENSORS = "safetensors"

    @classmethod
    def from_proto(cls, value: int) -> "MountedSourceFormatKind | None":
        if value == store_daemon_pb2.DISK_SOURCE_FORMAT_KIND_PARTITIONED:
            return cls.PARTITIONED
        if value == store_daemon_pb2.DISK_SOURCE_FORMAT_KIND_SAFETENSORS:
            return cls.SAFETENSORS
        return None

    def to_proto(self) -> store_daemon_pb2.DiskSourceFormatKind:
        if self is MountedSourceFormatKind.PARTITIONED:
            return store_daemon_pb2.DISK_SOURCE_FORMAT_KIND_PARTITIONED
        return store_daemon_pb2.DISK_SOURCE_FORMAT_KIND_SAFETENSORS


class MountedSourceMetadataCapability(str, Enum):
    TENSOR_AWARE = "tensor_aware"
    BYTE_ONLY = "byte_only"

    @classmethod
    def from_proto(cls, value: int) -> "MountedSourceMetadataCapability | None":
        if value == store_daemon_pb2.DISK_METADATA_CAPABILITY_TENSOR_AWARE:
            return cls.TENSOR_AWARE
        if value == store_daemon_pb2.DISK_METADATA_CAPABILITY_BYTE_ONLY:
            return cls.BYTE_ONLY
        return None

    def to_proto(self) -> store_daemon_pb2.DiskMetadataCapability:
        if self is MountedSourceMetadataCapability.BYTE_ONLY:
            return store_daemon_pb2.DISK_METADATA_CAPABILITY_BYTE_ONLY
        return store_daemon_pb2.DISK_METADATA_CAPABILITY_TENSOR_AWARE


class MountedSourceResolutionStrategy(str, Enum):
    ATTESTED_ONLY = "attested_only"
    ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT = "attested_with_trusted_descriptor_hint"

    @classmethod
    def from_proto(cls, value: int) -> "MountedSourceResolutionStrategy | None":
        if value == store_daemon_pb2.DISK_RESOLUTION_STRATEGY_ATTESTED_ONLY:
            return cls.ATTESTED_ONLY
        if (
            value
            == store_daemon_pb2.DISK_RESOLUTION_STRATEGY_ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT
        ):
            return cls.ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT
        return None

    def to_proto(self) -> store_daemon_pb2.DiskResolutionStrategy:
        if (
            self
            is MountedSourceResolutionStrategy.ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT
        ):
            return store_daemon_pb2.DISK_RESOLUTION_STRATEGY_ATTESTED_WITH_TRUSTED_DESCRIPTOR_HINT
        return store_daemon_pb2.DISK_RESOLUTION_STRATEGY_ATTESTED_ONLY


class MountedSourceValidationMode(str, Enum):
    VALIDATE_BEFORE_READ = "validate_before_read"

    @classmethod
    def from_proto(cls, value: int) -> "MountedSourceValidationMode | None":
        if value == store_daemon_pb2.DISK_VALIDATION_MODE_VALIDATE_BEFORE_READ:
            return cls.VALIDATE_BEFORE_READ
        return None

    def to_proto(self) -> store_daemon_pb2.DiskValidationMode:
        return store_daemon_pb2.DISK_VALIDATION_MODE_VALIDATE_BEFORE_READ


class CollectivePolicy(str, Enum):
    REQUIRE_COLLECTIVE = "require_collective"
    COLLECTIVE_FIRST = "collective_first"
    DISABLE_COLLECTIVE = "disable_collective"


class CollectiveFailureClass(str, Enum):
    NOT_ELIGIBLE = "not_eligible"
    EXECUTION_FAILED = "execution_failed"


class HashBackend(str, Enum):
    NONE = "none"
    GPU = "gpu"
    D2H_CPU = "d2h_cpu"
    CPU = "cpu"


class HashLocation(str, Enum):
    NONE = "none"
    SEAL = "seal"
    BINDING_CLOSEOUT = "binding_closeout"


class IdentityMintStrategy(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    SEAL_MINT = "seal_mint"
    SEAL_REUSE = "seal_reuse"
    CLOSEOUT_MINT = "closeout_mint"


_COLLECTIVE_POLICY_TO_PROTO: dict[
    CollectivePolicy, store_daemon_pb2.CollectivePolicy
] = {
    CollectivePolicy.REQUIRE_COLLECTIVE: store_daemon_pb2.COLLECTIVE_POLICY_REQUIRE_COLLECTIVE,
    CollectivePolicy.COLLECTIVE_FIRST: store_daemon_pb2.COLLECTIVE_POLICY_COLLECTIVE_FIRST,
    CollectivePolicy.DISABLE_COLLECTIVE: store_daemon_pb2.COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
}
_COLLECTIVE_POLICY_FROM_PROTO: dict[int, CollectivePolicy] = {
    value: key for key, value in _COLLECTIVE_POLICY_TO_PROTO.items()
}
_COLLECTIVE_FAILURE_CLASS_TO_PROTO: dict[
    CollectiveFailureClass, store_daemon_pb2.CollectiveFailureClass
] = {
    CollectiveFailureClass.NOT_ELIGIBLE: store_daemon_pb2.COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE,
    CollectiveFailureClass.EXECUTION_FAILED: store_daemon_pb2.COLLECTIVE_FAILURE_CLASS_EXECUTION_FAILED,
}
_COLLECTIVE_FAILURE_CLASS_FROM_PROTO: dict[int, CollectiveFailureClass] = {
    value: key for key, value in _COLLECTIVE_FAILURE_CLASS_TO_PROTO.items()
}
_HASH_BACKEND_TO_PROTO: dict[HashBackend, store_daemon_pb2.HashBackend] = {
    HashBackend.NONE: store_daemon_pb2.HASH_BACKEND_NONE,
    HashBackend.GPU: store_daemon_pb2.HASH_BACKEND_GPU,
    HashBackend.D2H_CPU: store_daemon_pb2.HASH_BACKEND_D2H_CPU,
    HashBackend.CPU: store_daemon_pb2.HASH_BACKEND_CPU,
}
_HASH_BACKEND_FROM_PROTO: dict[int, HashBackend] = {
    value: key for key, value in _HASH_BACKEND_TO_PROTO.items()
}
_HASH_LOCATION_TO_PROTO: dict[HashLocation, store_daemon_pb2.HashLocation] = {
    HashLocation.NONE: store_daemon_pb2.HASH_LOCATION_NONE,
    HashLocation.SEAL: store_daemon_pb2.HASH_LOCATION_SEAL,
    HashLocation.BINDING_CLOSEOUT: store_daemon_pb2.HASH_LOCATION_BINDING_CLOSEOUT,
}
_HASH_LOCATION_FROM_PROTO: dict[int, HashLocation] = {
    value: key for key, value in _HASH_LOCATION_TO_PROTO.items()
}
_IDENTITY_MINT_STRATEGY_TO_PROTO: dict[
    IdentityMintStrategy, store_daemon_pb2.IdentityMintStrategy
] = {
    IdentityMintStrategy.NOT_APPLICABLE: store_daemon_pb2.IDENTITY_MINT_STRATEGY_NOT_APPLICABLE,
    IdentityMintStrategy.SEAL_MINT: store_daemon_pb2.IDENTITY_MINT_STRATEGY_SEAL_MINT,
    IdentityMintStrategy.SEAL_REUSE: store_daemon_pb2.IDENTITY_MINT_STRATEGY_SEAL_REUSE,
    IdentityMintStrategy.CLOSEOUT_MINT: store_daemon_pb2.IDENTITY_MINT_STRATEGY_CLOSEOUT_MINT,
}
_IDENTITY_MINT_STRATEGY_FROM_PROTO: dict[int, IdentityMintStrategy] = {
    value: key for key, value in _IDENTITY_MINT_STRATEGY_TO_PROTO.items()
}


class ExecutionDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    collective_requested: bool = False
    collective_acknowledged: bool = False
    collective_used: bool = False
    collective_policy: CollectivePolicy = CollectivePolicy.DISABLE_COLLECTIVE
    collective_failure_class: CollectiveFailureClass | None = None
    dominant_executor: str | None = None
    direct_write_supported: bool = False
    fallback_bytes: int = 0
    residual_bytes: int = 0
    actual_collective_committed_bytes: int = 0
    actual_local_typed_bytes: int = 0
    actual_generic_backend_bytes: int = 0
    collective_unique_source_bytes: int = 0
    collective_peer_transfer_bytes: int = 0
    collective_peak_temporary_bytes: int = 0
    collective_batch_count: int = 0
    collective_dedup_saving_bytes: int = 0
    collective_skip_reason: str | None = None
    hash_rounds: int = 0
    hash_backend: HashBackend = HashBackend.NONE
    hash_bytes: int = 0
    hash_wall_time_ms: int = 0
    hash_identity_forming: bool = False
    hash_location: HashLocation = HashLocation.NONE
    identity_mint_strategy: IdentityMintStrategy = IdentityMintStrategy.NOT_APPLICABLE

    def to_proto(self) -> store_daemon_pb2.ExecutionDiagnostics:
        proto = store_daemon_pb2.ExecutionDiagnostics(
            collective_requested=bool(self.collective_requested),
            collective_acknowledged=bool(self.collective_acknowledged),
            collective_used=bool(self.collective_used),
            collective_policy=_COLLECTIVE_POLICY_TO_PROTO[self.collective_policy],
            dominant_executor=str(self.dominant_executor or ""),
            direct_write_supported=bool(self.direct_write_supported),
            fallback_bytes=int(self.fallback_bytes),
            residual_bytes=int(self.residual_bytes),
            actual_collective_committed_bytes=int(
                self.actual_collective_committed_bytes
            ),
            actual_local_typed_bytes=int(self.actual_local_typed_bytes),
            actual_generic_backend_bytes=int(self.actual_generic_backend_bytes),
            collective_unique_source_bytes=int(self.collective_unique_source_bytes),
            collective_peer_transfer_bytes=int(self.collective_peer_transfer_bytes),
            collective_peak_temporary_bytes=int(self.collective_peak_temporary_bytes),
            collective_batch_count=int(self.collective_batch_count),
            collective_dedup_saving_bytes=int(self.collective_dedup_saving_bytes),
            collective_skip_reason=str(self.collective_skip_reason or ""),
            hash_rounds=int(self.hash_rounds),
            hash_backend=_HASH_BACKEND_TO_PROTO[self.hash_backend],
            hash_bytes=int(self.hash_bytes),
            hash_wall_time_ms=int(self.hash_wall_time_ms),
            hash_identity_forming=bool(self.hash_identity_forming),
            hash_location=_HASH_LOCATION_TO_PROTO[self.hash_location],
            identity_mint_strategy=_IDENTITY_MINT_STRATEGY_TO_PROTO[
                self.identity_mint_strategy
            ],
        )
        if self.collective_failure_class is not None:
            proto.collective_failure_class = _COLLECTIVE_FAILURE_CLASS_TO_PROTO[
                self.collective_failure_class
            ]
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.ExecutionDiagnostics,
    ) -> "ExecutionDiagnostics":
        collective_failure_class_value = int(proto.collective_failure_class)
        return cls(
            collective_requested=bool(proto.collective_requested),
            collective_acknowledged=bool(proto.collective_acknowledged),
            collective_used=bool(proto.collective_used),
            collective_policy=_COLLECTIVE_POLICY_FROM_PROTO.get(
                int(proto.collective_policy),
                CollectivePolicy.DISABLE_COLLECTIVE,
            ),
            collective_failure_class=(
                _COLLECTIVE_FAILURE_CLASS_FROM_PROTO.get(collective_failure_class_value)
                if collective_failure_class_value
                != int(store_daemon_pb2.COLLECTIVE_FAILURE_CLASS_UNSPECIFIED)
                else None
            ),
            dominant_executor=str(proto.dominant_executor or "") or None,
            direct_write_supported=bool(proto.direct_write_supported),
            fallback_bytes=int(proto.fallback_bytes),
            residual_bytes=int(proto.residual_bytes),
            actual_collective_committed_bytes=int(
                getattr(proto, "actual_collective_committed_bytes", 0)
            ),
            actual_local_typed_bytes=int(getattr(proto, "actual_local_typed_bytes", 0)),
            actual_generic_backend_bytes=int(
                getattr(proto, "actual_generic_backend_bytes", 0)
            ),
            collective_unique_source_bytes=int(
                getattr(proto, "collective_unique_source_bytes", 0)
            ),
            collective_peer_transfer_bytes=int(
                getattr(proto, "collective_peer_transfer_bytes", 0)
            ),
            collective_peak_temporary_bytes=int(
                getattr(proto, "collective_peak_temporary_bytes", 0)
            ),
            collective_batch_count=int(getattr(proto, "collective_batch_count", 0)),
            collective_dedup_saving_bytes=int(
                getattr(proto, "collective_dedup_saving_bytes", 0)
            ),
            collective_skip_reason=str(
                getattr(proto, "collective_skip_reason", "") or ""
            )
            or None,
            hash_rounds=int(proto.hash_rounds),
            hash_backend=_HASH_BACKEND_FROM_PROTO.get(
                int(getattr(proto, "hash_backend", 0) or 0),
                HashBackend.NONE,
            ),
            hash_bytes=int(getattr(proto, "hash_bytes", 0) or 0),
            hash_wall_time_ms=int(getattr(proto, "hash_wall_time_ms", 0) or 0),
            hash_identity_forming=bool(getattr(proto, "hash_identity_forming", False)),
            hash_location=_HASH_LOCATION_FROM_PROTO.get(
                int(proto.hash_location),
                HashLocation.NONE,
            ),
            identity_mint_strategy=_IDENTITY_MINT_STRATEGY_FROM_PROTO.get(
                int(proto.identity_mint_strategy),
                IdentityMintStrategy.NOT_APPLICABLE,
            ),
        )


class SourceBoundPlanDiagnostics(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_plan_kind: str | None = None
    planned_collective_candidate_bytes: int = 0
    planned_collective_admitted_bytes: int = 0
    planned_local_typed_bytes: int = 0
    planned_non_admitted_typed_bytes: int = 0
    planned_generic_residual_bytes: int = 0
    collective_lowered_bytes: int = 0
    planner_reject_reason_buckets: dict[str, int] = Field(default_factory=dict)
    planner_version: str | None = None
    plan_hash: str | None = None
    estimated_collective_peak_temporary_bytes: int = 0
    estimated_collective_batch_bytes: int = 0
    estimated_collective_dedup_saving_bytes: int = 0

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.SourceBoundPlanDiagnostics,
    ) -> "SourceBoundPlanDiagnostics":
        return cls(
            execution_plan_kind=str(getattr(proto, "execution_plan_kind", "") or "")
            or None,
            planned_collective_candidate_bytes=int(
                getattr(proto, "planned_collective_candidate_bytes", 0)
            ),
            planned_collective_admitted_bytes=int(
                getattr(proto, "planned_collective_admitted_bytes", 0)
            ),
            planned_local_typed_bytes=int(
                getattr(proto, "planned_local_typed_bytes", 0)
            ),
            planned_non_admitted_typed_bytes=int(
                getattr(proto, "planned_non_admitted_typed_bytes", 0)
            ),
            planned_generic_residual_bytes=int(
                getattr(proto, "planned_generic_residual_bytes", 0)
            ),
            collective_lowered_bytes=int(getattr(proto, "collective_lowered_bytes", 0)),
            planner_reject_reason_buckets={
                str(key): int(value)
                for key, value in dict(
                    getattr(proto, "planner_reject_reason_buckets", {}) or {}
                ).items()
            },
            planner_version=str(getattr(proto, "planner_version", "") or "") or None,
            plan_hash=str(getattr(proto, "plan_hash", "") or "") or None,
            estimated_collective_peak_temporary_bytes=int(
                getattr(proto, "estimated_collective_peak_temporary_bytes", 0)
            ),
            estimated_collective_batch_bytes=int(
                getattr(proto, "estimated_collective_batch_bytes", 0)
            ),
            estimated_collective_dedup_saving_bytes=int(
                getattr(proto, "estimated_collective_dedup_saving_bytes", 0)
            ),
        )


# ----------------------------- Handshake models ----------------------------


class CoalescedHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["coalesced"] = "coalesced"
    daemon_ipc_handle: bytes


# CPU handshake variants removed


class LeaseHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["lease"] = "lease"


class StableDramHandshake(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["dram_stable"] = "dram_stable"
    staging_cuda_ipc_handle: bytes = b""
    publish_cpu_memfd_size_bytes: int = 0
    publish_cpu_memfd_offset_bytes: int = 0
    publish_cpu_memfd_lease_token: bytes = b""


Handshake = Union[
    CoalescedHandshake,
    LeaseHandshake,
    StableDramHandshake,
]


class BeginRegisterArtifactResult(BaseModel):
    """Result for BeginRegisterArtifact, replacing mixed dict outputs.

    Always includes a concrete handshake variant to avoid optional fields.
    """

    model_config = ConfigDict(frozen=True)

    registration_id: str
    device_id: int
    total_size: int
    handshake: Handshake


class ArtifactDescriptor(BaseModel):
    """Artifact descriptor (RFC-0007) returned at commit time."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    index_multihash: str | None = None
    data_multihash: str | None = None
    schema_version: str | None = None
    encoding: str | None = None
    total_size: int
    id_kind: ArtifactIdKind = ArtifactIdKind.MI2

    @field_validator(
        "index_multihash",
        "data_multihash",
        "schema_version",
        "encoding",
        mode="before",
    )
    @classmethod
    def _empty_str_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("id_kind", mode="before")
    @classmethod
    def _coerce_id_kind(cls, value: object) -> ArtifactIdKind:
        if isinstance(value, ArtifactIdKind):
            return value
        if isinstance(value, str):
            upper = value.upper()
            if upper == "MI2":
                return ArtifactIdKind.MI2
            if upper == "CGID":
                return ArtifactIdKind.CGID
            if upper == "MSA1":
                return ArtifactIdKind.MSA1
        if isinstance(value, int):
            if value == 1:
                return ArtifactIdKind.MI2
            if value == 2:
                return ArtifactIdKind.CGID
        raise ValueError(f"Unsupported artifact id kind: {value!r}")


class CanonicalRange(BaseModel):
    """Canonical byte range populated during view registration."""

    model_config = ConfigDict(frozen=True)

    offset: int
    length: int


LocalStableTierStatus = Literal["ready", "degraded", "skipped"]
ViewRegistrationKind = Literal["canonical", "piece"]


class LocalStableTierResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: LocalStableTierStatus
    message: str | None = None


class CommitResult(BaseModel):
    """Commit result with descriptor and idempotency flag."""

    model_config = ConfigDict(frozen=True)

    descriptor: ArtifactDescriptor
    existed: bool = False
    view_id: str | None = None
    view_index_json: bytes | None = None
    view_data_hash: str | None = None
    canonical_ranges: tuple[CanonicalRange, ...] = ()
    registration_kind: ViewRegistrationKind = "canonical"
    local_stable_tier: LocalStableTierResult | None = None


class SealAssemblyResult(BaseModel):
    """Seal outcome with the bound MI2 descriptor."""

    model_config = ConfigDict(frozen=True)

    sealed_artifact_id: str
    descriptor: ArtifactDescriptor
    already_sealed: bool = False


ContributionKind = Literal["piece_partial", "canonical_full"]

_CONTRIBUTION_KIND_TO_PROTO: dict[ContributionKind, int] = {
    "piece_partial": int(store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL),
    "canonical_full": int(store_daemon_pb2.BINDING_CONTRIBUTION_KIND_CANONICAL_FULL),
}

_CONTRIBUTION_KIND_FROM_PROTO: dict[int, ContributionKind] = {
    int(store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL): "piece_partial",
    int(store_daemon_pb2.BINDING_CONTRIBUTION_KIND_CANONICAL_FULL): "canonical_full",
}

AssemblyTargetKind = Literal["structural_view", "canonical_layout"]
AssemblyContractFamily = Literal["pp", "ep", "canonical_full"]
AssemblyContributorLivenessMode = Literal[
    "require_live_until_cut",
    "allow_durable_occupancy",
]
AssemblyCloseoutKind = Literal[
    "source_publish_only",
    "representation_publish",
    "rollout_gated_publish",
]

_ASSEMBLY_TARGET_KIND_TO_PROTO: dict[
    AssemblyTargetKind, store_daemon_pb2.AssemblyTargetKind
] = {
    "structural_view": store_daemon_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW,
    "canonical_layout": store_daemon_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT,
}
_ASSEMBLY_TARGET_KIND_FROM_PROTO: dict[int, AssemblyTargetKind] = {
    int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW): "structural_view",
    int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT): "canonical_layout",
}
_ASSEMBLY_LIVENESS_MODE_TO_PROTO: dict[
    AssemblyContributorLivenessMode,
    store_daemon_pb2.AssemblyContributorLivenessMode,
] = {
    "require_live_until_cut": (
        store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ),
    "allow_durable_occupancy": (
        store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY
    ),
}
_ASSEMBLY_LIVENESS_MODE_FROM_PROTO: dict[int, AssemblyContributorLivenessMode] = {
    int(
        store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ): "require_live_until_cut",
    int(
        store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY
    ): "allow_durable_occupancy",
}
_ASSEMBLY_CLOSEOUT_KIND_TO_PROTO: dict[
    AssemblyCloseoutKind, store_daemon_pb2.AssemblyCloseoutKind
] = {
    "source_publish_only": store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY,
    "representation_publish": store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH,
    "rollout_gated_publish": store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH,
}
_ASSEMBLY_CLOSEOUT_KIND_FROM_PROTO: dict[int, AssemblyCloseoutKind] = {
    int(store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY): (
        "source_publish_only"
    ),
    int(store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH): (
        "representation_publish"
    ),
    int(store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH): (
        "rollout_gated_publish"
    ),
}
_ASSEMBLY_CANONICAL_SLOT_ID = "__canonical_full__"
_ASSEMBLY_PP_PIECE_COVERAGE_CONTRACT = "pp_structural_view"
_ASSEMBLY_EP_PIECE_COVERAGE_CONTRACT = "ep_structural_view"
_ASSEMBLY_CANONICAL_COVERAGE_CONTRACT = "canonical_full"
SERVING_MANIFEST_TENSOR_NAME = "__tensorcast_meta__.manifest_json"
SERVING_BUILD_DIGEST_VERSION = "tensorcast.serving_build_digest.v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _multibase_multihash_sha256(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("SHA256 digest must be 32 bytes")
    multihash = b"\x12\x20" + digest
    encoded = base64.b32encode(multihash).decode("ascii").lower().rstrip("=")
    return f"b{encoded}"


def _hash_payload_to_multihash(payload: object) -> str:
    return _multibase_multihash_sha256(
        hashlib.sha256(_canonical_json_bytes(payload)).digest()
    )


def build_serving_manifest_ref(
    tensor_name: str = SERVING_MANIFEST_TENSOR_NAME,
) -> str:
    name = str(tensor_name).strip()
    if not name:
        raise ValueError("tensor_name must not be empty")
    return f"tensor:{name}"


def parse_serving_manifest_ref(ref: str) -> str:
    value = str(ref).strip()
    prefix = "tensor:"
    if not value.startswith(prefix):
        raise ValueError("serving_manifest_ref must use the tensor:<name> carrier")
    tensor_name = value[len(prefix) :].strip()
    if not tensor_name:
        raise ValueError("serving_manifest_ref tensor carrier requires a tensor name")
    return tensor_name


class BuilderMode(str, Enum):
    PURE_TRANSFORM = "pure_transform"
    BINDING_FINALIZE = "binding_finalize"


class FinalizeClass(str, Enum):
    RUNTIME_ONLY = "runtime_only"
    REPRESENTATION_CHANGING = "representation_changing"
    UNKNOWN_BLOCKED = "unknown_blocked"


class ServingSupportLevel(str, Enum):
    BLOCKED = "blocked"
    SOURCE_BIND_BOOTSTRAP_ONLY = "source_bind_bootstrap_only"
    BUILDER_PUBLICATION_READY = "builder_publication_ready"
    RUNTIME_BIND_SWAP_READY = "runtime_bind_swap_ready"


_PUBLICATION_BUILDER_MODE_TO_PROTO: dict[BuilderMode, publication_pb2.BuilderMode] = {
    BuilderMode.PURE_TRANSFORM: publication_pb2.BUILDER_MODE_PURE_TRANSFORM,
    BuilderMode.BINDING_FINALIZE: publication_pb2.BUILDER_MODE_BINDING_FINALIZE,
}
_PUBLICATION_BUILDER_MODE_FROM_PROTO: dict[int, BuilderMode] = {
    int(publication_pb2.BUILDER_MODE_PURE_TRANSFORM): BuilderMode.PURE_TRANSFORM,
    int(publication_pb2.BUILDER_MODE_BINDING_FINALIZE): BuilderMode.BINDING_FINALIZE,
}
_PUBLICATION_FINALIZE_CLASS_TO_PROTO: dict[
    FinalizeClass, publication_pb2.FinalizeClass
] = {
    FinalizeClass.RUNTIME_ONLY: publication_pb2.FINALIZE_CLASS_RUNTIME_ONLY,
    FinalizeClass.REPRESENTATION_CHANGING: publication_pb2.FINALIZE_CLASS_REPRESENTATION_CHANGING,
    FinalizeClass.UNKNOWN_BLOCKED: publication_pb2.FINALIZE_CLASS_UNKNOWN_BLOCKED,
}
_PUBLICATION_FINALIZE_CLASS_FROM_PROTO: dict[int, FinalizeClass] = {
    int(publication_pb2.FINALIZE_CLASS_RUNTIME_ONLY): FinalizeClass.RUNTIME_ONLY,
    int(publication_pb2.FINALIZE_CLASS_REPRESENTATION_CHANGING): (
        FinalizeClass.REPRESENTATION_CHANGING
    ),
    int(publication_pb2.FINALIZE_CLASS_UNKNOWN_BLOCKED): (
        FinalizeClass.UNKNOWN_BLOCKED
    ),
}
_PUBLICATION_SERVING_SUPPORT_LEVEL_TO_PROTO: dict[
    ServingSupportLevel, publication_pb2.ServingSupportLevel
] = {
    ServingSupportLevel.BLOCKED: publication_pb2.SERVING_SUPPORT_LEVEL_BLOCKED,
    ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: (
        publication_pb2.SERVING_SUPPORT_LEVEL_SOURCE_BIND_BOOTSTRAP_ONLY
    ),
    ServingSupportLevel.BUILDER_PUBLICATION_READY: (
        publication_pb2.SERVING_SUPPORT_LEVEL_BUILDER_PUBLICATION_READY
    ),
    ServingSupportLevel.RUNTIME_BIND_SWAP_READY: (
        publication_pb2.SERVING_SUPPORT_LEVEL_RUNTIME_BIND_SWAP_READY
    ),
}
_PUBLICATION_SERVING_SUPPORT_LEVEL_FROM_PROTO: dict[int, ServingSupportLevel] = {
    int(publication_pb2.SERVING_SUPPORT_LEVEL_BLOCKED): ServingSupportLevel.BLOCKED,
    int(publication_pb2.SERVING_SUPPORT_LEVEL_SOURCE_BIND_BOOTSTRAP_ONLY): (
        ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY
    ),
    int(publication_pb2.SERVING_SUPPORT_LEVEL_BUILDER_PUBLICATION_READY): (
        ServingSupportLevel.BUILDER_PUBLICATION_READY
    ),
    int(publication_pb2.SERVING_SUPPORT_LEVEL_RUNTIME_BIND_SWAP_READY): (
        ServingSupportLevel.RUNTIME_BIND_SWAP_READY
    ),
}
_PUBLICATION_ASSEMBLY_TARGET_KIND_TO_PROTO: dict[
    AssemblyTargetKind, publication_pb2.AssemblyTargetKind
] = {
    "structural_view": publication_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW,
    "canonical_layout": publication_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT,
}
_PUBLICATION_ASSEMBLY_TARGET_KIND_FROM_PROTO: dict[int, AssemblyTargetKind] = {
    int(publication_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW): "structural_view",
    int(publication_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT): "canonical_layout",
}
_PUBLICATION_ASSEMBLY_LIVENESS_MODE_TO_PROTO: dict[
    AssemblyContributorLivenessMode,
    publication_pb2.AssemblyContributorLivenessMode,
] = {
    "require_live_until_cut": (
        publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ),
    "allow_durable_occupancy": (
        publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY
    ),
}
_PUBLICATION_ASSEMBLY_LIVENESS_MODE_FROM_PROTO: dict[
    int, AssemblyContributorLivenessMode
] = {
    int(
        publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ): "require_live_until_cut",
    int(
        publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_ALLOW_DURABLE_OCCUPANCY
    ): "allow_durable_occupancy",
}


class AssemblyTargetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AssemblyTargetKind
    structural_view_id: str | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> "AssemblyTargetRef":
        if self.kind == "structural_view" and not self.structural_view_id:
            raise ValueError("structural_view targets require structural_view_id")
        if self.kind == "canonical_layout" and self.structural_view_id:
            raise ValueError("canonical_layout targets must not set structural_view_id")
        return self

    def to_proto(self) -> store_daemon_pb2.AssemblyTargetRef:
        proto = store_daemon_pb2.AssemblyTargetRef(
            kind=_ASSEMBLY_TARGET_KIND_TO_PROTO[self.kind]
        )
        if self.structural_view_id:
            proto.structural_view_id = str(self.structural_view_id)
        return proto

    @classmethod
    def from_proto(
        cls, proto: store_daemon_pb2.AssemblyTargetRef
    ) -> "AssemblyTargetRef":
        return cls(
            kind=_ASSEMBLY_TARGET_KIND_FROM_PROTO[int(proto.kind)],
            structural_view_id=str(proto.structural_view_id or "") or None,
        )

    def to_publication_proto(self) -> publication_pb2.AssemblyTargetRef:
        proto = publication_pb2.AssemblyTargetRef(
            kind=_PUBLICATION_ASSEMBLY_TARGET_KIND_TO_PROTO[self.kind]
        )
        if self.structural_view_id:
            proto.structural_view_id = str(self.structural_view_id)
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.AssemblyTargetRef,
    ) -> "AssemblyTargetRef":
        return cls(
            kind=_PUBLICATION_ASSEMBLY_TARGET_KIND_FROM_PROTO[int(proto.kind)],
            structural_view_id=str(proto.structural_view_id or "") or None,
        )


class AssemblyRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot_id: str
    target: AssemblyTargetRef
    coverage_contract: str

    def to_proto(self) -> store_daemon_pb2.AssemblyRequirement:
        proto = store_daemon_pb2.AssemblyRequirement(
            slot_id=str(self.slot_id),
            coverage_contract=str(self.coverage_contract),
        )
        proto.target.CopyFrom(self.target.to_proto())
        return proto

    @classmethod
    def from_proto(
        cls, proto: store_daemon_pb2.AssemblyRequirement
    ) -> "AssemblyRequirement":
        return cls(
            slot_id=str(proto.slot_id),
            target=AssemblyTargetRef.from_proto(proto.target),
            coverage_contract=str(proto.coverage_contract),
        )

    def to_publication_proto(self) -> publication_pb2.AssemblyRequirement:
        proto = publication_pb2.AssemblyRequirement(
            slot_id=str(self.slot_id),
            coverage_contract=str(self.coverage_contract),
        )
        proto.target.CopyFrom(self.target.to_publication_proto())
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.AssemblyRequirement,
    ) -> "AssemblyRequirement":
        return cls(
            slot_id=str(proto.slot_id),
            target=AssemblyTargetRef.from_publication_proto(proto.target),
            coverage_contract=str(proto.coverage_contract),
        )


class AssemblyRequirementSetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirements_digest: str = ""
    requirement_count: int = 0
    carrier_form: str = "inline"
    inline_requirements: tuple[AssemblyRequirement, ...] = ()

    def to_proto(self) -> store_daemon_pb2.AssemblyRequirementSetRef:
        proto = store_daemon_pb2.AssemblyRequirementSetRef(
            requirements_digest=str(self.requirements_digest),
            requirement_count=int(
                self.requirement_count
                if self.requirement_count > 0
                else len(self.inline_requirements)
            ),
            carrier_form=str(self.carrier_form),
        )
        proto.inline_requirements.extend(
            requirement.to_proto() for requirement in self.inline_requirements
        )
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.AssemblyRequirementSetRef,
    ) -> "AssemblyRequirementSetRef":
        return cls(
            requirements_digest=str(proto.requirements_digest),
            requirement_count=int(proto.requirement_count),
            carrier_form=str(proto.carrier_form or "inline"),
            inline_requirements=tuple(
                AssemblyRequirement.from_proto(requirement)
                for requirement in proto.inline_requirements
            ),
        )

    def to_publication_proto(self) -> publication_pb2.AssemblyRequirementSetRef:
        proto = publication_pb2.AssemblyRequirementSetRef(
            requirements_digest=str(self.requirements_digest),
            requirement_count=int(
                self.requirement_count
                if self.requirement_count > 0
                else len(self.inline_requirements)
            ),
            carrier_form=str(self.carrier_form),
        )
        proto.inline_requirements.extend(
            requirement.to_publication_proto()
            for requirement in self.inline_requirements
        )
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.AssemblyRequirementSetRef,
    ) -> "AssemblyRequirementSetRef":
        return cls(
            requirements_digest=str(proto.requirements_digest),
            requirement_count=int(proto.requirement_count),
            carrier_form=str(proto.carrier_form or "inline"),
            inline_requirements=tuple(
                AssemblyRequirement.from_publication_proto(requirement)
                for requirement in proto.inline_requirements
            ),
        )

    @staticmethod
    def _dedupe_structural_view_ids(
        structural_view_ids: Iterable[str],
    ) -> tuple[str, ...]:
        deduped_view_ids = tuple(
            sorted(
                {
                    str(view_id).strip()
                    for view_id in structural_view_ids
                    if str(view_id).strip()
                }
            )
        )
        return deduped_view_ids

    @classmethod
    def canonical_full(cls) -> "AssemblyRequirementSetRef":
        requirements = (
            AssemblyRequirement(
                slot_id=_ASSEMBLY_CANONICAL_SLOT_ID,
                target=AssemblyTargetRef(kind="canonical_layout"),
                coverage_contract=_ASSEMBLY_CANONICAL_COVERAGE_CONTRACT,
            ),
        )
        return cls(
            requirement_count=1,
            carrier_form="inline",
            inline_requirements=requirements,
        )

    @classmethod
    def pp_from_structural_views(
        cls, structural_view_ids: Iterable[str]
    ) -> "AssemblyRequirementSetRef":
        deduped_view_ids = cls._dedupe_structural_view_ids(structural_view_ids)
        if not deduped_view_ids:
            raise ValueError(
                "PP requirements require at least one structural_view_id; "
                "use canonical_full() for single-rank canonical publish"
            )
        return cls._piece_family_requirements(
            deduped_view_ids,
            coverage_contract=_ASSEMBLY_PP_PIECE_COVERAGE_CONTRACT,
        )

    @classmethod
    def ep_from_structural_views(
        cls, structural_view_ids: Iterable[str]
    ) -> "AssemblyRequirementSetRef":
        deduped_view_ids = cls._dedupe_structural_view_ids(structural_view_ids)
        if not deduped_view_ids:
            raise ValueError(
                "EP requirements require at least one structural_view_id; "
                "use canonical_full() for single-rank canonical publish"
            )
        return cls._piece_family_requirements(
            deduped_view_ids,
            coverage_contract=_ASSEMBLY_EP_PIECE_COVERAGE_CONTRACT,
        )

    @classmethod
    def from_contract_family(
        cls,
        *,
        family: AssemblyContractFamily,
        structural_view_ids: Iterable[str] = (),
    ) -> "AssemblyRequirementSetRef":
        if family == "canonical_full":
            return cls.canonical_full()
        if family == "pp":
            return cls.pp_from_structural_views(structural_view_ids)
        return cls.ep_from_structural_views(structural_view_ids)

    @classmethod
    def _piece_family_requirements(
        cls,
        structural_view_ids: tuple[str, ...],
        *,
        coverage_contract: str,
    ) -> "AssemblyRequirementSetRef":
        requirements = tuple(
            AssemblyRequirement(
                slot_id=view_id,
                target=AssemblyTargetRef(
                    kind="structural_view",
                    structural_view_id=view_id,
                ),
                coverage_contract=coverage_contract,
            )
            for view_id in structural_view_ids
        )
        return cls(
            requirement_count=len(requirements),
            carrier_form="inline",
            inline_requirements=requirements,
        )


class AssemblyReadinessPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    contributor_liveness_mode: AssemblyContributorLivenessMode = (
        "require_live_until_cut"
    )

    def to_proto(self) -> store_daemon_pb2.AssemblyReadinessPolicy:
        return store_daemon_pb2.AssemblyReadinessPolicy(
            contributor_liveness_mode=_ASSEMBLY_LIVENESS_MODE_TO_PROTO[
                self.contributor_liveness_mode
            ]
        )

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.AssemblyReadinessPolicy,
    ) -> "AssemblyReadinessPolicy":
        if int(proto.contributor_liveness_mode) == int(
            store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_UNSPECIFIED
        ):
            return cls()
        return cls(
            contributor_liveness_mode=_ASSEMBLY_LIVENESS_MODE_FROM_PROTO[
                int(proto.contributor_liveness_mode)
            ]
        )

    def to_publication_proto(self) -> publication_pb2.AssemblyReadinessPolicy:
        return publication_pb2.AssemblyReadinessPolicy(
            contributor_liveness_mode=_PUBLICATION_ASSEMBLY_LIVENESS_MODE_TO_PROTO[
                self.contributor_liveness_mode
            ]
        )

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.AssemblyReadinessPolicy,
    ) -> "AssemblyReadinessPolicy":
        if int(proto.contributor_liveness_mode) == int(
            publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_UNSPECIFIED
        ):
            return cls()
        return cls(
            contributor_liveness_mode=_PUBLICATION_ASSEMBLY_LIVENESS_MODE_FROM_PROTO[
                int(proto.contributor_liveness_mode)
            ]
        )


class ServingBuildIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    representation_contract_hash: str | None = None
    builder_mode: BuilderMode
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    build_pipeline_version: str
    source_artifact_ref: str | None = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "ServingBuildIntent":
        if (
            self.representation_contract_hash is not None
            and not self.representation_contract_hash
        ):
            raise ValueError("representation_contract_hash must not be empty")
        if not self.framework_name:
            raise ValueError("framework_name must not be empty")
        if not self.adapter_version:
            raise ValueError("adapter_version must not be empty")
        if not self.serving_abi_version:
            raise ValueError("serving_abi_version must not be empty")
        if not self.build_pipeline_version:
            raise ValueError("build_pipeline_version must not be empty")
        return self

    def compute_serving_build_digest(self) -> str:
        payload = {
            "builder_mode": self.builder_mode.value,
            "framework_name": self.framework_name,
            "adapter_version": self.adapter_version,
            "serving_abi_version": self.serving_abi_version,
            "build_pipeline_version": self.build_pipeline_version,
        }
        return _hash_payload_to_multihash(payload)

    @staticmethod
    def serving_build_digest_version() -> str:
        return SERVING_BUILD_DIGEST_VERSION

    def to_publication_proto(self) -> publication_pb2.ServingBuildIntent:
        proto = publication_pb2.ServingBuildIntent(
            builder_mode=_PUBLICATION_BUILDER_MODE_TO_PROTO[self.builder_mode],
            framework_name=str(self.framework_name),
            adapter_version=str(self.adapter_version),
            serving_abi_version=str(self.serving_abi_version),
            build_pipeline_version=str(self.build_pipeline_version),
        )
        if self.representation_contract_hash is not None:
            proto.representation_contract_hash = str(self.representation_contract_hash)
        if self.source_artifact_ref is not None:
            proto.source_artifact_ref = str(self.source_artifact_ref)
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.ServingBuildIntent,
    ) -> "ServingBuildIntent":
        builder_mode = BuilderMode.PURE_TRANSFORM
        if int(proto.builder_mode) != int(publication_pb2.BUILDER_MODE_UNSPECIFIED):
            builder_mode = _PUBLICATION_BUILDER_MODE_FROM_PROTO[int(proto.builder_mode)]
        return cls(
            representation_contract_hash=(
                str(proto.representation_contract_hash or "") or None
            ),
            builder_mode=builder_mode,
            framework_name=str(proto.framework_name),
            adapter_version=str(proto.adapter_version),
            serving_abi_version=str(proto.serving_abi_version),
            build_pipeline_version=str(proto.build_pipeline_version),
            source_artifact_ref=str(proto.source_artifact_ref or "") or None,
        )


class PureTransformPublicationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    build_intent: ServingBuildIntent
    contract_family: AssemblyContractFamily | None = None
    source_version_key: str | None = None
    serving_version_key: str | None = None
    logical_topology_json: str | None = None
    serving_manifest_ref: str | None = None
    layout_id: str | None = None
    requirements: AssemblyRequirementSetRef | None = None
    readiness_policy: AssemblyReadinessPolicy | None = None
    structural_view_ids: tuple[str, ...] = ()
    admission_facts: ServingAdmissionFacts | None = None

    @model_validator(mode="after")
    def _validate_publication_spec(self) -> "PureTransformPublicationSpec":
        if self.contract_family is not None and self.contract_family not in {
            "pp",
            "ep",
            "canonical_full",
        }:
            raise ValueError("contract_family must be one of: pp, ep, canonical_full")
        if self.serving_manifest_ref is not None:
            parse_serving_manifest_ref(self.serving_manifest_ref)
        if self.admission_facts is not None:
            self.admission_facts.validate_for_representation_publish(
                builder_mode=self.build_intent.builder_mode
            )
        return self

    def to_proto(self) -> publication_pb2.PureTransformPublicationSpec:
        proto = publication_pb2.PureTransformPublicationSpec()
        proto.build_intent.CopyFrom(self.build_intent.to_publication_proto())
        if self.contract_family is not None:
            proto.contract_family = str(self.contract_family)
        if self.source_version_key is not None:
            proto.source_version_key = str(self.source_version_key)
        if self.serving_version_key is not None:
            proto.serving_version_key = str(self.serving_version_key)
        if self.logical_topology_json is not None:
            proto.logical_topology_json = str(self.logical_topology_json)
        if self.serving_manifest_ref is not None:
            proto.serving_manifest_ref = str(self.serving_manifest_ref)
        if self.layout_id is not None:
            proto.layout_id = str(self.layout_id)
        if self.requirements is not None:
            proto.requirements.CopyFrom(self.requirements.to_publication_proto())
        if self.readiness_policy is not None:
            proto.readiness_policy.CopyFrom(
                self.readiness_policy.to_publication_proto()
            )
        proto.structural_view_ids.extend(str(item) for item in self.structural_view_ids)
        if self.admission_facts is not None:
            proto.admission_facts.CopyFrom(self.admission_facts.to_publication_proto())
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: publication_pb2.PureTransformPublicationSpec,
    ) -> "PureTransformPublicationSpec":
        return cls(
            build_intent=ServingBuildIntent.from_publication_proto(proto.build_intent),
            contract_family=cast(
                AssemblyContractFamily | None,
                str(proto.contract_family or "") or None,
            ),
            source_version_key=str(proto.source_version_key or "") or None,
            serving_version_key=str(proto.serving_version_key or "") or None,
            logical_topology_json=str(proto.logical_topology_json or "") or None,
            serving_manifest_ref=str(proto.serving_manifest_ref or "") or None,
            layout_id=str(proto.layout_id or "") or None,
            requirements=(
                AssemblyRequirementSetRef.from_publication_proto(proto.requirements)
                if proto.HasField("requirements")
                else None
            ),
            readiness_policy=(
                AssemblyReadinessPolicy.from_publication_proto(proto.readiness_policy)
                if proto.HasField("readiness_policy")
                else None
            ),
            structural_view_ids=tuple(str(item) for item in proto.structural_view_ids),
            admission_facts=(
                ServingAdmissionFacts.from_publication_proto(proto.admission_facts)
                if proto.HasField("admission_facts")
                else None
            ),
        )


class ServingAdmissionFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    finalize_class: FinalizeClass
    support_level: ServingSupportLevel
    topology_admission_digest: str | None = None
    same_binding_fast_path_validated: bool = False

    @field_validator("topology_admission_digest", mode="before")
    @classmethod
    def _empty_digest_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _validate_admission_facts(self) -> "ServingAdmissionFacts":
        if (
            self.finalize_class == FinalizeClass.REPRESENTATION_CHANGING
            and not self.same_binding_fast_path_validated
        ):
            raise ValueError(
                "representation_changing admission requires "
                "same_binding_fast_path_validated=True"
            )
        return self

    def validate_for_representation_publish(self, *, builder_mode: BuilderMode) -> None:
        if self.finalize_class == FinalizeClass.UNKNOWN_BLOCKED:
            raise ValueError(
                "representation publish requires a non-blocked finalize_class"
            )
        if self.support_level not in {
            ServingSupportLevel.BUILDER_PUBLICATION_READY,
            ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        }:
            raise ValueError(
                "representation publish requires support_level to admit builder publication"
            )
        if (
            builder_mode == BuilderMode.PURE_TRANSFORM
            and self.finalize_class == FinalizeClass.REPRESENTATION_CHANGING
        ):
            raise ValueError(
                "PURE_TRANSFORM publication cannot use finalize_class=REPRESENTATION_CHANGING"
            )
        if (
            builder_mode == BuilderMode.BINDING_FINALIZE
            and self.finalize_class != FinalizeClass.REPRESENTATION_CHANGING
        ):
            raise ValueError(
                "BINDING_FINALIZE publication requires finalize_class=REPRESENTATION_CHANGING"
            )
        if (
            builder_mode == BuilderMode.BINDING_FINALIZE
            and not self.same_binding_fast_path_validated
        ):
            raise ValueError(
                "BINDING_FINALIZE publication requires same_binding_fast_path_validated=True"
            )

    def admits_builder_publication(self) -> bool:
        return self.support_level in {
            ServingSupportLevel.BUILDER_PUBLICATION_READY,
            ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        }

    def admits_runtime_bind_swap(self) -> bool:
        return self.support_level == ServingSupportLevel.RUNTIME_BIND_SWAP_READY

    def require_runtime_bind_swap_ready(self) -> None:
        if not self.admits_runtime_bind_swap():
            raise ValueError(
                "serving runtime requires support_level=RUNTIME_BIND_SWAP_READY"
            )

    def require_serving_key_activation_ready(self) -> None:
        if not self.admits_runtime_bind_swap():
            raise ValueError(
                "serving_version_key activation requires support_level=RUNTIME_BIND_SWAP_READY"
            )

    def to_publication_proto(self) -> publication_pb2.ServingAdmissionFacts:
        proto = publication_pb2.ServingAdmissionFacts(
            finalize_class=_PUBLICATION_FINALIZE_CLASS_TO_PROTO[self.finalize_class],
            support_level=_PUBLICATION_SERVING_SUPPORT_LEVEL_TO_PROTO[
                self.support_level
            ],
            same_binding_fast_path_validated=bool(
                self.same_binding_fast_path_validated
            ),
        )
        if self.topology_admission_digest is not None:
            proto.topology_admission_digest = str(self.topology_admission_digest)
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.ServingAdmissionFacts,
    ) -> "ServingAdmissionFacts":
        if int(proto.finalize_class) == int(publication_pb2.FINALIZE_CLASS_UNSPECIFIED):
            raise ValueError("ServingAdmissionFacts.finalize_class must be specified")
        if int(proto.support_level) == int(
            publication_pb2.SERVING_SUPPORT_LEVEL_UNSPECIFIED
        ):
            raise ValueError("ServingAdmissionFacts.support_level must be specified")
        return cls(
            finalize_class=_PUBLICATION_FINALIZE_CLASS_FROM_PROTO[
                int(proto.finalize_class)
            ],
            support_level=_PUBLICATION_SERVING_SUPPORT_LEVEL_FROM_PROTO[
                int(proto.support_level)
            ],
            topology_admission_digest=(
                str(proto.topology_admission_digest or "") or None
            ),
            same_binding_fast_path_validated=bool(
                proto.same_binding_fast_path_validated
            ),
        )


class ServingArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    artifact_kind: str = "serving"
    framework_name: str
    adapter_version: str
    serving_abi_version: str
    representation_contract_hash: str
    serving_build_digest: str
    serving_build_digest_version: str = SERVING_BUILD_DIGEST_VERSION
    tensor_schema_hash: str
    canonical_tensor_count: int
    serving_manifest_ref: str = Field(default_factory=build_serving_manifest_ref)
    source_artifact_ref: str | None = None
    builder_mode: BuilderMode
    build_pipeline_version: str
    logical_topology_json: str | None = None
    topology_admission_digest: str | None = None

    @model_validator(mode="after")
    def _validate_manifest(self) -> "ServingArtifactManifest":
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.artifact_kind != "serving":
            raise ValueError("artifact_kind must be 'serving'")
        if not self.framework_name:
            raise ValueError("framework_name must not be empty")
        if not self.adapter_version:
            raise ValueError("adapter_version must not be empty")
        if not self.serving_abi_version:
            raise ValueError("serving_abi_version must not be empty")
        if not self.representation_contract_hash:
            raise ValueError("representation_contract_hash must not be empty")
        if not self.serving_build_digest:
            raise ValueError("serving_build_digest must not be empty")
        if not self.serving_build_digest_version:
            raise ValueError("serving_build_digest_version must not be empty")
        if not self.tensor_schema_hash:
            raise ValueError("tensor_schema_hash must not be empty")
        if self.canonical_tensor_count < 0:
            raise ValueError("canonical_tensor_count must be non-negative")
        parse_serving_manifest_ref(self.serving_manifest_ref)
        if not self.build_pipeline_version:
            raise ValueError("build_pipeline_version must not be empty")
        return self

    @classmethod
    def from_build_intent(
        cls,
        *,
        intent: ServingBuildIntent,
        representation_contract_hash: str | None = None,
        tensor_schema_hash: str,
        canonical_tensor_count: int,
        serving_manifest_ref: str | None = None,
        logical_topology_json: str | None = None,
        topology_admission_digest: str | None = None,
    ) -> "ServingArtifactManifest":
        resolved_representation_contract_hash = (
            representation_contract_hash or intent.representation_contract_hash
        )
        if not resolved_representation_contract_hash:
            raise ValueError(
                "representation_contract_hash must be resolved before building a serving manifest"
            )
        return cls(
            framework_name=intent.framework_name,
            adapter_version=intent.adapter_version,
            serving_abi_version=intent.serving_abi_version,
            representation_contract_hash=resolved_representation_contract_hash,
            serving_build_digest=intent.compute_serving_build_digest(),
            serving_build_digest_version=intent.serving_build_digest_version(),
            tensor_schema_hash=str(tensor_schema_hash),
            canonical_tensor_count=int(canonical_tensor_count),
            serving_manifest_ref=(
                build_serving_manifest_ref()
                if serving_manifest_ref is None
                else str(serving_manifest_ref)
            ),
            source_artifact_ref=intent.source_artifact_ref,
            builder_mode=intent.builder_mode,
            build_pipeline_version=intent.build_pipeline_version,
            logical_topology_json=logical_topology_json,
            topology_admission_digest=topology_admission_digest,
        )

    def to_bytes(self) -> bytes:
        return _canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def from_bytes(cls, payload: bytes | bytearray | str) -> "ServingArtifactManifest":
        raw = (
            payload.decode("utf-8")
            if isinstance(payload, (bytes, bytearray))
            else str(payload)
        )
        return cls.model_validate_json(raw)

    def to_runtime_policy(
        self,
        *,
        require_manifest: bool = True,
    ) -> "ServingRuntimePolicy":
        return ServingRuntimePolicy(
            require_manifest=bool(require_manifest),
            serving_manifest_ref=str(self.serving_manifest_ref),
            expected_representation_contract_hash=str(
                self.representation_contract_hash
            ),
            expected_serving_build_digest=str(self.serving_build_digest),
            expected_topology_admission_digest=(
                str(self.topology_admission_digest)
                if self.topology_admission_digest
                else None
            ),
        )


class ServingRuntimePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    require_manifest: bool = True
    serving_manifest_ref: str | None = None
    expected_representation_contract_hash: str | None = None
    expected_serving_build_digest: str | None = None
    expected_topology_admission_digest: str | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> "ServingRuntimePolicy":
        if self.serving_manifest_ref is not None:
            parse_serving_manifest_ref(self.serving_manifest_ref)
        return self

    def to_proto(self) -> store_daemon_pb2.ServingArtifactRuntimePolicy:
        proto = store_daemon_pb2.ServingArtifactRuntimePolicy(
            require_manifest=bool(
                self.require_manifest
                or self.serving_manifest_ref is not None
                or self.expected_representation_contract_hash is not None
                or self.expected_serving_build_digest is not None
                or self.expected_topology_admission_digest is not None
            )
        )
        if self.serving_manifest_ref is not None:
            proto.serving_manifest_ref = str(self.serving_manifest_ref)
        if self.expected_representation_contract_hash is not None:
            proto.expected_representation_contract_hash = str(
                self.expected_representation_contract_hash
            )
        if self.expected_serving_build_digest is not None:
            proto.expected_serving_build_digest = str(
                self.expected_serving_build_digest
            )
        if self.expected_topology_admission_digest is not None:
            proto.expected_topology_admission_digest = str(
                self.expected_topology_admission_digest
            )
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.ServingArtifactRuntimePolicy,
    ) -> "ServingRuntimePolicy":
        return cls(
            require_manifest=bool(proto.require_manifest),
            serving_manifest_ref=str(proto.serving_manifest_ref or "") or None,
            expected_representation_contract_hash=(
                str(proto.expected_representation_contract_hash or "") or None
            ),
            expected_serving_build_digest=(
                str(proto.expected_serving_build_digest or "") or None
            ),
            expected_topology_admission_digest=(
                str(proto.expected_topology_admission_digest or "") or None
            ),
        )


class BindingValueRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int

    @model_validator(mode="after")
    def _validate_ref(self) -> "BindingValueRef":
        if not self.binding_id:
            raise ValueError("binding_id must not be empty")
        if not self.binding_layout_id:
            raise ValueError("binding_layout_id must not be empty")
        if not self.binding_value_id:
            raise ValueError("binding_value_id must not be empty")
        if int(self.seal_generation) < 0:
            raise ValueError("seal_generation must be non-negative")
        return self

    def to_proto(self) -> publication_pb2.BindingValueRef:
        return publication_pb2.BindingValueRef(
            binding_id=str(self.binding_id),
            binding_layout_id=str(self.binding_layout_id),
            binding_value_id=str(self.binding_value_id),
            seal_generation=int(self.seal_generation),
        )

    def to_store_proto(self) -> publication_pb2.BindingValueRef:
        return publication_pb2.BindingValueRef(
            binding_id=str(self.binding_id),
            binding_layout_id=str(self.binding_layout_id),
            binding_value_id=str(self.binding_value_id),
            seal_generation=int(self.seal_generation),
        )

    @classmethod
    def from_proto(
        cls,
        proto: publication_pb2.BindingValueRef,
    ) -> "BindingValueRef":
        return cls(
            binding_id=str(proto.binding_id),
            binding_layout_id=str(proto.binding_layout_id),
            binding_value_id=str(proto.binding_value_id),
            seal_generation=int(proto.seal_generation),
        )


ServingBindingReadiness = Literal[
    "serving_reserved",
    "serving_local_ready",
    "serving_published_ready",
]

_SERVING_READINESS_TO_PROTO: dict[
    ServingBindingReadiness, operation_pb2.ServingBindingReadiness
] = {
    "serving_reserved": operation_pb2.SERVING_BINDING_READINESS_RESERVED,
    "serving_local_ready": operation_pb2.SERVING_BINDING_READINESS_LOCAL_READY,
    "serving_published_ready": operation_pb2.SERVING_BINDING_READINESS_PUBLISHED_READY,
}
_SERVING_READINESS_FROM_PROTO: dict[int, ServingBindingReadiness] = {
    int(value): key for key, value in _SERVING_READINESS_TO_PROTO.items()
}


class ServingTopologyRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    schema_topology_digest: str
    admission_topology_digest: str | None = None
    logical_topology_ref: str | None = None
    runtime_topology_diagnostics_ref: str | None = None

    @model_validator(mode="after")
    def _validate_topology(self) -> "ServingTopologyRef":
        if int(self.schema_version) <= 0:
            raise ValueError("schema_version must be positive")
        if not self.schema_topology_digest:
            raise ValueError("schema_topology_digest must not be empty")
        return self

    def to_proto(self) -> operation_pb2.ServingTopologyRef:
        proto = operation_pb2.ServingTopologyRef(
            schema_version=int(self.schema_version),
            schema_topology_digest=str(self.schema_topology_digest),
        )
        if self.admission_topology_digest is not None:
            proto.admission_topology_digest = str(self.admission_topology_digest)
        if self.logical_topology_ref is not None:
            proto.logical_topology_ref = str(self.logical_topology_ref)
        if self.runtime_topology_diagnostics_ref is not None:
            proto.runtime_topology_diagnostics_ref = str(
                self.runtime_topology_diagnostics_ref
            )
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingTopologyRef
    ) -> "ServingTopologyRef":
        return cls(
            schema_version=int(proto.schema_version),
            schema_topology_digest=str(proto.schema_topology_digest),
            admission_topology_digest=(
                str(proto.admission_topology_digest)
                if proto.HasField("admission_topology_digest")
                else None
            ),
            logical_topology_ref=(
                str(proto.logical_topology_ref)
                if proto.HasField("logical_topology_ref")
                else None
            ),
            runtime_topology_diagnostics_ref=(
                str(proto.runtime_topology_diagnostics_ref)
                if proto.HasField("runtime_topology_diagnostics_ref")
                else None
            ),
        )


class ServingBindingMemberRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    member_id: str
    member_index: int
    member_count: int
    group_id: str | None = None

    @model_validator(mode="after")
    def _validate_member(self) -> "ServingBindingMemberRef":
        if not self.member_id:
            raise ValueError("member_id must not be empty")
        if int(self.member_index) < 0:
            raise ValueError("member_index must be non-negative")
        if int(self.member_count) <= 0:
            raise ValueError("member_count must be positive")
        if int(self.member_index) >= int(self.member_count):
            raise ValueError("member_index must be less than member_count")
        if self.group_id is not None and not self.group_id:
            raise ValueError("group_id must not be empty when provided")
        return self

    def to_proto(self) -> operation_pb2.ServingBindingMemberRef:
        proto = operation_pb2.ServingBindingMemberRef(
            member_id=str(self.member_id),
            member_index=int(self.member_index),
            member_count=int(self.member_count),
        )
        if self.group_id is not None:
            proto.group_id = str(self.group_id)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingMemberRef
    ) -> "ServingBindingMemberRef":
        return cls(
            member_id=str(proto.member_id),
            member_index=int(proto.member_index),
            member_count=int(proto.member_count),
            group_id=str(proto.group_id) if proto.HasField("group_id") else None,
        )


class BlobRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    size_bytes: int

    @model_validator(mode="after")
    def _validate_blob(self) -> "BlobRef":
        if not self.path:
            raise ValueError("path must not be empty")
        if not self.sha256:
            raise ValueError("sha256 must not be empty")
        if int(self.size_bytes) < 0:
            raise ValueError("size_bytes must be non-negative")
        return self

    def to_proto(self) -> operation_pb2.BlobRef:
        return operation_pb2.BlobRef(
            path=str(self.path),
            sha256=str(self.sha256),
            size_bytes=int(self.size_bytes),
        )

    @classmethod
    def from_proto(cls, proto: operation_pb2.BlobRef) -> "BlobRef":
        return cls(
            path=str(proto.path),
            sha256=str(proto.sha256),
            size_bytes=int(proto.size_bytes),
        )


ServingBindingSourceKind = Literal[
    "checkpoint_artifact",
    "serving_artifact",
    "serving_artifact_set",
]
ServingBindingSourceReuseMode = Literal[
    "checkpoint_to_serving",
    "serving_direct_member_copy",
    "serving_transform_required",
    "unsupported",
]

_SOURCE_KIND_TO_PROTO: dict[
    ServingBindingSourceKind, operation_pb2.ServingBindingSourceKind
] = {
    "checkpoint_artifact": operation_pb2.SERVING_BINDING_SOURCE_KIND_CHECKPOINT_ARTIFACT,
    "serving_artifact": operation_pb2.SERVING_BINDING_SOURCE_KIND_SERVING_ARTIFACT,
    "serving_artifact_set": operation_pb2.SERVING_BINDING_SOURCE_KIND_SERVING_ARTIFACT_SET,
}
_SOURCE_KIND_FROM_PROTO: dict[int, ServingBindingSourceKind] = {
    int(value): key for key, value in _SOURCE_KIND_TO_PROTO.items()
}
_SOURCE_REUSE_TO_PROTO: dict[
    ServingBindingSourceReuseMode, operation_pb2.ServingBindingSourceReuseMode
] = {
    "checkpoint_to_serving": operation_pb2.SERVING_BINDING_SOURCE_REUSE_MODE_CHECKPOINT_TO_SERVING,
    "serving_direct_member_copy": operation_pb2.SERVING_BINDING_SOURCE_REUSE_MODE_SERVING_DIRECT_MEMBER_COPY,
    "serving_transform_required": operation_pb2.SERVING_BINDING_SOURCE_REUSE_MODE_SERVING_TRANSFORM_REQUIRED,
    "unsupported": operation_pb2.SERVING_BINDING_SOURCE_REUSE_MODE_UNSUPPORTED,
}
_SOURCE_REUSE_FROM_PROTO: dict[int, ServingBindingSourceReuseMode] = {
    int(value): key for key, value in _SOURCE_REUSE_TO_PROTO.items()
}


class ServingBindingSourceMemberRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    member: ServingBindingMemberRef
    artifact_ref: str
    serving_manifest_ref: str | None = None
    tensor_schema_hash: str | None = None
    target_layout_hash: str | None = None

    @model_validator(mode="after")
    def _validate_source_member(self) -> "ServingBindingSourceMemberRef":
        if not self.artifact_ref:
            raise ValueError("artifact_ref must not be empty")
        for field_name in (
            "serving_manifest_ref",
            "tensor_schema_hash",
            "target_layout_hash",
        ):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty when provided")
        return self

    def to_proto(self) -> operation_pb2.ServingBindingSourceMemberRef:
        proto = operation_pb2.ServingBindingSourceMemberRef(
            artifact_ref=str(self.artifact_ref)
        )
        proto.member.CopyFrom(self.member.to_proto())
        if self.serving_manifest_ref is not None:
            proto.serving_manifest_ref = str(self.serving_manifest_ref)
        if self.tensor_schema_hash is not None:
            proto.tensor_schema_hash = str(self.tensor_schema_hash)
        if self.target_layout_hash is not None:
            proto.target_layout_hash = str(self.target_layout_hash)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingSourceMemberRef
    ) -> "ServingBindingSourceMemberRef":
        return cls(
            member=ServingBindingMemberRef.from_proto(proto.member),
            artifact_ref=str(proto.artifact_ref),
            serving_manifest_ref=(
                str(proto.serving_manifest_ref)
                if proto.HasField("serving_manifest_ref")
                else None
            ),
            tensor_schema_hash=(
                str(proto.tensor_schema_hash)
                if proto.HasField("tensor_schema_hash")
                else None
            ),
            target_layout_hash=(
                str(proto.target_layout_hash)
                if proto.HasField("target_layout_hash")
                else None
            ),
        )


class ServingBindingSourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_kind: ServingBindingSourceKind
    artifact_selection_digest: str
    source_artifact_ref: str | None = None
    source_schema_hash: str
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    tensor_schema_hash: str | None = None
    topology: ServingTopologyRef | None = None
    members: tuple[ServingBindingSourceMemberRef, ...] = ()

    @model_validator(mode="after")
    def _validate_source(self) -> "ServingBindingSourceRef":
        if not self.artifact_selection_digest:
            raise ValueError("artifact_selection_digest must not be empty")
        if not self.source_schema_hash:
            raise ValueError("source_schema_hash must not be empty")
        if self.source_kind == "checkpoint_artifact":
            if not self.source_artifact_ref:
                raise ValueError(
                    "source_artifact_ref is required for checkpoint_artifact sources"
                )
            if self.members:
                raise ValueError("checkpoint_artifact sources must not carry members")
        if self.source_kind == "serving_artifact_set":
            if self.topology is None:
                raise ValueError(
                    "topology is required for serving_artifact_set sources"
                )
            if not self.members:
                raise ValueError(
                    "members are required for serving_artifact_set sources"
                )
        return self

    def to_proto(self) -> operation_pb2.ServingBindingSourceRef:
        proto = operation_pb2.ServingBindingSourceRef(
            source_kind=_SOURCE_KIND_TO_PROTO[self.source_kind],
            artifact_selection_digest=str(self.artifact_selection_digest),
            source_schema_hash=str(self.source_schema_hash),
        )
        if self.source_artifact_ref is not None:
            proto.source_artifact_ref = str(self.source_artifact_ref)
        if self.representation_contract_hash is not None:
            proto.representation_contract_hash = str(self.representation_contract_hash)
        if self.serving_build_digest is not None:
            proto.serving_build_digest = str(self.serving_build_digest)
        if self.tensor_schema_hash is not None:
            proto.tensor_schema_hash = str(self.tensor_schema_hash)
        if self.topology is not None:
            proto.topology.CopyFrom(self.topology.to_proto())
        proto.members.extend(member.to_proto() for member in self.members)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingSourceRef
    ) -> "ServingBindingSourceRef":
        source_kind = _SOURCE_KIND_FROM_PROTO.get(int(proto.source_kind))
        if source_kind is None:
            raise ValueError("ServingBindingSourceRef source_kind is required")
        return cls(
            source_kind=source_kind,
            artifact_selection_digest=str(proto.artifact_selection_digest),
            source_artifact_ref=(
                str(proto.source_artifact_ref)
                if proto.HasField("source_artifact_ref")
                else None
            ),
            source_schema_hash=str(proto.source_schema_hash),
            representation_contract_hash=(
                str(proto.representation_contract_hash)
                if proto.HasField("representation_contract_hash")
                else None
            ),
            serving_build_digest=(
                str(proto.serving_build_digest)
                if proto.HasField("serving_build_digest")
                else None
            ),
            tensor_schema_hash=(
                str(proto.tensor_schema_hash)
                if proto.HasField("tensor_schema_hash")
                else None
            ),
            topology=(
                ServingTopologyRef.from_proto(proto.topology)
                if proto.HasField("topology")
                else None
            ),
            members=tuple(
                ServingBindingSourceMemberRef.from_proto(member)
                for member in proto.members
            ),
        )


class ServingBindingSourceReuseDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: ServingBindingSourceReuseMode
    representation_contract_hash: str | None = None
    work_plan_hash: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_reuse(self) -> "ServingBindingSourceReuseDecision":
        for field_name in ("representation_contract_hash", "work_plan_hash", "reason"):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty when provided")
        if self.mode == "serving_transform_required" and not (
            self.work_plan_hash or self.reason
        ):
            raise ValueError(
                "serving_transform_required requires work_plan_hash or reason"
            )
        if self.mode == "unsupported" and not self.reason:
            raise ValueError("unsupported source reuse requires reason")
        return self

    def to_proto(self) -> operation_pb2.ServingBindingSourceReuseDecision:
        proto = operation_pb2.ServingBindingSourceReuseDecision(
            mode=_SOURCE_REUSE_TO_PROTO[self.mode]
        )
        if self.representation_contract_hash is not None:
            proto.representation_contract_hash = str(self.representation_contract_hash)
        if self.work_plan_hash is not None:
            proto.work_plan_hash = str(self.work_plan_hash)
        if self.reason is not None:
            proto.reason = str(self.reason)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingSourceReuseDecision
    ) -> "ServingBindingSourceReuseDecision":
        mode = _SOURCE_REUSE_FROM_PROTO.get(int(proto.mode))
        if mode is None:
            raise ValueError("ServingBindingSourceReuseDecision mode is required")
        return cls(
            mode=mode,
            representation_contract_hash=(
                str(proto.representation_contract_hash)
                if proto.HasField("representation_contract_hash")
                else None
            ),
            work_plan_hash=(
                str(proto.work_plan_hash) if proto.HasField("work_plan_hash") else None
            ),
            reason=str(proto.reason) if proto.HasField("reason") else None,
        )


def plan_serving_binding_source_reuse(
    *,
    source: ServingBindingSourceRef,
    topology: ServingTopologyRef,
    member: ServingBindingMemberRef,
    tensor_schema_hash: str,
    target_layout_hash: str,
    representation_contract_hash: str | None = None,
) -> ServingBindingSourceReuseDecision:
    if source.source_kind == "checkpoint_artifact":
        return ServingBindingSourceReuseDecision(
            mode="checkpoint_to_serving",
            representation_contract_hash=representation_contract_hash,
        )
    if source.source_kind not in {"serving_artifact", "serving_artifact_set"}:
        return ServingBindingSourceReuseDecision(
            mode="unsupported",
            reason=f"unsupported serving binding source kind: {source.source_kind}",
        )
    if (
        representation_contract_hash is not None
        and source.representation_contract_hash is not None
        and representation_contract_hash != source.representation_contract_hash
    ):
        return ServingBindingSourceReuseDecision(
            mode="serving_transform_required",
            reason="source representation contract does not match target",
        )
    if source.topology is not None and source.topology != topology:
        return ServingBindingSourceReuseDecision(
            mode="serving_transform_required",
            reason="source topology does not match target topology",
        )
    if (
        source.tensor_schema_hash is not None
        and source.tensor_schema_hash != tensor_schema_hash
    ):
        return ServingBindingSourceReuseDecision(
            mode="serving_transform_required",
            reason="source tensor schema does not match target tensor schema",
        )
    matching_members = [
        source_member
        for source_member in source.members
        if source_member.member == member
    ]
    if source.source_kind == "serving_artifact_set" and not matching_members:
        return ServingBindingSourceReuseDecision(
            mode="serving_transform_required",
            reason="source serving set does not contain target member",
        )
    for source_member in matching_members:
        if (
            source_member.tensor_schema_hash is not None
            and source_member.tensor_schema_hash != tensor_schema_hash
        ):
            return ServingBindingSourceReuseDecision(
                mode="serving_transform_required",
                reason="source member tensor schema does not match target",
            )
        if (
            source_member.target_layout_hash is not None
            and source_member.target_layout_hash != target_layout_hash
        ):
            return ServingBindingSourceReuseDecision(
                mode="serving_transform_required",
                reason="source member layout does not match target layout",
            )
    return ServingBindingSourceReuseDecision(
        mode="serving_direct_member_copy",
        representation_contract_hash=representation_contract_hash
        or source.representation_contract_hash,
    )


class ServingBindingResolvedLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    binding_layout_id: str
    source: ServingBindingSourceRef
    source_reuse: ServingBindingSourceReuseDecision
    topology: ServingTopologyRef
    member: ServingBindingMemberRef
    target_layout: bytes
    target_index_bytes: bytes
    target_layout_hash: str
    tensor_schema_hash: str
    spec_digest: str
    source_schema_hash: str | None = None
    copy_plan_bytes: bytes | None = None
    dst_specs_bytes: bytes | None = None

    @model_validator(mode="after")
    def _validate_layout(self) -> "ServingBindingResolvedLayout":
        if not self.binding_layout_id:
            raise ValueError("binding_layout_id must not be empty")
        if not self.target_layout:
            raise ValueError("target_layout must not be empty")
        if not self.target_index_bytes:
            raise ValueError("target_index_bytes must not be empty")
        if not self.target_layout_hash:
            raise ValueError("target_layout_hash must not be empty")
        if not self.tensor_schema_hash:
            raise ValueError("tensor_schema_hash must not be empty")
        if not self.spec_digest:
            raise ValueError("spec_digest must not be empty")
        if self.source_reuse.mode == "serving_direct_member_copy":
            if self.source.source_kind not in {
                "serving_artifact",
                "serving_artifact_set",
            }:
                raise ValueError(
                    "serving_direct_member_copy requires a serving artifact source"
                )
            if (
                self.source.representation_contract_hash is not None
                and self.source_reuse.representation_contract_hash is not None
                and self.source_reuse.representation_contract_hash
                != self.source.representation_contract_hash
            ):
                raise ValueError(
                    "source_reuse representation_contract_hash must match source"
                )
            if self.source.tensor_schema_hash is not None and (
                self.source.tensor_schema_hash != self.tensor_schema_hash
            ):
                raise ValueError(
                    "serving_direct_member_copy tensor_schema_hash must match target"
                )
            matching_members = [
                source_member
                for source_member in self.source.members
                if source_member.member == self.member
            ]
            if (
                self.source.source_kind == "serving_artifact_set"
                and not matching_members
            ):
                raise ValueError(
                    "serving_direct_member_copy requires a matching source member"
                )
            for source_member in matching_members:
                if (
                    source_member.target_layout_hash is not None
                    and source_member.target_layout_hash != self.target_layout_hash
                ):
                    raise ValueError(
                        "serving_direct_member_copy target_layout_hash must match source member"
                    )
                if (
                    source_member.tensor_schema_hash is not None
                    and source_member.tensor_schema_hash != self.tensor_schema_hash
                ):
                    raise ValueError(
                        "serving_direct_member_copy tensor_schema_hash must match source member"
                    )
        return self

    def to_proto(self) -> operation_pb2.ServingBindingResolvedLayout:
        proto = operation_pb2.ServingBindingResolvedLayout(
            binding_layout_id=str(self.binding_layout_id),
            target_layout=bytes(self.target_layout),
            target_index_bytes=bytes(self.target_index_bytes),
            target_layout_hash=str(self.target_layout_hash),
            tensor_schema_hash=str(self.tensor_schema_hash),
            spec_digest=str(self.spec_digest),
        )
        proto.source.CopyFrom(self.source.to_proto())
        proto.source_reuse.CopyFrom(self.source_reuse.to_proto())
        proto.topology.CopyFrom(self.topology.to_proto())
        proto.member.CopyFrom(self.member.to_proto())
        if self.source_schema_hash is not None:
            proto.source_schema_hash = str(self.source_schema_hash)
        if self.copy_plan_bytes is not None:
            proto.copy_plan_bytes = bytes(self.copy_plan_bytes)
        if self.dst_specs_bytes is not None:
            proto.dst_specs_bytes = bytes(self.dst_specs_bytes)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingResolvedLayout
    ) -> "ServingBindingResolvedLayout":
        return cls(
            binding_layout_id=str(proto.binding_layout_id),
            source=ServingBindingSourceRef.from_proto(proto.source),
            source_reuse=ServingBindingSourceReuseDecision.from_proto(
                proto.source_reuse
            ),
            topology=ServingTopologyRef.from_proto(proto.topology),
            member=ServingBindingMemberRef.from_proto(proto.member),
            target_layout=bytes(proto.target_layout),
            target_index_bytes=bytes(proto.target_index_bytes),
            target_layout_hash=str(proto.target_layout_hash),
            tensor_schema_hash=str(proto.tensor_schema_hash),
            spec_digest=str(proto.spec_digest),
            source_schema_hash=(
                str(proto.source_schema_hash)
                if proto.HasField("source_schema_hash")
                else None
            ),
            copy_plan_bytes=(
                bytes(proto.copy_plan_bytes)
                if proto.HasField("copy_plan_bytes")
                else None
            ),
            dst_specs_bytes=(
                bytes(proto.dst_specs_bytes)
                if proto.HasField("dst_specs_bytes")
                else None
            ),
        )


class ServingBindingTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    device: str | int
    device_uuid: str | None = None
    source: ServingBindingSourceRef
    topology: ServingTopologyRef
    member: ServingBindingMemberRef
    model_config_digest: str
    load_config_digest: str | None = None
    serving_build_digest: str
    resolved_layout: ServingBindingResolvedLayout

    @model_validator(mode="after")
    def _validate_target(self) -> "ServingBindingTarget":
        if not self.runtime:
            raise ValueError("runtime must not be empty")
        if str(self.device) == "":
            raise ValueError("device must not be empty")
        if self.device_uuid is not None and not self.device_uuid:
            raise ValueError("device_uuid must not be empty when provided")
        if not self.model_config_digest:
            raise ValueError("model_config_digest must not be empty")
        if self.load_config_digest is not None and not self.load_config_digest:
            raise ValueError("load_config_digest must not be empty when provided")
        if not self.serving_build_digest:
            raise ValueError("serving_build_digest must not be empty")
        if self.source != self.resolved_layout.source:
            raise ValueError("resolved_layout.source must match target source")
        if self.source.topology is not None and self.source.topology != self.topology:
            raise ValueError("source topology must match target topology when provided")
        if (
            self.resolved_layout.source_reuse.mode == "serving_direct_member_copy"
            and self.source.serving_build_digest is not None
            and self.source.serving_build_digest != self.serving_build_digest
        ):
            raise ValueError(
                "serving_direct_member_copy serving_build_digest must match source"
            )
        if self.topology != self.resolved_layout.topology:
            raise ValueError("resolved_layout.topology must match target topology")
        if self.member != self.resolved_layout.member:
            raise ValueError("resolved_layout.member must match target member")
        return self

    def to_proto(self) -> operation_pb2.ServingBindingTarget:
        proto = operation_pb2.ServingBindingTarget(
            runtime=str(self.runtime),
            device=str(self.device),
            model_config_digest=str(self.model_config_digest),
            serving_build_digest=str(self.serving_build_digest),
        )
        if self.device_uuid is not None:
            proto.device_uuid = str(self.device_uuid)
        if self.load_config_digest is not None:
            proto.load_config_digest = str(self.load_config_digest)
        proto.source.CopyFrom(self.source.to_proto())
        proto.topology.CopyFrom(self.topology.to_proto())
        proto.member.CopyFrom(self.member.to_proto())
        proto.resolved_layout.CopyFrom(self.resolved_layout.to_proto())
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingTarget
    ) -> "ServingBindingTarget":
        return cls(
            runtime=str(proto.runtime),
            device=str(proto.device),
            device_uuid=str(proto.device_uuid)
            if proto.HasField("device_uuid")
            else None,
            source=ServingBindingSourceRef.from_proto(proto.source),
            topology=ServingTopologyRef.from_proto(proto.topology),
            member=ServingBindingMemberRef.from_proto(proto.member),
            model_config_digest=str(proto.model_config_digest),
            load_config_digest=(
                str(proto.load_config_digest)
                if proto.HasField("load_config_digest")
                else None
            ),
            serving_build_digest=str(proto.serving_build_digest),
            resolved_layout=ServingBindingResolvedLayout.from_proto(
                proto.resolved_layout
            ),
        )


class ServingBindingSetTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    source: ServingBindingSourceRef
    topology: ServingTopologyRef
    group_id: str
    members: tuple[ServingBindingTarget, ...]

    @model_validator(mode="after")
    def _validate_set_target(self) -> "ServingBindingSetTarget":
        if not self.runtime:
            raise ValueError("runtime must not be empty")
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if not self.members:
            raise ValueError("members must not be empty")
        for member in self.members:
            if member.runtime != self.runtime:
                raise ValueError("all members must use the set runtime")
            if member.source != self.source:
                raise ValueError("all members must use the set source")
            if member.topology != self.topology:
                raise ValueError("all members must use the set topology")
            if member.member.group_id not in {None, self.group_id}:
                raise ValueError("member group_id must match set group_id")
        return self

    def to_proto(self) -> operation_pb2.ServingBindingSetTarget:
        proto = operation_pb2.ServingBindingSetTarget(
            runtime=str(self.runtime),
            group_id=str(self.group_id),
        )
        proto.source.CopyFrom(self.source.to_proto())
        proto.topology.CopyFrom(self.topology.to_proto())
        proto.members.extend(member.to_proto() for member in self.members)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingSetTarget
    ) -> "ServingBindingSetTarget":
        return cls(
            runtime=str(proto.runtime),
            source=ServingBindingSourceRef.from_proto(proto.source),
            topology=ServingTopologyRef.from_proto(proto.topology),
            group_id=str(proto.group_id),
            members=tuple(
                ServingBindingTarget.from_proto(member) for member in proto.members
            ),
        )


class ServingBindingResolvedSpecCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int
    cache_key_digest: str
    spec_digest: str
    runtime: str
    source: ServingBindingSourceRef
    source_reuse: ServingBindingSourceReuseDecision
    topology: ServingTopologyRef
    member: ServingBindingMemberRef
    source_schema_hash: str
    model_config_digest: str
    load_config_digest: str | None = None
    serving_build_digest: str
    binding_layout_id: str
    target_layout_hash: str
    tensor_schema_hash: str
    blob_refs: Mapping[str, BlobRef]

    @model_validator(mode="after")
    def _validate_cache_entry(self) -> "ServingBindingResolvedSpecCacheEntry":
        if int(self.schema_version) <= 0:
            raise ValueError("schema_version must be positive")
        for field_name in (
            "cache_key_digest",
            "spec_digest",
            "runtime",
            "source_schema_hash",
            "model_config_digest",
            "serving_build_digest",
            "binding_layout_id",
            "target_layout_hash",
            "tensor_schema_hash",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if not self.blob_refs:
            raise ValueError("blob_refs must not be empty")
        return self

    def canonical_key_json(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "runtime": self.runtime,
            "source_schema_hash": self.source_schema_hash,
            "model_config_digest": self.model_config_digest,
            "load_config_digest": self.load_config_digest,
            "topology": self.topology.model_dump(mode="json", exclude_none=True),
            "member": self.member.model_dump(mode="json", exclude_none=True),
            "serving_build_digest": self.serving_build_digest,
            "source": self.source.model_dump(mode="json", exclude_none=True),
            "source_reuse": self.source_reuse.model_dump(
                mode="json", exclude_none=True
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def computed_cache_key_digest(self) -> str:
        return hashlib.sha256(self.canonical_key_json().encode("utf-8")).hexdigest()

    def canonical_spec_core_json(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "canonical_key": json.loads(self.canonical_key_json()),
            "binding_layout_id": self.binding_layout_id,
            "target_layout_hash": self.target_layout_hash,
            "tensor_schema_hash": self.tensor_schema_hash,
            "source_schema_hash": self.source_schema_hash,
            "serving_build_digest": self.serving_build_digest,
            "source_reuse": self.source_reuse.model_dump(
                mode="json", exclude_none=True
            ),
            "blob_refs": {
                key: value.model_dump(mode="json", exclude_none=True)
                for key, value in sorted(self.blob_refs.items())
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def computed_spec_digest(self) -> str:
        return hashlib.sha256(
            self.canonical_spec_core_json().encode("utf-8")
        ).hexdigest()

    def to_proto(self) -> operation_pb2.ServingBindingResolvedSpecCacheEntry:
        proto = operation_pb2.ServingBindingResolvedSpecCacheEntry(
            schema_version=int(self.schema_version),
            cache_key_digest=str(self.cache_key_digest),
            spec_digest=str(self.spec_digest),
            runtime=str(self.runtime),
            source_schema_hash=str(self.source_schema_hash),
            model_config_digest=str(self.model_config_digest),
            serving_build_digest=str(self.serving_build_digest),
            binding_layout_id=str(self.binding_layout_id),
            target_layout_hash=str(self.target_layout_hash),
            tensor_schema_hash=str(self.tensor_schema_hash),
        )
        proto.source.CopyFrom(self.source.to_proto())
        proto.source_reuse.CopyFrom(self.source_reuse.to_proto())
        proto.topology.CopyFrom(self.topology.to_proto())
        proto.member.CopyFrom(self.member.to_proto())
        if self.load_config_digest is not None:
            proto.load_config_digest = str(self.load_config_digest)
        for key, value in self.blob_refs.items():
            proto.blob_refs[str(key)].CopyFrom(value.to_proto())
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.ServingBindingResolvedSpecCacheEntry
    ) -> "ServingBindingResolvedSpecCacheEntry":
        return cls(
            schema_version=int(proto.schema_version),
            cache_key_digest=str(proto.cache_key_digest),
            spec_digest=str(proto.spec_digest),
            runtime=str(proto.runtime),
            source=ServingBindingSourceRef.from_proto(proto.source),
            source_reuse=ServingBindingSourceReuseDecision.from_proto(
                proto.source_reuse
            ),
            topology=ServingTopologyRef.from_proto(proto.topology),
            member=ServingBindingMemberRef.from_proto(proto.member),
            source_schema_hash=str(proto.source_schema_hash),
            model_config_digest=str(proto.model_config_digest),
            load_config_digest=(
                str(proto.load_config_digest)
                if proto.HasField("load_config_digest")
                else None
            ),
            serving_build_digest=str(proto.serving_build_digest),
            binding_layout_id=str(proto.binding_layout_id),
            target_layout_hash=str(proto.target_layout_hash),
            tensor_schema_hash=str(proto.tensor_schema_hash),
            blob_refs={
                str(key): BlobRef.from_proto(value)
                for key, value in proto.blob_refs.items()
            },
        )


class PrefetchRetentionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    expire_if_unacquired_after_ms: int | None = None
    idle_ttl_after_last_release_ms: int | None = None
    materialization_timeout_ms: int | None = None
    allow_acquire_after_creator_exit: bool = True

    @model_validator(mode="after")
    def _validate_policy(self) -> "PrefetchRetentionPolicy":
        for field_name in (
            "expire_if_unacquired_after_ms",
            "idle_ttl_after_last_release_ms",
            "materialization_timeout_ms",
        ):
            value = getattr(self, field_name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        return self

    def to_proto(self) -> operation_pb2.PrefetchRetentionPolicy:
        proto = operation_pb2.PrefetchRetentionPolicy(
            allow_acquire_after_creator_exit=bool(self.allow_acquire_after_creator_exit)
        )
        if self.expire_if_unacquired_after_ms is not None:
            proto.expire_if_unacquired_after_ms = int(
                self.expire_if_unacquired_after_ms
            )
        if self.idle_ttl_after_last_release_ms is not None:
            proto.idle_ttl_after_last_release_ms = int(
                self.idle_ttl_after_last_release_ms
            )
        if self.materialization_timeout_ms is not None:
            proto.materialization_timeout_ms = int(self.materialization_timeout_ms)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.PrefetchRetentionPolicy
    ) -> "PrefetchRetentionPolicy":
        return cls(
            expire_if_unacquired_after_ms=(
                int(proto.expire_if_unacquired_after_ms)
                if proto.HasField("expire_if_unacquired_after_ms")
                else None
            ),
            idle_ttl_after_last_release_ms=(
                int(proto.idle_ttl_after_last_release_ms)
                if proto.HasField("idle_ttl_after_last_release_ms")
                else None
            ),
            materialization_timeout_ms=(
                int(proto.materialization_timeout_ms)
                if proto.HasField("materialization_timeout_ms")
                else None
            ),
            allow_acquire_after_creator_exit=bool(
                proto.allow_acquire_after_creator_exit
            ),
        )


class BindingReservationCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    capability_id: str
    binding_value_ref: BindingValueRef
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member: ServingBindingMemberRef
    reservation_bytes: int
    scope_digest: str
    expires_at_ms: int | None = None

    @model_validator(mode="after")
    def _validate_capability(self) -> "BindingReservationCapability":
        for field_name in (
            "capability_id",
            "daemon_id",
            "daemon_session_id",
            "device_uuid",
            "scope_digest",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if int(self.reservation_bytes) < 0:
            raise ValueError("reservation_bytes must be non-negative")
        if self.expires_at_ms is not None and int(self.expires_at_ms) < 0:
            raise ValueError("expires_at_ms must be non-negative")
        return self

    def to_proto(self) -> operation_pb2.BindingReservationCapability:
        proto = operation_pb2.BindingReservationCapability(
            capability_id=str(self.capability_id),
            daemon_id=str(self.daemon_id),
            daemon_session_id=str(self.daemon_session_id),
            device_uuid=str(self.device_uuid),
            reservation_bytes=int(self.reservation_bytes),
            scope_digest=str(self.scope_digest),
        )
        proto.binding_value_ref.CopyFrom(self.binding_value_ref.to_proto())
        proto.member.CopyFrom(self.member.to_proto())
        if self.expires_at_ms is not None:
            proto.expires_at_ms = int(self.expires_at_ms)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.BindingReservationCapability
    ) -> "BindingReservationCapability":
        return cls(
            capability_id=str(proto.capability_id),
            binding_value_ref=BindingValueRef.from_proto(proto.binding_value_ref),
            daemon_id=str(proto.daemon_id),
            daemon_session_id=str(proto.daemon_session_id),
            device_uuid=str(proto.device_uuid),
            member=ServingBindingMemberRef.from_proto(proto.member),
            reservation_bytes=int(proto.reservation_bytes),
            scope_digest=str(proto.scope_digest),
            expires_at_ms=(
                int(proto.expires_at_ms) if proto.HasField("expires_at_ms") else None
            ),
        )


class GroupRealizationAcquireRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    version_set_id: str
    part_id: str
    staging_token: str
    wait_for_publish: bool = False
    wait_timeout_ms: int = 0

    @model_validator(mode="after")
    def _validate_ref(self) -> "GroupRealizationAcquireRef":
        for field_name in (
            "transaction_id",
            "version_set_id",
            "part_id",
            "staging_token",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if int(self.wait_timeout_ms) < 0:
            raise ValueError("wait_timeout_ms must be non-negative")
        return self

    def to_proto(self) -> store_daemon_pb2.GroupRealizationAcquireRef:
        return store_daemon_pb2.GroupRealizationAcquireRef(
            transaction_id=str(self.transaction_id),
            version_set_id=str(self.version_set_id),
            part_id=str(self.part_id),
            staging_token=str(self.staging_token),
            wait_for_publish=bool(self.wait_for_publish),
            wait_timeout_ms=int(self.wait_timeout_ms),
        )

    @classmethod
    def from_proto(
        cls, proto: store_daemon_pb2.GroupRealizationAcquireRef
    ) -> "GroupRealizationAcquireRef":
        return cls(
            transaction_id=str(proto.transaction_id),
            version_set_id=str(proto.version_set_id),
            part_id=str(proto.part_id),
            staging_token=str(proto.staging_token),
            wait_for_publish=bool(proto.wait_for_publish),
            wait_timeout_ms=int(proto.wait_timeout_ms),
        )


class PrefetchedServingBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    local_serving_ref: str | None = None
    binding_value_ref: BindingValueRef
    daemon_id: str
    daemon_session_id: str
    device_uuid: str
    member: ServingBindingMemberRef
    reservation_bytes: int
    reservation_capability: BindingReservationCapability
    readiness: ServingBindingReadiness
    verification_state: BindingValueVerificationState
    serving_artifact_id: str | None = None
    expires_at_ms: int | None = None
    staged_value: bool = False
    group_realization_acquire: GroupRealizationAcquireRef | None = None
    report: object | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _validate_result(self) -> "PrefetchedServingBinding":
        if self.local_serving_ref is not None and not self.local_serving_ref:
            raise ValueError("local_serving_ref must not be empty when provided")
        for field_name in ("daemon_id", "daemon_session_id", "device_uuid"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if int(self.reservation_bytes) < 0:
            raise ValueError("reservation_bytes must be non-negative")
        if self.reservation_capability.binding_value_ref != self.binding_value_ref:
            raise ValueError(
                "reservation_capability.binding_value_ref must match binding_value_ref"
            )
        if self.expires_at_ms is not None and int(self.expires_at_ms) < 0:
            raise ValueError("expires_at_ms must be non-negative")
        if self.staged_value and self.group_realization_acquire is None:
            raise ValueError(
                "group_realization_acquire must be provided for staged values"
            )
        return self

    def to_proto(self) -> operation_pb2.PrefetchServingBindingResult:
        proto = operation_pb2.PrefetchServingBindingResult(
            daemon_id=str(self.daemon_id),
            daemon_session_id=str(self.daemon_session_id),
            device_uuid=str(self.device_uuid),
            reservation_bytes=int(self.reservation_bytes),
            readiness=_SERVING_READINESS_TO_PROTO[self.readiness],
            verification_state=str(self.verification_state.value),
        )
        if self.local_serving_ref is not None:
            proto.local_serving_ref = str(self.local_serving_ref)
        proto.binding_value_ref.CopyFrom(self.binding_value_ref.to_proto())
        proto.member.CopyFrom(self.member.to_proto())
        proto.reservation_capability.CopyFrom(self.reservation_capability.to_proto())
        if self.serving_artifact_id is not None:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        if self.expires_at_ms is not None:
            proto.expires_at_ms = int(self.expires_at_ms)
        proto.staged_value = bool(self.staged_value)
        if self.group_realization_acquire is not None:
            proto.group_realization_transaction_id = (
                self.group_realization_acquire.transaction_id
            )
            proto.group_realization_version_set_id = (
                self.group_realization_acquire.version_set_id
            )
            proto.group_realization_part_id = self.group_realization_acquire.part_id
            proto.group_realization_staging_token = (
                self.group_realization_acquire.staging_token
            )
            proto.group_realization_wait_for_publish = (
                self.group_realization_acquire.wait_for_publish
            )
            proto.group_realization_wait_timeout_ms = (
                self.group_realization_acquire.wait_timeout_ms
            )
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.PrefetchServingBindingResult
    ) -> "PrefetchedServingBinding":
        readiness = _SERVING_READINESS_FROM_PROTO.get(int(proto.readiness))
        if readiness is None:
            raise ValueError("PrefetchServingBindingResult readiness is required")
        group_realization_acquire = None
        if bool(proto.staged_value):
            group_realization_acquire = GroupRealizationAcquireRef(
                transaction_id=str(proto.group_realization_transaction_id),
                version_set_id=str(proto.group_realization_version_set_id),
                part_id=str(proto.group_realization_part_id),
                staging_token=str(proto.group_realization_staging_token),
                wait_for_publish=(
                    bool(proto.group_realization_wait_for_publish)
                    if proto.HasField("group_realization_wait_for_publish")
                    else False
                ),
                wait_timeout_ms=(
                    int(proto.group_realization_wait_timeout_ms)
                    if proto.HasField("group_realization_wait_timeout_ms")
                    else 0
                ),
            )
        return cls(
            local_serving_ref=(
                str(proto.local_serving_ref)
                if proto.HasField("local_serving_ref")
                else None
            ),
            binding_value_ref=BindingValueRef.from_proto(proto.binding_value_ref),
            daemon_id=str(proto.daemon_id),
            daemon_session_id=str(proto.daemon_session_id),
            device_uuid=str(proto.device_uuid),
            member=ServingBindingMemberRef.from_proto(proto.member),
            reservation_bytes=int(proto.reservation_bytes),
            reservation_capability=BindingReservationCapability.from_proto(
                proto.reservation_capability
            ),
            readiness=readiness,
            verification_state=BindingValueVerificationState(
                str(proto.verification_state)
            ),
            serving_artifact_id=(
                str(proto.serving_artifact_id)
                if proto.HasField("serving_artifact_id")
                else None
            ),
            expires_at_ms=(
                int(proto.expires_at_ms) if proto.HasField("expires_at_ms") else None
            ),
            staged_value=bool(proto.staged_value),
            group_realization_acquire=group_realization_acquire,
        )


class PrefetchedServingBindingMemberFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    member: ServingBindingMemberRef
    code: str
    message: str
    phase: str | None = None
    cache_key_digest: str | None = None
    spec_digest: str | None = None

    @model_validator(mode="after")
    def _validate_failure(self) -> "PrefetchedServingBindingMemberFailure":
        if not self.code:
            raise ValueError("code must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")
        if self.phase is not None and not self.phase:
            raise ValueError("phase must not be empty when provided")
        if self.cache_key_digest is not None and not self.cache_key_digest:
            raise ValueError("cache_key_digest must not be empty when provided")
        if self.spec_digest is not None and not self.spec_digest:
            raise ValueError("spec_digest must not be empty when provided")
        return self

    def to_proto(self) -> operation_pb2.PrefetchServingBindingMemberFailure:
        proto = operation_pb2.PrefetchServingBindingMemberFailure(
            code=str(self.code),
            message=str(self.message),
        )
        proto.member.CopyFrom(self.member.to_proto())
        if self.phase is not None:
            proto.phase = str(self.phase)
        if self.cache_key_digest is not None:
            proto.cache_key_digest = str(self.cache_key_digest)
        if self.spec_digest is not None:
            proto.spec_digest = str(self.spec_digest)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.PrefetchServingBindingMemberFailure
    ) -> "PrefetchedServingBindingMemberFailure":
        return cls(
            member=ServingBindingMemberRef.from_proto(proto.member),
            code=str(proto.code),
            message=str(proto.message),
            phase=str(proto.phase) if proto.HasField("phase") else None,
            cache_key_digest=(
                str(proto.cache_key_digest)
                if proto.HasField("cache_key_digest")
                else None
            ),
            spec_digest=(
                str(proto.spec_digest) if proto.HasField("spec_digest") else None
            ),
        )


class PrefetchedServingBindingSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: str
    topology: ServingTopologyRef
    group_id: str
    members: tuple[PrefetchedServingBinding, ...]
    readiness: ServingBindingReadiness
    expires_at_ms: int | None = None
    member_failures: tuple[PrefetchedServingBindingMemberFailure, ...] = ()
    partial: bool = False
    report: object | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _validate_result_set(self) -> "PrefetchedServingBindingSet":
        if not self.runtime:
            raise ValueError("runtime must not be empty")
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if not self.members:
            raise ValueError("members must not be empty")
        if self.expires_at_ms is not None and int(self.expires_at_ms) < 0:
            raise ValueError("expires_at_ms must be non-negative")
        if self.partial and not self.member_failures:
            raise ValueError("partial serving binding set requires member_failures")
        success_member_ids = {member.member.member_id for member in self.members}
        failed_member_ids = {
            failure.member.member_id for failure in self.member_failures
        }
        overlap = success_member_ids & failed_member_ids
        if overlap:
            raise ValueError(
                "serving binding set member cannot be both success and failure"
            )
        return self

    def to_proto(self) -> operation_pb2.PrefetchServingBindingSetResult:
        proto = operation_pb2.PrefetchServingBindingSetResult(
            runtime=str(self.runtime),
            group_id=str(self.group_id),
            readiness=_SERVING_READINESS_TO_PROTO[self.readiness],
        )
        proto.topology.CopyFrom(self.topology.to_proto())
        proto.members.extend(member.to_proto() for member in self.members)
        proto.member_failures.extend(
            failure.to_proto() for failure in self.member_failures
        )
        proto.partial = bool(self.partial)
        if self.expires_at_ms is not None:
            proto.expires_at_ms = int(self.expires_at_ms)
        return proto

    @classmethod
    def from_proto(
        cls, proto: operation_pb2.PrefetchServingBindingSetResult
    ) -> "PrefetchedServingBindingSet":
        readiness = _SERVING_READINESS_FROM_PROTO.get(int(proto.readiness))
        if readiness is None:
            raise ValueError("PrefetchServingBindingSetResult readiness is required")
        return cls(
            runtime=str(proto.runtime),
            topology=ServingTopologyRef.from_proto(proto.topology),
            group_id=str(proto.group_id),
            members=tuple(
                PrefetchedServingBinding.from_proto(member) for member in proto.members
            ),
            readiness=readiness,
            expires_at_ms=(
                int(proto.expires_at_ms) if proto.HasField("expires_at_ms") else None
            ),
            member_failures=tuple(
                PrefetchedServingBindingMemberFailure.from_proto(failure)
                for failure in proto.member_failures
            ),
            partial=bool(proto.partial),
        )


class ServingPublicationSubject(BaseModel):
    model_config = ConfigDict(frozen=True)

    serving_artifact_id: str | None = None
    binding_value_ref: BindingValueRef | None = None

    @model_validator(mode="after")
    def _validate_subject(self) -> "ServingPublicationSubject":
        artifact_id = self.serving_artifact_id
        binding_value_ref = self.binding_value_ref
        if artifact_id is not None and not artifact_id:
            raise ValueError("serving_artifact_id must not be empty")
        if (artifact_id is None) == (binding_value_ref is None):
            raise ValueError(
                "ServingPublicationSubject requires exactly one of serving_artifact_id or binding_value_ref"
            )
        return self

    @property
    def is_artifact_subject(self) -> bool:
        return self.serving_artifact_id is not None

    @property
    def is_binding_subject(self) -> bool:
        return self.binding_value_ref is not None

    def require_serving_artifact_id(self) -> str:
        if self.serving_artifact_id is None:
            raise ValueError(
                "serving publication subject does not yet resolve to a serving_artifact_id"
            )
        return self.serving_artifact_id

    def require_binding_value_ref(self) -> BindingValueRef:
        if self.binding_value_ref is None:
            raise ValueError(
                "serving publication subject does not carry a binding_value_ref"
            )
        return self.binding_value_ref

    def to_proto(self) -> publication_pb2.ServingPublicationSubject:
        proto = publication_pb2.ServingPublicationSubject()
        if self.serving_artifact_id is not None:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        elif self.binding_value_ref is not None:
            proto.binding_value.CopyFrom(self.binding_value_ref.to_proto())
        return proto

    def to_store_proto(self) -> publication_pb2.ServingPublicationSubject:
        proto = publication_pb2.ServingPublicationSubject()
        if self.serving_artifact_id is not None:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        elif self.binding_value_ref is not None:
            proto.binding_value.CopyFrom(self.binding_value_ref.to_store_proto())
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: publication_pb2.ServingPublicationSubject
        | publication_pb2.ServingPublicationSubject,
    ) -> "ServingPublicationSubject":
        ref_case = proto.WhichOneof("ref")
        if ref_case == "serving_artifact_id":
            return cls(serving_artifact_id=str(proto.serving_artifact_id))
        if ref_case == "binding_value":
            return cls(
                binding_value_ref=BindingValueRef.from_proto(proto.binding_value)
            )
        raise ValueError("ServingPublicationSubject requires exactly one ref")


class RepresentationPublishContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    subject: ServingPublicationSubject
    serving_manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    serving_build_digest_version: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> "RepresentationPublishContract":
        parse_serving_manifest_ref(self.serving_manifest_ref)
        if not self.representation_contract_hash:
            raise ValueError("representation_contract_hash must not be empty")
        if not self.serving_build_digest:
            raise ValueError("serving_build_digest must not be empty")
        if (
            self.serving_build_digest_version is not None
            and not self.serving_build_digest_version
        ):
            raise ValueError("serving_build_digest_version must not be empty")
        return self

    @property
    def serving_artifact_id(self) -> str | None:
        return self.subject.serving_artifact_id

    @property
    def binding_value_ref(self) -> BindingValueRef | None:
        return self.subject.binding_value_ref

    def require_serving_artifact_id(self) -> str:
        return self.subject.require_serving_artifact_id()

    def to_proto(self) -> store_daemon_pb2.RepresentationPublishContract:
        proto = store_daemon_pb2.RepresentationPublishContract(
            serving_manifest_ref=str(self.serving_manifest_ref),
            representation_contract_hash=str(self.representation_contract_hash),
            serving_build_digest=str(self.serving_build_digest),
        )
        if self.serving_artifact_id is not None:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        proto.subject.CopyFrom(self.subject.to_store_proto())
        if self.serving_build_digest_version is not None:
            proto.serving_build_digest_version = str(self.serving_build_digest_version)
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.RepresentationPublishContract,
    ) -> "RepresentationPublishContract":
        if not proto.HasField("subject"):
            raise ValueError(
                "RepresentationPublishContract requires a serving publication subject"
            )
        subject = ServingPublicationSubject.from_proto(proto.subject)
        return cls(
            subject=subject,
            serving_manifest_ref=str(proto.serving_manifest_ref),
            representation_contract_hash=str(proto.representation_contract_hash),
            serving_build_digest=str(proto.serving_build_digest),
            serving_build_digest_version=(
                str(proto.serving_build_digest_version or "") or None
            ),
        )

    def validate_against_manifest(
        self,
        manifest: ServingArtifactManifest,
    ) -> None:
        if manifest.serving_manifest_ref != self.serving_manifest_ref:
            raise ValueError(
                "RepresentationPublishContract.serving_manifest_ref does not match the serving manifest"
            )
        if manifest.representation_contract_hash != self.representation_contract_hash:
            raise ValueError(
                "RepresentationPublishContract.representation_contract_hash does not match the serving manifest"
            )
        if manifest.serving_build_digest != self.serving_build_digest:
            raise ValueError(
                "RepresentationPublishContract.serving_build_digest does not match the serving manifest"
            )
        if (
            self.serving_build_digest_version is not None
            and manifest.serving_build_digest_version
            != self.serving_build_digest_version
        ):
            raise ValueError(
                "RepresentationPublishContract.serving_build_digest_version does not match the serving manifest"
            )

    def to_runtime_policy(
        self,
        *,
        require_manifest: bool = True,
    ) -> ServingRuntimePolicy:
        serving_artifact_id = self.serving_artifact_id
        if serving_artifact_id is None:
            raise ValueError(
                "binding publication subjects do not resolve to a serving runtime policy until closeout promotion completes"
            )
        return ServingRuntimePolicy(
            require_manifest=bool(require_manifest),
            serving_manifest_ref=str(self.serving_manifest_ref),
            expected_representation_contract_hash=str(
                self.representation_contract_hash
            ),
            expected_serving_build_digest=str(self.serving_build_digest),
        )

    def to_publication_proto(self) -> publication_pb2.RepresentationPublishContract:
        proto = publication_pb2.RepresentationPublishContract(
            serving_manifest_ref=str(self.serving_manifest_ref),
            representation_contract_hash=str(self.representation_contract_hash),
            serving_build_digest=str(self.serving_build_digest),
        )
        if self.serving_artifact_id is not None:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        proto.subject.CopyFrom(self.subject.to_proto())
        if self.serving_build_digest_version is not None:
            proto.serving_build_digest_version = str(self.serving_build_digest_version)
        return proto

    @classmethod
    def from_publication_proto(
        cls,
        proto: publication_pb2.RepresentationPublishContract,
    ) -> "RepresentationPublishContract":
        if not proto.HasField("subject"):
            raise ValueError(
                "RepresentationPublishContract requires a serving publication subject"
            )
        subject = ServingPublicationSubject.from_proto(proto.subject)
        return cls(
            subject=subject,
            serving_manifest_ref=str(proto.serving_manifest_ref),
            representation_contract_hash=str(proto.representation_contract_hash),
            serving_build_digest=str(proto.serving_build_digest),
            serving_build_digest_version=(
                str(proto.serving_build_digest_version or "") or None
            ),
        )


class AssemblyCloseoutContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AssemblyCloseoutKind = "source_publish_only"
    closeout_contract_digest: str = ""
    source_version_key: str | None = None
    serving_version_key: str | None = None
    serving_artifact_id: str | None = None
    serving_manifest_ref: str | None = None
    representation_publish_contract: RepresentationPublishContract | None = None

    @model_validator(mode="after")
    def _validate_closeout_fields(self) -> "AssemblyCloseoutContract":
        if self.kind == "source_publish_only":
            if self.serving_version_key is not None:
                raise ValueError(
                    "source_publish_only closeout contracts may not set serving_version_key"
                )
            if self.serving_artifact_id is not None:
                raise ValueError(
                    "source_publish_only closeout contracts may not set serving_artifact_id"
                )
            if self.serving_manifest_ref is not None:
                raise ValueError(
                    "source_publish_only closeout contracts may not set serving_manifest_ref"
                )
            if self.representation_publish_contract is not None:
                raise ValueError(
                    "source_publish_only closeout contracts may not set representation_publish_contract"
                )
            return self

        if self.kind == "representation_publish":
            if self.representation_publish_contract is None:
                raise ValueError(
                    "representation_publish closeout contracts require representation_publish_contract"
                )
            if self.serving_artifact_id is not None:
                raise ValueError(
                    "representation_publish closeout contracts must not set serving_artifact_id; "
                    "use representation_publish_contract.subject"
                )
            if (
                self.serving_manifest_ref is not None
                and self.serving_manifest_ref
                != self.representation_publish_contract.serving_manifest_ref
            ):
                raise ValueError(
                    "serving_manifest_ref must match representation_publish_contract.serving_manifest_ref"
                )
        return self

    def to_proto(self) -> store_daemon_pb2.AssemblyCloseoutContract:
        proto = store_daemon_pb2.AssemblyCloseoutContract(
            kind=_ASSEMBLY_CLOSEOUT_KIND_TO_PROTO[self.kind],
            closeout_contract_digest=str(self.closeout_contract_digest),
        )
        if self.source_version_key:
            proto.source_version_key = str(self.source_version_key)
        if self.serving_version_key:
            proto.serving_version_key = str(self.serving_version_key)
        if self.serving_artifact_id:
            proto.serving_artifact_id = str(self.serving_artifact_id)
        if self.serving_manifest_ref:
            proto.serving_manifest_ref = str(self.serving_manifest_ref)
        if self.representation_publish_contract is not None:
            proto.representation_publish_contract.CopyFrom(
                self.representation_publish_contract.to_proto()
            )
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.AssemblyCloseoutContract,
    ) -> "AssemblyCloseoutContract":
        kind = "source_publish_only"
        if int(proto.kind) != int(store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED):
            kind = _ASSEMBLY_CLOSEOUT_KIND_FROM_PROTO[int(proto.kind)]
        representation_publish_contract = (
            RepresentationPublishContract.from_proto(
                proto.representation_publish_contract
            )
            if proto.HasField("representation_publish_contract")
            else None
        )
        return cls(
            kind=kind,
            closeout_contract_digest=str(proto.closeout_contract_digest),
            source_version_key=str(proto.source_version_key or "") or None,
            serving_version_key=str(proto.serving_version_key or "") or None,
            serving_artifact_id=(
                str(proto.serving_artifact_id or "") or None
                if not representation_publish_contract
                else None
            ),
            serving_manifest_ref=(
                str(proto.serving_manifest_ref or "") or None
                if not representation_publish_contract
                else representation_publish_contract.serving_manifest_ref
            ),
            representation_publish_contract=representation_publish_contract,
        )


class RepresentationPublishSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    serving_artifact_id: str | None = None
    serving_manifest_ref: str
    serving_manifest: ServingArtifactManifest
    serving_manifest_bytes: bytes
    canonical_index: object | None = None
    representation_publish_contract: RepresentationPublishContract
    closeout_contract: AssemblyCloseoutContract
    source_artifact_ref: str | None = None
    contract_family: AssemblyContractFamily | None = None
    structural_view_ids: tuple[str, ...] = ()
    layout_id: str | None = None
    requirements: AssemblyRequirementSetRef | None = None
    readiness_policy: AssemblyReadinessPolicy | None = None
    admission_facts: ServingAdmissionFacts | None = None

    @model_validator(mode="after")
    def _validate_representation_publish_spec(self) -> "RepresentationPublishSpec":
        parse_serving_manifest_ref(self.serving_manifest_ref)
        if self.contract_family is not None and self.contract_family not in {
            "pp",
            "ep",
            "canonical_full",
        }:
            raise ValueError("contract_family must be one of: pp, ep, canonical_full")
        manifest_from_bytes = ServingArtifactManifest.from_bytes(
            self.serving_manifest_bytes
        )
        if manifest_from_bytes != self.serving_manifest:
            raise ValueError(
                "serving_manifest_bytes must round-trip to serving_manifest"
            )
        self.representation_publish_contract.validate_against_manifest(
            self.serving_manifest
        )
        if self.serving_manifest_ref != self.serving_manifest.serving_manifest_ref:
            raise ValueError(
                "serving_manifest_ref must match serving_manifest.serving_manifest_ref"
            )
        if (
            self.serving_manifest_ref
            != self.representation_publish_contract.serving_manifest_ref
        ):
            raise ValueError(
                "serving_manifest_ref must match representation_publish_contract.serving_manifest_ref"
            )
        subject_artifact_id = self.representation_publish_contract.serving_artifact_id
        if subject_artifact_id is not None and self.serving_artifact_id not in {
            None,
            subject_artifact_id,
        }:
            raise ValueError(
                "serving_artifact_id must match representation_publish_contract.serving_artifact_id"
            )
        if subject_artifact_id is None and self.serving_artifact_id is not None:
            raise ValueError(
                "binding-native representation publish specs must not set serving_artifact_id before closeout promotion"
            )
        if (
            self.serving_manifest.builder_mode == BuilderMode.BINDING_FINALIZE
            and self.representation_publish_contract.binding_value_ref is None
        ):
            raise ValueError(
                "BINDING_FINALIZE representation publish requires a binding_value_ref subject"
            )
        if (
            self.serving_manifest.builder_mode == BuilderMode.BINDING_FINALIZE
            and self.admission_facts is None
        ):
            raise ValueError(
                "BINDING_FINALIZE representation publish requires admission_facts"
            )
        if self.closeout_contract.kind != "representation_publish":
            raise ValueError(
                "RepresentationPublishSpec.closeout_contract must use kind='representation_publish'"
            )
        if (
            self.closeout_contract.representation_publish_contract
            != self.representation_publish_contract
        ):
            raise ValueError(
                "RepresentationPublishSpec.closeout_contract must carry the same representation_publish_contract"
            )
        if self.admission_facts is not None:
            self.admission_facts.validate_for_representation_publish(
                builder_mode=self.serving_manifest.builder_mode
            )
            if self.closeout_contract.serving_version_key is not None:
                self.admission_facts.require_serving_key_activation_ready()
        return self

    @property
    def manifest_tensor_name(self) -> str:
        return parse_serving_manifest_ref(self.serving_manifest_ref)

    def require_serving_runtime_policy(
        self,
        *,
        require_manifest: bool = True,
    ) -> ServingRuntimePolicy:
        if self.admission_facts is not None:
            self.admission_facts.require_runtime_bind_swap_ready()
        return self.representation_publish_contract.to_runtime_policy(
            require_manifest=require_manifest
        )

    def to_proto(self) -> publication_pb2.RepresentationPublishSpec:
        proto = publication_pb2.RepresentationPublishSpec(
            serving_manifest_bytes=bytes(self.serving_manifest_bytes)
        )
        if self.layout_id is not None:
            proto.layout_id = str(self.layout_id)
        if self.requirements is not None:
            proto.requirements.CopyFrom(self.requirements.to_publication_proto())
        if self.readiness_policy is not None:
            proto.readiness_policy.CopyFrom(
                self.readiness_policy.to_publication_proto()
            )
        if self.closeout_contract.source_version_key is not None:
            proto.source_version_key = str(self.closeout_contract.source_version_key)
        if self.closeout_contract.serving_version_key is not None:
            proto.serving_version_key = str(self.closeout_contract.serving_version_key)
        proto.representation_publish_contract.CopyFrom(
            self.representation_publish_contract.to_publication_proto()
        )
        if self.source_artifact_ref is not None:
            proto.source_artifact_ref = str(self.source_artifact_ref)
        if self.contract_family is not None:
            proto.contract_family = str(self.contract_family)
        proto.structural_view_ids.extend(str(item) for item in self.structural_view_ids)
        if self.admission_facts is not None:
            proto.admission_facts.CopyFrom(self.admission_facts.to_publication_proto())
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: publication_pb2.RepresentationPublishSpec,
    ) -> "RepresentationPublishSpec":
        representation_publish_contract = (
            RepresentationPublishContract.from_publication_proto(
                proto.representation_publish_contract
            )
        )
        closeout_contract = AssemblyCloseoutContract(
            kind="representation_publish",
            source_version_key=str(proto.source_version_key or "") or None,
            serving_version_key=str(proto.serving_version_key or "") or None,
            serving_manifest_ref=representation_publish_contract.serving_manifest_ref,
            representation_publish_contract=representation_publish_contract,
        )
        manifest_bytes = bytes(proto.serving_manifest_bytes)
        manifest = ServingArtifactManifest.from_bytes(manifest_bytes)
        return cls(
            serving_artifact_id=representation_publish_contract.serving_artifact_id,
            serving_manifest_ref=representation_publish_contract.serving_manifest_ref,
            serving_manifest=manifest,
            serving_manifest_bytes=manifest_bytes,
            representation_publish_contract=representation_publish_contract,
            closeout_contract=closeout_contract,
            source_artifact_ref=str(proto.source_artifact_ref or "") or None,
            contract_family=cast(
                AssemblyContractFamily | None,
                str(proto.contract_family or "") or None,
            ),
            structural_view_ids=tuple(str(item) for item in proto.structural_view_ids),
            layout_id=str(proto.layout_id or "") or None,
            requirements=(
                AssemblyRequirementSetRef.from_publication_proto(proto.requirements)
                if proto.HasField("requirements")
                else None
            ),
            readiness_policy=(
                AssemblyReadinessPolicy.from_publication_proto(proto.readiness_policy)
                if proto.HasField("readiness_policy")
                else None
            ),
            admission_facts=(
                ServingAdmissionFacts.from_publication_proto(proto.admission_facts)
                if proto.HasField("admission_facts")
                else None
            ),
        )

    def with_attempt_inputs(
        self,
        *,
        layout_id: str | None = None,
        requirements: AssemblyRequirementSetRef | None = None,
        readiness_policy: AssemblyReadinessPolicy | None = None,
    ) -> "RepresentationPublishSpec":
        return self.model_copy(
            update={
                "layout_id": (
                    self.layout_id if layout_id is None else str(layout_id) or None
                ),
                "requirements": (
                    self.requirements if requirements is None else requirements
                ),
                "readiness_policy": (
                    self.readiness_policy
                    if readiness_policy is None
                    else readiness_policy
                ),
            }
        )


class PublicDiskSourceHandle(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    canonical_index_bytes: bytes
    artifact_id: str
    generation: int = 0
    verify_checksums: bool = True
    trusted_content_artifact_id: str | None = None
    source_index_bytes: bytes | None = None
    format_kind: MountedSourceFormatKind | None = None
    metadata_capability: MountedSourceMetadataCapability | None = None
    resolution_strategy: MountedSourceResolutionStrategy | None = None
    validation_mode: MountedSourceValidationMode | None = None
    policy_id: str | None = None
    exact_size_bytes: int = 0

    @model_validator(mode="after")
    def _validate_handle(self) -> "PublicDiskSourceHandle":
        if not self.path:
            raise ValueError("path must not be empty")
        if not self.canonical_index_bytes:
            raise ValueError("canonical_index_bytes must not be empty")
        if not self.artifact_id:
            raise ValueError("artifact_id must not be empty")
        if int(self.generation) < 0:
            raise ValueError("generation must be non-negative")
        if (
            self.trusted_content_artifact_id is not None
            and not self.trusted_content_artifact_id
        ):
            raise ValueError("trusted_content_artifact_id must not be empty")
        if self.policy_id is not None and not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if int(self.exact_size_bytes) < 0:
            raise ValueError("exact_size_bytes must be non-negative")
        return self

    def to_proto(self) -> store_daemon_pb2.PublicDiskSourceHandle:
        proto = store_daemon_pb2.PublicDiskSourceHandle(
            path=str(self.path),
            canonical_index_bytes=bytes(self.canonical_index_bytes),
            artifact_id=str(self.artifact_id),
            generation=int(self.generation),
            verify_checksums=bool(self.verify_checksums),
            exact_size_bytes=int(self.exact_size_bytes),
        )
        if self.trusted_content_artifact_id is not None:
            proto.trusted_content_artifact_id = str(self.trusted_content_artifact_id)
        if self.source_index_bytes is not None:
            proto.source_index_bytes = bytes(self.source_index_bytes)
        if self.format_kind is not None:
            proto.format_kind = self.format_kind.to_proto()
        if self.metadata_capability is not None:
            proto.metadata_capability = self.metadata_capability.to_proto()
        if self.resolution_strategy is not None:
            proto.resolution_strategy = self.resolution_strategy.to_proto()
        if self.validation_mode is not None:
            proto.validation_mode = self.validation_mode.to_proto()
        if self.policy_id is not None:
            proto.policy_id = str(self.policy_id)
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.PublicDiskSourceHandle,
    ) -> "PublicDiskSourceHandle":
        return cls(
            path=str(proto.path),
            canonical_index_bytes=bytes(proto.canonical_index_bytes),
            artifact_id=str(proto.artifact_id or ""),
            generation=int(proto.generation),
            verify_checksums=bool(proto.verify_checksums),
            trusted_content_artifact_id=(
                str(proto.trusted_content_artifact_id or "") or None
            ),
            source_index_bytes=(
                bytes(proto.source_index_bytes)
                if bytes(proto.source_index_bytes)
                else None
            ),
            format_kind=MountedSourceFormatKind.from_proto(int(proto.format_kind)),
            metadata_capability=MountedSourceMetadataCapability.from_proto(
                int(proto.metadata_capability)
            ),
            resolution_strategy=MountedSourceResolutionStrategy.from_proto(
                int(proto.resolution_strategy)
            ),
            validation_mode=MountedSourceValidationMode.from_proto(
                int(proto.validation_mode)
            ),
            policy_id=str(proto.policy_id or "") or None,
            exact_size_bytes=int(proto.exact_size_bytes),
        )


class AssemblyAttemptRef(BaseModel):
    """Durable assembly-attempt reference."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    attempt_id: str
    workspace_assembly_id: str
    layout_id: str
    attempt_intent_digest: str
    coordinator_generation: int = 0
    coordinator_operation: operation_pb2.OperationRef = Field(
        default_factory=operation_pb2.OperationRef
    )

    @property
    def coordinator_operation_id(self) -> str:
        return str(self.coordinator_operation.operation_id or "")


class PartialSealResult(BaseModel):
    """Accepted assembly contribution rooted in one open attempt."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    workspace_assembly_id: str
    slot_id: str | None = None
    binding_id: str
    binding_value_id: str
    contribution_kind: Literal["piece_partial", "canonical_full"]
    view_id: str | None = None
    coverage_plan_hash: str
    accepted: bool = True
    already_exists: bool = False


class PublishedModelVersion(BaseModel):
    """Published model-version lineage for an assembly attempt."""

    model_config = ConfigDict(frozen=True)

    assembly_id: str
    source_artifact_id: str
    source_descriptor: ArtifactDescriptor
    serving_artifact_id: str | None = None
    serving_descriptor: ArtifactDescriptor | None = None
    source_version_key: str | None = None
    serving_version_key: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    serving_manifest_ref: str | None = None
    serving_execution_diagnostics: ExecutionDiagnostics | None = None

    def require_serving_runtime_policy(self) -> ServingRuntimePolicy:
        if not self.serving_manifest_ref:
            raise ValueError(
                "PublishedModelVersion does not carry serving_manifest_ref"
            )
        if not self.representation_contract_hash:
            raise ValueError(
                "PublishedModelVersion does not carry representation_contract_hash"
            )
        if not self.serving_build_digest:
            raise ValueError(
                "PublishedModelVersion does not carry serving_build_digest"
            )
        return ServingRuntimePolicy(
            require_manifest=True,
            serving_manifest_ref=str(self.serving_manifest_ref),
            expected_representation_contract_hash=str(
                self.representation_contract_hash
            ),
            expected_serving_build_digest=str(self.serving_build_digest),
        )


ServingRuntimePolicyInput = Union[
    ServingRuntimePolicy,
    ServingArtifactManifest,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    PublishedModelVersion,
]


def coerce_serving_runtime_policy(
    value: ServingRuntimePolicyInput | None,
) -> ServingRuntimePolicy | None:
    if value is None:
        return None
    if isinstance(value, ServingRuntimePolicy):
        return value
    if isinstance(value, ServingArtifactManifest):
        return value.to_runtime_policy()
    if isinstance(value, RepresentationPublishContract):
        return value.to_runtime_policy()
    if isinstance(value, RepresentationPublishSpec):
        return value.require_serving_runtime_policy()
    if isinstance(value, PublishedModelVersion):
        return value.require_serving_runtime_policy()
    raise TypeError(
        "serving runtime policy requires ServingRuntimePolicy, ServingArtifactManifest, "
        "RepresentationPublishContract, RepresentationPublishSpec, or PublishedModelVersion"
    )


# ------------------------------ Plan models --------------------------------


class PlanBase(BaseModel):
    """Abstract base for plan variants with a typed applicator to the proto."""

    model_config = ConfigDict(frozen=True)

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        raise NotImplementedError


class CoalescedPlan(PlanBase):
    # Aliases supported for coalesced semantics
    kind: Literal["coalesced", "vram_coalesced"] = "coalesced"
    max_inflight_bytes: int = 512 * 1024 * 1024
    release_on_tensor_commit: bool = True

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        co = store_daemon_pb2.CoalescedOptions()
        co.max_inflight_bytes = int(self.max_inflight_bytes)
        co.release_on_tensor_commit = bool(self.release_on_tensor_commit)
        req.coalesced.CopyFrom(co)


# CPU plan removed


class LeasePlan(PlanBase):
    kind: Literal["lease", "vram_leased"] = "lease"
    min_tensor_bytes: int = 64 * 1024
    max_tensor_count: int = 8192
    lease_bytes_limit: int = 0
    in_place: bool = False

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        lo = store_daemon_pb2.LeaseOptions()
        lo.min_tensor_bytes = int(self.min_tensor_bytes)
        lo.max_tensor_count = int(self.max_tensor_count)
        lo.lease_bytes_limit = int(self.lease_bytes_limit)
        lo.in_place = bool(self.in_place)
        req.lease.CopyFrom(lo)


class StableDramPlan(PlanBase):
    kind: Literal["dram_stable"] = "dram_stable"
    stage_on_gpu: bool = False
    release_gpu_on_commit: bool = True

    def apply_to_begin_request(
        self, req: store_daemon_pb2.BeginRegisterArtifactRequest
    ) -> None:
        opts = store_daemon_pb2.StableDramOptions()
        opts.stage_on_gpu = bool(self.stage_on_gpu)
        opts.release_gpu_on_commit = bool(self.release_gpu_on_commit)
        req.stable_dram.CopyFrom(opts)


Plan = Union[CoalescedPlan, LeasePlan, StableDramPlan]

# ---------------------------- Segment feed model ---------------------------


class LeaseSegment(BaseModel):
    """A single lease segment exported from CUDA IPC and fed to the daemon.

    A LeaseSegment maps a physical storage window (referenced by storage_id) into
    the artifact's logical address space.
    """

    model_config = ConfigDict(frozen=True)

    storage_id: str
    storage_offset: int = 0
    artifact_offset: int
    length: int

    @model_validator(mode="after")
    def _validate_offsets(self) -> "LeaseSegment":
        if not self.storage_id:
            raise ValueError("LeaseSegment.storage_id must not be empty")
        if self.storage_offset < 0:
            raise ValueError("LeaseSegment.storage_offset must be non-negative")
        if self.artifact_offset < 0:
            raise ValueError("LeaseSegment.artifact_offset must be non-negative")
        if self.length <= 0:
            raise ValueError("LeaseSegment.length must be positive")
        return self


class RegisterStorage(BaseModel):
    """Deduplicated storage descriptor used during registration feeds."""

    model_config = ConfigDict(frozen=True)

    storage_id: str
    device_id: int
    cuda_ipc_handle: bytes | None = None
    storage_length: int
    vram_region_id: str | None = None
    mapping_base_offset: int = 0

    @model_validator(mode="after")
    def _validate_source(self) -> "RegisterStorage":
        has_handle = self.cuda_ipc_handle is not None
        has_region = self.vram_region_id is not None
        if has_handle == has_region:
            raise ValueError(
                "RegisterStorage requires exactly one of cuda_ipc_handle or vram_region_id"
            )
        if self.mapping_base_offset < 0:
            raise ValueError("RegisterStorage.mapping_base_offset must be non-negative")
        return self


class RegisterTensorAlias(BaseModel):
    """Logical tensor metadata referencing a deduplicated storage."""

    model_config = ConfigDict(frozen=True)

    name: str
    storage_id: str
    storage_offset: int
    logical_length: int
    shape: list[int]
    stride: list[int]
    dtype: str


class RegionMemoryKind(str, Enum):
    VRAM = "VRAM"
    HOST_SHARED = "HOST_SHARED"


class HostSharedRegionClass(str, Enum):
    SCRATCH = "SCRATCH"
    ALLOCATOR = "ALLOCATOR"


class LocalRegionHandle(BaseModel):
    """Registered local region descriptor returned by the daemon."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    memory_kind: RegionMemoryKind
    ttl_ms: int
    size_bytes: int
    device_id: int | None = None
    attach_token: bytes = b""
    daemon_managed: bool = False
    host_shared_region_class: HostSharedRegionClass | None = None
    expires_at: datetime | None = None


class VramRegionHandle(BaseModel):
    """Registered VRAM region descriptor returned by the daemon."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    ttl_ms: int
    expires_at: datetime | None = None


class HostSharedRegionAttachment(BaseModel):
    """Local memfd attachment returned for a daemon-managed HOST_SHARED region."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    size_bytes: int
    attach_token: bytes
    fd: int


class DeregisterArtifactOutcome(BaseModel):
    """Result of a deregister_artifact invocation."""

    model_config = ConfigDict(frozen=True)

    drained: bool
    removed: bool
    released_region_ids: tuple[str, ...] = ()
    message: str | None = None


__all__ = [
    "ServerConfig",
    "SourceBoundCapability",
    "BindingValueVerificationState",
    "BindingPromotionStatusState",
    "MountedSourceFormatKind",
    "MountedSourceMetadataCapability",
    "MountedSourceResolutionStrategy",
    "MountedSourceValidationMode",
    "CollectivePolicy",
    "CollectiveFailureClass",
    "HashBackend",
    "HashLocation",
    "IdentityMintStrategy",
    "ExecutionDiagnostics",
    "CoalescedHandshake",
    "LeaseHandshake",
    "StableDramHandshake",
    "Handshake",
    "BeginRegisterArtifactResult",
    "ArtifactDescriptor",
    "BindingValueRef",
    "ServingBindingReadiness",
    "ServingBindingSourceKind",
    "ServingBindingSourceReuseMode",
    "ServingTopologyRef",
    "ServingBindingMemberRef",
    "BlobRef",
    "ServingBindingSourceMemberRef",
    "ServingBindingSourceRef",
    "ServingBindingSourceReuseDecision",
    "plan_serving_binding_source_reuse",
    "ServingBindingResolvedLayout",
    "ServingBindingTarget",
    "ServingBindingSetTarget",
    "ServingBindingResolvedSpecCacheEntry",
    "PrefetchRetentionPolicy",
    "BindingReservationCapability",
    "GroupRealizationAcquireRef",
    "PrefetchedServingBinding",
    "PrefetchedServingBindingMemberFailure",
    "PrefetchedServingBindingSet",
    "BuilderMode",
    "ServingPublicationSubject",
    "AssemblyCloseoutContract",
    "AssemblyAttemptRef",
    "AssemblyContractFamily",
    "AssemblyReadinessPolicy",
    "AssemblyRequirement",
    "AssemblyRequirementSetRef",
    "AssemblyTargetRef",
    "FinalizeClass",
    "PartialSealResult",
    "CanonicalRange",
    "CommitResult",
    "PublishedModelVersion",
    "PureTransformPublicationSpec",
    "RepresentationPublishContract",
    "RepresentationPublishSpec",
    "PublicDiskSourceHandle",
    "ServingAdmissionFacts",
    "ViewRegistrationKind",
    "SealAssemblyResult",
    "ServingArtifactManifest",
    "ServingBuildIntent",
    "SERVING_BUILD_DIGEST_VERSION",
    "ServingRuntimePolicy",
    "ServingRuntimePolicyInput",
    "ServingSupportLevel",
    "SERVING_MANIFEST_TENSOR_NAME",
    "PlanBase",
    "CoalescedPlan",
    "LeasePlan",
    "StableDramPlan",
    "Plan",
    "LeaseSegment",
    "RegisterStorage",
    "RegisterTensorAlias",
    "RegionMemoryKind",
    "HostSharedRegionClass",
    "HostSharedRegionAttachment",
    "LocalRegionHandle",
    "VramRegionHandle",
    "DeregisterArtifactOutcome",
    "build_serving_manifest_ref",
    "coerce_serving_runtime_policy",
    "parse_serving_manifest_ref",
]
