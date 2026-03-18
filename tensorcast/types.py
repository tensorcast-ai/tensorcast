#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tensorcast.common.identity import ArtifactIdKind
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2

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
AssemblyContributorLivenessMode = Literal[
    "require_live_until_cut",
    "allow_durable_occupancy",
]
AssemblyCloseoutKind = Literal[
    "source_publish_only",
    "representation_publish",
    "rollout_gated_publish",
]

_ASSEMBLY_TARGET_KIND_TO_PROTO: dict[AssemblyTargetKind, int] = {
    "structural_view": int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW),
    "canonical_layout": int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT),
}
_ASSEMBLY_TARGET_KIND_FROM_PROTO: dict[int, AssemblyTargetKind] = {
    int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW): "structural_view",
    int(store_daemon_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT): "canonical_layout",
}
_ASSEMBLY_LIVENESS_MODE_TO_PROTO: dict[AssemblyContributorLivenessMode, int] = {
    "require_live_until_cut": int(
        store_daemon_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ),
    "allow_durable_occupancy": int(
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
_ASSEMBLY_CLOSEOUT_KIND_TO_PROTO: dict[AssemblyCloseoutKind, int] = {
    "source_publish_only": int(
        store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY
    ),
    "representation_publish": int(
        store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH
    ),
    "rollout_gated_publish": int(
        store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH
    ),
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


class AssemblyCloseoutContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AssemblyCloseoutKind = "source_publish_only"
    closeout_contract_digest: str = ""
    source_version_key: str | None = None
    serving_version_key: str | None = None
    serving_artifact_id: str | None = None
    serving_manifest_ref: str | None = None

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
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.AssemblyCloseoutContract,
    ) -> "AssemblyCloseoutContract":
        kind = "source_publish_only"
        if int(proto.kind) != int(store_daemon_pb2.ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED):
            kind = _ASSEMBLY_CLOSEOUT_KIND_FROM_PROTO[int(proto.kind)]
        return cls(
            kind=kind,
            closeout_contract_digest=str(proto.closeout_contract_digest),
            source_version_key=str(proto.source_version_key or "") or None,
            serving_version_key=str(proto.serving_version_key or "") or None,
            serving_artifact_id=str(proto.serving_artifact_id or "") or None,
            serving_manifest_ref=str(proto.serving_manifest_ref or "") or None,
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
    serving_manifest_ref: str | None = None


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
    stage_on_gpu: bool = True
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


class VramRegionHandle(BaseModel):
    """Registered VRAM region descriptor returned by the daemon."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    ttl_ms: int
    expires_at: datetime | None = None


class DeregisterArtifactOutcome(BaseModel):
    """Result of a deregister_artifact invocation."""

    model_config = ConfigDict(frozen=True)

    drained: bool
    removed: bool
    released_region_ids: tuple[str, ...] = ()
    message: str | None = None


__all__ = [
    "ServerConfig",
    "CoalescedHandshake",
    "LeaseHandshake",
    "StableDramHandshake",
    "Handshake",
    "BeginRegisterArtifactResult",
    "ArtifactDescriptor",
    "AssemblyCloseoutContract",
    "AssemblyAttemptRef",
    "AssemblyReadinessPolicy",
    "AssemblyRequirement",
    "AssemblyRequirementSetRef",
    "AssemblyTargetRef",
    "PartialSealResult",
    "CanonicalRange",
    "CommitResult",
    "PublishedModelVersion",
    "ViewRegistrationKind",
    "SealAssemblyResult",
    "PlanBase",
    "CoalescedPlan",
    "LeasePlan",
    "StableDramPlan",
    "Plan",
    "LeaseSegment",
    "RegisterStorage",
    "RegisterTensorAlias",
    "VramRegionHandle",
    "DeregisterArtifactOutcome",
]
