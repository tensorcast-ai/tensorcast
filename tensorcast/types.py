#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from enum import Enum, IntFlag
from typing import Iterable, Literal, Union

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


class CollectivePolicy(str, Enum):
    REQUIRE_COLLECTIVE = "require_collective"
    ALLOW_NOT_ELIGIBLE_FALLBACK = "allow_not_eligible_fallback"
    DISABLE_COLLECTIVE = "disable_collective"


class CollectiveFailureClass(str, Enum):
    NOT_ELIGIBLE = "not_eligible"
    EXECUTION_FAILED = "execution_failed"


class HashLocation(str, Enum):
    NONE = "none"
    SEAL = "seal"
    BINDING_CLOSEOUT = "binding_closeout"


class IdentityMintStrategy(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    SEAL_MINT = "seal_mint"
    SEAL_REUSE = "seal_reuse"
    CLOSEOUT_MINT = "closeout_mint"


_COLLECTIVE_POLICY_TO_PROTO: dict[CollectivePolicy, int] = {
    CollectivePolicy.REQUIRE_COLLECTIVE: int(
        store_daemon_pb2.COLLECTIVE_POLICY_REQUIRE_COLLECTIVE
    ),
    CollectivePolicy.ALLOW_NOT_ELIGIBLE_FALLBACK: int(
        store_daemon_pb2.COLLECTIVE_POLICY_ALLOW_NOT_ELIGIBLE_FALLBACK
    ),
    CollectivePolicy.DISABLE_COLLECTIVE: int(
        store_daemon_pb2.COLLECTIVE_POLICY_DISABLE_COLLECTIVE
    ),
}
_COLLECTIVE_POLICY_FROM_PROTO: dict[int, CollectivePolicy] = {
    value: key for key, value in _COLLECTIVE_POLICY_TO_PROTO.items()
}
_COLLECTIVE_FAILURE_CLASS_TO_PROTO: dict[CollectiveFailureClass, int] = {
    CollectiveFailureClass.NOT_ELIGIBLE: int(
        store_daemon_pb2.COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE
    ),
    CollectiveFailureClass.EXECUTION_FAILED: int(
        store_daemon_pb2.COLLECTIVE_FAILURE_CLASS_EXECUTION_FAILED
    ),
}
_COLLECTIVE_FAILURE_CLASS_FROM_PROTO: dict[int, CollectiveFailureClass] = {
    value: key for key, value in _COLLECTIVE_FAILURE_CLASS_TO_PROTO.items()
}
_HASH_LOCATION_TO_PROTO: dict[HashLocation, int] = {
    HashLocation.NONE: int(store_daemon_pb2.HASH_LOCATION_NONE),
    HashLocation.SEAL: int(store_daemon_pb2.HASH_LOCATION_SEAL),
    HashLocation.BINDING_CLOSEOUT: int(store_daemon_pb2.HASH_LOCATION_BINDING_CLOSEOUT),
}
_HASH_LOCATION_FROM_PROTO: dict[int, HashLocation] = {
    value: key for key, value in _HASH_LOCATION_TO_PROTO.items()
}
_IDENTITY_MINT_STRATEGY_TO_PROTO: dict[IdentityMintStrategy, int] = {
    IdentityMintStrategy.NOT_APPLICABLE: int(
        store_daemon_pb2.IDENTITY_MINT_STRATEGY_NOT_APPLICABLE
    ),
    IdentityMintStrategy.SEAL_MINT: int(
        store_daemon_pb2.IDENTITY_MINT_STRATEGY_SEAL_MINT
    ),
    IdentityMintStrategy.SEAL_REUSE: int(
        store_daemon_pb2.IDENTITY_MINT_STRATEGY_SEAL_REUSE
    ),
    IdentityMintStrategy.CLOSEOUT_MINT: int(
        store_daemon_pb2.IDENTITY_MINT_STRATEGY_CLOSEOUT_MINT
    ),
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
    hash_rounds: int = 0
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
            hash_rounds=int(self.hash_rounds),
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
            hash_rounds=int(proto.hash_rounds),
            hash_location=_HASH_LOCATION_FROM_PROTO.get(
                int(proto.hash_location),
                HashLocation.NONE,
            ),
            identity_mint_strategy=_IDENTITY_MINT_STRATEGY_FROM_PROTO.get(
                int(proto.identity_mint_strategy),
                IdentityMintStrategy.NOT_APPLICABLE,
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


class RealizationProtocol(str, Enum):
    SAME_BINDING_FAST_PATH = "same_binding_fast_path"
    SCRATCH_THEN_COMMIT = "scratch_then_commit"


class FinalizeClass(str, Enum):
    RUNTIME_ONLY = "runtime_only"
    REPRESENTATION_CHANGING = "representation_changing"
    UNKNOWN_BLOCKED = "unknown_blocked"


class ServingSupportLevel(str, Enum):
    BLOCKED = "blocked"
    SOURCE_BIND_BOOTSTRAP_ONLY = "source_bind_bootstrap_only"
    BUILDER_PUBLICATION_READY = "builder_publication_ready"
    RUNTIME_BIND_SWAP_READY = "runtime_bind_swap_ready"


_PUBLICATION_BUILDER_MODE_TO_PROTO: dict[BuilderMode, int] = {
    BuilderMode.PURE_TRANSFORM: int(publication_pb2.BUILDER_MODE_PURE_TRANSFORM),
    BuilderMode.BINDING_FINALIZE: int(publication_pb2.BUILDER_MODE_BINDING_FINALIZE),
}
_PUBLICATION_BUILDER_MODE_FROM_PROTO: dict[int, BuilderMode] = {
    int(publication_pb2.BUILDER_MODE_PURE_TRANSFORM): BuilderMode.PURE_TRANSFORM,
    int(publication_pb2.BUILDER_MODE_BINDING_FINALIZE): BuilderMode.BINDING_FINALIZE,
}
_PUBLICATION_REALIZATION_PROTOCOL_TO_PROTO: dict[RealizationProtocol, int] = {
    RealizationProtocol.SAME_BINDING_FAST_PATH: int(
        publication_pb2.REALIZATION_PROTOCOL_SAME_BINDING_FAST_PATH
    ),
    RealizationProtocol.SCRATCH_THEN_COMMIT: int(
        publication_pb2.REALIZATION_PROTOCOL_SCRATCH_THEN_COMMIT
    ),
}
_PUBLICATION_REALIZATION_PROTOCOL_FROM_PROTO: dict[int, RealizationProtocol] = {
    int(publication_pb2.REALIZATION_PROTOCOL_SAME_BINDING_FAST_PATH): (
        RealizationProtocol.SAME_BINDING_FAST_PATH
    ),
    int(publication_pb2.REALIZATION_PROTOCOL_SCRATCH_THEN_COMMIT): (
        RealizationProtocol.SCRATCH_THEN_COMMIT
    ),
}
_PUBLICATION_FINALIZE_CLASS_TO_PROTO: dict[FinalizeClass, int] = {
    FinalizeClass.RUNTIME_ONLY: int(publication_pb2.FINALIZE_CLASS_RUNTIME_ONLY),
    FinalizeClass.REPRESENTATION_CHANGING: int(
        publication_pb2.FINALIZE_CLASS_REPRESENTATION_CHANGING
    ),
    FinalizeClass.UNKNOWN_BLOCKED: int(publication_pb2.FINALIZE_CLASS_UNKNOWN_BLOCKED),
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
_PUBLICATION_SERVING_SUPPORT_LEVEL_TO_PROTO: dict[ServingSupportLevel, int] = {
    ServingSupportLevel.BLOCKED: int(publication_pb2.SERVING_SUPPORT_LEVEL_BLOCKED),
    ServingSupportLevel.SOURCE_BIND_BOOTSTRAP_ONLY: int(
        publication_pb2.SERVING_SUPPORT_LEVEL_SOURCE_BIND_BOOTSTRAP_ONLY
    ),
    ServingSupportLevel.BUILDER_PUBLICATION_READY: int(
        publication_pb2.SERVING_SUPPORT_LEVEL_BUILDER_PUBLICATION_READY
    ),
    ServingSupportLevel.RUNTIME_BIND_SWAP_READY: int(
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
_PUBLICATION_ASSEMBLY_TARGET_KIND_TO_PROTO: dict[AssemblyTargetKind, int] = {
    "structural_view": int(publication_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW),
    "canonical_layout": int(publication_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT),
}
_PUBLICATION_ASSEMBLY_TARGET_KIND_FROM_PROTO: dict[int, AssemblyTargetKind] = {
    int(publication_pb2.ASSEMBLY_TARGET_KIND_STRUCTURAL_VIEW): "structural_view",
    int(publication_pb2.ASSEMBLY_TARGET_KIND_CANONICAL_LAYOUT): "canonical_layout",
}
_PUBLICATION_ASSEMBLY_LIVENESS_MODE_TO_PROTO: dict[
    AssemblyContributorLivenessMode, int
] = {
    "require_live_until_cut": int(
        publication_pb2.ASSEMBLY_CONTRIBUTOR_LIVENESS_MODE_REQUIRE_LIVE_UNTIL_CUT
    ),
    "allow_durable_occupancy": int(
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
            contract_family=str(proto.contract_family or "") or None,
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
    realization_protocol: RealizationProtocol
    support_level: ServingSupportLevel
    topology_admission_digest: str | None = None
    fast_path_validated: bool = False

    @field_validator("topology_admission_digest", mode="before")
    @classmethod
    def _empty_digest_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _validate_admission_facts(self) -> "ServingAdmissionFacts":
        if (
            self.realization_protocol == RealizationProtocol.SAME_BINDING_FAST_PATH
            and not self.fast_path_validated
        ):
            raise ValueError("same_binding_fast_path requires fast_path_validated=True")
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
            realization_protocol=_PUBLICATION_REALIZATION_PROTOCOL_TO_PROTO[
                self.realization_protocol
            ],
            support_level=_PUBLICATION_SERVING_SUPPORT_LEVEL_TO_PROTO[
                self.support_level
            ],
            fast_path_validated=bool(self.fast_path_validated),
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
        if int(proto.realization_protocol) == int(
            publication_pb2.REALIZATION_PROTOCOL_UNSPECIFIED
        ):
            raise ValueError(
                "ServingAdmissionFacts.realization_protocol must be specified"
            )
        if int(proto.support_level) == int(
            publication_pb2.SERVING_SUPPORT_LEVEL_UNSPECIFIED
        ):
            raise ValueError("ServingAdmissionFacts.support_level must be specified")
        return cls(
            finalize_class=_PUBLICATION_FINALIZE_CLASS_FROM_PROTO[
                int(proto.finalize_class)
            ],
            realization_protocol=_PUBLICATION_REALIZATION_PROTOCOL_FROM_PROTO[
                int(proto.realization_protocol)
            ],
            support_level=_PUBLICATION_SERVING_SUPPORT_LEVEL_FROM_PROTO[
                int(proto.support_level)
            ],
            topology_admission_digest=(
                str(proto.topology_admission_digest or "") or None
            ),
            fast_path_validated=bool(proto.fast_path_validated),
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
        )


class ServingRuntimePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    require_manifest: bool = True
    serving_manifest_ref: str | None = None
    expected_representation_contract_hash: str | None = None
    expected_serving_build_digest: str | None = None

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
        if int(self.seal_generation) <= 0:
            raise ValueError("seal_generation must be positive")
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
        proto: publication_pb2.BindingValueRef | store_daemon_pb2.BindingValueRef,
    ) -> "BindingValueRef":
        return cls(
            binding_id=str(proto.binding_id),
            binding_layout_id=str(proto.binding_layout_id),
            binding_value_id=str(proto.binding_value_id),
            seal_generation=int(proto.seal_generation),
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
            contract_family=str(proto.contract_family or "") or None,
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
    artifact_id: str | None = None
    generation: int = 0
    verify_checksums: bool = True

    @model_validator(mode="after")
    def _validate_handle(self) -> "PublicDiskSourceHandle":
        if not self.path:
            raise ValueError("path must not be empty")
        if not self.canonical_index_bytes:
            raise ValueError("canonical_index_bytes must not be empty")
        if self.artifact_id is not None and not self.artifact_id:
            raise ValueError("artifact_id must not be empty")
        if int(self.generation) < 0:
            raise ValueError("generation must be non-negative")
        return self

    def to_proto(self) -> store_daemon_pb2.PublicDiskSourceHandle:
        proto = store_daemon_pb2.PublicDiskSourceHandle(
            path=str(self.path),
            canonical_index_bytes=bytes(self.canonical_index_bytes),
            generation=int(self.generation),
            verify_checksums=bool(self.verify_checksums),
        )
        if self.artifact_id is not None:
            proto.artifact_id = str(self.artifact_id)
        return proto

    @classmethod
    def from_proto(
        cls,
        proto: store_daemon_pb2.PublicDiskSourceHandle,
    ) -> "PublicDiskSourceHandle":
        return cls(
            path=str(proto.path),
            canonical_index_bytes=bytes(proto.canonical_index_bytes),
            artifact_id=str(proto.artifact_id or "") or None,
            generation=int(proto.generation),
            verify_checksums=bool(proto.verify_checksums),
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
    "SourceBoundCapability",
    "CollectivePolicy",
    "CollectiveFailureClass",
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
    "RealizationProtocol",
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
    "VramRegionHandle",
    "DeregisterArtifactOutcome",
    "build_serving_manifest_ref",
    "coerce_serving_runtime_policy",
    "parse_serving_manifest_ref",
]
