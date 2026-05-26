#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import threading
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, SupportsIndex, SupportsInt, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from tensorcast.api._errors import InvalidPlan
from tensorcast.api.context import CollectiveLoadGroup

if TYPE_CHECKING:
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

# Module-level constants
DEFAULT_ALIGN: int = 8
DEFAULT_PINNED_TIMEOUT_MS: int = 30000
DEFAULT_COLD_TTL_MS: int = 60_000
# Final model: no default CPU stream chunking heuristic


# Global daemon address configuration
_daemon_address_lock = threading.RLock()
_global_daemon_address: str | None = None


def set_daemon_address(address: str) -> None:
    with _daemon_address_lock:
        global _global_daemon_address
        _global_daemon_address = address


def get_daemon_address() -> str:
    with _daemon_address_lock:
        if _global_daemon_address is None:
            raise RuntimeError(
                "TensorCast runtime is not initialized. "
                "Call tensorcast.startup.init(mode='connect'|'create'|'auto') first."
            )
        return _global_daemon_address


def clear_daemon_address() -> None:
    with _daemon_address_lock:
        global _global_daemon_address
        _global_daemon_address = None


def has_daemon_address() -> bool:
    with _daemon_address_lock:
        return _global_daemon_address is not None


class PlanType(Enum):
    DRAM_STABLE = "dram_stable"
    VRAM_COALESCED = "vram_coalesced"
    VRAM_LEASED = "vram_leased"

    @staticmethod
    def parse(value: object) -> "PlanType":
        if isinstance(value, PlanType):
            return value
        s = str(value).strip().lower()
        if s in ("dram_stable", "dram"):
            return PlanType.DRAM_STABLE
        if s in ("vram_coalesced", "coalesced", "copy"):
            return PlanType.VRAM_COALESCED
        if s in ("vram_leased", "lease"):
            return PlanType.VRAM_LEASED
        raise InvalidPlan(
            "Unknown plan "
            f"'{value}'; expected 'dram', 'lease', or 'copy' (or PlanType enum)."
        )


class StorePolicyProfile(Enum):
    CACHE = "cache"
    DURABLE = "durable"
    HA = "ha"
    COLD = "cold"
    WARM = "warm"
    PINNED = "pinned"

    @staticmethod
    def parse(value: object) -> "StorePolicyProfile":
        if isinstance(value, StorePolicyProfile):
            return value
        normalized = str(value).strip().lower()
        for item in StorePolicyProfile:
            if item.value == normalized:
                return item
        raise ValueError(
            "Unknown policy profile "
            f"'{value}'; expected cache, durable, ha, cold, warm, or pinned."
        )


class PolicyTier(Enum):
    STABLE_DRAM = "stable_dram"
    SHARED_DISK = "shared_disk"

    @staticmethod
    def parse(value: object) -> "PolicyTier":
        if isinstance(value, PolicyTier):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"stable_dram", "stable", "dram"}:
            return PolicyTier.STABLE_DRAM
        if normalized in {"shared_disk", "disk"}:
            return PolicyTier.SHARED_DISK
        raise ValueError(
            f"Unknown tier '{value}'; expected stable_dram or shared_disk."
        )


class PolicyScope(Enum):
    LOCAL = "local"
    REMOTE = "remote"
    ANY = "any"

    @staticmethod
    def parse(value: object) -> "PolicyScope":
        if isinstance(value, PolicyScope):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"local"}:
            return PolicyScope.LOCAL
        if normalized in {"remote"}:
            return PolicyScope.REMOTE
        if normalized in {"any", "either"}:
            return PolicyScope.ANY
        raise ValueError(f"Unknown scope '{value}'; expected local, remote, or any.")


class RetentionPolicy(Enum):
    BEST_EFFORT = "best_effort"
    TTL = "ttl"
    PINNED = "pinned"

    @staticmethod
    def parse(value: object) -> "RetentionPolicy":
        if isinstance(value, RetentionPolicy):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"best_effort", "best-effort", "best"}:
            return RetentionPolicy.BEST_EFFORT
        if normalized in {"ttl"}:
            return RetentionPolicy.TTL
        if normalized in {"pinned", "pin"}:
            return RetentionPolicy.PINNED
        raise ValueError(
            f"Unknown retention_policy '{value}'; expected best_effort, ttl, or pinned."
        )


class OverflowPolicy(Enum):
    EVICT = "evict"
    SPILL = "spill"
    REJECT = "reject"

    @staticmethod
    def parse(value: object) -> "OverflowPolicy":
        if isinstance(value, OverflowPolicy):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"evict"}:
            return OverflowPolicy.EVICT
        if normalized in {"spill"}:
            return OverflowPolicy.SPILL
        if normalized in {"reject"}:
            return OverflowPolicy.REJECT
        raise ValueError(
            f"Unknown overflow_policy '{value}'; expected evict, spill, or reject."
        )


class PolicyLayout(Enum):
    AUTO = "auto"
    UNSHARDED = "unsharded"
    SHARDED = "sharded"

    @staticmethod
    def parse(value: object) -> "PolicyLayout":
        if isinstance(value, PolicyLayout):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"auto"}:
            return PolicyLayout.AUTO
        if normalized in {"unsharded", "unshard"}:
            return PolicyLayout.UNSHARDED
        if normalized in {"sharded", "shard"}:
            return PolicyLayout.SHARDED
        raise ValueError(
            f"Unknown layout '{value}'; expected auto, unsharded, or sharded."
        )


class TierSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: PolicyTier
    scope: PolicyScope = PolicyScope.ANY
    min_replicas: int = 1
    retention_policy: RetentionPolicy = RetentionPolicy.BEST_EFFORT
    retention_ttl_ms: int | None = None

    @field_validator("tier", mode="before")
    @classmethod
    def _normalize_tier(cls, value: object) -> PolicyTier:
        return PolicyTier.parse(value)

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope(cls, value: object) -> PolicyScope:
        return PolicyScope.parse(value)

    @field_validator("retention_policy", mode="before")
    @classmethod
    def _normalize_retention(cls, value: object) -> RetentionPolicy:
        return RetentionPolicy.parse(value)

    @field_validator("min_replicas", mode="before")
    @classmethod
    def _normalize_min_replicas(cls, value: object) -> int:
        if value is None:
            return 1
        count = int(cast(SupportsInt | SupportsIndex | str | bytes | bytearray, value))
        if count <= 0:
            raise ValueError("min_replicas must be >= 1")
        return count

    @field_validator("retention_ttl_ms", mode="before")
    @classmethod
    def _normalize_ttl(cls, value: object) -> int | None:
        if value is None:
            return None
        ttl = int(cast(SupportsInt | SupportsIndex | str | bytes | bytearray, value))
        if ttl <= 0:
            raise ValueError("retention_ttl_ms must be > 0")
        return ttl

    @model_validator(mode="after")
    def _validate_constraints(self) -> "TierSpec":
        if self.tier is PolicyTier.SHARED_DISK:
            if self.scope is not PolicyScope.ANY:
                raise ValueError("shared_disk scope must be any")
            if self.min_replicas != 1:
                raise ValueError("shared_disk min_replicas must be 1")
            if self.retention_policy is not RetentionPolicy.BEST_EFFORT:
                raise ValueError("shared_disk does not support retention_policy")
            if self.retention_ttl_ms is not None:
                raise ValueError("shared_disk does not support retention_ttl_ms")
        if self.tier is PolicyTier.STABLE_DRAM:
            if self.min_replicas != 1:
                raise ValueError("stable_dram min_replicas must be 1")
            if self.scope is PolicyScope.REMOTE:
                if self.retention_policy is not RetentionPolicy.BEST_EFFORT:
                    raise ValueError(
                        "retention_policy is only valid for local stable_dram"
                    )
                if self.retention_ttl_ms is not None:
                    raise ValueError(
                        "retention_ttl_ms is only valid for local stable_dram"
                    )
        if (
            self.retention_policy is RetentionPolicy.TTL
            and self.retention_ttl_ms is None
        ):
            raise ValueError("retention_policy=ttl requires retention_ttl_ms")
        if (
            self.retention_policy is not RetentionPolicy.TTL
            and self.retention_ttl_ms is not None
        ):
            raise ValueError("retention_ttl_ms is only valid when retention_policy=ttl")
        return self


def _policy_proto_maps() -> tuple[
    dict[StorePolicyProfile, "store_daemon_pb2.PolicyProfile"],
    dict[PolicyTier, "store_daemon_pb2.PolicyTier"],
    dict[PolicyScope, "store_daemon_pb2.PolicyScope"],
    dict[RetentionPolicy, "store_daemon_pb2.RetentionPolicy"],
    dict[OverflowPolicy, "store_daemon_pb2.OverflowPolicy"],
    dict[PolicyLayout, "store_daemon_pb2.PolicyLayout"],
]:
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    return (
        {
            StorePolicyProfile.CACHE: store_daemon_pb2.POLICY_PROFILE_CACHE,
            StorePolicyProfile.DURABLE: store_daemon_pb2.POLICY_PROFILE_DURABLE,
            StorePolicyProfile.HA: store_daemon_pb2.POLICY_PROFILE_HA,
            StorePolicyProfile.COLD: store_daemon_pb2.POLICY_PROFILE_COLD,
            StorePolicyProfile.WARM: store_daemon_pb2.POLICY_PROFILE_WARM,
            StorePolicyProfile.PINNED: store_daemon_pb2.POLICY_PROFILE_PINNED,
        },
        {
            PolicyTier.STABLE_DRAM: store_daemon_pb2.POLICY_TIER_STABLE_DRAM,
            PolicyTier.SHARED_DISK: store_daemon_pb2.POLICY_TIER_SHARED_DISK,
        },
        {
            PolicyScope.LOCAL: store_daemon_pb2.POLICY_SCOPE_LOCAL,
            PolicyScope.REMOTE: store_daemon_pb2.POLICY_SCOPE_REMOTE,
            PolicyScope.ANY: store_daemon_pb2.POLICY_SCOPE_ANY,
        },
        {
            RetentionPolicy.BEST_EFFORT: store_daemon_pb2.RETENTION_POLICY_BEST_EFFORT,
            RetentionPolicy.TTL: store_daemon_pb2.RETENTION_POLICY_TTL,
            RetentionPolicy.PINNED: store_daemon_pb2.RETENTION_POLICY_PINNED,
        },
        {
            OverflowPolicy.EVICT: store_daemon_pb2.OVERFLOW_POLICY_EVICT,
            OverflowPolicy.SPILL: store_daemon_pb2.OVERFLOW_POLICY_SPILL,
            OverflowPolicy.REJECT: store_daemon_pb2.OVERFLOW_POLICY_REJECT,
        },
        {
            PolicyLayout.AUTO: store_daemon_pb2.POLICY_LAYOUT_AUTO,
            PolicyLayout.UNSHARDED: store_daemon_pb2.POLICY_LAYOUT_UNSHARDED,
            PolicyLayout.SHARDED: store_daemon_pb2.POLICY_LAYOUT_SHARDED,
        },
    )


(
    _POLICY_PROFILE_TO_PROTO,
    _POLICY_TIER_TO_PROTO,
    _POLICY_SCOPE_TO_PROTO,
    _RETENTION_POLICY_TO_PROTO,
    _OVERFLOW_POLICY_TO_PROTO,
    _POLICY_LAYOUT_TO_PROTO,
) = _policy_proto_maps()


def _tier_spec_to_proto(tier: TierSpec) -> "store_daemon_pb2.TierSpec":
    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    spec = store_daemon_pb2.TierSpec(
        tier=_POLICY_TIER_TO_PROTO[tier.tier],
        scope=_POLICY_SCOPE_TO_PROTO[tier.scope],
        min_replicas=int(tier.min_replicas),
    )
    if tier.tier is PolicyTier.STABLE_DRAM:
        if tier.retention_policy is not RetentionPolicy.BEST_EFFORT:
            spec.retention_policy = _RETENTION_POLICY_TO_PROTO[tier.retention_policy]
        if tier.retention_ttl_ms is not None:
            spec.retention_policy = _RETENTION_POLICY_TO_PROTO[tier.retention_policy]
            spec.retention_ttl_ms = int(tier.retention_ttl_ms)
    return spec


class StorePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: StorePolicyProfile | None = None
    must: tuple[TierSpec, ...] = ()
    should: tuple[TierSpec, ...] = ()
    may: tuple[TierSpec, ...] = ()
    overflow_policy: OverflowPolicy = OverflowPolicy.EVICT
    layout: PolicyLayout = PolicyLayout.AUTO

    @field_validator("profile", mode="before")
    @classmethod
    def _normalize_profile(cls, value: object) -> StorePolicyProfile | None:
        if value is None or value == "":
            return None
        return StorePolicyProfile.parse(value)

    @field_validator("overflow_policy", mode="before")
    @classmethod
    def _normalize_overflow(cls, value: object) -> OverflowPolicy:
        return OverflowPolicy.parse(value)

    @field_validator("layout", mode="before")
    @classmethod
    def _normalize_layout(cls, value: object) -> PolicyLayout:
        return PolicyLayout.parse(value)

    @model_validator(mode="after")
    def _validate_policy(self) -> "StorePolicy":
        has_tiers = bool(self.must or self.should or self.may)
        if self.profile is not None and has_tiers:
            raise ValueError("profile cannot be set when must/should/may are provided")
        if (
            self.overflow_policy is OverflowPolicy.SPILL
            and not self._has_shared_disk_tier()
        ):
            raise ValueError(
                "overflow_policy=spill requires shared_disk in must or should"
            )
        for tier in self.must:
            if (
                tier.tier is PolicyTier.STABLE_DRAM
                and tier.scope
                in {
                    PolicyScope.LOCAL,
                    PolicyScope.ANY,
                }
                and tier.retention_policy is not RetentionPolicy.PINNED
            ):
                raise ValueError(
                    "must local stable_dram requires retention_policy=pinned"
                )
        return self

    def _has_shared_disk_tier(self) -> bool:
        if self.profile in {
            StorePolicyProfile.DURABLE,
            StorePolicyProfile.HA,
            StorePolicyProfile.COLD,
        }:
            return True
        for tier in (*self.must, *self.should):
            if tier.tier is PolicyTier.SHARED_DISK:
                return True
        return False

    @staticmethod
    def from_profile(profile: StorePolicyProfile) -> "StorePolicy":
        if profile is StorePolicyProfile.CACHE:
            return StorePolicy(
                profile=None,
                may=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.BEST_EFFORT,
                    ),
                ),
                overflow_policy=OverflowPolicy.EVICT,
                layout=PolicyLayout.AUTO,
            )
        if profile is StorePolicyProfile.DURABLE:
            return StorePolicy(
                profile=None,
                must=(TierSpec(tier=PolicyTier.SHARED_DISK),),
                should=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.BEST_EFFORT,
                    ),
                ),
                overflow_policy=OverflowPolicy.EVICT,
                layout=PolicyLayout.AUTO,
            )
        if profile is StorePolicyProfile.HA:
            return StorePolicy(
                profile=None,
                must=(TierSpec(tier=PolicyTier.SHARED_DISK),),
                should=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.REMOTE,
                        min_replicas=1,
                    ),
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.BEST_EFFORT,
                    ),
                ),
                overflow_policy=OverflowPolicy.EVICT,
                layout=PolicyLayout.AUTO,
            )
        if profile is StorePolicyProfile.COLD:
            return StorePolicy(
                profile=None,
                must=(TierSpec(tier=PolicyTier.SHARED_DISK),),
                should=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.TTL,
                        retention_ttl_ms=DEFAULT_COLD_TTL_MS,
                    ),
                ),
                overflow_policy=OverflowPolicy.EVICT,
                layout=PolicyLayout.AUTO,
            )
        if profile is StorePolicyProfile.WARM:
            return StorePolicy(
                profile=None,
                should=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.BEST_EFFORT,
                    ),
                ),
                overflow_policy=OverflowPolicy.REJECT,
                layout=PolicyLayout.AUTO,
            )
        if profile is StorePolicyProfile.PINNED:
            return StorePolicy(
                profile=None,
                must=(
                    TierSpec(
                        tier=PolicyTier.STABLE_DRAM,
                        scope=PolicyScope.LOCAL,
                        retention_policy=RetentionPolicy.PINNED,
                    ),
                ),
                overflow_policy=OverflowPolicy.REJECT,
                layout=PolicyLayout.AUTO,
            )
        raise ValueError(f"Unknown profile {profile}")

    def expanded(self) -> "StorePolicy":
        if self.profile is None:
            return self
        return StorePolicy.from_profile(self.profile)

    @classmethod
    def parse(cls, value: object) -> "StorePolicy | None":
        if value is None:
            return None
        if isinstance(value, StorePolicy):
            return value
        if isinstance(value, str):
            if value == "":
                return StorePolicy(profile=None)
            return StorePolicy(profile=StorePolicyProfile.parse(value))
        return StorePolicy.model_validate(value)

    def to_proto(self) -> "store_daemon_pb2.StorePolicy":
        from tensorcast.proto.daemon.v2 import store_daemon_pb2

        policy = store_daemon_pb2.StorePolicy()
        if self.profile is not None:
            policy.profile = _POLICY_PROFILE_TO_PROTO[self.profile]
        policy.overflow_policy = _OVERFLOW_POLICY_TO_PROTO[self.overflow_policy]
        policy.layout = _POLICY_LAYOUT_TO_PROTO[self.layout]
        for tier in self.must:
            policy.must.add().CopyFrom(_tier_spec_to_proto(tier))
        for tier in self.should:
            policy.should.add().CopyFrom(_tier_spec_to_proto(tier))
        for tier in self.may:
            policy.may.add().CopyFrom(_tier_spec_to_proto(tier))
        return policy


def policy_requires_persistence(policy: StorePolicy | None) -> bool:
    resolved = StorePolicy.parse(policy) or StorePolicy(
        profile=StorePolicyProfile.CACHE
    )
    resolved = resolved.expanded()
    for tier in (*resolved.must, *resolved.should, *resolved.may):
        if tier.tier is PolicyTier.SHARED_DISK:
            return True
        if tier.tier is PolicyTier.STABLE_DRAM and tier.scope in {
            PolicyScope.REMOTE,
            PolicyScope.ANY,
        }:
            return True
    return False


class RegionBackedMode(Enum):
    AUTO = "auto"
    REQUIRE = "require"
    DISABLE = "disable"

    @staticmethod
    def parse(value: object) -> "RegionBackedMode":
        if isinstance(value, RegionBackedMode):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"auto", ""}:
            return RegionBackedMode.AUTO
        if normalized in {"require", "required"}:
            return RegionBackedMode.REQUIRE
        if normalized in {"disable", "disabled", "off", "false"}:
            return RegionBackedMode.DISABLE
        raise ValueError(
            f"Unknown region_backed_mode '{value}'; expected auto, require, or disable."
        )


class RetrievalPreference(str, Enum):
    AUTO = "auto"
    PREFER_P2P = "prefer_p2p"
    PREFER_DISK = "prefer_disk"

    @staticmethod
    def parse(value: object) -> "RetrievalPreference":
        if isinstance(value, RetrievalPreference):
            return value
        normalized = "auto" if value is None else str(value).strip().lower()
        if normalized == "auto":
            return RetrievalPreference.AUTO
        if normalized in {"prefer_p2p", "p2p_first"}:
            return RetrievalPreference.PREFER_P2P
        if normalized in {"prefer_disk", "disk_first"}:
            return RetrievalPreference.PREFER_DISK
        raise ValueError(
            "Unknown retrieval preference "
            f"'{value}'; expected auto, prefer_p2p, or prefer_disk."
        )


class RetrievalPreset(str, Enum):
    AUTO = "auto"
    LOCAL_ONLY = "local_only"
    DISK_FIRST = "disk_first"
    DISK_ONLY = "disk_only"

    @staticmethod
    def parse(value: object) -> "RetrievalPreset":
        if isinstance(value, RetrievalPreset):
            return value
        normalized = str(value).strip().lower()
        if normalized == "auto":
            return RetrievalPreset.AUTO
        if normalized == "local_only":
            return RetrievalPreset.LOCAL_ONLY
        if normalized == "disk_first":
            return RetrievalPreset.DISK_FIRST
        if normalized == "disk_only":
            return RetrievalPreset.DISK_ONLY
        raise ValueError(
            "Unknown retrieval preset "
            f"'{value}'; expected auto, local_only, disk_first, or disk_only."
        )


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    preference: RetrievalPreference = RetrievalPreference.AUTO
    allow_p2p: bool = True
    allow_disk: bool = True

    @field_validator("preference", mode="before")
    @classmethod
    def _normalize_preference(cls, value: object) -> RetrievalPreference:
        return RetrievalPreference.parse(value)

    @model_validator(mode="after")
    def _validate_policy(self) -> "RetrievalPolicy":
        if self.preference is RetrievalPreference.PREFER_P2P and not self.allow_p2p:
            raise ValueError(
                "RetrievalPolicy disallows P2P but preference=prefer_p2p was requested"
            )
        if self.preference is RetrievalPreference.PREFER_DISK and not self.allow_disk:
            raise ValueError(
                "RetrievalPolicy disallows disk but preference=prefer_disk was requested"
            )
        return self

    @classmethod
    def from_preset(cls, preset: RetrievalPreset) -> "RetrievalPolicy":
        if preset is RetrievalPreset.AUTO:
            return cls()
        if preset is RetrievalPreset.LOCAL_ONLY:
            return cls(
                preference=RetrievalPreference.AUTO,
                allow_p2p=False,
                allow_disk=False,
            )
        if preset is RetrievalPreset.DISK_FIRST:
            return cls(preference=RetrievalPreference.PREFER_DISK)
        if preset is RetrievalPreset.DISK_ONLY:
            return cls(
                preference=RetrievalPreference.PREFER_DISK,
                allow_p2p=False,
                allow_disk=True,
            )
        raise ValueError(f"Unsupported retrieval preset {preset!r}")

    @classmethod
    def parse(cls, value: object) -> "RetrievalPolicy | None":
        if value is None:
            return None
        if isinstance(value, RetrievalPolicy):
            return value
        if isinstance(value, RetrievalPreset):
            return cls.from_preset(value)
        if isinstance(value, str):
            return cls.from_preset(RetrievalPreset.parse(value))
        if isinstance(value, Mapping):
            return cls(**value)
        raise ValueError(
            "source must be a RetrievalPolicy, RetrievalPreset, or mapping"
        )


class SourceLocalityHint(str, Enum):
    AUTO = "auto"
    HOST_LOCAL = "host_local"
    SHARED_SOURCE = "shared_source"

    @staticmethod
    def parse(value: object) -> "SourceLocalityHint":
        if isinstance(value, SourceLocalityHint):
            return value
        normalized = "auto" if value is None else str(value).strip().lower()
        if normalized == "auto":
            return SourceLocalityHint.AUTO
        if normalized == "host_local":
            return SourceLocalityHint.HOST_LOCAL
        if normalized == "shared_source":
            return SourceLocalityHint.SHARED_SOURCE
        raise ValueError(
            "Unknown source_locality "
            f"'{value}'; expected auto, host_local, or shared_source."
        )


class CollectivePolicyMode(str, Enum):
    REQUIRE_COLLECTIVE = "require_collective"
    COLLECTIVE_FIRST = "collective_first"
    DISABLE_COLLECTIVE = "disable_collective"

    @staticmethod
    def parse(value: object) -> "CollectivePolicyMode":
        if isinstance(value, CollectivePolicyMode):
            return value
        if value is None or value == "":
            raise ValueError(
                "collective_policy must be explicit when parsed directly; "
                "leave it unset for request normalization to choose the default."
            )
        normalized = str(value).strip().lower()
        if normalized == "require_collective":
            return CollectivePolicyMode.REQUIRE_COLLECTIVE
        if normalized == "collective_first":
            return CollectivePolicyMode.COLLECTIVE_FIRST
        if normalized == "disable_collective":
            return CollectivePolicyMode.DISABLE_COLLECTIVE
        raise ValueError(
            "Unknown collective_policy "
            f"'{value}'; expected require_collective, "
            "collective_first, or disable_collective."
        )


class ExecutionTopologyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    collective_group: CollectiveLoadGroup | None = None
    collective_policy: CollectivePolicyMode | None = None
    source_locality: SourceLocalityHint = SourceLocalityHint.AUTO
    source_sharing_domain: str | None = None

    @field_validator("collective_policy", mode="before")
    @classmethod
    def _normalize_collective_policy(cls, value: object) -> CollectivePolicyMode | None:
        if value is None or value == "":
            return None
        return CollectivePolicyMode.parse(value)

    @field_validator("source_locality", mode="before")
    @classmethod
    def _normalize_source_locality(cls, value: object) -> SourceLocalityHint:
        return SourceLocalityHint.parse(value)


class RegisterArtifactOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: PlanType = PlanType.DRAM_STABLE
    policy: StorePolicy | None = None
    p2p_prefer: str = "vram"
    max_inflight_bytes: int = 512 * 1024 * 1024
    release_on_tensor_commit: bool = True
    min_tensor_bytes: int = 64 * 1024
    max_tensor_count: int = 8192
    lease_bytes_limit: int = 0
    # Lease/LIP specific: opt-in in-place mode for LIP flows.
    lease_in_place: bool = False
    # Stable DRAM options. Whole-replica GPU staging is disabled; keep
    # stage_on_gpu=false unless the backend is redesigned.
    stage_on_gpu: bool = False
    release_gpu_on_commit: bool = True
    # When stage_on_gpu=false, require the daemon to return a CPU memfd publish
    # handshake and fail fast instead of silently falling back to cpu_stream.
    require_cpu_memfd_publish: bool = True
    key: str | None = None
    disk_path: str | None = None

    @field_validator("plan", mode="before")
    @classmethod
    def _normalize_plan(cls, value: object) -> PlanType:
        try:
            return PlanType.parse(value)
        except InvalidPlan:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InvalidPlan(str(exc)) from exc

    @field_validator("policy", mode="before")
    @classmethod
    def _normalize_policy(cls, value: object) -> StorePolicy | None:
        if value is None or value == "":
            return None
        try:
            return StorePolicy.parse(value)
        except Exception as exc:  # noqa: BLE001
            raise InvalidPlan(str(exc)) from exc


class GetArtifactOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: RetrievalPolicy | None = None
    execution_topology: ExecutionTopologyContext | None = None
    replica_uuid: str | None = None
    verify_checksums: bool = True
    export_policy: str = "never"  # "never" | "auto" | "force"
    need_view_data_hash: bool = True
    pinned_allocation_timeout_ms: int = DEFAULT_PINNED_TIMEOUT_MS
    # When >0 and the initial retrieval fails, the daemon can wait for a managed
    # shared-disk location to become ready before retrying disk-only.
    wait_for_shared_disk_ms: int = 0
    wait_for_completion: bool = True
    enable_verification: bool = True
    region_backed_mode: RegionBackedMode = RegionBackedMode.AUTO
    # Hint for P2P transport lock TTL extension; forwarded by daemons
    transport_hold_ms: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, value: object) -> object:
        if isinstance(value, Mapping) and "prefer" in value:
            raise ValueError(
                "GetArtifactOptions.prefer has been removed; use source=..."
            )
        return value

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> RetrievalPolicy | None:
        return RetrievalPolicy.parse(value)

    @field_validator("export_policy", mode="before")
    @classmethod
    def _normalize_export_policy(cls, value: object) -> str:
        normalized = "never" if value is None else str(value).strip().lower()
        if normalized not in {"never", "auto", "force"}:
            raise ValueError(
                "GetArtifactOptions.export_policy must be one of: never, auto, force"
            )
        return normalized

    @field_validator("region_backed_mode", mode="before")
    @classmethod
    def _normalize_region_backed_mode(cls, value: object) -> RegionBackedMode:
        return RegionBackedMode.parse(value)

    @field_validator("wait_for_shared_disk_ms", mode="before")
    @classmethod
    def _normalize_wait_for_shared_disk_ms(cls, value: object) -> int:
        if value is None or value == "":
            return 0
        ms = int(cast(SupportsInt, value))
        if ms < 0:
            raise ValueError("GetArtifactOptions.wait_for_shared_disk_ms must be >= 0")
        return ms


__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_PINNED_TIMEOUT_MS",
    "DEFAULT_COLD_TTL_MS",
    "ExecutionTopologyContext",
    "GetArtifactOptions",
    "PlanType",
    "RegionBackedMode",
    "RegisterArtifactOptions",
    "RetrievalPolicy",
    "RetrievalPreference",
    "RetrievalPreset",
    "SourceLocalityHint",
    "StorePolicy",
    "StorePolicyProfile",
    "PolicyTier",
    "PolicyScope",
    "RetentionPolicy",
    "OverflowPolicy",
    "PolicyLayout",
    "TierSpec",
    "policy_requires_persistence",
    "clear_daemon_address",
    "get_daemon_address",
    "has_daemon_address",
    "set_daemon_address",
]
