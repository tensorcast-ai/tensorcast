#  Copyright (c) 2025-2026, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tensorcast.api._config import (
    GetArtifactOptions,
    PlanType,
    PolicyScope,
    PolicyTier,
    RegisterArtifactOptions,
    RetentionPolicy,
    StorePolicy,
    StorePolicyProfile,
    TierSpec,
    OverflowPolicy,
)
from tensorcast.api._errors import InvalidPlan
from tensorcast.api.store.types import ArtifactError, FallbackOptions, StoreOptions


def test_register_options_coerce_plan_string() -> None:
    opts = RegisterArtifactOptions(plan="copy")
    assert opts.plan is PlanType.VRAM_COALESCED

    lease_opts = RegisterArtifactOptions(plan="lease")
    assert lease_opts.plan is PlanType.VRAM_LEASED


def test_register_options_reject_unknown_plan() -> None:
    with pytest.raises(InvalidPlan):
        RegisterArtifactOptions(plan="gpu-only")


def test_register_options_are_frozen() -> None:
    opts = RegisterArtifactOptions()
    assert opts.plan is PlanType.DRAM_STABLE
    with pytest.raises((TypeError, ValidationError)):
        opts.plan = PlanType.VRAM_LEASED


def test_register_options_policy_parsing() -> None:
    default_opts = RegisterArtifactOptions()
    assert default_opts.policy is None

    durable = RegisterArtifactOptions(policy="durable")
    assert durable.policy is not None
    assert durable.policy.profile is StorePolicyProfile.DURABLE

    with pytest.raises(InvalidPlan):
        RegisterArtifactOptions(policy="unknown")


def test_store_policy_rejects_profile_with_tiers() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            profile=StorePolicyProfile.CACHE,
            must=(TierSpec(tier=PolicyTier.SHARED_DISK),),
        )


def test_store_policy_requires_shared_disk_for_spill() -> None:
    with pytest.raises(ValueError):
        StorePolicy(overflow_policy=OverflowPolicy.SPILL)


def test_store_policy_shared_disk_constraints() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.SHARED_DISK,
                    scope=PolicyScope.LOCAL,
                ),
            )
        )
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.SHARED_DISK,
                    min_replicas=2,
                ),
            )
        )


def test_store_policy_shared_disk_rejects_retention() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.SHARED_DISK,
                    retention_policy=RetentionPolicy.PINNED,
                ),
            )
        )
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.SHARED_DISK,
                    retention_policy=RetentionPolicy.TTL,
                    retention_ttl_ms=5000,
                ),
            )
        )


def test_store_policy_stable_dram_rejects_remote_retention() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.STABLE_DRAM,
                    scope=PolicyScope.REMOTE,
                    retention_policy=RetentionPolicy.PINNED,
                ),
            )
        )
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.STABLE_DRAM,
                    scope=PolicyScope.REMOTE,
                    retention_policy=RetentionPolicy.TTL,
                    retention_ttl_ms=5000,
                ),
            )
        )


def test_store_policy_stable_dram_requires_single_replica() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.STABLE_DRAM,
                    scope=PolicyScope.LOCAL,
                    min_replicas=2,
                ),
            )
        )


def test_store_policy_must_local_stable_requires_pinned() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.STABLE_DRAM,
                    scope=PolicyScope.LOCAL,
                    retention_policy=RetentionPolicy.BEST_EFFORT,
                ),
            )
        )
    StorePolicy(
        must=(
            TierSpec(
                tier=PolicyTier.STABLE_DRAM,
                scope=PolicyScope.LOCAL,
                retention_policy=RetentionPolicy.PINNED,
            ),
        )
    )


def test_store_policy_ttl_requires_deadline() -> None:
    with pytest.raises(ValueError):
        StorePolicy(
            must=(
                TierSpec(
                    tier=PolicyTier.STABLE_DRAM,
                    scope=PolicyScope.LOCAL,
                    retention_policy=RetentionPolicy.TTL,
                ),
            )
        )


def test_store_policy_profile_warm_expands_to_local_should() -> None:
    policy = StorePolicy(profile="warm")
    assert policy.profile is StorePolicyProfile.WARM

    expanded = policy.expanded()
    assert expanded.profile is None
    assert expanded.overflow_policy is OverflowPolicy.REJECT
    assert expanded.should == (
        TierSpec(
            tier=PolicyTier.STABLE_DRAM,
            scope=PolicyScope.LOCAL,
            retention_policy=RetentionPolicy.BEST_EFFORT,
        ),
    )

    from tensorcast.proto.daemon.v2 import store_daemon_pb2

    proto = policy.to_proto()
    assert proto.profile == store_daemon_pb2.POLICY_PROFILE_WARM


def test_get_options_validate_prefer_values() -> None:
    valid = GetArtifactOptions(prefer="p2p")
    assert valid.prefer == "p2p"

    with pytest.raises(ValidationError):
        GetArtifactOptions(prefer="bluetooth")


def test_get_options_validate_wait_for_shared_disk_ms() -> None:
    assert GetArtifactOptions().wait_for_shared_disk_ms == 0
    assert GetArtifactOptions(wait_for_shared_disk_ms=123).wait_for_shared_disk_ms == 123
    assert GetArtifactOptions(wait_for_shared_disk_ms=None).wait_for_shared_disk_ms == 0

    with pytest.raises(ValidationError):
        GetArtifactOptions(wait_for_shared_disk_ms=-1)


def test_store_options_parse_fallback_string() -> None:
    opts = StoreOptions(fallback="disk")
    assert isinstance(opts.fallback, FallbackOptions)
    assert opts.fallback.prefer == "disk"
    with pytest.raises(ArtifactError):
        StoreOptions(fallback="disk:/tmp/cache")


def test_fallback_parse_accepts_existing_instance() -> None:
    instance = FallbackOptions(prefer="local")
    parsed = FallbackOptions.parse(instance)
    assert parsed is instance
