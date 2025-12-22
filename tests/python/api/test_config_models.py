#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tensorcast.api._config import (
    GetArtifactOptions,
    PlacementPolicy,
    PlanType,
    RegisterArtifactOptions,
)
from tensorcast.api._errors import InvalidPlan
from tensorcast.api.store.types import FallbackOptions, StoreOptions


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


def test_register_options_placement_policy_parsing() -> None:
    default_opts = RegisterArtifactOptions()
    assert default_opts.placement_policy is PlacementPolicy.LOCAL_ONLY

    replicated = RegisterArtifactOptions(placement_policy="replicated")
    assert replicated.placement_policy is PlacementPolicy.REPLICATED

    with pytest.raises(InvalidPlan):
        RegisterArtifactOptions(placement_policy="unknown")


def test_get_options_validate_prefer_values() -> None:
    valid = GetArtifactOptions(prefer="p2p")
    assert valid.prefer == "p2p"

    with pytest.raises(ValidationError):
        GetArtifactOptions(prefer="bluetooth")


def test_store_options_parse_fallback_string() -> None:
    opts = StoreOptions(fallback="disk:/tmp/cache")
    assert isinstance(opts.fallback, FallbackOptions)
    assert opts.fallback.prefer == "disk"
    assert opts.fallback.disk_path == "/tmp/cache"
    # Ensure frozen model cannot be mutated
    with pytest.raises((TypeError, ValidationError)):
        opts.fallback.disk_path = "/other"


def test_fallback_parse_accepts_existing_instance() -> None:
    instance = FallbackOptions(prefer="local")
    parsed = FallbackOptions.parse(instance)
    assert parsed is instance
