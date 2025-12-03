#  Copyright (c) 2025, TensorCast Team.

# Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tensorcast.api._config import GetArtifactOptions, RegisterArtifactOptions
from tensorcast.api._errors import InvalidPlan
from tensorcast.api.store.types import FallbackOptions, StoreOptions
from tensorcast.api._config import PlanType


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
    with pytest.raises(TypeError):
        opts.plan = PlanType.VRAM_LEASED


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
    with pytest.raises(TypeError):
        opts.fallback.disk_path = "/other"


def test_fallback_parse_accepts_existing_instance() -> None:
    instance = FallbackOptions(prefer="local")
    parsed = FallbackOptions.parse(instance)
    assert parsed is instance
