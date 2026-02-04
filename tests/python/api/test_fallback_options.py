#  Copyright (c) 2025-2026, TensorCast Team.

import pytest

from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api.store.types import ArtifactError, FallbackOptions, StoreOptions


def test_fallback_options_compat_prefer_disk():
    opts = FallbackOptions(prefer="auto", prefer_disk=True, allow_disk=True)
    assert opts.prefer == "auto"
    assert opts.prefer_disk is True
    disk_opts = FallbackOptions.parse("disk")
    assert disk_opts is not None
    assert disk_opts.prefer == "disk"
    assert disk_opts.allow_disk is True


def test_fallback_options_local_only():
    opts = FallbackOptions.local_only()
    assert opts.prefer == "local"
    assert opts.allow_p2p is False
    assert opts.allow_disk is False


def test_fallback_options_parse_shortcuts():
    disk = FallbackOptions.parse("disk")
    assert disk is not None
    assert disk.prefer == "disk"
    assert disk.allow_disk is True

    local = FallbackOptions.parse("local")
    assert local is not None
    assert local.prefer == "local"
    assert local.allow_p2p is False
    assert local.allow_disk is False

    auto = FallbackOptions.parse("auto")
    assert auto is not None
    assert auto.prefer == "auto"


def test_fallback_options_parse_rejects_unknown():
    with pytest.raises(ArtifactError):
        FallbackOptions.parse("unknown-mode")
    with pytest.raises(ArtifactError):
        FallbackOptions.parse("disk:/tmp/x")


def test_store_options_coerces_fallback_strings():
    opts = StoreOptions(fallback="disk")
    assert isinstance(opts.fallback, FallbackOptions)
    assert opts.fallback.prefer == "disk"


def test_register_options_accepts_plan_strings():
    lease = RegisterArtifactOptions(plan="lease")
    assert lease.plan is PlanType.VRAM_LEASED
    copy = RegisterArtifactOptions(plan="copy")
    assert copy.plan is PlanType.VRAM_COALESCED
