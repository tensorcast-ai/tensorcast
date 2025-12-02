#  Copyright (c) 2025, TensorCast Team.

from tensorcast.api.store.types import FallbackOptions


def test_fallback_options_compat_prefer_disk():
    opts = FallbackOptions(prefer="auto", prefer_disk=True, disk_path="/tmp/x")
    assert opts.prefer == "auto"
    assert opts.prefer_disk is True
    disk_opts = FallbackOptions.for_disk("/tmp/y")
    assert disk_opts.prefer == "disk"
    assert disk_opts.allow_p2p is False
    assert disk_opts.disk_path == "/tmp/y"


def test_fallback_options_local_only():
    opts = FallbackOptions.local_only()
    assert opts.prefer == "local"
    assert opts.allow_p2p is False
