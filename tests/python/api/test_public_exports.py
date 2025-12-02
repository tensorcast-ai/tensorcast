#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations


def test_tensorcast_exposes_lazy_artifact_surface() -> None:
    import tensorcast

    assert hasattr(tensorcast, "artifact")
    assert hasattr(tensorcast, "artifact_async")
    assert hasattr(tensorcast, "from_disk")
    assert hasattr(tensorcast, "BatchContext")
    assert hasattr(tensorcast, "PrefetchTicket")
