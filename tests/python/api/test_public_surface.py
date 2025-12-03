#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import tensorcast as tc


def test_tensorcast_exports_artifact_helpers() -> None:
    assert hasattr(tc, "artifact")
    assert callable(tc.artifact)
    assert hasattr(tc, "artifact_async")
    assert callable(tc.artifact_async)

