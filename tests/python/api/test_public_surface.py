#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import inspect

import tensorcast as tc
from tensorcast.api.store import Store


def test_tensorcast_exports_artifact_helpers() -> None:
    assert hasattr(tc, "artifact")
    assert callable(tc.artifact)
    assert hasattr(tc, "artifact_async")
    assert callable(tc.artifact_async)


def test_tensorcast_exports_programmable_primitives() -> None:
    assert hasattr(tc, "context")
    assert callable(tc.context)
    assert hasattr(tc, "CallContext")
    assert hasattr(tc, "CollectiveLoadGroup")
    assert hasattr(tc, "Operation")
    assert hasattr(tc, "plan")
    assert callable(tc.plan)
    assert hasattr(tc, "Plan")


def test_store_register_and_put_accept_policy_argument() -> None:
    for func in (
        Store.register,
        Store.register_async,
        Store.put,
        Store.put_async,
        tc.register,
        tc.register_async,
        tc.put,
        tc.put_async,
    ):
        sig = inspect.signature(func)
        assert "policy" in sig.parameters
        assert sig.parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
