#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import tensorcast as tc


def test_context_is_pure_data_container() -> None:
    ctx = tc.context(
        request_id="req-1",
        qos="background",
        deadline_ms=123,
        idempotency_key="idem-1",
        tags={"stage": "warm", "attempt": 2},
    )
    assert isinstance(ctx, tc.CallContext)
    assert ctx.request_id == "req-1"
    assert ctx.qos == "background"
    assert ctx.deadline_ms == 123
    assert ctx.idempotency_key == "idem-1"
    assert ctx.tags is not None and ctx.tags["stage"] == "warm"
    assert not hasattr(ctx, "artifact")

    with pytest.raises(FrozenInstanceError):
        ctx.request_id = "req-2"  # type: ignore[misc]


def test_handle_factories_are_context_free() -> None:
    sig = inspect.signature(tc.artifact)
    assert "ctx" not in sig.parameters

