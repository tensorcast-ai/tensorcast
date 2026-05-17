#  Copyright (c) 2026, TensorCast Team.

"""Common serving binding session state shell."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from tensorcast.serving.policy import ServingSelector
from tensorcast.types import BindingValueRef


class ServingBindingState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: str
    selector: ServingSelector | None = None
    serving_artifact_ref: str | None = None
    manifest_ref: str | None = None
    representation_contract_hash: str | None = None
    serving_build_digest: str | None = None
    binding_value_ref: BindingValueRef | None = None
    local_serving_ref: str | None = None
    readiness: str | None = None
    updated_at: str | None = None

    def to_response(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class ServingBindingSession:
    """Lifecycle shell for framework-owned serving bind/swap callables."""

    def __init__(self, state: ServingBindingState | None = None) -> None:
        self.state = state

    def bind(
        self,
        callback: Callable[[], ServingBindingState],
    ) -> ServingBindingState:
        state = callback()
        if not isinstance(state, ServingBindingState):
            raise TypeError(
                "ServingBindingSession.bind callback must return ServingBindingState"
            )
        self.state = state
        return state

    def swap(
        self,
        callback: Callable[[], ServingBindingState],
    ) -> ServingBindingState:
        state = callback()
        if not isinstance(state, ServingBindingState):
            raise TypeError(
                "ServingBindingSession.swap callback must return ServingBindingState"
            )
        self.state = state
        return state
