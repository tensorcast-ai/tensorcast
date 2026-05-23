#  Copyright (c) 2026, TensorCast Team.

"""Common serving binding session state shell."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from tensorcast.serving.policy import ServingArtifactLocator
from tensorcast.types import BindingValueRef


class ServingBindingState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: str
    artifact_locator: ServingArtifactLocator | None = None
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
