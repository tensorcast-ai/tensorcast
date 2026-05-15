#  Copyright (c) 2026, TensorCast Team.

"""Serving artifact runtime DTOs shared by framework integrations."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from tensorcast.serving.policy import ServingSelector


class PreparedServingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_ref: str
    serving_artifact_ref: str
    manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str
    family: str
    tensor_schema_hash: str
    selector: ServingSelector | None = None


class BootstrapSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_artifact_ref: str
    serving_artifact_ref: str
    manifest_ref: str
    representation_contract_hash: str
    serving_build_digest: str


class FamilyReadiness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    process_after_load_class: Any | None = None
    post_bind_finalize_class: Any | None = None
    support_level: Any | None = None
    publication_modes: tuple[str, ...] = ()
    runtime_bind_swap_allowed: bool = False
    notes: str = ""


class FrameworkAdapter(Protocol):
    def framework_name(self) -> str: ...

    def adapter_version(self) -> str: ...
