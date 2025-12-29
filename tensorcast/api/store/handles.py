#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass

from tensorcast.api._register import RegistrationResult
from tensorcast.api.store.types import (
    CanonicalIndex,
    LeaseHandle,
    ReplicaInfo,
    TensorDict,
)
from tensorcast.types import LocalStableTierResult


@dataclass(frozen=True)
class RegisteredArtifact:
    artifact_id: str
    replica: ReplicaInfo
    canonical_index: CanonicalIndex
    lease: LeaseHandle | None
    state_dict: TensorDict | None = None
    registration_result: RegistrationResult | None = None
    persistence_task_id: str | None = None
    local_stable_tier: LocalStableTierResult | None = None


__all__ = ["RegisteredArtifact"]
