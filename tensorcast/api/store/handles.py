#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tensorcast.api._register import RegistrationResult
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    LeaseHandle,
    ReplicaInfo,
    TensorDict,
)
from tensorcast.types import LocalStableTierResult

if TYPE_CHECKING:
    from tensorcast.api.context import CallContext
    from tensorcast.api.operation import Operation
    from tensorcast.api.store.types import PersistenceStatusResult


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
    _daemon_endpoint: str | None = field(default=None, repr=False, compare=False)

    def persistence_operation(
        self, *, ctx: "CallContext | None" = None
    ) -> "Operation[PersistenceStatusResult]":
        if not self.persistence_task_id:
            raise ArtifactError(
                "persistence_task_id missing; persistence operation unavailable",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not self._daemon_endpoint:
            raise ArtifactError(
                "daemon endpoint missing; persistence operation unavailable",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        from tensorcast.api.store import Store

        store = Store(self._daemon_endpoint)
        return store.persistence_operation(task_id=self.persistence_task_id, ctx=ctx)


__all__ = ["RegisteredArtifact"]
