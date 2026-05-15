#  Copyright (c) 2026, TensorCast Team.

"""Source-bound serving runtime contract readiness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tensorcast as tc
from tensorcast.types import SourceBoundCapability

MIN_SOURCE_BOUND_CONTRACT_VERSION = 4
SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4 = "collective_first_v4"
REQUIRED_SOURCE_BOUND_CAPABILITIES = (
    SourceBoundCapability.FIRST_CLASS_COLLECTIVE_INGRESS,
    SourceBoundCapability.TYPED_EXECUTION_DIAGNOSTICS,
    SourceBoundCapability.SINGLE_MINT_BINDING_CLOSEOUT,
)


@dataclass(frozen=True)
class SourceBoundContractState:
    server_config_present: bool
    source_bound_contract_version: int
    source_bound_capability_flags: int
    source_bound_capability_names: tuple[str, ...]
    source_bound_contract_ready: bool

    @classmethod
    def unavailable(cls) -> SourceBoundContractState:
        return cls(
            server_config_present=False,
            source_bound_contract_version=0,
            source_bound_capability_flags=0,
            source_bound_capability_names=(),
            source_bound_contract_ready=False,
        )

    @classmethod
    def from_server_config(
        cls,
        server_config: Any | None,
    ) -> SourceBoundContractState:
        if server_config is None:
            return cls.unavailable()
        flags = int(getattr(server_config, "source_bound_capability_flags", 0) or 0)
        version = int(getattr(server_config, "source_bound_contract_version", 0) or 0)
        capability_names = tuple(
            capability.name
            for capability in SourceBoundCapability
            if flags & int(capability)
        )
        contract_ready = version >= MIN_SOURCE_BOUND_CONTRACT_VERSION and all(
            flags & int(capability) for capability in REQUIRED_SOURCE_BOUND_CAPABILITIES
        )
        return cls(
            server_config_present=True,
            source_bound_contract_version=version,
            source_bound_capability_flags=flags,
            source_bound_capability_names=capability_names,
            source_bound_contract_ready=contract_ready,
        )


def read_source_bound_contract_state(
    *,
    store_fn: Callable[[], Any] | None = None,
) -> SourceBoundContractState:
    try:
        store = (store_fn or tc.store)()
        capabilities = store.capabilities
        server_config = getattr(capabilities, "server_config", None)
    except Exception:
        return SourceBoundContractState.unavailable()
    return SourceBoundContractState.from_server_config(server_config)


__all__ = [
    "MIN_SOURCE_BOUND_CONTRACT_VERSION",
    "REQUIRED_SOURCE_BOUND_CAPABILITIES",
    "SOURCE_BOUND_CONTRACT_PATH_COLLECTIVE_FIRST_V4",
    "SourceBoundContractState",
    "read_source_bound_contract_state",
]
