#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

from tensorcast.serving.runtime_contract import (
    SourceBoundContractState,
    read_source_bound_contract_state,
)
from tensorcast.types import SourceBoundCapability


def _flags() -> int:
    return (
        int(SourceBoundCapability.FIRST_CLASS_COLLECTIVE_INGRESS)
        | int(SourceBoundCapability.TYPED_EXECUTION_DIAGNOSTICS)
        | int(SourceBoundCapability.SINGLE_MINT_BINDING_CLOSEOUT)
    )


def test_source_bound_contract_state_requires_v4_and_capabilities() -> None:
    state = SourceBoundContractState.from_server_config(
        SimpleNamespace(
            source_bound_capability_flags=_flags(),
            source_bound_contract_version=4,
        )
    )

    assert state.server_config_present is True
    assert state.source_bound_contract_version == 4
    assert state.source_bound_capability_flags == _flags()
    assert state.source_bound_capability_names == (
        "FIRST_CLASS_COLLECTIVE_INGRESS",
        "TYPED_EXECUTION_DIAGNOSTICS",
        "SINGLE_MINT_BINDING_CLOSEOUT",
    )
    assert state.source_bound_contract_ready is True


def test_source_bound_contract_state_rejects_pre_v4() -> None:
    state = SourceBoundContractState.from_server_config(
        SimpleNamespace(
            source_bound_capability_flags=_flags(),
            source_bound_contract_version=3,
        )
    )

    assert state.server_config_present is True
    assert state.source_bound_contract_ready is False


def test_source_bound_contract_state_rejects_missing_capability() -> None:
    state = SourceBoundContractState.from_server_config(
        SimpleNamespace(
            source_bound_capability_flags=int(
                SourceBoundCapability.FIRST_CLASS_COLLECTIVE_INGRESS
            ),
            source_bound_contract_version=4,
        )
    )

    assert state.source_bound_capability_names == ("FIRST_CLASS_COLLECTIVE_INGRESS",)
    assert state.source_bound_contract_ready is False


def test_read_source_bound_contract_state_fails_closed() -> None:
    assert (
        read_source_bound_contract_state(
            store_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        == SourceBoundContractState.unavailable()
    )
