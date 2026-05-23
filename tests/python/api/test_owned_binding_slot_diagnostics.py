#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

from tensorcast.api.store.owned_binding_layout import BindingLayout
from tensorcast.api.store.owned_binding_slot import (
    _materialization_diagnostics_from_response,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_create_owned_binding_source_diagnostics_use_target_storage_extent():
    layout = BindingLayout(
        binding_layout_id="bl1:test",
        target_layout=store_daemon_pb2.TargetLayout(),
        target_index_bytes=(
            b'{"tensor":[0,4096,[1024],[1],"torch.float32",0]}'
        ),
    )

    diagnostics = _materialization_diagnostics_from_response(
        SimpleNamespace(
            source=store_daemon_pb2.MATERIALIZATION_SOURCE_P2P,
            selected_source_replica_id="replica-source-1",
            selected_source_transport_id="transport-1",
        ),
        layout=layout,
    )

    assert diagnostics == {
        "source": "p2p",
        "source_code": int(store_daemon_pb2.MATERIALIZATION_SOURCE_P2P),
        "total_bytes": 4096,
        "retry_attempts": 1,
        "retry_reason_buckets": {},
        "replica_id": "replica-source-1",
        "transport_id": "transport-1",
    }
