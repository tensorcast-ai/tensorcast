#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.api.store.binding_state import (
    binding_value_from_proto,
    parse_binding_value_or_raise,
)
from tensorcast.api.store.types import ArtifactError
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_binding_value_from_proto_validates_artifact_backed_identity() -> None:
    selection = common_pb2.ArtifactSelection(artifact_id="artifact-1")
    value = store_daemon_pb2.BindingValue(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=3,
        source_artifact_id="artifact-1",
        is_artifact_backed=True,
    )
    value.selection.CopyFrom(selection)

    metadata = binding_value_from_proto(
        value,
        expected_binding_id="binding-1",
        expected_binding_layout_id="layout-1",
    )

    assert metadata is not None
    assert metadata.binding_id == "binding-1"
    assert metadata.binding_layout_id == "layout-1"
    assert metadata.source_artifact_id == "artifact-1"
    assert metadata.is_artifact_backed is True


def test_binding_value_from_proto_rejects_binding_identity_mismatch() -> None:
    value = store_daemon_pb2.BindingValue(
        binding_id="binding-other",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
        is_artifact_backed=False,
    )

    with pytest.raises(ValueError, match="binding_id mismatch"):
        binding_value_from_proto(
            value,
            expected_binding_id="binding-1",
            expected_binding_layout_id="layout-1",
        )


def test_parse_binding_value_or_raise_maps_malformed_proto_to_data_loss() -> None:
    value = store_daemon_pb2.BindingValue(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
        is_artifact_backed=True,
    )

    with pytest.raises(ArtifactError) as excinfo:
        parse_binding_value_or_raise(
            value,
            rpc_name="SealBinding",
            expected_binding_id="binding-1",
            expected_binding_layout_id="layout-1",
        )

    assert excinfo.value.status_code == "DATA_LOSS"
