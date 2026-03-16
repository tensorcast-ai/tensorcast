#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import uuid
from dataclasses import dataclass

from tensorcast.api.store.types import ArtifactError
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


@dataclass(frozen=True, slots=True)
class BindingValueMetadata:
    binding_id: str
    binding_layout_id: str
    binding_value_id: str
    seal_generation: int
    source_artifact_id: str | None
    selection: common_pb2.ArtifactSelection | None
    is_artifact_backed: bool


def clone_selection(
    selection: common_pb2.ArtifactSelection | None,
) -> common_pb2.ArtifactSelection | None:
    if selection is None:
        return None
    clone = common_pb2.ArtifactSelection()
    clone.CopyFrom(selection)
    return clone


def binding_value_from_proto(
    value: store_daemon_pb2.BindingValue | None,
    *,
    expected_binding_id: str | None = None,
    expected_binding_layout_id: str | None = None,
) -> BindingValueMetadata | None:
    if value is None:
        return None
    binding_id = str(getattr(value, "binding_id", "") or "")
    if not binding_id:
        raise ValueError("binding_id is required")
    if expected_binding_id is not None and binding_id != str(expected_binding_id):
        raise ValueError(f"binding_id mismatch ({binding_id} != {expected_binding_id})")
    binding_layout_id = str(getattr(value, "binding_layout_id", "") or "")
    if not binding_layout_id:
        raise ValueError("binding_layout_id is required")
    if expected_binding_layout_id is not None and binding_layout_id != str(
        expected_binding_layout_id
    ):
        raise ValueError(
            "binding_layout_id mismatch "
            f"({binding_layout_id} != {expected_binding_layout_id})"
        )
    binding_value_id = str(getattr(value, "binding_value_id", "") or "")
    if not binding_value_id:
        raise ValueError("binding_value_id is required")
    selection = None
    if value.HasField("selection"):
        selection = clone_selection(value.selection)
    source_artifact_id = None
    if value.HasField("source_artifact_id"):
        source_artifact_id = str(value.source_artifact_id)
    is_artifact_backed = bool(value.is_artifact_backed)
    if is_artifact_backed:
        if source_artifact_id is None:
            raise ValueError("artifact-backed values require source_artifact_id")
        if selection is None:
            raise ValueError("artifact-backed values require selection")
        if not str(selection.artifact_id or ""):
            raise ValueError("artifact-backed values require selection.artifact_id")
        if str(selection.artifact_id) != source_artifact_id:
            raise ValueError(
                "selection.artifact_id must match source_artifact_id "
                f"({selection.artifact_id} != {source_artifact_id})"
            )
    else:
        if source_artifact_id is not None:
            raise ValueError("local-only values must not carry source_artifact_id")
        if selection is not None:
            raise ValueError("local-only values must not carry selection")
    return BindingValueMetadata(
        binding_id=binding_id,
        binding_layout_id=binding_layout_id,
        binding_value_id=binding_value_id,
        seal_generation=int(value.seal_generation),
        source_artifact_id=source_artifact_id,
        selection=selection,
        is_artifact_backed=is_artifact_backed,
    )


def mint_artifact_backed_value(
    *,
    binding_id: str,
    binding_layout_id: str,
    source_artifact_id: str,
    selection: common_pb2.ArtifactSelection,
    seal_generation: int,
) -> BindingValueMetadata:
    return BindingValueMetadata(
        binding_id=str(binding_id),
        binding_layout_id=str(binding_layout_id),
        binding_value_id=uuid.uuid4().hex,
        seal_generation=int(seal_generation),
        source_artifact_id=str(source_artifact_id),
        selection=clone_selection(selection),
        is_artifact_backed=True,
    )


def mint_local_value(
    *,
    binding_id: str,
    binding_layout_id: str,
    seal_generation: int,
) -> BindingValueMetadata:
    return BindingValueMetadata(
        binding_id=str(binding_id),
        binding_layout_id=str(binding_layout_id),
        binding_value_id=uuid.uuid4().hex,
        seal_generation=int(seal_generation),
        source_artifact_id=None,
        selection=None,
        is_artifact_backed=False,
    )


def parse_binding_value_or_raise(
    value: store_daemon_pb2.BindingValue | None,
    *,
    rpc_name: str,
    expected_binding_id: str,
    expected_binding_layout_id: str,
) -> BindingValueMetadata | None:
    try:
        return binding_value_from_proto(
            value,
            expected_binding_id=expected_binding_id,
            expected_binding_layout_id=expected_binding_layout_id,
        )
    except ValueError as exc:
        raise ArtifactError(
            f"{rpc_name} returned malformed current_value: {exc}",
            status_code="DATA_LOSS",
            retryable=False,
        ) from exc


__all__ = [
    "BindingValueMetadata",
    "binding_value_from_proto",
    "clone_selection",
    "mint_artifact_backed_value",
    "mint_local_value",
    "parse_binding_value_or_raise",
]
