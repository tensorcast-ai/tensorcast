#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from collections.abc import Sequence

from tensorcast._c_ext import compute_view_index_bytes
from tensorcast.common.identity import is_byte_artifact_id, validate_byte_artifact_cgid
from tensorcast.common.selection_identity import (
    compute_byte_artifact_logical_layout_hash,
    compute_byte_artifact_selection_hash,
    compute_logical_layout_hash,
    compute_selection_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2


def _normalized_view_ops(
    view_spec: common_pb2.ViewSpec | None,
) -> dict[str, list[dict[str, int | str]]]:
    if view_spec is None or not view_spec.tensors:
        return {}
    normalized_ops: dict[str, list[dict[str, int | str]]] = {}
    for name, ops in view_spec.tensors.items():
        op_list: list[dict[str, int | str]] = []
        for op in ops.ops:
            if op.HasField("narrow"):
                op_list.append(
                    {
                        "type": "narrow",
                        "dim": int(op.narrow.dim),
                        "start": int(op.narrow.start),
                        "length": int(op.narrow.length),
                    }
                )
            elif op.HasField("transpose"):
                op_list.append(
                    {
                        "type": "transpose",
                        "dim0": int(op.transpose.dim0),
                        "dim1": int(op.transpose.dim1),
                    }
                )
        if op_list:
            normalized_ops[str(name)] = op_list
    return normalized_ops


def compute_selected_index_bytes(
    *,
    canonical_index_bytes: bytes,
    view_spec: common_pb2.ViewSpec | None,
    tensor_names: Sequence[str] | None,
) -> bytes:
    canonical_bytes = bytes(canonical_index_bytes or b"")
    if not canonical_bytes:
        raise ValueError(
            "canonical index bytes are required for selected index computation"
        )
    subset_names = tuple(str(name) for name in (tensor_names or ()))
    if len(set(subset_names)) != len(subset_names):
        raise ValueError("tensor_names must be unique")
    subset_payload = list(subset_names) if subset_names else None
    payload = compute_view_index_bytes(
        canonical_bytes,
        _normalized_view_ops(view_spec),
        subset_payload,
    )
    return bytes(payload["view_index_bytes"])


def _build_byte_artifact_selection(
    *,
    artifact_id: str,
    view_spec: common_pb2.ViewSpec | None,
    tensor_names: Sequence[str] | None,
    view_subset_hash: bytes | None,
    view_id: str | None,
) -> common_pb2.ArtifactSelection:
    if view_spec is not None and view_spec.tensors:
        raise ValueError("byte artifact selection does not support view transforms")
    if view_id:
        raise ValueError("byte artifact selection does not support view_id")

    ordered_names = tuple(str(name) for name in (tensor_names or ()))
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("tensor_names must be unique")
    if ordered_names:
        raise ValueError("byte artifact selection supports full selection only")

    provided_subset_hash = bytes(view_subset_hash or b"")
    if provided_subset_hash:
        raise ValueError("byte artifact selection does not support view_subset_hash")

    return common_pb2.ArtifactSelection(
        artifact_id=artifact_id,
        view_id="",
        logical_layout_hash=compute_byte_artifact_logical_layout_hash(),
        selection_hash=compute_byte_artifact_selection_hash(),
    )


def build_artifact_selection(
    *,
    artifact_id: str,
    canonical_index_bytes: bytes,
    layout_index_bytes: bytes | None,
    view_spec: common_pb2.ViewSpec | None,
    tensor_names: Sequence[str] | None,
    view_subset_hash: bytes | None = None,
    view_id: str | None = None,
    allow_view_id_without_spec: bool = False,
) -> common_pb2.ArtifactSelection:
    if not artifact_id:
        raise ValueError("artifact_id is required for selection")

    if is_byte_artifact_id(artifact_id):
        validate_byte_artifact_cgid(artifact_id)
        return _build_byte_artifact_selection(
            artifact_id=artifact_id,
            view_spec=view_spec,
            tensor_names=tensor_names,
            view_subset_hash=view_subset_hash,
            view_id=view_id,
        )

    canonical_bytes = bytes(canonical_index_bytes or b"")
    if not canonical_bytes:
        raise ValueError("canonical index bytes are required for selection hashing")

    ordered_names = tuple(str(name) for name in (tensor_names or ()))
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("tensor_names must be unique")

    provided_subset_hash = bytes(view_subset_hash or b"")
    computed_subset_hash = (
        compute_view_subset_hash(ordered_names) if ordered_names else b""
    )
    if ordered_names:
        if provided_subset_hash and provided_subset_hash != computed_subset_hash:
            raise ValueError("view_subset_hash does not match tensor_names")
    elif provided_subset_hash:
        raise ValueError("view_subset_hash requires tensor_names")
    resolved_subset_hash = computed_subset_hash

    has_transform = bool(view_spec is not None and view_spec.tensors)
    resolved_view_id = ""
    if has_transform:
        if view_spec is None:
            raise ValueError("view_spec is required for view transform selection")
        from tensorcast.api.store.view_composer import compute_view_id

        resolved_view_id = compute_view_id(view_spec, canonical_bytes)
        if view_id and str(view_id) != resolved_view_id:
            raise ValueError("view_id does not match view_spec")
    elif view_id:
        if not allow_view_id_without_spec:
            raise ValueError("view_id requires a non-identity view spec")
        resolved_view_id = str(view_id)

    has_subset = bool(ordered_names)
    needs_view_index = has_subset or bool(resolved_view_id)
    if needs_view_index:
        index_bytes = bytes(layout_index_bytes or b"")
        if not index_bytes:
            raise ValueError("view_index bytes are required for selected selection")
    else:
        index_bytes = canonical_bytes

    selection = common_pb2.ArtifactSelection(
        artifact_id=artifact_id,
        view_id=resolved_view_id,
        logical_layout_hash=compute_logical_layout_hash(
            index_bytes=index_bytes,
            needs_view_index=needs_view_index,
        ),
        selection_hash=compute_selection_hash(
            view_id=resolved_view_id,
            view_subset_hash=resolved_subset_hash if resolved_subset_hash else None,
        ),
    )
    if resolved_subset_hash:
        selection.view_subset_hash = resolved_subset_hash
        selection.tensor_names.extend(ordered_names)
    if has_transform and view_spec is not None:
        selection.view_spec.CopyFrom(view_spec)
    return selection


__all__ = ["build_artifact_selection", "compute_selected_index_bytes"]
