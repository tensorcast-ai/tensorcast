#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.common.identity import (
    build_byte_artifact_cgid,
    decode_cgid_segment,
    parse_byte_artifact_cgid,
)
from tensorcast.common.selection_contract import build_artifact_selection
from tensorcast.common.selection_identity import (
    compute_byte_artifact_logical_layout_hash,
    compute_byte_artifact_selection_hash,
    compute_view_subset_hash,
)
from tensorcast.proto.common.v1 import common_pb2


def test_byte_artifact_cgid_build_parse_and_decode_vectors() -> None:
    artifact_id = build_byte_artifact_cgid(
        namespace="tenantA",
        engine="sglang",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_version="v1",
        layout_id="layout_v1",
        engine_key="request-0001:blk-42",
    )
    assert (
        artifact_id == "cgid:byte_artifact~tenantA~sglang~"
        "b64u.bWV0YS1sbGFtYS9MbGFtYS0zLjEtOEItSW5zdHJ1Y3Q~"
        "b64u.djE~"
        "layout_v1~b64u.cmVxdWVzdC0wMDAxOmJsay00Mg"
    )

    parsed = parse_byte_artifact_cgid(artifact_id)
    assert parsed.namespace == "tenantA"
    assert parsed.engine == "sglang"
    assert parsed.layout_id == "layout_v1"
    assert decode_cgid_segment(parsed.model_id_enc).decode("utf-8") == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert decode_cgid_segment(parsed.model_version_enc).decode("utf-8") == "v1"
    assert decode_cgid_segment(parsed.engine_key_enc).decode("utf-8") == (
        "request-0001:blk-42"
    )


def test_byte_artifact_selection_profile_requires_canonical_full_selection() -> None:
    artifact_id = (
        "cgid:byte_artifact~tenantA~sglang~"
        "b64u.bWV0YS1sbGFtYS9MbGFtYS0zLjEtOEItSW5zdHJ1Y3Q~"
        "b64u.djE~"
        "layout_v1~b64u.cmVxdWVzdC0wMDAxOmJsay00Mg"
    )
    selection = build_artifact_selection(
        artifact_id=artifact_id,
        canonical_index_bytes=b"",
        layout_index_bytes=None,
        view_spec=None,
        tensor_names=None,
        view_subset_hash=None,
    )
    assert selection.view_id == ""
    assert not selection.tensor_names
    assert selection.view_subset_hash == b""
    assert selection.logical_layout_hash == compute_byte_artifact_logical_layout_hash()
    assert selection.selection_hash == compute_byte_artifact_selection_hash()

    with pytest.raises(ValueError, match="full selection only"):
        build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=["payload"],
            view_subset_hash=None,
        )

    with pytest.raises(ValueError, match="does not support view_subset_hash"):
        build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=None,
            view_subset_hash=compute_view_subset_hash(["payload"]),
        )


def test_byte_artifact_selection_profile_rejects_view_and_non_payload_subset() -> None:
    artifact_id = (
        "cgid:byte_artifact~tenantA~sglang~"
        "b64u.bWV0YS1sbGFtYS9MbGFtYS0zLjEtOEItSW5zdHJ1Y3Q~"
        "b64u.djE~"
        "layout_v1~b64u.cmVxdWVzdC0wMDAxOmJsay00Mg"
    )
    with pytest.raises(ValueError, match="full selection only"):
        build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=["other"],
        )

    view_spec = common_pb2.ViewSpec()
    view_spec.tensors["payload"].ops.add().narrow.dim = 0
    view_spec.tensors["payload"].ops[0].narrow.start = 0
    view_spec.tensors["payload"].ops[0].narrow.length = 1
    with pytest.raises(ValueError, match="does not support view transforms"):
        build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=view_spec,
            tensor_names=None,
        )


def test_byte_artifact_selection_rejects_malformed_profile_cgid() -> None:
    with pytest.raises(ValueError, match="byte_artifact cgid must match"):
        build_artifact_selection(
            artifact_id="cgid:byte_artifact~tenant~engine~layout_only",
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=None,
        )


def test_byte_artifact_legacy_cgid_still_parses() -> None:
    artifact_id = "cgid:byte_artifact~tenant~engine~b64u.bQ~layout_v1~b64u.azQ"
    parsed = parse_byte_artifact_cgid(artifact_id)
    assert parsed.namespace == "tenant"
    assert parsed.engine == "engine"
    assert parsed.model_id_enc == "b64u.bQ"
    assert parsed.model_version_enc == ""
    assert parsed.layout_id == "layout_v1"
    assert parsed.engine_key_enc == "b64u.azQ"
