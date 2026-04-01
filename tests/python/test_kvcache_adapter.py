#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.api.errors import ArtifactError
from tensorcast.engine_adapter.kvcache_adapter import (
    MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA,
    ManifestResult,
    compute_key_set_digest_hex,
    open_byte_artifact,
    seal_byte_artifact,
)
from tensorcast.proto.common.v1 import common_pb2


def test_byte_artifact_open_to_seal_enforces_invariants() -> None:
    artifact_id = "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE"
    opened = open_byte_artifact(
        artifact_id=artifact_id,
        layout_id="layout-v1",
        payload=b"abc123",
    )
    sealed = opened.seal()
    assert sealed.artifact_id == artifact_id
    assert sealed.invariant.layout_id == "layout-v1"
    assert sealed.invariant.byte_length == 6
    assert sealed.invariant.payload_digest_alg == "sha256"
    assert (
        sealed.invariant.payload_digest_hex
        == "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090"
    )
    assert (
        sealed.invariant.verification_mode
        == "BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256"
    )


def test_byte_artifact_open_rejects_non_profile_artifact_id() -> None:
    with pytest.raises(ArtifactError, match="byte artifact id"):
        open_byte_artifact(
            artifact_id="cgid:not-byte-artifact",
            layout_id="layout-v1",
            payload=b"x",
        )


def test_byte_artifact_open_rejects_malformed_profile_cgid() -> None:
    with pytest.raises(ArtifactError, match="valid byte artifact cgid"):
        open_byte_artifact(
            artifact_id="cgid:byte_artifact~tenant~engine~layout_only",
            layout_id="layout-v1",
            payload=b"x",
        )


def test_byte_artifact_key_set_digest_is_order_insensitive() -> None:
    ids_a = (
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",
        "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azI",
    )
    ids_b = (ids_a[1], ids_a[0], ids_a[0])
    digest_a = compute_key_set_digest_hex(layout_id="layout-v1", artifact_ids=ids_a)
    digest_b = compute_key_set_digest_hex(layout_id="layout-v1", artifact_ids=ids_b)
    assert digest_a == digest_b


def test_manifest_result_helper_uses_key_set_digest() -> None:
    result = ManifestResult.from_artifact_ids(
        engine_request_id="rid-1",
        layout_id="layout-v1",
        artifact_ids=("cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",),
    )
    assert result.key_set_digest_alg == "sha256"
    assert len(result.key_set_digest_hex) == 64


def test_manifest_result_from_artifact_selections_emits_explicit_bridge() -> None:
    manifest_selection = common_pb2.ArtifactSelection(
        artifact_id="engine-manifest:rid-1",
        logical_layout_hash=b"manifest-logical",
        selection_hash=b"manifest-selection",
    )
    item_a = common_pb2.ArtifactSelection(
        artifact_id="cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azA",
        logical_layout_hash=b"logical-a",
        selection_hash=b"selection-a",
    )
    item_b = common_pb2.ArtifactSelection(
        artifact_id="cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",
        logical_layout_hash=b"logical-b",
        selection_hash=b"selection-b",
    )

    result = ManifestResult.from_artifact_selections(
        engine_request_id="rid-1",
        layout_id="layout-v1",
        manifest_selection=manifest_selection,
        artifact_selections=(item_b, item_a, item_b),
    )

    bridge = result.require_artifact_set_bridge()
    assert bridge.bridge_schema == MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA
    assert bridge.bridge_version == 1
    assert bridge.artifact_set_ref.carrier_form == "manifest_backed"
    assert bridge.artifact_set_ref.item_count == 2
    assert bridge.artifact_set_ref.manifest_selection == manifest_selection
    assert len(bridge.resolved_items) == 2
    assert bridge.resolved_items[0].artifact_id == item_a.artifact_id
    assert bridge.resolved_items[1].artifact_id == item_b.artifact_id
    assert result.require_artifact_set_ref() == bridge.artifact_set_ref
    assert result.key_set_digest_hex != bridge.artifact_set_ref.set_digest_hex


def test_manifest_result_require_bridge_fails_closed_when_missing() -> None:
    result = ManifestResult.from_artifact_ids(
        engine_request_id="rid-1",
        layout_id="layout-v1",
        artifact_ids=("cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE",),
    )

    with pytest.raises(
        ArtifactError, match="does not carry an explicit ManifestArtifactSetBridge"
    ):
        result.require_artifact_set_bridge()


def test_seal_byte_artifact_direct_helper_matches_open_seal() -> None:
    artifact_id = "cgid:byte_artifact~ns~eng~b64u.bW9kZWw~b64u.djE~layout-v1~b64u.azE"
    direct = seal_byte_artifact(
        artifact_id=artifact_id,
        layout_id="layout-v1",
        payload=b"abc123",
    )
    opened = open_byte_artifact(
        artifact_id=artifact_id,
        layout_id="layout-v1",
        payload=b"abc123",
    ).seal()
    assert direct.invariant == opened.invariant
