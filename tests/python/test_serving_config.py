#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest

from tensorcast.serving import ServingConfig, ServingPolicy, ServingSelector


def test_serving_config_parses_nested_schema_defaults() -> None:
    config = ServingConfig.from_mapping({
        "runtime": {
            "mode": "CONNECT",
            "daemon": {
                "show_logs": "true",
            },
            "global_store": {
                "address": "127.0.0.1:50051",
            },
        },
        "serving": {
            "selector": {
                "kind": "version_key",
                "value": " models/demo/serving/v1 ",
            },
        },
        "bootstrap": {
            "mode": "required",
            "verify_source_checksums": "false",
        },
        "materialization": {
            "collective": "required",
        },
    })

    assert config.runtime.mode == "connect"
    assert config.runtime.daemon.show_logs is True
    assert config.runtime.global_store.resolved_mode("connect") == "connect"
    assert config.serving.selector == ServingSelector(
        kind="version_key",
        value="models/demo/serving/v1",
    )
    assert config.bootstrap.serving_metadata_mode() == "require"
    assert config.bootstrap.verify_source_checksums is False
    assert config.materialization.collective_policy_value() == \
        "require_collective"


def test_serving_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Unexpected TensorCast serving config"):
        ServingConfig.from_mapping({"preload_mode": "external"})


def test_serving_policy_pinned_requires_identity_fields() -> None:
    with pytest.raises(ValueError, match="manifest_ref"):
        ServingPolicy(mode="pinned")

    policy = ServingPolicy(
        mode="pinned",
        manifest_ref="tensor:__tensorcast_meta__.manifest_json",
        representation_contract_hash="repr-hash",
        serving_build_digest="build-digest",
    )

    assert policy.manifest_ref == "tensor:__tensorcast_meta__.manifest_json"


def test_serving_config_parses_external_preload_authority() -> None:
    config = ServingConfig.from_mapping({
        "preload": {
            "mode": "external",
            "authority": {
                "group_id": "group-1",
                "member_ref": {
                    "member_id": "member-0",
                    "member_index": 0,
                    "member_count": 1,
                    "group_id": "group-1",
                },
                "daemon_id": "daemon-1",
                "daemon_session_id": "session-1",
                "device_uuid": "GPU-0",
                "binding_value_ref": {
                    "binding_id": "binding-1",
                    "binding_layout_id": "layout-1",
                    "binding_value_id": "value-1",
                    "seal_generation": 1,
                },
                "reservation_capability": {
                    "capability_id": "capability-1",
                },
                "readiness": "serving_published_ready",
                "serving_artifact_id": "mi2:test:serving",
                "trusted_reservation_bytes": 4096,
                "expected": {
                    "target_layout_hash": "layout-hash",
                    "tensor_schema_hash": "schema-hash",
                    "serving_build_digest": "build-digest",
                    "resolved_spec_digest": "spec-digest",
                },
            },
        },
    })

    assert config.preload.mode == "external"
    assert config.preload.authority is not None
    assert config.preload.authority.serving_artifact_id == "mi2:test:serving"
    assert config.preload.authority.trusted_reservation_bytes == 4096
