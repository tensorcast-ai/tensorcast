#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tensorcast.api._config import (
    ExecutionTopologyContext,
    GetArtifactOptions,
)
from tensorcast.api.context import CollectiveLoadGroup
from tensorcast.api.operation import OperationRefMetadata
from tensorcast.artifact_runtime.local_ready_prewarm import (
    DAEMON_ADDRESS_ENV,
    PREWARM_RETENTION_TTL_MS_ENV,
    SOURCE_PATH_FILTER_ENV,
    TARGET_PLAN_MANIFEST_B64_ENV,
    TARGET_PLAN_MANIFEST_CACHE_DIR_ENV,
    TARGET_PLAN_MANIFEST_ENV,
    TARGET_PLAN_MANIFEST_JSON_ENV,
    TARGET_PLAN_MANIFEST_SHA256_ENV,
    TARGET_PLAN_MANIFEST_WRITE_ENV,
    decode_local_ready_target_plan_record,
    initialize_runtime_from_env,
    main,
    prewarm_local_ready_target_plan_manifest,
    prewarm_local_ready_target_plans_from_env,
    retained_local_ready_cache_key_for_intent,
    target_plan_manifest_cache_path,
    target_plan_manifest_cache_source_index_path,
    update_target_plan_manifest_cache_record,
    wait_retained_manifest_record_operation_handoff,
)
from tensorcast.types import (
    PrefetchHandoff,
    PrefetchHandoffSet,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeBindingMemberRef,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeTopologyRef,
)


def _realization_target() -> RealizationTarget:
    topology = RuntimeTopologyRef(schema_topology_digest="topology-schema")
    member = RuntimeBindingMemberRef(
        member_id="rank-3",
        member_index=3,
        member_count=8,
        group_id="same-host-tp-load",
    )
    source = RuntimeBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:source",
        source_schema_hash="source-schema",
    )
    source_reuse = RuntimeBindingSourceReuseDecision(
        mode="checkpoint_to_runtime",
        representation_contract_hash="repr-contract",
    )
    resolved_layout = RuntimeBindingResolvedLayout(
        binding_layout_id="layout-3",
        source=source,
        source_reuse=source_reuse,
        topology=topology,
        member=member,
        target_layout=b"target-layout",
        target_index_bytes=b"target-index",
        target_layout_hash="target-layout-hash",
        tensor_schema_hash="tensor-schema",
        spec_digest="spec-digest",
        source_schema_hash="source-schema",
        copy_plan_bytes=b"copy-plan",
        dst_specs_bytes=b"dst-specs",
    )
    return RealizationTarget(
        runtime="vllm",
        device="cuda:3",
        device_uuid="GPU-3",
        source=source,
        topology=topology,
        member=member,
        model_config_digest="model-config",
        runtime_build_digest="serving-build",
        resolved_layout=resolved_layout,
    )


def _target_plan_record(source_path: str) -> dict[str, Any]:
    target = _realization_target()
    target_proto = target.to_proto()
    target_proto_bytes = target_proto.SerializeToString(deterministic=True)
    options = GetArtifactOptions(
        source="disk_first",
        execution_topology=ExecutionTopologyContext(
            collective_group=CollectiveLoadGroup(
                group_id="same-host-tp-load",
                world_size=8,
                rank=3,
            ),
            collective_policy="collective_first",
            source_locality="shared_source",
            source_sharing_domain="node-a",
        ),
    )
    return {
        "schema_version":
        1,
        "intent_key":
        "intent-key",
        "source_path":
        source_path,
        "model_hash":
        "model-hash",
        "target_device":
        "cuda:3",
        "device_uuid":
        "GPU-3",
        "runtime":
        "vllm",
        "expected_member":
        target.member.model_dump(mode="python"),
        "framework_identity": {
            "framework_name": "vllm",
            "framework_version": "test",
            "adapter_version": "test-adapter",
            "serving_abi_version": "test-abi",
        },
        "runtime_config_digest":
        "runtime-config-digest",
        "source_artifact_ref":
        "mi2:source",
        "source_schema_hash":
        "source-schema",
        "source_selection_digest":
        "selection-digest",
        "spec_cache_key_digest":
        "spec-cache-key",
        "spec_digest":
        "spec-digest",
        "target_layout_hash":
        "target-layout-hash",
        "tensor_schema_hash":
        "tensor-schema",
        "serving_build_digest":
        "serving-build",
        "materialization_options":
        options.model_dump(mode="json", exclude_none=True),
        "target_proto_type":
        target_proto.DESCRIPTOR.full_name,
        "target_proto_encoding":
        "base64",
        "target_proto_b64":
        base64.b64encode(target_proto_bytes).decode("ascii"),
        "target_proto_sha256":
        hashlib.sha256(target_proto_bytes).hexdigest(),
        "retention_ttl_ms":
        12_345,
        "written_at_ms":
        1_000,
    }


def _target_for_member(member_index: int, member_count: int) -> RealizationTarget:
    base = _realization_target()
    member = RuntimeBindingMemberRef(
        member_id=f"rank-{member_index}",
        member_index=member_index,
        member_count=member_count,
        group_id="same-host-tp-load",
    )
    source = base.source.model_copy(
        update={
            "representation_contract_hash": f"repr-contract-{member_index}",
        })
    topology = RuntimeTopologyRef(
        schema_topology_digest=f"topology-schema-{member_index}",
        logical_topology_ref=f"tensorcast://placement/{member_index}",
    )
    resolved_layout = base.resolved_layout.model_copy(
        update={
            "binding_layout_id": f"layout-{member_index}",
            "source": source,
            "topology": topology,
            "member": member,
            "target_layout": f"target-layout-{member_index}".encode(),
            "target_index_bytes": f"target-index-{member_index}".encode(),
            "target_layout_hash": f"target-layout-hash-{member_index}",
            "spec_digest": f"spec-digest-{member_index}",
        })
    return base.model_copy(
        update={
            "device": f"cuda:{member_index}",
            "device_uuid": f"GPU-{member_index}",
            "source": source,
            "topology": topology,
            "member": member,
            "resolved_layout": resolved_layout,
        })


def _target_plan_record_for_member(
    source_path: str,
    *,
    member_index: int,
    member_count: int,
) -> dict[str, Any]:
    target = _target_for_member(member_index, member_count)
    record = _target_plan_record(source_path)
    target_proto = target.to_proto()
    target_proto_bytes = target_proto.SerializeToString(deterministic=True)
    options = GetArtifactOptions(
        source="disk_first",
        execution_topology=ExecutionTopologyContext(
            collective_group=CollectiveLoadGroup(
                group_id="same-host-tp-load",
                world_size=member_count,
                rank=member_index,
            ),
            collective_policy="collective_first",
            source_locality="shared_source",
            source_sharing_domain="node-a",
        ),
    )
    record.update({
        "intent_key":
        f"intent-key-{member_index}",
        "target_device":
        str(target.device),
        "device_uuid":
        target.device_uuid,
        "expected_member":
        target.member.model_dump(mode="python"),
        "spec_digest":
        target.resolved_layout.spec_digest,
        "target_layout_hash":
        target.resolved_layout.target_layout_hash,
        "materialization_options":
        options.model_dump(mode="json", exclude_none=True),
        "target_proto_b64":
        base64.b64encode(target_proto_bytes).decode("ascii"),
        "target_proto_sha256":
        hashlib.sha256(target_proto_bytes).hexdigest(),
    })
    return record


def _write_manifest(path, record: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "producer": "vllm.tensorcast.loader",
        "intent_kind": "local_ready_target_plan",
        "records_by_source_path": {
            record["source_path"]: [record],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_initialize_runtime_from_env_connects_to_daemon(monkeypatch) -> None:
    import tensorcast.startup as startup

    calls: list[dict[str, str]] = []
    monkeypatch.setenv(DAEMON_ADDRESS_ENV, "127.0.0.1:12345")
    monkeypatch.setattr(startup, "is_initialized", lambda: False)
    monkeypatch.setattr(startup, "init", lambda **kwargs: calls.append(kwargs))

    initialize_runtime_from_env()

    assert calls == [{
        "mode": "connect",
        "address": "127.0.0.1:12345",
    }]


def _handoff(
    target: RealizationTarget,
    *,
    readiness: str = "runtime_local_ready",
    local_serving_ref: str = "local-serving-ref",
) -> SimpleNamespace:
    return SimpleNamespace(
        member=target.member,
        local_serving_ref=local_serving_ref,
        daemon_id="daemon-id",
        daemon_session_id="daemon-session-id",
        serving_artifact_id="serving-artifact-id",
        reservation_bytes=4096,
        expires_at_ms=99_999,
        device_uuid="GPU-3",
        readiness=readiness,
        staged_value=False,
        verification_state="verified",
        binding_value_ref={
            "binding_id": "binding-id",
            "binding_epoch": 1,
        },
        reservation_capability={
            "binding_value_ref": {
                "binding_id": "binding-id",
                "binding_epoch": 1,
            },
            "daemon_id": "daemon-id",
            "daemon_session_id": "daemon-session-id",
            "device_uuid": "GPU-3",
            "member": target.member.model_dump(mode="python"),
            "reservation_bytes": 4096,
        },
        group_realization_acquire=None,
    )


def _prefetch_handoff(
    target: RealizationTarget,
    *,
    local_serving_ref: str,
) -> PrefetchHandoff:
    payload = dict(vars(_handoff(
        target,
        local_serving_ref=local_serving_ref,
    )))
    binding_value_ref = {
        "binding_id": f"binding-{target.member.member_index}",
        "binding_layout_id": target.resolved_layout.binding_layout_id,
        "binding_value_id": f"value-{target.member.member_index}",
        "seal_generation": 1,
    }
    payload.update({
        "binding_value_ref":
        binding_value_ref,
        "device_uuid":
        target.device_uuid,
        "reservation_capability": {
            "capability_id":
            f"capability-{target.member.member_index}",
            "binding_value_ref":
            binding_value_ref,
            "daemon_id":
            "daemon-id",
            "daemon_session_id":
            "daemon-session-id",
            "device_uuid":
            target.device_uuid,
            "member":
            target.member.model_dump(mode="python"),
            "reservation_bytes":
            4096,
            "scope_digest":
            f"scope-{target.member.member_index}",
            "expires_at_ms":
            99_999,
        },
    })
    return PrefetchHandoff.model_validate(payload)


def test_decode_local_ready_target_plan_record_restores_target_and_options(
        tmp_path) -> None:
    record = _target_plan_record(str(tmp_path / "model"))

    intent = decode_local_ready_target_plan_record(record)

    assert intent.source_path == str(tmp_path / "model")
    assert intent.intent_key == "intent-key"
    assert intent.source_artifact_ref == "mi2:source"
    assert intent.target == _realization_target()
    assert intent.materialization_options is not None
    topology = intent.materialization_options.execution_topology
    assert topology is not None
    assert topology.collective_group is not None
    assert topology.collective_group.group_id == "same-host-tp-load"
    assert topology.collective_group.rank == 3
    assert str(topology.collective_policy.value) == "collective_first"
    assert intent.retention is not None
    assert intent.retention.expire_if_unacquired_after_ms == 12_345
    assert intent.retention.idle_ttl_after_last_release_ms == 12_345
    assert intent.retention.materialization_timeout_ms == 12_345
    assert intent.retention.allow_acquire_after_creator_exit is True


def test_decode_local_ready_target_plan_record_rejects_bad_target_hash(
        tmp_path) -> None:
    record = _target_plan_record(str(tmp_path / "model"))
    record["target_proto_sha256"] = "bad"

    with pytest.raises(ValueError, match="target_proto_sha256"):
        decode_local_ready_target_plan_record(record)


def test_prewarm_manifest_writes_vllm_retained_manifest(tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    record = _target_plan_record(source_path)
    _write_manifest(input_manifest, record)
    handoff = _handoff(_realization_target())
    calls: list[dict[str, Any]] = []

    class _FakeOperation:
        operation_id = "operation-id"
        operation_ref_metadata = OperationRefMetadata(
            operation_id="operation-id",
            kind="prefetch_serving_binding",
            target_artifact_id="serving-artifact-id",
            authority_scope_kind="workflow_owner",
            authority_scope_id="workflow-id",
            attachment_kind="target_publication",
            recovery_class="ephemeral_process_local",
            fencing_digest="operation-digest",
        )

        def result(self, *, timeout_s=None):
            calls[-1]["result_timeout_s"] = timeout_s
            return handoff

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            return _FakeOperation()

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retained_manifest_write=retained_manifest,
        timeout_s=3.0,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    intent = results[0].intent
    assert calls[0]["target"] == intent.target
    assert calls[0]["readiness"] == "runtime_local_ready"
    assert calls[0]["options"] == intent.materialization_options
    assert calls[0]["retention"] == intent.retention
    assert calls[0]["ctx"].idempotency_key == "local-ready-prewarm:intent-key"
    assert calls[0]["result_timeout_s"] == 3.0
    assert results[0].operation_id == "operation-id"

    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained_records = payload["records_by_source_path"][source_path]
    assert len(retained_records) == 1
    retained = retained_records[0]
    assert retained["cache_key"] == retained_local_ready_cache_key_for_intent(
        intent)
    assert len(retained["cache_key"]) == 64
    assert retained["local_serving_ref"] == "local-serving-ref"
    assert retained["expected_member"] == intent.target.member.model_dump(
        mode="python")
    assert retained["expected_tensor_schema_hash"] == "tensor-schema"
    assert retained["expected_serving_build_digest"] == "serving-build"
    assert retained["expected_target_layout_hash"] == "target-layout-hash"
    assert retained["expected_daemon_id"] == "daemon-id"
    assert retained["expected_daemon_session_id"] == "daemon-session-id"
    assert retained["serving_artifact_id"] == "serving-artifact-id"
    assert retained["device_uuid"] == "GPU-3"
    assert retained["reservation_bytes"] == 4096
    assert retained["expires_at_ms"] == 99_999
    assert retained["cache_identity_version"] == 2
    assert retained["source_path"] == source_path
    assert retained["model_hash"] == "model-hash"
    assert retained["target_device"] == "cuda:3"
    assert retained["retained_readiness"] == "runtime_local_ready"
    assert retained["retained_realization_claim_extra"][
        "retained_binding_acquire"]["mode"] == "external"
    assert retained["retained_realization_claim_extra"][
        "retained_binding_acquire"]["authority"][
            "readiness"] == "runtime_local_ready"
    assert retained["operation_id"] == "operation-id"
    assert retained["operation_ref"]["kind"] == "prefetch_serving_binding"
    assert retained["operation_ref"]["authority_scope_id"] == "workflow-id"
    assert retained["framework_identity"]["framework_name"] == "vllm"
    assert retained["runtime_config_digest"] == "runtime-config-digest"
    assert retained["source_selection_digest"] == "selection-digest"

    older_record = dict(retained)
    older_record.update({
        "cache_key": "old-cache-key",
        "local_serving_ref": "old-local-serving-ref",
        "source_selection_digest": "old-selection-digest",
        "written_at_ms": retained["written_at_ms"] - 1,
    })
    retained_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "records_by_source_path": {
                source_path: [older_record],
            },
        }),
        encoding="utf-8",
    )
    prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retained_manifest_write=retained_manifest,
        timeout_s=3.0,
        artifact_factory=lambda intent: _FakeArtifact(),
    )
    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained_records = payload["records_by_source_path"][source_path]
    assert len(retained_records) == 1
    assert retained_records[0]["local_serving_ref"] == "local-serving-ref"


def test_prewarm_manifest_can_override_retention_ttl(tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))
    calls: list[dict[str, Any]] = []

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            return _FakeOperation(kwargs["target"])

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retention_ttl_ms=67_890,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    retention = calls[0]["retention"]
    assert retention.expire_if_unacquired_after_ms == 67_890
    assert retention.idle_ttl_after_last_release_ms == 67_890
    assert retention.materialization_timeout_ms == 67_890
    assert results[0].intent.retention == retention


def test_prewarm_env_can_override_retention_ttl(monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))
    calls: list[dict[str, Any]] = []

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            return _FakeOperation(kwargs["target"])

    monkeypatch.setenv(TARGET_PLAN_MANIFEST_ENV, str(input_manifest))
    monkeypatch.setenv(PREWARM_RETENTION_TTL_MS_ENV, "45678")

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    assert calls[0]["retention"].expire_if_unacquired_after_ms == 45_678


def test_source_path_mode_rebinds_target_plan_to_current_artifact_selection(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    record = _target_plan_record(source_path)
    record["source_artifact_ref"] = "msa1:old-session"
    record["source_selection_digest"] = "old-selection-digest"
    _write_manifest(input_manifest, record)
    calls: list[dict[str, Any]] = []

    class _FakeArtifact:

        def _resolve_realization_selection(self):
            return SimpleNamespace(
                artifact_id="msa1:new-session",
                source_selection_digest="new-selection-digest",
            )

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            target = kwargs["target"]
            return SimpleNamespace(
                operation_id="operation-id",
                result=lambda *, timeout_s=None: _handoff(target),
            )

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retained_manifest_write=retained_manifest,
        source_mode="source_path",
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    intent = results[0].intent
    target = calls[0]["target"]
    assert target.source.source_artifact_ref == "msa1:new-session"
    assert target.source.artifact_selection_digest == "new-selection-digest"
    assert target.resolved_layout.source == target.source
    assert intent.source_artifact_ref == "msa1:new-session"
    assert intent.record["source_rebind_mode"] == "source_path"
    assert intent.record[
        "source_rebound_from_artifact_ref"] == "msa1:old-session"
    assert intent.record["source_selection_digest"] == "new-selection-digest"
    assert intent.record["target_proto_sha256"] != record[
        "target_proto_sha256"]
    assert calls[0]["ctx"].idempotency_key != "local-ready-prewarm:intent-key"

    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained = payload["records_by_source_path"][source_path][0]
    assert retained["source_selection_digest"] == "new-selection-digest"
    assert retained["runtime_config_digest"] == ""
    assert retained[
        "runtime_config_digest_source"] == "unbound_source_path_rebind"
    assert retained["cache_key"] == retained_local_ready_cache_key_for_intent(
        intent)


def test_prewarm_manifest_issues_all_operations_before_waiting(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    record_1 = _target_plan_record(source_path)
    record_2 = _target_plan_record(source_path)
    record_2["intent_key"] = "intent-key-2"
    input_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "records": [record_1, record_2],
        }),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    waits: list[str] = []

    class _FakeOperation:

        def __init__(self, operation_id: str) -> None:
            self.operation_id = operation_id

        def result(self, *, timeout_s=None):
            assert len(calls) == 2
            waits.append(self.operation_id)
            return _handoff(
                calls[int(self.operation_id.split("-")[-1]) - 1]["target"],
                local_serving_ref=f"local-serving-ref-{self.operation_id}",
            )

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            return _FakeOperation(f"operation-{len(calls)}")

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert [result.operation_id for result in results] == [
        "operation-1",
        "operation-2",
    ]
    assert waits == ["operation-1", "operation-2"]
    assert len(calls) == 2


def test_prewarm_manifest_groups_complete_target_set(tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    records = [
        _target_plan_record_for_member(
            source_path,
            member_index=0,
            member_count=2,
        ),
        _target_plan_record_for_member(
            source_path,
            member_index=1,
            member_count=2,
        ),
    ]
    input_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "records": list(reversed(records)),
        }),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    class _FakeOperation:
        operation_id = "target-set-operation"

        def __init__(self, target: RealizationTargetSet) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            assert timeout_s == 3.0
            return PrefetchHandoffSet(
                runtime=self._target.runtime,
                topology=self._target.topology,
                group_id=self._target.group_id,
                members=tuple(
                    _prefetch_handoff(
                        target,
                        local_serving_ref=(
                            "local-serving-ref-"
                            f"{target.member.member_index}"),
                    ) for target in self._target.members),
                readiness="runtime_local_ready",
            )

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            assert isinstance(kwargs["target"], RealizationTargetSet)
            return _FakeOperation(kwargs["target"])

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retained_manifest_write=retained_manifest,
        timeout_s=3.0,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(calls) == 1
    target_set = calls[0]["target"]
    assert isinstance(target_set, RealizationTargetSet)
    assert [target.member.member_index for target in target_set.members] == [
        0,
        1,
    ]
    assert target_set.members[0].source != target_set.members[1].source
    assert target_set.members[0].topology != target_set.members[1].topology
    assert calls[0]["ctx"].idempotency_key.startswith(
        "local-ready-prewarm-set:")
    assert [result.operation_id for result in results] == [
        "target-set-operation",
        "target-set-operation",
    ]
    assert [result.handoff.local_serving_ref for result in results] == [
        "local-serving-ref-0",
        "local-serving-ref-1",
    ]

    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained_records = payload["records_by_source_path"][source_path]
    assert [record["local_serving_ref"] for record in retained_records] == [
        "local-serving-ref-0",
        "local-serving-ref-1",
    ]


def test_prewarm_uses_target_plan_intent_key_as_idempotency_key(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    record = _target_plan_record(source_path)
    _write_manifest(input_manifest, record)
    idempotency_keys: list[str | None] = []

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(
                self._target,
                readiness="runtime_local_ready",
                local_serving_ref="local-ready-serving-ref",
            )

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            ctx = kwargs["ctx"]
            idempotency_keys.append(getattr(ctx, "idempotency_key", None))
            return _FakeOperation(kwargs["target"])

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    assert idempotency_keys == [f"local-ready-prewarm:{record['intent_key']}"]


def test_prewarm_manifest_writes_materializing_record_before_terminal_wait(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    ready_marker = tmp_path / "ready.json"
    record = _target_plan_record(source_path)
    _write_manifest(input_manifest, record)
    manifest_seen_during_result: dict[str, Any] = {}
    ready_marker_seen_during_result: dict[str, Any] = {}

    class _FakeOperation:
        operation_id = "operation-id"
        operation_ref_metadata = OperationRefMetadata(
            operation_id="operation-id",
            kind="prefetch_serving_binding",
            target_artifact_id="serving-artifact-id",
            authority_scope_kind="workflow_owner",
            authority_scope_id="workflow-id",
            attachment_kind="target_publication",
            recovery_class="ephemeral_process_local",
            fencing_digest="operation-digest",
        )

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def latest_result(self):
            return _handoff(
                self._target,
                readiness="runtime_reserved",
                local_serving_ref="reserved-serving-ref",
            )

        def result(self, *, timeout_s=None):
            del timeout_s
            payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
            retained = payload["records_by_source_path"][source_path][0]
            manifest_seen_during_result.update(retained)
            ready_marker_seen_during_result.update(
                json.loads(ready_marker.read_text(encoding="utf-8")))
            return _handoff(
                self._target,
                readiness="runtime_local_ready",
                local_serving_ref="local-ready-serving-ref",
            )

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        retained_manifest_write=retained_manifest,
        write_materializing_records=True,
        materializing_record_timeout_s=0.0,
        materializing_ready_write=ready_marker,
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    assert manifest_seen_during_result["retained_readiness"] == (
        "runtime_reserved")
    assert manifest_seen_during_result["local_serving_ref"] == (
        "reserved-serving-ref")
    assert manifest_seen_during_result["operation_ref"][
        "authority_scope_id"] == "workflow-id"
    assert manifest_seen_during_result["retained_realization_claim_extra"][
        "retained_binding_acquire"]["authority"][
            "readiness"] == "runtime_reserved"
    assert ready_marker_seen_during_result["ready"] is True
    assert ready_marker_seen_during_result["expected_records"] == 1
    assert ready_marker_seen_during_result["retained_records"] == 1
    assert ready_marker_seen_during_result["readiness"] == {
        "runtime_reserved": 1
    }
    assert ready_marker_seen_during_result["retained_manifest"] == str(
        retained_manifest)
    assert ready_marker_seen_during_result["source_paths"] == [source_path]

    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained = payload["records_by_source_path"][source_path][0]
    assert retained["retained_readiness"] == "runtime_local_ready"
    assert retained["local_serving_ref"] == "local-ready-serving-ref"
    assert retained["retained_realization_claim_extra"][
        "retained_binding_acquire"]["authority"][
            "readiness"] == "runtime_local_ready"


def test_wait_retained_manifest_record_operation_handoff_uses_operation_ref(
        monkeypatch) -> None:
    from tensorcast.api import operation as operation_mod
    from tensorcast.api.store import runtime as store_runtime_mod

    captured: dict[str, Any] = {}

    class _FakeRuntime:

        def __init__(self) -> None:
            self.initialized = False

        def ensure_initialized(self) -> None:
            self.initialized = True

    class _FakeStoreRuntime:
        pass

    class _FakeOperation:

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def result(self, *, timeout_s=None):
            captured["timeout_s"] = timeout_s
            return "handoff"

    monkeypatch.setattr(operation_mod, "DaemonGlobalStoreOperation",
                        _FakeOperation)
    store_runtime = _FakeStoreRuntime()
    monkeypatch.setattr(store_runtime_mod, "get_context",
                        lambda: store_runtime)
    runtime = _FakeRuntime()
    record = {
        "operation_ref":
        OperationRefMetadata(
            operation_id="operation-id",
            kind="prefetch_serving_binding",
            target_artifact_id="serving-artifact-id",
        ).to_dict(),
    }

    handoff = wait_retained_manifest_record_operation_handoff(
        record,
        runtime=runtime,
        timeout_s=12.5,
    )

    assert handoff == "handoff"
    assert runtime.initialized is True
    assert captured["operation_id"] == "operation-id"
    assert captured["runtime_ref"]() is store_runtime
    assert captured["operation_ref"].kind == "prefetch_serving_binding"
    assert captured["timeout_s"] == 12.5


def test_prewarm_env_materializing_ready_marker_enables_materializing_records(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    ready_marker = tmp_path / "ready.json"
    record = _target_plan_record(source_path)
    _write_manifest(input_manifest, record)

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def latest_result(self):
            return _handoff(
                self._target,
                readiness="runtime_reserved",
                local_serving_ref="reserved-serving-ref",
            )

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(
                self._target,
                readiness="runtime_local_ready",
                local_serving_ref="local-ready-serving-ref",
            )

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    monkeypatch.setenv("TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST",
                       str(input_manifest))
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))
    monkeypatch.setenv("TENSORCAST_LOCAL_READY_MATERIALIZING_READY_WRITE",
                       str(ready_marker))
    monkeypatch.setenv("TENSORCAST_LOCAL_READY_MATERIALIZING_RECORD_TIMEOUT_S",
                       "0")

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is True
    assert marker["readiness"] == {"runtime_reserved": 1}
    payload = json.loads(retained_manifest.read_text(encoding="utf-8"))
    retained = payload["records_by_source_path"][source_path][0]
    assert retained["retained_readiness"] == "runtime_local_ready"


def test_prewarm_env_accepts_inline_target_plan_manifest_json(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    retained_manifest = tmp_path / "retained.json"
    record = _target_plan_record(source_path)
    payload = {
        "schema_version": 1,
        "records_by_source_path": {
            source_path: [record],
        },
    }
    calls: list[dict[str, Any]] = []

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            calls.append(kwargs)
            return _FakeOperation(kwargs["target"])

    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_B64_ENV, raising=False)
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_JSON_ENV, json.dumps(payload))
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    assert len(calls) == 1
    retained_payload = json.loads(
        retained_manifest.read_text(encoding="utf-8"))
    retained = retained_payload["records_by_source_path"][source_path][0]
    assert retained["local_serving_ref"] == "local-serving-ref"


def test_prewarm_env_accepts_inline_target_plan_manifest_b64(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    retained_manifest = tmp_path / "retained.json"
    record = _target_plan_record(source_path)
    payload = {
        "schema_version": 1,
        "records": [record],
    }

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    encoded = base64.b64encode(
        json.dumps(payload).encode("utf-8")).decode("ascii")
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_JSON_ENV, raising=False)
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_B64_ENV, encoded)
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    retained_payload = json.loads(
        retained_manifest.read_text(encoding="utf-8"))
    retained = retained_payload["records_by_source_path"][source_path][0]
    assert retained["local_serving_ref"] == "local-serving-ref"


def test_prewarm_manifest_validates_sha256(tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    record = _target_plan_record(source_path)
    _write_manifest(input_manifest, record)
    expected_sha256 = hashlib.sha256(input_manifest.read_bytes()).hexdigest()

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    results = prewarm_local_ready_target_plan_manifest(
        input_manifest,
        manifest_sha256s=[expected_sha256],
        artifact_factory=lambda intent: _FakeArtifact(),
    )

    assert len(results) == 1
    assert results[0].operation_id == "operation-id"


def test_prewarm_manifest_sha256_mismatch_writes_failure_marker(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    ready_marker = tmp_path / "ready.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))

    class _UnexpectedArtifact:

        def prefetch(self, **kwargs):
            del kwargs
            raise AssertionError("prefetch should not be called")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        prewarm_local_ready_target_plan_manifest(
            input_manifest,
            manifest_sha256s=["0" * 64],
            materializing_ready_write=ready_marker,
            artifact_factory=lambda intent: _UnexpectedArtifact(),
        )

    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is False
    assert marker["event"] == "materializing_records_failed"
    assert marker["phase"] == "materializing"
    assert marker["error_type"] == "ValueError"
    assert "sha256 mismatch" in marker["error_message"]


def test_prewarm_env_accepts_target_plan_manifest_sha256(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))
    expected_sha256 = hashlib.sha256(input_manifest.read_bytes()).hexdigest()

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    monkeypatch.setenv(TARGET_PLAN_MANIFEST_ENV, str(input_manifest))
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_SHA256_ENV, expected_sha256)
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    retained_payload = json.loads(
        retained_manifest.read_text(encoding="utf-8"))
    retained = retained_payload["records_by_source_path"][source_path][0]
    assert retained["local_serving_ref"] == "local-serving-ref"


def test_prewarm_env_accepts_target_plan_manifest_cache_identity(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    source_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    _write_manifest(source_manifest, _target_plan_record(source_path))
    expected_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    cache_dir = tmp_path / "target-plan-cache"
    cached_manifest = target_plan_manifest_cache_path(
        cache_dir,
        expected_sha256,
    )
    cached_manifest.parent.mkdir(parents=True, exist_ok=True)
    cached_manifest.write_bytes(source_manifest.read_bytes())

    class _FakeOperation:
        operation_id = "operation-id"

        def __init__(self, target: RealizationTarget) -> None:
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(self._target)

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            return _FakeOperation(kwargs["target"])

    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_JSON_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_B64_ENV, raising=False)
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_SHA256_ENV, expected_sha256)
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: _FakeArtifact())

    assert len(results) == 1
    assert results[0].operation_id == "operation-id"
    retained_payload = json.loads(
        retained_manifest.read_text(encoding="utf-8"))
    retained = retained_payload["records_by_source_path"][source_path][0]
    assert retained["local_serving_ref"] == "local-serving-ref"


def test_prewarm_env_resolves_target_plan_manifest_cache_source_index(
        monkeypatch, tmp_path) -> None:
    source_path = str(tmp_path / "model")
    retained_manifest = tmp_path / "retained.json"
    cache_dir = tmp_path / "target-plan-cache"
    record_1 = _target_plan_record(source_path)
    record_2 = _target_plan_record(source_path)
    record_1["expected_member"] = dict(record_1["expected_member"])
    record_1["expected_member"]["member_count"] = 2
    record_2["intent_key"] = "intent-key-rank-2"
    record_2["expected_member"] = dict(record_2["expected_member"])
    record_2["expected_member"]["member_id"] = "rank-1"
    record_2["expected_member"]["member_index"] = 1
    record_2["expected_member"]["member_count"] = 2
    target_2_member = RuntimeBindingMemberRef(
        member_id="rank-1",
        member_index=1,
        member_count=2,
        group_id="same-host-tp-load",
    )
    target_2 = _realization_target()
    target_2_layout = target_2.resolved_layout.model_copy(
        update={"member": target_2_member})
    target_2 = target_2.model_copy(
        update={
            "member": target_2_member,
            "resolved_layout": target_2_layout,
        })
    target_2_proto_bytes = target_2.to_proto().SerializeToString(
        deterministic=True)
    record_2["target_proto_b64"] = base64.b64encode(
        target_2_proto_bytes).decode("ascii")
    record_2["target_proto_sha256"] = hashlib.sha256(
        target_2_proto_bytes).hexdigest()

    not_ready = update_target_plan_manifest_cache_record(
        cache_dir,
        source_path=source_path,
        record=record_1,
        producer="test",
    )
    assert not_ready["ready"] is False
    ready = update_target_plan_manifest_cache_record(
        cache_dir,
        source_path=source_path,
        record=record_2,
        producer="test",
    )
    assert ready["ready"] is True
    assert ready["record_count"] == 2
    assert ready["expected_records"] == 2
    index = json.loads(
        target_plan_manifest_cache_source_index_path(
            cache_dir, source_path).read_text(encoding="utf-8"))
    assert index["manifest_sha256"] == ready["manifest_sha256"]
    assert Path(index["manifest_path"]).exists()

    class _FakeOperation:

        def __init__(self, operation_id: str, target: RealizationTarget) -> None:
            self.operation_id = operation_id
            self._target = target

        def result(self, *, timeout_s=None):
            del timeout_s
            return _handoff(
                self._target,
                local_serving_ref=f"local-serving-ref-{self.operation_id}",
            )

    class _FakeArtifact:

        def __init__(self) -> None:
            self.calls = 0

        def prefetch(self, **kwargs):
            self.calls += 1
            return _FakeOperation(f"operation-{self.calls}", kwargs["target"])

    fake_artifact = _FakeArtifact()
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_JSON_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_B64_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_SHA256_ENV, raising=False)
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV, str(cache_dir))
    monkeypatch.setenv(SOURCE_PATH_FILTER_ENV, source_path)
    monkeypatch.setenv("TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE",
                       str(retained_manifest))

    results = prewarm_local_ready_target_plans_from_env(
        artifact_factory=lambda intent: fake_artifact)

    assert len(results) == 2
    assert [result.operation_id for result in results] == [
        "operation-1",
        "operation-2",
    ]
    retained_payload = json.loads(
        retained_manifest.read_text(encoding="utf-8"))
    retained_records = retained_payload["records_by_source_path"][source_path]
    assert len(retained_records) == 2


def test_prewarm_materializing_ready_marker_records_prefetch_failure(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    ready_marker = tmp_path / "ready.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))

    class _FailingArtifact:

        def prefetch(self, **kwargs):
            del kwargs
            raise RuntimeError("prefetch boom")

    with pytest.raises(RuntimeError, match="prefetch boom"):
        prewarm_local_ready_target_plan_manifest(
            input_manifest,
            retained_manifest_write=retained_manifest,
            materializing_ready_write=ready_marker,
            artifact_factory=lambda intent: _FailingArtifact(),
        )

    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is False
    assert marker["event"] == "materializing_records_failed"
    assert marker["phase"] == "materializing"
    assert marker["error_type"] == "RuntimeError"
    assert "prefetch boom" in marker["error_message"]


def test_prewarm_materializing_ready_marker_fails_when_reserved_records_missing(
        tmp_path) -> None:
    source_path = str(tmp_path / "model")
    input_manifest = tmp_path / "target_plan.json"
    retained_manifest = tmp_path / "retained.json"
    ready_marker = tmp_path / "ready.json"
    _write_manifest(input_manifest, _target_plan_record(source_path))
    terminal_waits = 0

    class _SlowOperation:
        operation_id = "operation-id"

        def latest_result(self):
            return None

        def result(self, *, timeout_s=None):
            nonlocal terminal_waits
            del timeout_s
            terminal_waits += 1
            raise AssertionError("terminal wait should not run")

    class _FakeArtifact:

        def prefetch(self, **kwargs):
            del kwargs
            return _SlowOperation()

    with pytest.raises(RuntimeError,
                       match="materializing retained records were not ready"):
        prewarm_local_ready_target_plan_manifest(
            input_manifest,
            retained_manifest_write=retained_manifest,
            materializing_record_timeout_s=0.0,
            materializing_ready_write=ready_marker,
            artifact_factory=lambda intent: _FakeArtifact(),
        )

    assert terminal_waits == 0
    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is False
    assert marker["event"] == "materializing_records_failed"
    assert marker["expected_records"] == 1
    assert marker["retained_records"] == 0
    assert marker["error_type"] == "RuntimeError"
    assert "expected=1 retained=0" in marker["error_message"]


def test_main_writes_failure_marker_when_manifest_is_missing(
        monkeypatch, tmp_path) -> None:
    ready_marker = tmp_path / "ready.json"
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_JSON_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_B64_ENV, raising=False)

    with pytest.raises(SystemExit, match="no target-plan manifest supplied"):
        main(["--materializing-ready-write", str(ready_marker)])

    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is False
    assert marker["event"] == "materializing_records_failed"
    assert marker["phase"] == "argument_validation"
    assert marker["error_type"] == "ValueError"
    assert "no target-plan manifest supplied" in marker["error_message"]


def test_main_writes_failure_marker_when_inline_manifest_is_invalid(
        monkeypatch, tmp_path) -> None:
    ready_marker = tmp_path / "ready.json"
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_WRITE_ENV, raising=False)
    monkeypatch.delenv(TARGET_PLAN_MANIFEST_B64_ENV, raising=False)
    monkeypatch.setenv(TARGET_PLAN_MANIFEST_JSON_ENV, "{")

    with pytest.raises(SystemExit, match=TARGET_PLAN_MANIFEST_JSON_ENV):
        main(["--materializing-ready-write", str(ready_marker)])

    marker = json.loads(ready_marker.read_text(encoding="utf-8"))
    assert marker["ready"] is False
    assert marker["event"] == "materializing_records_failed"
    assert marker["phase"] == "argument_validation"
    assert marker["error_type"] == "ValueError"
    assert TARGET_PLAN_MANIFEST_JSON_ENV in marker["error_message"]
