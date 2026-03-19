#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures

import pytest

from tensorcast.api.store import AssemblyRequirementSetRef, Store
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import AssemblyAttemptRef


class FakeAttemptClient:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.seal_calls: list[dict[str, object]] = []
        self.wait_calls: list[dict[str, object]] = []

    def start_assembly_attempt(self, **kwargs: object) -> AssemblyAttemptRef:
        self.start_calls.append(dict(kwargs))
        operation_ref = operation_pb2.OperationRef(
            operation_id="bafkattemptop",
            kind="assembly_attempt",
            target_artifact_id="cgid:assembly-workspace-1",
            authority_scope_kind="assembly_attempt",
            authority_scope_id="cgid:assembly-attempt-1",
            attachment_kind="assembly_attempt",
            recovery_class="cluster_durable",
            fencing_digest="bafkattemptintent",
        )
        return AssemblyAttemptRef(
            attempt_id="cgid:assembly-attempt-1",
            workspace_assembly_id="cgid:assembly-workspace-1",
            layout_id=str(kwargs["layout_id"]),
            attempt_intent_digest="bafkattemptintent",
            coordinator_generation=1,
            coordinator_operation=operation_ref,
        )

    def seal_assembly_attempt(
        self,
        *,
        attempt_id: str,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.SealAssemblyAttemptResponse:
        self.seal_calls.append(
            {
                "attempt_id": attempt_id,
                "timeout_s": timeout_s,
            }
        )
        return store_daemon_pb2.SealAssemblyAttemptResponse(
            operation=operation_pb2.OperationRef(
                operation_id="bafkattemptop",
                kind="assembly_attempt",
                target_artifact_id="cgid:assembly-workspace-9",
                authority_scope_kind="assembly_attempt",
                authority_scope_id=attempt_id,
                attachment_kind="assembly_attempt",
                recovery_class="cluster_durable",
                fencing_digest="bafk-intent-9",
            )
        )

    def get_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_s: float = 10.0,
    ) -> operation_pb2.GetOperationResponse:
        del operation_id
        del operation_ref
        del timeout_s
        response = operation_pb2.GetOperationResponse()
        response.status.state = operation_pb2.OPERATION_STATE_PENDING
        return response

    def wait_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_ms: int,
        timeout_s: float,
    ) -> operation_pb2.GetOperationResponse:
        self.wait_calls.append(
            {
                "operation_id": operation_id,
                "operation_ref": operation_ref,
                "timeout_ms": timeout_ms,
                "timeout_s": timeout_s,
            }
        )
        payload = store_daemon_pb2.SealAssemblyResult(
            artifact=common_pb2.ArtifactDescriptor(
                artifact_id="mi2:test:artifact",
                index_multihash="bafkindex",
                data_multihash="bafkdata",
                schema_version="v3",
                encoding="json",
                total_size=128,
            ),
            source_version_key="models/demo/source/v1",
        )
        response = operation_pb2.GetOperationResponse()
        response.status.state = operation_pb2.OPERATION_STATE_SUCCESS
        response.status.result.Pack(payload)
        return response


class FakeRuntime:
    def __init__(self, client: FakeAttemptClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def ensure_client(self) -> FakeAttemptClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None


def test_start_assembly_attempt_returns_attempt_ref() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.pp_from_structural_views(
        ["view-a", "view-b"]
    )

    attempt = store.start_assembly_attempt(
        layout_id="layout-1",
        requirements=requirements,
    )

    assert attempt.attempt_id == "cgid:assembly-attempt-1"
    assert attempt.workspace_assembly_id == "cgid:assembly-workspace-1"
    assert attempt.layout_id == "layout-1"
    assert attempt.attempt_intent_digest == "bafkattemptintent"
    assert attempt.coordinator_operation.operation_id == "bafkattemptop"
    assert attempt.coordinator_operation.kind == "assembly_attempt"
    assert client.start_calls == [
        {
            "layout_id": "layout-1",
            "requirements": requirements,
        }
    ]


def test_start_assembly_attempt_requires_explicit_requirements() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)

    with pytest.raises(ValueError) as exc_info:
        store.start_assembly_attempt(layout_id="layout-1")

    assert "requirements are required" in str(exc_info.value)


def test_seal_assembly_attempt_decodes_source_lineage() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    operation_ref = operation_pb2.OperationRef(
        operation_id="bafkattemptop",
        kind="assembly_attempt",
        target_artifact_id="cgid:assembly-workspace-9",
        authority_scope_kind="assembly_attempt",
        authority_scope_id="cgid:assembly-attempt-9",
        attachment_kind="assembly_attempt",
        recovery_class="cluster_durable",
        fencing_digest="bafk-intent-9",
    )
    attempt = AssemblyAttemptRef(
        attempt_id="cgid:assembly-attempt-9",
        workspace_assembly_id="cgid:assembly-workspace-9",
        layout_id="layout-9",
        attempt_intent_digest="bafk-intent-9",
        coordinator_generation=1,
        coordinator_operation=operation_ref,
    )

    operation = store.seal_assembly_attempt(attempt)
    result = operation.wait(timeout_s=5.0)

    assert result.assembly_id == "cgid:assembly-workspace-9"
    assert result.source_artifact_id == "mi2:test:artifact"
    assert result.source_descriptor.artifact_id == "mi2:test:artifact"
    assert result.source_version_key == "models/demo/source/v1"
    assert result.serving_artifact_id is None
    assert result.representation_contract_hash is None
    assert client.seal_calls == [
        {"attempt_id": "cgid:assembly-attempt-9", "timeout_s": 10.0}
    ]
    assert len(client.wait_calls) == 1
    wait_call = client.wait_calls[0]
    assert wait_call["operation_id"] == "bafkattemptop"
    assert wait_call["operation_ref"] == operation_ref
    assert 4900 <= int(wait_call["timeout_ms"]) <= 5000
    assert 9.0 <= float(wait_call["timeout_s"]) <= 10.0


def test_requirement_family_builders_encode_distinct_contracts() -> None:
    pp = AssemblyRequirementSetRef.pp_from_structural_views(
        ["view-b", "view-a", "view-a"]
    )
    ep = AssemblyRequirementSetRef.ep_from_structural_views(["view-b", "view-a"])
    canonical = AssemblyRequirementSetRef.canonical_full()

    assert tuple(req.slot_id for req in pp.inline_requirements) == (
        "view-a",
        "view-b",
    )
    assert tuple(req.coverage_contract for req in pp.inline_requirements) == (
        "pp_structural_view",
        "pp_structural_view",
    )

    assert tuple(req.slot_id for req in ep.inline_requirements) == (
        "view-a",
        "view-b",
    )
    assert tuple(req.coverage_contract for req in ep.inline_requirements) == (
        "ep_structural_view",
        "ep_structural_view",
    )

    assert canonical.requirement_count == 1
    assert canonical.inline_requirements[0].slot_id == "__canonical_full__"
    assert canonical.inline_requirements[0].coverage_contract == "canonical_full"
