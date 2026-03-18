#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import types

from tensorcast.api.store import Store
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import AssemblyAttemptRef


def _build_piece_contract(layout_id: str, *, view_ids: tuple[str, ...]) -> store_daemon_pb2.ContributionContractSnapshot:
    snapshot = store_daemon_pb2.ContributionContractSnapshot(
        layout_id=layout_id,
        require_live_contributions_until_readiness_cut=True,
    )
    for view_id in view_ids:
        snapshot.required_slots.add(
            slot_key=view_id,
            structural_view_id=view_id,
            contribution_kind=store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL,
            coverage_semantics="phase1_layout_expected_view",
        )
    return snapshot


def _build_attempt_spec(
    *,
    assembly_id: str,
    layout_id: str,
    contribution_contract_hash: str,
    attempt_spec_hash: str,
    view_ids: tuple[str, ...],
) -> store_daemon_pb2.AssemblyAttemptSpec:
    spec = store_daemon_pb2.AssemblyAttemptSpec(
        assembly_id=assembly_id,
        layout_id=layout_id,
        contribution_contract_hash=contribution_contract_hash,
        attempt_spec_hash=attempt_spec_hash,
        closeout_policy=store_daemon_pb2.CloseoutPolicySnapshot(
            policy_json="",
            source_policy_version=0,
            closeout_policy_hash="bafkcloseout",
        ),
    )
    spec.contribution_contract.CopyFrom(_build_piece_contract(layout_id, view_ids=view_ids))
    return spec


class FakeAttemptClient:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.start_seal_calls: list[dict[str, object]] = []
        self.wait_calls: list[dict[str, object]] = []

    def start_assembly_attempt(self, **kwargs: object) -> AssemblyAttemptRef:
        self.start_calls.append(dict(kwargs))
        operation_ref = operation_pb2.OperationRef(
            operation_id="bafkattemptop",
            kind="assembly_attempt",
            target_artifact_id="cgid:assembly-attempt-1",
            authority_scope_kind="assembly_attempt",
            authority_scope_id="cgid:assembly-attempt-1",
            attachment_kind="assembly_attempt",
            recovery_class="cluster_durable",
            fencing_digest="bafkattemptspec",
        )
        spec = _build_attempt_spec(
            assembly_id="cgid:assembly-attempt-1",
            layout_id=str(kwargs["layout_id"]),
            contribution_contract_hash="bafkqaaa",
            attempt_spec_hash="bafkattemptspec",
            view_ids=("view-a", "view-b"),
        )
        return AssemblyAttemptRef(
            assembly_id="cgid:assembly-attempt-1",
            layout_id=str(kwargs["layout_id"]),
            attempt_spec_hash="bafkattemptspec",
            contribution_contract_hash="bafkqaaa",
            coordinator_operation=operation_ref,
            coordinator_generation=1,
            attempt_spec_proto=spec.SerializeToString(),
        )

    def start_seal_assembly(
        self,
        *,
        assembly_id: str,
        layout_id: str | None = None,
        expected_coordinator_generation: int | None = None,
        attempt_spec: object | None = None,
        timeout_s: float = 10.0,
    ) -> object:
        self.start_seal_calls.append(
            {
                "assembly_id": assembly_id,
                "layout_id": layout_id,
                "expected_coordinator_generation": expected_coordinator_generation,
                "attempt_spec": attempt_spec,
                "timeout_s": timeout_s,
            }
        )
        return types.SimpleNamespace(
            operation=types.SimpleNamespace(
                operation_id="bafksealop",
                kind="assembly_attempt",
                target_artifact_id=assembly_id,
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

    attempt = store.start_assembly_attempt(layout_id="layout-1")

    assert attempt.assembly_id == "cgid:assembly-attempt-1"
    assert attempt.layout_id == "layout-1"
    assert attempt.attempt_spec_hash == "bafkattemptspec"
    assert attempt.contribution_contract_hash == "bafkqaaa"
    assert attempt.coordinator_operation.operation_id == "bafkattemptop"
    assert attempt.coordinator_operation.kind == "assembly_attempt"
    decoded_spec = attempt.decode_attempt_spec()
    assert decoded_spec.layout_id == "layout-1"
    assert tuple(slot.slot_key for slot in decoded_spec.contribution_contract.required_slots) == (
        "view-a",
        "view-b",
    )
    assert client.start_calls == [{"layout_id": "layout-1"}]


def test_wait_assembly_attempt_decodes_source_lineage() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    operation_ref = operation_pb2.OperationRef(
        operation_id="bafksealop",
        kind="assembly_attempt",
        target_artifact_id="cgid:assembly-attempt-9",
        authority_scope_kind="assembly_attempt",
        authority_scope_id="cgid:assembly-attempt-9",
        attachment_kind="assembly_attempt",
        recovery_class="cluster_durable",
        fencing_digest="bafk-spec-9",
    )
    spec = _build_attempt_spec(
        assembly_id="cgid:assembly-attempt-9",
        layout_id="layout-9",
        contribution_contract_hash="bafkcontract",
        attempt_spec_hash="bafk-spec-9",
        view_ids=("view-a",),
    )
    attempt = AssemblyAttemptRef(
        assembly_id="cgid:assembly-attempt-9",
        layout_id="layout-9",
        attempt_spec_hash="bafk-spec-9",
        contribution_contract_hash="bafkcontract",
        coordinator_operation=operation_ref,
        coordinator_generation=1,
        attempt_spec_proto=spec.SerializeToString(),
    )

    result = store.wait_assembly_attempt(attempt, timeout_s=5.0)

    assert result.assembly_id == "cgid:assembly-attempt-9"
    assert result.source_artifact_id == "mi2:test:artifact"
    assert result.source_descriptor.artifact_id == "mi2:test:artifact"
    assert result.serving_artifact_id is None
    assert result.representation_contract_hash is None
    assert len(client.start_seal_calls) == 1
    assert client.start_seal_calls[0]["assembly_id"] == "cgid:assembly-attempt-9"
    assert client.start_seal_calls[0]["layout_id"] == "layout-9"
    assert client.start_seal_calls[0]["expected_coordinator_generation"] == 1
    attempt_spec = client.start_seal_calls[0]["attempt_spec"]
    assert isinstance(attempt_spec, store_daemon_pb2.AssemblyAttemptSpec)
    assert attempt_spec.assembly_id == "cgid:assembly-attempt-9"
    assert attempt_spec.layout_id == "layout-9"
    assert attempt_spec.attempt_spec_hash == "bafk-spec-9"
    assert client.start_seal_calls[0]["timeout_s"] == 5.0
    assert client.wait_calls == [
        {
            "operation_id": "bafksealop",
            "operation_ref": operation_ref,
            "timeout_ms": 5000,
            "timeout_s": 10.0,
        }
    ]
