#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import types

from tensorcast.api.store import Store
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import AssemblyAttemptRef


class FakeAttemptClient:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.start_seal_calls: list[dict[str, object]] = []
        self.wait_calls: list[dict[str, object]] = []

    def start_assembly_attempt(self, **kwargs: object) -> AssemblyAttemptRef:
        self.start_calls.append(dict(kwargs))
        contract_proto = store_daemon_pb2.ContributionContractSnapshot(
            layout_id=str(kwargs["layout_id"]),
            require_live_contributions=True,
        ).SerializeToString()
        return AssemblyAttemptRef(
            assembly_id="cgid:assembly-attempt-1",
            layout_id=str(kwargs["layout_id"]),
            contribution_contract_hash="bafkqaaa",
            coordinator_operation_id="bafksealop",
            coordinator_generation=1,
            expected_view_ids=("view-a", "view-b"),
            contribution_contract_proto=contract_proto,
        )

    def start_seal_assembly(
        self,
        *,
        assembly_id: str,
        layout_id: str | None = None,
        expected_coordinator_generation: int | None = None,
        attempt_snapshot: object | None = None,
        timeout_s: float = 10.0,
    ) -> object:
        self.start_seal_calls.append(
            {
                "assembly_id": assembly_id,
                "layout_id": layout_id,
                "expected_coordinator_generation": expected_coordinator_generation,
                "attempt_snapshot": attempt_snapshot,
                "timeout_s": timeout_s,
            }
        )
        return types.SimpleNamespace(
            operation=types.SimpleNamespace(
                operation_id="bafksealop",
                kind="seal_assembly",
                target_artifact_id=assembly_id,
            )
        )

    def get_operation(
        self,
        operation_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> operation_pb2.GetOperationResponse:
        del operation_id
        del timeout_s
        response = operation_pb2.GetOperationResponse()
        response.status.state = operation_pb2.OPERATION_STATE_PENDING
        return response

    def wait_operation(
        self,
        operation_id: str,
        *,
        timeout_ms: int,
        timeout_s: float,
    ) -> operation_pb2.GetOperationResponse:
        self.wait_calls.append(
            {
                "operation_id": operation_id,
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
            representation_contract_hash="bafkcontract",
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
    assert attempt.contribution_contract_hash == "bafkqaaa"
    assert attempt.expected_view_ids == ("view-a", "view-b")
    assert client.start_calls == [{"layout_id": "layout-1"}]


def test_wait_assembly_attempt_decodes_source_lineage() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    attempt = AssemblyAttemptRef(
        assembly_id="cgid:assembly-attempt-9",
        layout_id="layout-9",
        contribution_contract_hash="bafkcontract",
        coordinator_operation_id="bafksealop",
        coordinator_generation=1,
        expected_view_ids=(),
        contribution_contract_proto=store_daemon_pb2.ContributionContractSnapshot(
            layout_id="layout-9",
            require_live_contributions=True,
        ).SerializeToString(),
    )

    result = store.wait_assembly_attempt(attempt, timeout_s=5.0)

    assert result.assembly_id == "cgid:assembly-attempt-9"
    assert result.source_artifact_id == "mi2:test:artifact"
    assert result.source_descriptor.artifact_id == "mi2:test:artifact"
    assert result.serving_artifact_id is None
    assert result.representation_contract_hash == "bafkcontract"
    assert len(client.start_seal_calls) == 1
    assert client.start_seal_calls[0]["assembly_id"] == "cgid:assembly-attempt-9"
    assert client.start_seal_calls[0]["layout_id"] == "layout-9"
    assert client.start_seal_calls[0]["expected_coordinator_generation"] == 1
    snapshot = client.start_seal_calls[0]["attempt_snapshot"]
    assert isinstance(snapshot, store_daemon_pb2.SealAssemblySnapshot)
    assert snapshot.assembly_id == "cgid:assembly-attempt-9"
    assert snapshot.layout_id == "layout-9"
    assert snapshot.contribution_contract_hash == "bafkcontract"
    assert client.start_seal_calls[0]["timeout_s"] == 5.0
    assert client.wait_calls == [
        {
            "operation_id": "bafksealop",
            "timeout_ms": 5000,
            "timeout_s": 10.0,
        }
    ]
