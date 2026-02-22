#  Copyright (c) 2026, TensorCast Team.

import grpc

from tensorcast.api.store.retry import (
    map_materialization_error,
    retry_reason_bucket,
)


class _DummyRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


def test_materialization_error_global_store_not_connected_adds_hint() -> None:
    err = _DummyRpcError(
        grpc.StatusCode.FAILED_PRECONDITION,
        "GlobalStoreClient not connected",
    )
    mapped = map_materialization_error(err)
    assert mapped.status_code == "FAILED_PRECONDITION"
    assert "Hint:" in str(mapped)
    assert "Global Store" in str(mapped)
    assert "tc.init" in str(mapped)
    assert "global_store_address" in str(mapped)
    assert "tensorcast-cli daemon start" in str(mapped)
    assert "--global-store-address" in str(mapped)
    assert "mode='connect'" in str(mapped)
    assert "do not reconfigure" in str(mapped)


def test_materialization_runtime_error_global_store_not_connected_adds_hint() -> None:
    mapped = map_materialization_error(RuntimeError("GlobalStoreClient not connected"))
    assert mapped.status_code == "FAILED_PRECONDITION"
    assert "Hint:" in str(mapped)
    assert "tc.init" in str(mapped)
    assert "global_store_address" in str(mapped)
    assert "do not reconfigure" in str(mapped)


def test_materialization_runtime_error_confirm_failed_is_retryable() -> None:
    mapped = map_materialization_error(
        RuntimeError(
            "Failed to confirm artifact loading: "
            "artifact_id=mi2:foo, replica_uuid=abc, disk_path="
        )
    )
    assert mapped.status_code == "UNAVAILABLE"
    assert mapped.retryable is True
    assert "Hint:" in str(mapped)
    assert "stale/unavailable" in str(mapped)


def test_materialization_runtime_error_transport_timeout_is_retryable() -> None:
    mapped = map_materialization_error(
        RuntimeError("No available replica for artifact foo within timeout")
    )
    assert mapped.status_code == "UNAVAILABLE"
    assert mapped.retryable is True
    assert retry_reason_bucket(mapped) in {"transport_unavailable", "unavailable"}
