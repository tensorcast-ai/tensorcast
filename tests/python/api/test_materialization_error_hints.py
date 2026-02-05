#  Copyright (c) 2026, TensorCast Team.

import grpc

from tensorcast.api.store.retry import map_materialization_error


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
