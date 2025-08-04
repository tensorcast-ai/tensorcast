#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

import grpc
import pytest

from scstore.proto import global_store_pb2, global_store_pb2_grpc


@pytest.mark.integration  # mark for explicit grouping
def test_request_timeout_raises(global_store_service):
    """RequestModelReplicaTransport should propagate client-side gRPC deadline.

    Unlike existing tests which only assert ``Status.TIMED_OUT`` on the *response*,
    this verifies that a **gRPC deadline** triggers ``grpc.RpcError`` with
    ``StatusCode.DEADLINE_EXCEEDED`` on the client side (Scenario 7 in §九).
    """

    # ------------------------------------------------------------------
    # Build a real gRPC stub against the in-process Global-Store server.
    # ------------------------------------------------------------------
    addr = global_store_service._address
    with grpc.insecure_channel(addr) as channel:
        stub = global_store_pb2_grpc.GlobalModelStoreStub(channel)

        # Craft a request that is guaranteed to wait because the model name has
        # *no* registered replicas.
        req = global_store_pb2.RequestModelReplicaTransportRequest(
            model_name="non_existent_model",
            wait_timeout_ms=200,  # Server will loop for ~200ms then respond TIMED_OUT
        )

        # Set the *client* deadline to a tiny value (20ms) so that the gRPC
        # runtime aborts the call **before** the server responds.
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.RequestModelReplicaTransport(req, timeout=0.02)  # 20ms

        rpc_err = exc_info.value
        assert rpc_err.code() == grpc.StatusCode.DEADLINE_EXCEEDED  # pyright: ignore[reportAttributeAccessIssue]
        # Optional: gRPC details string may mention "Deadline Exceeded"