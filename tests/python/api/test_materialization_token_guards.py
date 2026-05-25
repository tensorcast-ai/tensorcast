#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch

import tensorcast.api._materialize as materialize_mod
from tensorcast.api._config import GetArtifactOptions
from tensorcast.api._errors import DaemonUnavailable, IndexParseError
from tensorcast.api._materialize import materialize_artifact
from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import ServerConfig


def _canonical_index_bytes() -> bytes:
    return canonical_index_to_bytes(
        CanonicalIndex(
            entries=(
                CanonicalIndexEntry(
                    name="w",
                    dtype=torch.float32,
                    shape=(1,),
                    stride=(1,),
                    storage_offset=0,
                    segment_offset=0,
                    size_bytes=4,
                ),
            ),
            total_size_bytes=4,
            avbs_hash="",
        )
    )


class _MaterializeClient:
    def __init__(
        self,
        response: store_daemon_pb2.MaterializeReplicaResponse,
        *,
        expected_device_type: int = store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
    ) -> None:
        self._response = response
        self._expected_device_type = expected_device_type

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        assert artifact_id == "mi2:test:artifact"
        return _canonical_index_bytes()

    def materialize_by_artifact_id(
        self,
        **kwargs: object,
    ) -> store_daemon_pb2.MaterializeReplicaResponse:
        assert kwargs["target_device_type"] == self._expected_device_type
        return self._response


def _response(
    *,
    lease_token: bytes,
    cpu_memfd_size_bytes: int = 4,
) -> store_daemon_pb2.MaterializeReplicaResponse:
    response = store_daemon_pb2.MaterializeReplicaResponse(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=_canonical_index_bytes(),
        source=store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA,
    )
    response.mem_handle.cpu_memfd.size_bytes = int(cpu_memfd_size_bytes)
    response.mem_handle.cpu_memfd.offset_bytes = 0
    response.mem_handle.lease_token = bytes(lease_token)
    payload = response.payloads.add()
    payload.name = "w"
    payload.shape.append(1)
    payload.stride.append(1)
    payload.buffer_offset = 0
    payload.byte_length = 4
    payload.storage_offset = 0
    payload.dtype = "torch.float32"
    return response


def _cuda_response() -> store_daemon_pb2.MaterializeReplicaResponse:
    response = store_daemon_pb2.MaterializeReplicaResponse(
        artifact_id="mi2:test:artifact",
        canonical_index_bytes=_canonical_index_bytes(),
        source=store_daemon_pb2.MATERIALIZATION_SOURCE_LOCAL_REPLICA,
    )
    response.mem_handle.cuda_ipc_handle = b"cuda-ipc-handle"
    response.mem_handle.lease_token = b"lease-token"
    payload = response.payloads.add()
    payload.name = "w"
    payload.shape.append(1)
    payload.stride.append(1)
    payload.buffer_offset = 0
    payload.byte_length = 4
    payload.storage_offset = 0
    payload.dtype = "torch.float32"
    payload.device_uuid = "GPU-0"
    return response


def _server_config(**updates: object) -> ServerConfig:
    values: dict[str, Any] = {
        "tx_slice_bytes": 4096,
        "mem_pool_size": 1 << 20,
        "artifact_chunk_bytes": 1 << 18,
        "local_handle_socket_path": "/tmp/tensorcast-local-handle.sock",
        "cpu_shared_memory_enabled": True,
    }
    values.update(updates)
    return ServerConfig(**values)


@pytest.mark.parametrize(
    ("response_kwargs", "server_updates", "message"),
    [
        (
            {"lease_token": b""},
            {},
            "empty lease_token",
        ),
        (
            {"lease_token": b"lease-token"},
            {"local_handle_socket_path": ""},
            "local_handle_socket_path is missing",
        ),
        (
            {"lease_token": b"lease-token"},
            {"cpu_shared_memory_enabled": False},
            "cpu_shared_memory_enabled is false",
        ),
        (
            {"lease_token": b"lease-token", "cpu_memfd_size_bytes": 0},
            {},
            "empty cpu_memfd handle",
        ),
    ],
)
def test_cpu_memfd_materialization_fails_before_tensor_restore_without_export_authority(
    response_kwargs: Mapping[str, object],
    server_updates: Mapping[str, object],
    message: str,
) -> None:
    client = _MaterializeClient(
        _response(
            lease_token=cast(bytes, response_kwargs["lease_token"]),
            cpu_memfd_size_bytes=int(
                cast(int, response_kwargs.get("cpu_memfd_size_bytes", 4))
            ),
        )
    )

    with pytest.raises(DaemonUnavailable, match=message):
        materialize_artifact(
            client=cast(Any, client),
            daemon_address="fake://daemon",
            server_config=_server_config(**server_updates),
            device_id=torch.device("cpu"),
            artifact_id="mi2:test:artifact",
            key=None,
            options=GetArtifactOptions(wait_for_completion=True),
        )


def test_cuda_ipc_materialization_fails_before_ipc_open_without_local_handle_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(materialize_mod, "device_uuid_for", lambda _device_id: "GPU-0")

    def fail_get_cuda_memory_ptr(_device_id: int, _handle: bytes) -> int:
        raise AssertionError("CUDA IPC must not open without local handle authority")

    monkeypatch.setattr(
        materialize_mod,
        "get_cuda_memory_ptr",
        fail_get_cuda_memory_ptr,
    )
    payload = materialize_artifact(
        client=cast(
            Any,
            _MaterializeClient(
                _cuda_response(),
                expected_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            ),
        ),
        daemon_address="fake://daemon",
        server_config=_server_config(local_handle_socket_path=""),
        device_id=torch.device("cuda", 0),
        artifact_id="mi2:test:artifact",
        key=None,
        options=GetArtifactOptions(wait_for_completion=True),
    )

    assert payload.state_dict_loader is not None
    with pytest.raises(
        DaemonUnavailable,
        match="lease_token present but local_handle_socket_path is missing",
    ):
        payload.state_dict_loader()


def test_materialize_rejects_missing_daemon_canonical_index_bytes() -> None:
    response = _response(lease_token=b"lease-token")
    response.canonical_index_bytes = b""

    with pytest.raises(IndexParseError, match="missing canonical_index_bytes"):
        materialize_artifact(
            client=cast(Any, _MaterializeClient(response)),
            daemon_address="fake://daemon",
            server_config=_server_config(),
            device_id=torch.device("cpu"),
            artifact_id="mi2:test:artifact",
            key=None,
            options=GetArtifactOptions(wait_for_completion=True),
        )
