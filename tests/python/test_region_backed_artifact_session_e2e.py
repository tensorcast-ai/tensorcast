#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import multiprocessing
import os
import subprocess
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from typing import Any, NoReturn
from uuid import uuid4

import grpc
import pytest
import torch

from tensorcast.api.store import (
    AllocatorTransferOptions,
    ByteArtifactKeyspace,
    ByteArtifactSpec,
    HostMemorySpan,
    RegionArtifactTransfer,
    RegionBackedArtifactSession,
    RegionBackedArtifactSessionOptions,
    RegionSessionFailedError,
    RegionSessionFailureCode,
    ScratchTransferOptions,
)
from tensorcast.global_store.config.settings import GlobalStoreConfig, set_config
from tensorcast.global_store.grpc_service import (
    GlobalStoreServicer,
    register_global_store_servicers,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2, store_daemon_pb2_grpc
from tests.python.utils.daemon import start_daemon_binary
from tests.python.utils.ports import get_free_port

pytestmark = pytest.mark.integration

_SUBPROCESS_TIMEOUT_S = 45.0


def _keyspace(run_id: str) -> ByteArtifactKeyspace:
    return ByteArtifactKeyspace(
        namespace="region-session-e2e",
        engine="python-sdk",
        model_id="byte-artifacts",
        model_version=run_id,
        layout_id="page-v1",
    )


def _transfer(
    artifact: ByteArtifactSpec,
    tensor: torch.Tensor,
) -> RegionArtifactTransfer:
    return RegionArtifactTransfer(
        artifact=artifact,
        span=HostMemorySpan.from_tensor(
            tensor,
            offset_bytes=0,
            byte_length=artifact.byte_length,
        ),
    )


def _scratch_worker(daemon_address: str, run_id: str) -> dict[str, Any]:
    session = RegionBackedArtifactSession.attach(
        RegionBackedArtifactSessionOptions(
            daemon_address=daemon_address,
            session_name=f"scratch-{run_id}",
            transfer=ScratchTransferOptions(capacity_bytes=4096),
        )
    )
    keyspace = _keyspace(run_id)
    first_source = torch.tensor(list(b"alpha"), dtype=torch.uint8)
    second_source = torch.tensor(list(b"bravo!!"), dtype=torch.uint8)
    first_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=b"scratch:first",
        byte_length=first_source.numel(),
    )
    second_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=b"scratch:second",
        byte_length=second_source.numel(),
    )
    missing_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=b"scratch:missing",
        byte_length=4,
    )

    put_result = session.batch_put_from(
        (
            _transfer(first_artifact, first_source),
            _transfer(second_artifact, second_source),
        )
    )
    exists_result = session.batch_exists(
        (first_artifact, missing_artifact, second_artifact)
    )
    first_target = torch.full((first_artifact.byte_length,), 17, dtype=torch.uint8)
    missing_target = torch.full((missing_artifact.byte_length,), 23, dtype=torch.uint8)
    second_target = torch.full((second_artifact.byte_length,), 29, dtype=torch.uint8)
    get_result = session.batch_get_into(
        (
            _transfer(first_artifact, first_target),
            _transfer(missing_artifact, missing_target),
            _transfer(second_artifact, second_target),
        )
    )

    assert put_result.success_mask == (True, True)
    assert exists_result.existence_mask == (True, False, True)
    assert get_result.success_mask == (True, False, True)
    assert bytes(first_target.tolist()) == b"alpha"
    assert missing_target.tolist() == [23] * missing_artifact.byte_length
    assert bytes(second_target.tolist()) == b"bravo!!"

    conflict_source = torch.tensor(list(b"size"), dtype=torch.uint8)
    conflict_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=first_artifact.engine_key,
        byte_length=conflict_source.numel(),
    )
    try:
        session.batch_put_from((_transfer(conflict_artifact, conflict_source),))
    except RegionSessionFailedError as exc:
        failure = exc.failure
    else:
        raise AssertionError("daemon invariant conflict did not fail the Session")
    assert failure.code is RegionSessionFailureCode.DAEMON_STATUS

    def reject_late_rpc(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("failed Session issued a later RPC")

    session._client.batch_exists = reject_late_rpc  # type: ignore[method-assign]
    with pytest.raises(RegionSessionFailedError) as repeated_error:
        session.batch_exists((first_artifact,))
    assert repeated_error.value.failure == failure

    session._client.release_host_shared_region = (  # type: ignore[method-assign]
        reject_late_rpc
    )
    session._client.unregister_region = reject_late_rpc  # type: ignore[method-assign]
    session.terminate_process_session()
    return {
        "put_mask": put_result.success_mask,
        "exists_mask": exists_result.existence_mask,
        "get_mask": get_result.success_mask,
        "failure_code": failure.code.value,
    }


def _allocator_worker(daemon_address: str, run_id: str) -> dict[str, Any]:
    session = RegionBackedArtifactSession.attach(
        RegionBackedArtifactSessionOptions(
            daemon_address=daemon_address,
            session_name=f"allocator-{run_id}",
            transfer=AllocatorTransferOptions(),
        )
    )
    keyspace = _keyspace(run_id)
    first_source = session.allocate_host_tensor((32,), torch.uint8, name="k-region")
    second_source = session.allocate_host_tensor((32,), torch.uint8, name="v-region")
    first_payload = b"key-data"
    second_payload = b"val-data"
    first_source[: len(first_payload)] = torch.tensor(
        list(first_payload), dtype=torch.uint8
    )
    second_source[: len(second_payload)] = torch.tensor(
        list(second_payload), dtype=torch.uint8
    )
    first_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=b"allocator:k",
        byte_length=len(first_payload),
    )
    second_artifact = ByteArtifactSpec(
        keyspace=keyspace,
        engine_key=b"allocator:v",
        byte_length=len(second_payload),
    )
    put_storage_counts: list[int] = []
    original_put = session._client.batch_put_if_absent_from_region

    def observe_put(
        **kwargs: Any,
    ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
        put_storage_counts.append(len(kwargs["source_layout"].storages))
        return original_put(**kwargs)

    session._client.batch_put_if_absent_from_region = (  # type: ignore[method-assign]
        observe_put
    )
    put_result = session.batch_put_from(
        (
            _transfer(first_artifact, first_source[: len(first_payload)]),
            _transfer(second_artifact, second_source[: len(second_payload)]),
        )
    )
    assert put_result.success_mask == (True, True)

    first_source.zero_()
    second_source.zero_()
    first_target = session.allocate_host_tensor((32,), torch.uint8, name="k-target")
    second_target = session.allocate_host_tensor((32,), torch.uint8, name="v-target")
    first_target.fill_(31)
    second_target.fill_(37)
    exists_result = session.batch_exists((first_artifact, second_artifact))
    get_storage_counts: list[int] = []
    original_get = session._client.batch_get_into_region

    def observe_get(**kwargs: Any) -> store_daemon_pb2.BatchGetIntoRegionResponse:
        get_storage_counts.append(len(kwargs["target_layout"].storages))
        return original_get(**kwargs)

    session._client.batch_get_into_region = observe_get  # type: ignore[method-assign]
    get_result = session.batch_get_into(
        (
            _transfer(first_artifact, first_target[: len(first_payload)]),
            _transfer(second_artifact, second_target[: len(second_payload)]),
        )
    )

    assert exists_result.existence_mask == (True, True)
    assert get_result.success_mask == (True, True)
    assert bytes(first_target[: len(first_payload)].tolist()) == first_payload
    assert bytes(second_target[: len(second_payload)].tolist()) == second_payload

    def reject_termination_rpc(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise AssertionError("termination sent a region lifecycle RPC")

    session._client.release_host_shared_region = (  # type: ignore[method-assign]
        reject_termination_rpc
    )
    session._client.unregister_region = reject_termination_rpc  # type: ignore[method-assign]
    session.terminate_process_session()
    return {
        "put_mask": put_result.success_mask,
        "exists_mask": exists_result.existence_mask,
        "get_mask": get_result.success_mask,
        "put_storage_count": put_storage_counts,
        "get_storage_count": get_storage_counts,
    }


def _pid_cleanup_worker(daemon_address: str, run_id: str) -> dict[str, Any]:
    session = RegionBackedArtifactSession.attach(
        RegionBackedArtifactSessionOptions(
            daemon_address=daemon_address,
            session_name=f"pid-cleanup-{run_id}",
            transfer=AllocatorTransferOptions(),
        )
    )
    tensor = session.allocate_host_tensor((4096,), torch.uint8, name="pid-owned")
    tensor.fill_(41)
    record = session._diagnostic_allocation_records[0]
    return {"region_id": record.handle.region_id, "owner_pid": os.getpid()}


def _worker_entry(
    worker: Callable[[str, str], dict[str, Any]],
    daemon_address: str,
    run_id: str,
    connection: Connection,
) -> None:
    try:
        connection.send({"ok": True, "result": worker(daemon_address, run_id)})
    except BaseException:
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _run_worker(
    worker: Callable[[str, str], dict[str, Any]],
    daemon_address: str,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(worker, daemon_address, uuid4().hex, sending),
    )
    process.start()
    sending.close()
    try:
        if not receiving.poll(_SUBPROCESS_TIMEOUT_S):
            process.terminate()
            process.join(timeout=5.0)
            raise AssertionError("region Session acceptance subprocess timed out")
        payload = receiving.recv()
    finally:
        receiving.close()
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        raise AssertionError("region Session acceptance subprocess did not exit")
    assert process.exitcode == 0
    assert payload["ok"], payload.get("traceback", "subprocess failed")
    return payload["result"]


@pytest.fixture(scope="module")
def daemon_address(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    set_config(GlobalStoreConfig())
    global_store_servicer = GlobalStoreServicer()
    global_store_server = grpc.server(ThreadPoolExecutor(max_workers=8))
    register_global_store_servicers(global_store_server, global_store_servicer)
    global_store_port = global_store_server.add_insecure_port("127.0.0.1:0")
    if global_store_port <= 0:
        raise RuntimeError("failed to bind Global Store server port")
    global_store_server.start()

    daemon_port = get_free_port()
    address = f"127.0.0.1:{daemon_port}"
    root = tmp_path_factory.mktemp("region_session_e2e")
    daemon_process = start_daemon_binary(
        address,
        root / "storage",
        global_store_addr=f"127.0.0.1:{global_store_port}",
        local_handle_socket_path=str(root / "local_handle.sock"),
        stable_bytes=64 * 1024 * 1024,
    )
    try:
        yield address
    finally:
        daemon_process.terminate()
        try:
            daemon_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            daemon_process.kill()
            daemon_process.wait(timeout=5.0)
        global_store_server.stop(grace=None)


def test_scratch_round_trip_partial_hit_rejection_and_termination(
    daemon_address: str,
) -> None:
    result = _run_worker(_scratch_worker, daemon_address)

    assert result == {
        "put_mask": (True, True),
        "exists_mask": (True, False, True),
        "get_mask": (True, False, True),
        "failure_code": "daemon_status",
    }


def test_allocator_multi_region_round_trip_and_put_quiescence(
    daemon_address: str,
) -> None:
    result = _run_worker(_allocator_worker, daemon_address)

    assert result == {
        "put_mask": (True, True),
        "exists_mask": (True, True),
        "get_mask": (True, True),
        "put_storage_count": [2],
        "get_storage_count": [2],
    }


def test_subprocess_exit_reclaims_process_pinned_region(
    daemon_address: str,
) -> None:
    result = _run_worker(_pid_cleanup_worker, daemon_address)
    time.sleep(2.0)

    channel = grpc.insecure_channel(daemon_address)
    stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
    try:
        try:
            response = stub.UnregisterRegion(
                store_daemon_pb2.UnregisterRegionRequest(
                    region_id=result["region_id"],
                    owner_pid=result["owner_pid"],
                ),
                timeout=5.0,
            )
        except grpc.RpcError as exc:
            assert exc.code() is grpc.StatusCode.NOT_FOUND
        else:
            assert not response.released
    finally:
        channel.close()
