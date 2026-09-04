#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import gc
import mmap
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, NoReturn, Sequence

import grpc
import pytest
import torch
from pydantic import ValidationError

import tensorcast.api.store as store_api
import tensorcast.api.store.region_backed_artifact_session as region_session
import tensorcast.daemon_ctl as daemon_ctl
from tensorcast.api.store.region_backed_artifact_session import (
    AllocatorTransferOptions,
    ByteArtifactKeyspace,
    ByteArtifactSpec,
    HostMemorySpan,
    RegionArtifactExistsResult,
    RegionArtifactInputError,
    RegionArtifactTransfer,
    RegionArtifactTransferResult,
    RegionBackedArtifactSession,
    RegionBackedArtifactSessionOptions,
    RegionSessionAttachError,
    RegionSessionFailedError,
    RegionSessionFailure,
    RegionSessionFailureCode,
    RegionSessionHealth,
    RegionSessionLifecycleState,
    RegionSessionOperationKind,
    RegionSessionTerminatedError,
    RegionTransferMode,
    ScratchTransferOptions,
)
from tensorcast.common.identity import build_byte_artifact_cgid
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import (
    HostSharedRegionAttachment,
    HostSharedRegionClass,
    LocalRegionHandle,
    RegionMemoryKind,
    ServerConfig,
)


def _keyspace() -> ByteArtifactKeyspace:
    return ByteArtifactKeyspace(
        namespace="serving",
        engine="runtime",
        model_id="model-a",
        model_version="revision-42",
        layout_id="runtime-cache-v1",
    )


def _artifact(
    *,
    byte_length: int = 64,
    engine_key: bytes = b"layer-0:page-1",
    keyspace: ByteArtifactKeyspace | None = None,
) -> ByteArtifactSpec:
    return ByteArtifactSpec(
        keyspace=keyspace or _keyspace(),
        engine_key=engine_key,
        byte_length=byte_length,
    )


def _transfer(
    tensor: torch.Tensor,
    *,
    offset_bytes: int,
    byte_length: int,
    engine_key: bytes,
    keyspace: ByteArtifactKeyspace | None = None,
) -> RegionArtifactTransfer:
    return RegionArtifactTransfer(
        artifact=_artifact(
            byte_length=byte_length,
            engine_key=engine_key,
            keyspace=keyspace,
        ),
        span=HostMemorySpan.from_tensor(
            tensor,
            offset_bytes=offset_bytes,
            byte_length=byte_length,
        ),
    )


def _successful_direct_result(
    prepared: region_session._PreparedDirectTransfer,
) -> RegionArtifactTransferResult:
    response = store_daemon_pb2.BatchGetIntoRegionResponse()
    for expected in reversed(prepared.expected_outcomes()):
        response.outcomes.add(
            artifact_id=expected.artifact_id,
            status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            slot_index=expected.slot_index,
            slot_generation=expected.slot_generation,
        )
    success_mask = region_session._validate_batch_outcomes(
        response.outcomes,
        prepared.expected_outcomes(),
        operation_kind=prepared.operation_kind,
    )
    return RegionArtifactTransferResult(
        success_mask=success_mask,
        operation_id=prepared.operation_id,
        pack_elapsed_s=0.0,
        copy_elapsed_s=0.0,
        rpc_elapsed_s=0.0,
    )


def _session_options(
    *,
    daemon_address: str = "127.0.0.1:8073",
    session_name: str = "worker-0",
    region_name_prefix: str = "tensorcast_region_artifact",
    transfer: ScratchTransferOptions | AllocatorTransferOptions | None = None,
    transfer_timeout_s: float | None = None,
    exists_timeout_s: float = 30.0,
) -> RegionBackedArtifactSessionOptions:
    return RegionBackedArtifactSessionOptions(
        daemon_address=daemon_address,
        session_name=session_name,
        region_name_prefix=region_name_prefix,
        transfer=transfer or AllocatorTransferOptions(),
        transfer_timeout_s=transfer_timeout_s,
        exists_timeout_s=exists_timeout_s,
    )


def _server_config(
    *,
    startup_phase: int = store_daemon_pb2.DAEMON_STARTUP_PHASE_READY,
    cpu_shared_memory_enabled: bool = True,
    local_handle_socket_path: str = "/tmp/tensorcast-local-handle.sock",
) -> ServerConfig:
    return ServerConfig(
        tx_slice_bytes=1024,
        mem_pool_size=4096,
        startup_phase=startup_phase,
        cpu_shared_memory_enabled=cpu_shared_memory_enabled,
        local_handle_socket_path=local_handle_socket_path,
    )


@dataclass(frozen=True, slots=True)
class _GetRegionCall:
    selections: tuple[common_pb2.ArtifactSelection, ...]
    target_layout: store_daemon_pb2.TargetLayout
    pid: int
    device_uuid: str
    operation_id: str | None
    timeout_s: float | None
    retries: int


@dataclass(frozen=True, slots=True)
class _PutRegionCall:
    items: tuple[store_daemon_pb2.BatchPutIfAbsentFromRegionItem, ...]
    source_layout: store_daemon_pb2.TargetLayout
    pid: int
    device_uuid: str
    ttl_ms: int | None
    operation_id: str | None
    timeout_s: float | None
    retries: int


class _FakeDaemonClient:
    def __init__(
        self,
        config: ServerConfig,
        *,
        config_error: Exception | None = None,
    ) -> None:
        self.config = config
        self.config_error = config_error
        self.config_calls = 0
        self.close_calls = 0
        self.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            max_receive_message_bytes=daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES,
        )
        self.exists_calls: list[
            tuple[tuple[common_pb2.ArtifactSelection, ...], float, str | None]
        ] = []

    @property
    def _effective_grpc_message_limits(self) -> daemon_ctl._GrpcMessageLimits:
        return self.message_limits

    def get_server_config(self) -> ServerConfig:
        self.config_calls += 1
        if self.config_error is not None:
            raise self.config_error
        return self.config

    def close(self) -> None:
        self.close_calls += 1

    def batch_exists(
        self,
        *,
        selections: Sequence[common_pb2.ArtifactSelection],
        timeout_s: float,
        operation_id: str | None = None,
    ) -> store_daemon_pb2.BatchExistsResponse:
        selection_snapshot = tuple(selections)
        self.exists_calls.append((selection_snapshot, timeout_s, operation_id))
        response = store_daemon_pb2.BatchExistsResponse()
        for selection in selection_snapshot:
            response.outcomes.add(
                artifact_id=selection.artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            )
        return response


class _FakeRegionDaemonClient(_FakeDaemonClient):
    def __init__(self) -> None:
        super().__init__(_server_config())
        self.register_calls: list[dict[str, object]] = []
        self.attach_calls: list[LocalRegionHandle] = []
        self.release_calls: list[LocalRegionHandle] = []
        self.unregister_calls: list[str] = []
        self.registration_error: BaseException | None = None
        self.attachment_error: BaseException | None = None
        self.release_result = True
        self.unregister_result = True
        self._daemon_fds: dict[str, int] = {}
        self._attached_fds: list[int] = []
        self.get_region_calls: list[_GetRegionCall] = []
        self.put_region_calls: list[_PutRegionCall] = []

    def register_region(
        self,
        *,
        memory_kind: RegionMemoryKind,
        size_bytes: int,
        ttl_ms: int,
        daemon_managed: bool,
        host_shared_region_class: HostSharedRegionClass,
        region_name: str,
    ) -> LocalRegionHandle:
        self.register_calls.append(
            {
                "memory_kind": memory_kind,
                "size_bytes": size_bytes,
                "ttl_ms": ttl_ms,
                "daemon_managed": daemon_managed,
                "host_shared_region_class": host_shared_region_class,
                "region_name": region_name,
            }
        )
        if self.registration_error is not None:
            raise self.registration_error
        region_id = f"region:{len(self.register_calls)}"
        file_descriptor = os.memfd_create(region_id)
        os.ftruncate(file_descriptor, size_bytes)
        self._daemon_fds[region_id] = file_descriptor
        return LocalRegionHandle(
            region_id=region_id,
            memory_kind=memory_kind,
            ttl_ms=ttl_ms,
            size_bytes=size_bytes,
            attach_token=f"token:{region_id}".encode(),
            daemon_managed=daemon_managed,
            host_shared_region_class=host_shared_region_class,
        )

    def attach_host_shared_region(
        self,
        handle: LocalRegionHandle,
    ) -> HostSharedRegionAttachment:
        self.attach_calls.append(handle)
        if self.attachment_error is not None:
            raise self.attachment_error
        attached_fd = os.dup(self._daemon_fds[handle.region_id])
        self._attached_fds.append(attached_fd)
        return HostSharedRegionAttachment(
            region_id=handle.region_id,
            size_bytes=handle.size_bytes,
            attach_token=handle.attach_token,
            fd=attached_fd,
        )

    def release_host_shared_region(self, handle: LocalRegionHandle) -> bool:
        self.release_calls.append(handle)
        return self.release_result

    def unregister_region(self, region_id: str) -> bool:
        self.unregister_calls.append(region_id)
        if self.unregister_result:
            file_descriptor = self._daemon_fds.pop(region_id, None)
            if file_descriptor is not None:
                os.close(file_descriptor)
        return self.unregister_result

    def batch_get_into_region(
        self,
        *,
        selections: Sequence[common_pb2.ArtifactSelection],
        target_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        operation_id: str | None = None,
        timeout_s: float | None = 600.0,
        retries: int = 1,
    ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
        layout_snapshot = store_daemon_pb2.TargetLayout()
        layout_snapshot.CopyFrom(target_layout)
        selection_snapshot = tuple(selections)
        self.get_region_calls.append(
            _GetRegionCall(
                selections=selection_snapshot,
                target_layout=layout_snapshot,
                pid=pid,
                device_uuid=device_uuid,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
        )
        response = store_daemon_pb2.BatchGetIntoRegionResponse()
        offsets_by_name = {offset.name: offset for offset in target_layout.offsets}
        for selection in selection_snapshot:
            outcome = response.outcomes.add(
                artifact_id=selection.artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            )
            offset = offsets_by_name[selection.artifact_id]
            if offset.HasField("slot_index"):
                outcome.slot_index = offset.slot_index
                outcome.slot_generation = offset.slot_generation
        return response

    def batch_put_if_absent_from_region(
        self,
        *,
        items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
        source_layout: store_daemon_pb2.TargetLayout,
        pid: int,
        device_uuid: str,
        ttl_ms: int | None = None,
        operation_id: str | None = None,
        timeout_s: float | None = 600.0,
        retries: int = 1,
    ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
        layout_snapshot = store_daemon_pb2.TargetLayout()
        layout_snapshot.CopyFrom(source_layout)
        item_snapshot = tuple(items)
        self.put_region_calls.append(
            _PutRegionCall(
                items=item_snapshot,
                source_layout=layout_snapshot,
                pid=pid,
                device_uuid=device_uuid,
                ttl_ms=ttl_ms,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
        )
        response = store_daemon_pb2.BatchPutIfAbsentFromRegionResponse()
        offsets_by_name = {offset.name: offset for offset in source_layout.offsets}
        for item in item_snapshot:
            outcome = response.outcomes.add(
                artifact_id=item.selection.artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            )
            offset = offsets_by_name[item.selection.artifact_id]
            if offset.HasField("slot_index"):
                outcome.slot_index = offset.slot_index
                outcome.slot_generation = offset.slot_generation
        return response

    def read_region_bytes(
        self,
        region_id: str,
        *,
        offset: int,
        byte_length: int,
    ) -> bytes:
        return os.pread(self._daemon_fds[region_id], byte_length, offset)

    def write_region_bytes(
        self,
        region_id: str,
        payload: bytes,
        *,
        offset: int,
    ) -> None:
        written = os.pwrite(self._daemon_fds[region_id], payload, offset)
        assert written == len(payload)

    def close_test_file_descriptors(self) -> None:
        for file_descriptor in self._daemon_fds.values():
            with contextlib.suppress(OSError):
                os.close(file_descriptor)
        self._daemon_fds.clear()
        for file_descriptor in self._attached_fds:
            with contextlib.suppress(OSError):
                os.close(file_descriptor)
        self._attached_fds.clear()


def _install_fake_attach(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeDaemonClient,
) -> tuple[list[str], list[str]]:
    factory_addresses: list[str] = []
    probe_paths: list[str] = []

    def client_factory(address: str) -> _FakeDaemonClient:
        factory_addresses.append(address)
        return client

    def local_handle_probe(socket_path: str) -> None:
        probe_paths.append(socket_path)

    monkeypatch.setattr(region_session, "_DAEMON_CLIENT_FACTORY", client_factory)
    monkeypatch.setattr(
        region_session,
        "_LOCAL_HANDLE_SERVICE_PROBE",
        local_handle_probe,
    )
    return factory_addresses, probe_paths


def _attach_allocator_session(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeRegionDaemonClient,
) -> RegionBackedArtifactSession:
    _install_fake_attach(monkeypatch, client)
    return RegionBackedArtifactSession.attach(_session_options())


def _attach_scratch_session(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeRegionDaemonClient,
    *,
    capacity_bytes: int = 4096,
    transfer_timeout_s: float | None = None,
) -> RegionBackedArtifactSession:
    _install_fake_attach(monkeypatch, client)
    return RegionBackedArtifactSession.attach(
        _session_options(
            transfer=ScratchTransferOptions(capacity_bytes=capacity_bytes),
            transfer_timeout_s=transfer_timeout_s,
        )
    )


def _failure(
    message: str,
    *,
    operation_id: str = "operation-1",
) -> RegionSessionFailure:
    return RegionSessionFailure(
        code=RegionSessionFailureCode.TRANSPORT,
        message=message,
        operation_kind=RegionSessionOperationKind.GET_INTO,
        operation_id=operation_id,
        occurred_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def _isolate_process_session_registry() -> Iterator[None]:
    registry_lock = region_session._PROCESS_SESSION_REGISTRY_LOCK
    with registry_lock:
        region_session._PROCESS_SESSION_REGISTRY.clear()
    yield
    with registry_lock:
        region_session._PROCESS_SESSION_REGISTRY.clear()


class TestPublicContracts:
    def test_models_are_frozen_forbid_extra_and_discriminate_transfer_mode(
        self,
    ) -> None:
        options = RegionBackedArtifactSessionOptions.model_validate(
            {
                "daemon_address": "unix:///run/tensorcast/store.sock",
                "session_name": "worker-0",
                "transfer": {"mode": "scratch", "capacity_bytes": 4096},
            }
        )

        assert isinstance(options.transfer, ScratchTransferOptions)
        assert options.transfer.mode is RegionTransferMode.SCRATCH
        with pytest.raises(ValidationError, match="frozen"):
            options.session_name = "worker-1"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ByteArtifactKeyspace.model_validate(
                {
                    **_keyspace().model_dump(),
                    "artifact_id": "caller-must-not-supply-this",
                }
            )
        with pytest.raises(ValidationError):
            RegionBackedArtifactSessionOptions.model_validate(
                {
                    "daemon_address": "127.0.0.1:8073",
                    "session_name": "worker-0",
                    "transfer": {
                        "mode": "allocator",
                        "capacity_bytes": 4096,
                    },
                }
            )

        allocator_options = RegionBackedArtifactSessionOptions(
            daemon_address="127.0.0.1:8073",
            session_name="worker-0",
            transfer=AllocatorTransferOptions(),
        )
        assert isinstance(allocator_options.transfer, AllocatorTransferOptions)
        assert allocator_options.transfer.mode is RegionTransferMode.ALLOCATOR

    @pytest.mark.parametrize(
        "field_name",
        ["namespace", "engine", "model_id", "model_version", "layout_id"],
    )
    def test_rejects_empty_keyspace_identity(self, field_name: str) -> None:
        values = _keyspace().model_dump()
        values[field_name] = "  "

        with pytest.raises(ValidationError, match="must not be empty"):
            ByteArtifactKeyspace.model_validate(values)

    @pytest.mark.parametrize("byte_length", [0, -1])
    def test_rejects_invalid_artifact_lengths(self, byte_length: int) -> None:
        with pytest.raises(ValidationError):
            _artifact(byte_length=byte_length)

    def test_rejects_empty_engine_key_and_invalid_scratch_capacity(self) -> None:
        with pytest.raises(ValidationError, match="engine_key must not be empty"):
            ByteArtifactSpec(
                keyspace=_keyspace(),
                engine_key=b"",
                byte_length=1,
            )
        with pytest.raises(ValidationError):
            ScratchTransferOptions(capacity_bytes=0)

    @pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_timeouts(self, timeout_s: float) -> None:
        with pytest.raises(ValidationError):
            RegionBackedArtifactSessionOptions(
                daemon_address="127.0.0.1:8073",
                session_name="worker-0",
                transfer=AllocatorTransferOptions(),
                transfer_timeout_s=timeout_s,
            )
        with pytest.raises(ValidationError):
            RegionBackedArtifactSessionOptions(
                daemon_address="127.0.0.1:8073",
                session_name="worker-0",
                transfer=AllocatorTransferOptions(),
                exists_timeout_s=timeout_s,
            )

    @pytest.mark.parametrize(
        "field_name",
        ["daemon_address", "session_name", "region_name_prefix"],
    )
    def test_rejects_empty_session_options(self, field_name: str) -> None:
        values: dict[str, object] = {
            "daemon_address": "127.0.0.1:8073",
            "session_name": "worker-0",
            "transfer": {"mode": "allocator"},
            "region_name_prefix": "tensorcast_region_artifact",
        }
        values[field_name] = "\t"

        with pytest.raises(ValidationError, match="must not be empty"):
            RegionBackedArtifactSessionOptions.model_validate(values)

    def test_span_and_transfer_reject_malformed_lengths(self) -> None:
        owner = object()
        with pytest.raises(TypeError, match="from_tensor"):
            HostMemorySpan()
        with pytest.raises(RegionArtifactInputError, match="positive"):
            HostMemorySpan.from_address(4096, 0, owner=owner)
        with pytest.raises(RegionArtifactInputError, match="owner"):
            HostMemorySpan.from_address(4096, 64, owner=None)

        span = HostMemorySpan.from_address(4096, 32, owner=owner)
        with pytest.raises(RegionArtifactInputError, match="must equal"):
            RegionArtifactTransfer(artifact=_artifact(byte_length=64), span=span)

    def test_result_masks_are_tuples_and_empty_transfer_id_is_nullable(self) -> None:
        exists_result = RegionArtifactExistsResult(
            existence_mask=[True, False],
            rpc_elapsed_s=0.5,
        )
        transfer_result = RegionArtifactTransferResult(
            success_mask=[],
            operation_id=None,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
        )

        assert exists_result.existence_mask == (True, False)
        assert isinstance(exists_result.existence_mask, tuple)
        assert transfer_result.success_mask == ()
        assert isinstance(transfer_result.success_mask, tuple)
        assert transfer_result.operation_id is None
        with pytest.raises(ValidationError, match="requires an operation_id"):
            RegionArtifactTransferResult(
                success_mask=(True,),
                operation_id=None,
                pack_elapsed_s=0.0,
                copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
            )

    def test_public_exception_inheritance_and_failure_schema(self) -> None:
        occurred_at = datetime.now(timezone(timedelta(hours=8)))
        failure = RegionSessionFailure(
            code=RegionSessionFailureCode.TRANSPORT,
            message="daemon transport failed",
            operation_kind=RegionSessionOperationKind.GET_INTO,
            operation_id="get-17",
            occurred_at=occurred_at,
        )
        error = RegionSessionFailedError(failure)

        assert issubclass(RegionArtifactInputError, ValueError)
        assert issubclass(RegionSessionAttachError, RuntimeError)
        assert issubclass(RegionSessionFailedError, RuntimeError)
        assert issubclass(RegionSessionTerminatedError, RuntimeError)
        assert failure.occurred_at.tzinfo is timezone.utc
        assert error.failure is failure
        assert str(error) == failure.message
        with pytest.raises(ValidationError, match="timezone-aware"):
            RegionSessionFailure(
                code=RegionSessionFailureCode.INTERNAL,
                message="unexpected SDK error",
                operation_kind=RegionSessionOperationKind.ALLOCATE,
                operation_id=None,
                occurred_at=datetime.now(),
            )

    def test_caller_models_do_not_expose_internal_identity_or_wire_types(self) -> None:
        public_models = (
            ByteArtifactKeyspace,
            ByteArtifactSpec,
            ScratchTransferOptions,
            AllocatorTransferOptions,
            RegionBackedArtifactSessionOptions,
            RegionSessionFailure,
            RegionArtifactExistsResult,
            RegionArtifactTransferResult,
        )

        for model_type in public_models:
            assert "artifact_id" not in model_type.model_fields
            for field in model_type.model_fields.values():
                annotation = str(field.annotation)
                assert "tensorcast.proto" not in annotation
                assert "RegionHandle" not in annotation
        assert "artifact_id" not in RegionArtifactTransfer.__annotations__
        assert store_api.RegionBackedArtifactSession is RegionBackedArtifactSession
        assert "RegionBackedArtifactSession" in store_api.__all__
        assert store_api.HostMemorySpan is HostMemorySpan
        assert "HostMemorySpan" in store_api.__all__
        assert RegionSessionHealth.READY.value == "ready"
        assert RegionSessionLifecycleState.ATTACHED.value == "attached"


class TestHostMemorySpan:
    def test_tensor_view_address_offset_and_length_are_snapshotted(self) -> None:
        tensor = torch.arange(32, dtype=torch.uint8)[4:24]

        span = HostMemorySpan.from_tensor(
            tensor,
            offset_bytes=3,
            byte_length=11,
        )
        captured_address = span.address
        tensor.fill_(0)

        assert tensor.is_contiguous()
        assert captured_address == tensor.data_ptr() + 3
        assert span.address == captured_address
        assert span.byte_length == 11

    def test_rejects_non_contiguous_non_cpu_and_invalid_tensor_ranges(self) -> None:
        with pytest.raises(RegionArtifactInputError, match="contiguous"):
            HostMemorySpan.from_tensor(
                torch.empty((4, 4), dtype=torch.float32).t(),
                offset_bytes=0,
                byte_length=4,
            )
        with pytest.raises(RegionArtifactInputError, match="CPU"):
            HostMemorySpan.from_tensor(
                torch.empty((4,), device="meta"),
                offset_bytes=0,
                byte_length=4,
            )

        tensor = torch.empty((8,), dtype=torch.uint8)
        for offset_bytes, byte_length in ((-1, 1), (0, 0), (4, 5)):
            with pytest.raises(RegionArtifactInputError):
                HostMemorySpan.from_tensor(
                    tensor,
                    offset_bytes=offset_bytes,
                    byte_length=byte_length,
                )
        with pytest.raises(RegionArtifactInputError, match="integer"):
            HostMemorySpan.from_tensor(
                tensor,
                offset_bytes=0.5,
                byte_length=1,
            )

    def test_raw_address_requires_owner_and_rejects_invalid_ranges(self) -> None:
        owner = object()
        with pytest.raises(RegionArtifactInputError, match="owner"):
            HostMemorySpan.from_address(4096, 1, owner=None)
        for address, byte_length in ((0, 1), (-1, 1), (4096, 0), (4096, -1)):
            with pytest.raises(RegionArtifactInputError):
                HostMemorySpan.from_address(address, byte_length, owner=owner)
        with pytest.raises(RegionArtifactInputError, match="overflows"):
            HostMemorySpan.from_address(
                region_session._MAX_POINTER_VALUE,
                2,
                owner=owner,
            )
        with pytest.raises(RegionArtifactInputError, match="integer"):
            HostMemorySpan.from_address(
                4096,
                1.5,
                owner=owner,
            )

    def test_tensor_and_explicit_owner_are_strongly_retained(self) -> None:
        class Owner:
            pass

        tensor = torch.empty((16,), dtype=torch.uint8)
        tensor_reference = weakref.ref(tensor)
        tensor_span = HostMemorySpan.from_tensor(
            tensor,
            offset_bytes=0,
            byte_length=16,
        )
        del tensor
        gc.collect()
        assert tensor_reference() is not None

        owner = Owner()
        owner_reference = weakref.ref(owner)
        raw_span = HostMemorySpan.from_address(4096, 16, owner=owner)
        del owner
        gc.collect()
        assert owner_reference() is not None

        with pytest.raises(RegionArtifactInputError, match="must equal"):
            RegionArtifactTransfer(
                artifact=_artifact(byte_length=32),
                span=raw_span,
            )
        gc.collect()
        assert owner_reference() is not None

        del tensor_span, raw_span
        gc.collect()
        assert tensor_reference() is None
        assert owner_reference() is None


class _DirectRpcError(grpc.RpcError):
    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.UNKNOWN

    def details(self) -> str:
        return "direct RPC failed"


class TestGrpcMessageLimits:
    @pytest.fixture(autouse=True)
    def _reset_message_limit_caches(self) -> Iterator[None]:
        daemon_ctl._grpc_max_send_message_bytes.cache_clear()
        daemon_ctl._grpc_max_receive_message_bytes.cache_clear()
        yield
        daemon_ctl._grpc_max_send_message_bytes.cache_clear()
        daemon_ctl._grpc_max_receive_message_bytes.cache_clear()

    def test_defaults_and_environment_overrides_are_resolved_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES", raising=False)
        monkeypatch.delenv("TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES", raising=False)
        default_client = DaemonCtl("127.0.0.1:65535")
        try:
            assert (
                default_client._effective_grpc_message_limits.max_send_message_bytes
                == daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES
            )
            assert (
                default_client._effective_grpc_message_limits.max_receive_message_bytes
                == daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES
            )
        finally:
            default_client.close()

        daemon_ctl._grpc_max_send_message_bytes.cache_clear()
        daemon_ctl._grpc_max_receive_message_bytes.cache_clear()
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES", "2097152")
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES", "3145728")
        configured_client = DaemonCtl("127.0.0.1:65535")
        try:
            assert configured_client._effective_grpc_message_limits == (
                daemon_ctl._GrpcMessageLimits(
                    max_send_message_bytes=2 * 1024 * 1024,
                    max_receive_message_bytes=3 * 1024 * 1024,
                )
            )
        finally:
            configured_client.close()

    def test_refresh_reuses_snapshot_after_environment_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES", "2097152")
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES", "3145728")
        client = DaemonCtl("127.0.0.1:65535")
        original_limits = client._effective_grpc_message_limits

        try:
            monkeypatch.setenv("TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES", "4194304")
            monkeypatch.setenv("TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES", "5242880")
            daemon_ctl._grpc_max_send_message_bytes.cache_clear()
            daemon_ctl._grpc_max_receive_message_bytes.cache_clear()
            client._refresh_channel()

            assert client._effective_grpc_message_limits is original_limits
            assert (
                client._effective_grpc_message_limits
                == daemon_ctl._GrpcMessageLimits(
                    max_send_message_bytes=2 * 1024 * 1024,
                    max_receive_message_bytes=3 * 1024 * 1024,
                )
            )
        finally:
            client.close()


class TestAttachRegistry:
    def test_daemon_client_config_snapshot_retains_startup_phase(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = DaemonCtl("127.0.0.1:65535")
        response = store_daemon_pb2.GetServerConfigResponse(
            tx_slice_bytes=1024,
            mem_pool_size=4096,
            startup_phase=store_daemon_pb2.DAEMON_STARTUP_PHASE_READY,
            cpu_shared_memory_enabled=True,
            local_handle_socket_path="/tmp/tensorcast-local-handle.sock",
        )

        def fake_unary_call(
            method: object,
            request: object,
            *,
            timeout: float | int | None,
            retries: int,
            span: object,
        ) -> store_daemon_pb2.GetServerConfigResponse:
            del method, request, timeout, retries, span
            return response

        monkeypatch.setattr(client, "_unary_call", fake_unary_call)
        try:
            config = client.get_server_config()
        finally:
            client.close()

        assert config.startup_phase == store_daemon_pb2.DAEMON_STARTUP_PHASE_READY

    def test_same_normalized_options_return_one_object_and_first_diagnostics_win(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, probe_paths = _install_fake_attach(monkeypatch, client)

        first = RegionBackedArtifactSession.attach(
            _session_options(
                daemon_address=" 127.0.0.1:8073 ",
                session_name=" first-worker ",
                region_name_prefix=" first-prefix ",
            )
        )
        repeated = RegionBackedArtifactSession.attach(
            _session_options(
                daemon_address="127.0.0.1:8073",
                session_name="second-worker",
                region_name_prefix="second-prefix",
                transfer_timeout_s=None,
                exists_timeout_s=30,
            )
        )

        assert repeated is first
        assert first._owner_pid == os.getpid()
        assert first._canonical_daemon_address == "127.0.0.1:8073"
        assert first._options.session_name == "first-worker"
        assert first._options.region_name_prefix == "first-prefix"
        assert factory_addresses == ["127.0.0.1:8073"]
        assert probe_paths == ["/tmp/tensorcast-local-handle.sock"]
        assert client.config_calls == 1

    def test_different_endpoint_is_rejected_before_client_creation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, _ = _install_fake_attach(monkeypatch, client)
        RegionBackedArtifactSession.attach(_session_options())

        with pytest.raises(RegionSessionAttachError, match="already attached"):
            RegionBackedArtifactSession.attach(
                _session_options(daemon_address="127.0.0.1:8074")
            )

        assert factory_addresses == ["127.0.0.1:8073"]

    def test_equivalent_unix_endpoint_spellings_share_registry_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, _ = _install_fake_attach(monkeypatch, client)

        first = RegionBackedArtifactSession.attach(
            _session_options(
                daemon_address="unix:/tmp/tensorcast/../tensorcast/daemon.sock"
            )
        )
        repeated = RegionBackedArtifactSession.attach(
            _session_options(daemon_address="unix:///tmp/tensorcast/daemon.sock")
        )

        assert repeated is first
        assert factory_addresses == ["unix:///tmp/tensorcast/daemon.sock"]

    @pytest.mark.parametrize(
        ("first_transfer", "second_transfer", "expected_field"),
        [
            (
                AllocatorTransferOptions(),
                ScratchTransferOptions(capacity_bytes=4096),
                "transfer.mode",
            ),
            (
                ScratchTransferOptions(capacity_bytes=4096),
                ScratchTransferOptions(capacity_bytes=8192),
                "transfer.capacity_bytes",
            ),
        ],
    )
    def test_conflicting_transfer_fingerprint_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        first_transfer: ScratchTransferOptions | AllocatorTransferOptions,
        second_transfer: ScratchTransferOptions | AllocatorTransferOptions,
        expected_field: str,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        RegionBackedArtifactSession.attach(_session_options(transfer=first_transfer))

        with pytest.raises(RegionSessionAttachError, match=expected_field):
            RegionBackedArtifactSession.attach(
                _session_options(transfer=second_transfer)
            )

    @pytest.mark.parametrize(
        ("first_timeout", "second_timeout", "expected_field"),
        [
            (None, 10.0, "transfer_timeout_s"),
            (30.0, 31.0, "exists_timeout_s"),
        ],
    )
    def test_conflicting_timeout_fingerprint_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        first_timeout: float | None,
        second_timeout: float,
        expected_field: str,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        if expected_field == "transfer_timeout_s":
            first_options = _session_options(transfer_timeout_s=first_timeout)
            second_options = _session_options(transfer_timeout_s=second_timeout)
        else:
            assert first_timeout is not None
            first_options = _session_options(exists_timeout_s=first_timeout)
            second_options = _session_options(exists_timeout_s=second_timeout)
        RegionBackedArtifactSession.attach(first_options)

        with pytest.raises(RegionSessionAttachError, match=expected_field):
            RegionBackedArtifactSession.attach(second_options)

    def test_concurrent_attach_performs_one_handshake_and_publication(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, probe_paths = _install_fake_attach(monkeypatch, client)
        caller_count = 8
        start = threading.Barrier(caller_count)

        def attach_after_barrier() -> RegionBackedArtifactSession:
            start.wait()
            return RegionBackedArtifactSession.attach(_session_options())

        with ThreadPoolExecutor(max_workers=caller_count) as executor:
            sessions = tuple(
                executor.map(lambda _: attach_after_barrier(), range(caller_count))
            )

        assert all(session is sessions[0] for session in sessions)
        assert factory_addresses == ["127.0.0.1:8073"]
        assert probe_paths == ["/tmp/tensorcast-local-handle.sock"]
        assert client.config_calls == 1
        with region_session._PROCESS_SESSION_REGISTRY_LOCK:
            assert len(region_session._PROCESS_SESSION_REGISTRY) == 1

    @pytest.mark.parametrize(
        ("config", "expected_message"),
        [
            (
                _server_config(
                    startup_phase=store_daemon_pb2.DAEMON_STARTUP_PHASE_LISTENING
                ),
                "not ready",
            ),
            (
                _server_config(cpu_shared_memory_enabled=False),
                "CPU shared memory is disabled",
            ),
            (
                _server_config(local_handle_socket_path=""),
                "local_handle_socket_path is missing",
            ),
        ],
    )
    def test_missing_attach_capability_raises_without_publication(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ServerConfig,
        expected_message: str,
    ) -> None:
        client = _FakeDaemonClient(config)
        _install_fake_attach(monkeypatch, client)

        with pytest.raises(RegionSessionAttachError) as error_info:
            RegionBackedArtifactSession.attach(_session_options())

        assert expected_message in str(error_info.value.__cause__)
        with region_session._PROCESS_SESSION_REGISTRY_LOCK:
            assert not region_session._PROCESS_SESSION_REGISTRY

    def test_connection_failure_preserves_cause_and_publishes_no_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        connection_error = _DirectRpcError()
        failing_client = _FakeDaemonClient(
            _server_config(),
            config_error=connection_error,
        )
        _install_fake_attach(monkeypatch, failing_client)

        with pytest.raises(RegionSessionAttachError) as error_info:
            RegionBackedArtifactSession.attach(_session_options())

        assert error_info.value.__cause__ is connection_error
        with region_session._PROCESS_SESSION_REGISTRY_LOCK:
            assert not region_session._PROCESS_SESSION_REGISTRY

        healthy_client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, healthy_client)
        attached = RegionBackedArtifactSession.attach(_session_options())
        assert attached.health is RegionSessionHealth.READY

    def test_equal_reattach_returns_the_same_failed_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, _ = _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        failure = _failure("sticky failure")
        session._latch_failure(failure)

        repeated = RegionBackedArtifactSession.attach(_session_options())

        assert repeated is session
        assert repeated.failure is failure
        assert factory_addresses == ["127.0.0.1:8073"]

    def test_unreachable_local_handle_service_preserves_cause(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        connection_error = ConnectionRefusedError("local FD service unavailable")

        def fail_probe(socket_path: str) -> NoReturn:
            del socket_path
            raise connection_error

        monkeypatch.setattr(
            region_session,
            "_LOCAL_HANDLE_SERVICE_PROBE",
            fail_probe,
        )

        with pytest.raises(RegionSessionAttachError) as error_info:
            RegionBackedArtifactSession.attach(_session_options())

        assert error_info.value.__cause__ is connection_error

    def test_non_local_endpoint_is_rejected_before_client_creation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, _ = _install_fake_attach(monkeypatch, client)

        with pytest.raises(RegionSessionAttachError) as error_info:
            RegionBackedArtifactSession.attach(
                _session_options(daemon_address="192.0.2.1:8073")
            )

        assert "not node-local" in str(error_info.value.__cause__)
        assert not factory_addresses

    def test_registry_lock_is_not_used_by_empty_data_plane_methods(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())

        class FailIfAcquired:
            def __enter__(self) -> NoReturn:
                raise AssertionError("data plane acquired process registry lock")

            def __exit__(
                self,
                exc_type: object,
                exc_value: object,
                traceback: object,
            ) -> None:
                del exc_type, exc_value, traceback

        monkeypatch.setattr(
            region_session,
            "_PROCESS_SESSION_REGISTRY_LOCK",
            FailIfAcquired(),
        )

        assert session.batch_exists(()).existence_mask == ()
        assert session.batch_get_into(()).success_mask == ()
        assert session.batch_put_from(()).success_mask == ()


class TestRegionAllocation:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    def test_memfd_allocation_returns_exact_page_aligned_cpu_tensor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)

        tensor = session.allocate_host_tensor(
            (3, 5),
            torch.float32,
            name="K cache / layer 0",
        )

        assert tensor.shape == (3, 5)
        assert tensor.dtype is torch.float32
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert tensor.numel() * tensor.element_size() == 60
        assert tensor.data_ptr() % mmap.PAGESIZE == 0
        assert region_client.register_calls == [
            {
                "memory_kind": RegionMemoryKind.HOST_SHARED,
                "size_bytes": 60,
                "ttl_ms": 0,
                "daemon_managed": True,
                "host_shared_region_class": HostSharedRegionClass.ALLOCATOR,
                "region_name": (
                    "tensorcast_region_artifact-worker-0-"
                    f"{os.getpid()}-1-K_cache_layer_0"
                ),
            }
        ]
        record = session._diagnostic_allocation_records[0]
        assert (
            record.lifecycle is region_session._RegionAllocationLifecycle.PROCESS_PINNED
        )
        assert record.capacity_bytes == 60
        assert record.base_address == tensor.data_ptr()
        assert record.tensor_root is tensor
        assert record.storage_root is not None
        assert record.mapped_region is not None
        assert record.file_descriptor is not None
        assert record.view_escaped is True
        assert record.data_rpc_used is False

    def test_multiple_allocations_remain_live_and_resolve_independently(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        first = session.allocate_host_tensor((64,), torch.uint8, name="key")
        second = session.allocate_host_tensor((8,), torch.float32, name="value")
        first_reference = weakref.ref(first)
        first_span = HostMemorySpan.from_tensor(
            first,
            offset_bytes=7,
            byte_length=16,
        )
        second_span = HostMemorySpan.from_tensor(
            second,
            offset_bytes=8,
            byte_length=16,
        )

        first_resolution = session._resolve_allocation_span(first_span)
        second_resolution = session._resolve_allocation_span(second_span)

        assert len(session._diagnostic_allocation_records) == 2
        assert first_resolution.record.handle.region_id == "region:1"
        assert first_resolution.region_offset == 7
        assert second_resolution.record.handle.region_id == "region:2"
        assert second_resolution.region_offset == 8

        del first
        gc.collect()
        assert first_reference() is not None
        assert session._resolve_allocation_span(first_span) == first_resolution

    def test_cross_region_missing_and_ambiguous_ranges_fail_locally(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        first = session.allocate_host_tensor((64,), torch.uint8, name="first")
        second = session.allocate_host_tensor((64,), torch.uint8, name="second")

        crossing = HostMemorySpan.from_address(
            first.data_ptr() + 60,
            8,
            owner=first,
        )
        missing = HostMemorySpan.from_address(4096, 8, owner=object())
        with pytest.raises(RegionArtifactInputError, match="not fully contained"):
            session._resolve_allocation_span(crossing)
        with pytest.raises(RegionArtifactInputError, match="not fully contained"):
            session._resolve_allocation_span(missing)

        first_record, second_record = session._diagnostic_allocation_records
        original_second_base = second_record.base_address
        with session._region_lock:
            second_record.base_address = first_record.base_address
        ambiguous = HostMemorySpan.from_tensor(
            first,
            offset_bytes=0,
            byte_length=8,
        )
        with pytest.raises(RegionArtifactInputError, match="ambiguously contained"):
            session._resolve_allocation_span(ambiguous)
        with session._region_lock:
            second_record.base_address = original_second_base

        assert not region_client.release_calls
        assert not region_client.unregister_calls
        assert second.data_ptr() == original_second_base

    @pytest.mark.parametrize(
        ("shape", "dtype", "name"),
        [
            ((-1,), torch.uint8, "negative"),
            ((0,), torch.uint8, "zero"),
            ((2, region_session.sys.maxsize), torch.uint8, "overflow"),
            ((1,), object(), "bad-dtype"),
            ((1,), torch.uint8, "  "),
        ],
    )
    def test_invalid_allocation_input_sends_no_region_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        name: str,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)

        with pytest.raises(RegionArtifactInputError):
            session.allocate_host_tensor(shape, dtype, name=name)

        assert session.health is RegionSessionHealth.READY
        assert not region_client.register_calls

    def test_allocate_is_rejected_in_scratch_mode_without_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        _install_fake_attach(monkeypatch, region_client)
        session = RegionBackedArtifactSession.attach(
            _session_options(transfer=ScratchTransferOptions(capacity_bytes=4096))
        )

        with pytest.raises(RegionArtifactInputError, match="allocator mode"):
            session.allocate_host_tensor((1,), torch.uint8, name="invalid")

        assert not region_client.register_calls

    def test_exact_unexposed_rollback_releases_and_unregisters_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)

        def fail_tensor_construction(
            mapped_region: mmap.mmap,
            shape: tuple[int, ...],
            dtype: torch.dtype,
            element_count: int,
        ) -> NoReturn:
            del mapped_region, shape, dtype, element_count
            raise RuntimeError("local tensor construction failed")

        monkeypatch.setattr(
            region_session,
            "_MAPPED_TENSOR_FACTORY",
            fail_tensor_construction,
        )

        with pytest.raises(RegionArtifactInputError, match="rolled back"):
            session.allocate_host_tensor((32,), torch.uint8, name="rollback")

        assert session.health is RegionSessionHealth.READY
        assert len(region_client.release_calls) == 1
        assert region_client.unregister_calls == ["region:1"]
        record = session._diagnostic_allocation_records[0]
        assert record.lifecycle is region_session._RegionAllocationLifecycle.ROLLED_BACK
        assert record.rollback_attempted is True
        assert record.file_descriptor is None
        assert record.mapped_region is None
        assert record.tensor_root is None

    @pytest.mark.parametrize("failure_point", ["registration", "attachment"])
    def test_ambiguous_registration_or_attachment_latches_without_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
        failure_point: str,
    ) -> None:
        if failure_point == "registration":
            region_client.registration_error = RuntimeError("registration unavailable")
        else:
            region_client.attachment_error = RuntimeError("FD handoff unavailable")
        session = _attach_allocator_session(monkeypatch, region_client)

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.allocate_host_tensor((32,), torch.uint8, name=failure_point)

        assert error_info.value.failure.code is RegionSessionFailureCode.REGION_SETUP
        assert (
            error_info.value.failure.operation_kind
            is RegionSessionOperationKind.ALLOCATE
        )
        assert error_info.value.failure.operation_id
        assert session.health is RegionSessionHealth.FAILED
        assert not region_client.release_calls
        assert not region_client.unregister_calls
        records = session._diagnostic_allocation_records
        assert len(records) == (0 if failure_point == "registration" else 1)
        if records:
            assert (
                records[0].lifecycle
                is region_session._RegionAllocationLifecycle.BUILDING
            )

    def test_uncertain_rollback_latches_and_never_unregisters(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        region_client.release_result = False
        session = _attach_allocator_session(monkeypatch, region_client)

        def fail_mapping(file_descriptor: int, byte_length: int) -> NoReturn:
            del file_descriptor, byte_length
            raise OSError("mmap failed")

        monkeypatch.setattr(region_session, "_SHARED_REGION_MAPPER", fail_mapping)

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.allocate_host_tensor((32,), torch.uint8, name="ambiguous")

        assert error_info.value.failure.code is RegionSessionFailureCode.REGION_SETUP
        assert len(region_client.release_calls) == 1
        assert not region_client.unregister_calls
        record = session._diagnostic_allocation_records[0]
        assert record.lifecycle is region_session._RegionAllocationLifecycle.BUILDING
        assert record.rollback_attempted is True
        assert record.file_descriptor is None

    def test_process_pinned_region_survives_failure_and_termination(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="pinned")
        tensor_reference = weakref.ref(tensor)
        record = session._diagnostic_allocation_records[0]
        failure = _failure("later transfer failed")

        session._latch_failure(failure)
        session.terminate_process_session()
        del tensor
        gc.collect()

        assert tensor_reference() is record.tensor_root
        assert (
            record.lifecycle is region_session._RegionAllocationLifecycle.PROCESS_PINNED
        )
        assert record.mapped_region is not None
        assert record.file_descriptor is not None
        assert not region_client.release_calls
        assert not region_client.unregister_calls
        with pytest.raises(RegionSessionFailedError) as error_info:
            session.allocate_host_tensor((1,), torch.uint8, name="after-failure")
        assert error_info.value.failure is failure
        assert len(region_client.register_calls) == 1

    def test_attach_reachability_probe_allocates_no_region(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)

        assert session.health is RegionSessionHealth.READY
        assert not region_client.register_calls
        assert not session._diagnostic_allocation_records


class TestArtifactLowering:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    def test_uses_canonical_byte_artifact_identity_and_selection(self) -> None:
        artifact = _artifact(engine_key=b"layer-3:page-17")

        compiled = region_session._lower_artifacts((artifact,))[0]
        expected_id = build_byte_artifact_cgid(
            namespace=artifact.keyspace.namespace,
            engine=artifact.keyspace.engine,
            model_id=artifact.keyspace.model_id,
            model_version=artifact.keyspace.model_version,
            layout_id=artifact.keyspace.layout_id,
            engine_key=artifact.engine_key,
        )

        assert compiled.artifact_id == expected_id
        assert compiled.selection.artifact_id == expected_id
        assert compiled.selection.view_id == ""
        assert not compiled.selection.tensor_names

    def test_multiple_keyspaces_coexist_without_identity_cache(self) -> None:
        other_keyspace = ByteArtifactKeyspace(
            namespace="other-tenant",
            engine="vllm",
            model_id="model-b",
            model_version="revision-9",
            layout_id="kv-v2",
        )
        artifacts = (
            _artifact(engine_key=b"page-1"),
            _artifact(engine_key=b"page-1", keyspace=other_keyspace),
        )

        first = region_session._lower_artifacts(artifacts)
        second = region_session._lower_artifacts(artifacts)

        assert first[0].artifact_id != first[1].artifact_id
        assert first[0].selection is not second[0].selection
        assert first[0].selection == second[0].selection

    def test_duplicate_derived_identity_is_rejected(self) -> None:
        artifact = _artifact(engine_key=b"duplicate")

        with pytest.raises(RegionArtifactInputError, match="duplicate"):
            region_session._lower_artifacts((artifact, artifact.model_copy()))

    def test_scratch_layout_packs_one_storage_in_caller_order(self) -> None:
        source_a = torch.empty((16,), dtype=torch.uint8)
        source_b = torch.empty((24,), dtype=torch.uint8)
        transfers = (
            _transfer(
                source_a,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"a",
            ),
            _transfer(
                source_b,
                offset_bytes=0,
                byte_length=24,
                engine_key=b"b",
            ),
        )
        handle = LocalRegionHandle(
            region_id="region:scratch",
            memory_kind=RegionMemoryKind.HOST_SHARED,
            ttl_ms=0,
            size_bytes=64,
            attach_token=b"scratch-token",
            daemon_managed=True,
            host_shared_region_class=HostSharedRegionClass.SCRATCH,
        )
        record = region_session._RegionAllocationRecord(
            allocation_sequence=1,
            handle=handle,
            capacity_bytes=64,
            lifecycle=region_session._RegionAllocationLifecycle.PROCESS_PINNED,
        )

        prepared = region_session._compile_scratch_layout(
            region_session._lower_transfers(transfers),
            record,
        )

        assert len(prepared.layout.storages) == 1
        assert prepared.packed_offsets == (0, 16)
        assert [offset.name for offset in prepared.layout.offsets] == [
            compiled.artifact.artifact_id for compiled in prepared.compiled_transfers
        ]
        assert [offset.storage_offset for offset in prepared.layout.offsets] == [
            0,
            16,
        ]
        assert not any(
            offset.HasField("slot_index") for offset in prepared.layout.offsets
        )

    def test_allocator_layout_deduplicates_regions_and_builds_put_invariants(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        first = session.allocate_host_tensor((64,), torch.uint8, name="first")
        second = session.allocate_host_tensor((64,), torch.uint8, name="second")
        transfers = (
            _transfer(
                first,
                offset_bytes=16,
                byte_length=16,
                engine_key=b"first-1",
            ),
            _transfer(
                second,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"second-0",
            ),
            _transfer(
                first,
                offset_bytes=32,
                byte_length=16,
                engine_key=b"first-2",
            ),
        )

        prepared = session._compile_direct_transfers(
            transfers,
            operation_kind=RegionSessionOperationKind.PUT_FROM,
        )
        assert isinstance(
            prepared.request,
            store_daemon_pb2.BatchPutIfAbsentFromRegionRequest,
        )

        layout = prepared.request.source_layout
        assert [storage.storage_id for storage in layout.storages] == [
            "storage-0",
            "storage-1",
        ]
        assert [storage.region_ref.region_id for storage in layout.storages] == [
            "region:1",
            "region:2",
        ]
        assert [storage.device_id for storage in layout.storages] == [-1, -1]
        assert [storage.storage_length for storage in layout.storages] == [64, 64]
        assert [offset.storage_offset for offset in layout.offsets] == [16, 64, 32]
        assert [offset.storage_id for offset in layout.offsets] == [
            "storage-0",
            "storage-1",
            "storage-0",
        ]
        assert [offset.slot_index for offset in layout.offsets] == [1, 0, 2]
        assert all(
            offset.slot_generation == region_session._MAX_UINT64
            for offset in layout.offsets
        )
        assert [candidate.slot_bytes for candidate in prepared.geometry_candidates] == [
            16,
            16,
        ]
        for item, transfer in zip(prepared.request.items, transfers, strict=True):
            assert item.selection.artifact_id in [
                offset.name for offset in layout.offsets
            ]
            assert item.invariant.layout_id == transfer.artifact.keyspace.layout_id
            assert item.invariant.byte_length == transfer.artifact.byte_length
            assert item.invariant.verification_mode == (
                store_daemon_pb2.BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY
            )
            assert not item.invariant.payload_digest_alg
            assert not item.invariant.payload_digest_hex
            assert not item.inline_payload
            assert not item.payload_ref

    def test_public_results_expose_no_internal_identity_or_selection(self) -> None:
        assert "artifact_id" not in RegionArtifactExistsResult.model_fields
        assert "selection" not in RegionArtifactExistsResult.model_fields
        assert "artifact_id" not in RegionArtifactTransferResult.model_fields
        assert "selection" not in RegionArtifactTransferResult.model_fields


class TestWireBudget:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    @staticmethod
    def _artifacts(count: int) -> tuple[ByteArtifactSpec, ...]:
        return tuple(
            _artifact(engine_key=f"wire-item-{index:04d}".encode())
            for index in range(count)
        )

    def test_exists_partitions_at_completed_request_boundary_and_preserves_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        artifacts = self._artifacts(5)
        compiled = region_session._lower_artifacts(artifacts)
        two_item_request = region_session._build_exists_request(compiled[:2])
        three_item_request = region_session._build_exists_request(compiled[:3])
        two_item_bytes = region_session._request_wire_bytes(two_item_request)
        assert two_item_bytes < region_session._request_wire_bytes(three_item_request)
        client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=two_item_bytes,
            max_receive_message_bytes=1 << 20,
        )
        expected_by_id = {
            item.artifact_id: index % 2 == 0 for index, item in enumerate(compiled)
        }
        attempt_operation_ids: list[str] = []

        def batch_exists(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            timeout_s: float,
            operation_id: str | None = None,
        ) -> store_daemon_pb2.BatchExistsResponse:
            assert timeout_s == 30.0
            assert operation_id is not None
            attempt_operation_ids.extend((operation_id, operation_id))
            response = store_daemon_pb2.BatchExistsResponse()
            for selection in reversed(tuple(selections)):
                response.outcomes.add(
                    artifact_id=selection.artifact_id,
                    status=(
                        store_daemon_pb2.BATCH_ITEM_STATUS_OK
                        if expected_by_id[selection.artifact_id]
                        else store_daemon_pb2.BATCH_ITEM_STATUS_MISS
                    ),
                )
            return response

        monotonic_values = iter((0.0, 0.1, 1.0, 1.2, 2.0, 2.3))
        monkeypatch.setattr(client, "batch_exists", batch_exists)
        monkeypatch.setattr(
            region_session.time,
            "monotonic",
            lambda: next(monotonic_values),
        )

        result = session.batch_exists(artifacts)

        assert result.existence_mask == (True, False, True, False, True)
        assert result.rpc_elapsed_s == pytest.approx(0.6)
        assert len(attempt_operation_ids) == 6
        assert attempt_operation_ids[0] == attempt_operation_ids[1]
        assert attempt_operation_ids[2] == attempt_operation_ids[3]
        assert attempt_operation_ids[4] == attempt_operation_ids[5]
        assert len(set(attempt_operation_ids[::2])) == 3

    @pytest.mark.parametrize(
        ("operation_kind", "limited_direction"),
        [
            (RegionSessionOperationKind.GET_INTO, "send"),
            (RegionSessionOperationKind.GET_INTO, "receive"),
            (RegionSessionOperationKind.PUT_FROM, "send"),
            (RegionSessionOperationKind.PUT_FROM, "receive"),
        ],
    )
    def test_transfer_rejects_either_oversized_wire_direction_before_geometry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
        operation_kind: RegionSessionOperationKind,
        limited_direction: str,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="wire")
        transfers = (
            _transfer(
                tensor,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"wire-transfer",
            ),
        )
        prepared = session._compile_direct_transfers(
            transfers,
            operation_kind=operation_kind,
        )
        expected = tuple(
            region_session._ExpectedOutcome(
                artifact_id=compiled.artifact.artifact_id,
                slot_index=int(offset.slot_index),
                slot_generation=region_session._MAX_UINT64,
            )
            for compiled, offset in zip(
                prepared.compiled_transfers,
                prepared.request_layout.offsets,
                strict=True,
            )
        )
        request_bytes = region_session._request_wire_bytes(prepared.request)
        response_bytes = region_session._estimate_response_bytes(
            expected,
            direct=True,
        )
        region_client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=(
                request_bytes - 1 if limited_direction == "send" else request_bytes
            ),
            max_receive_message_bytes=(
                response_bytes - 1 if limited_direction == "receive" else response_bytes
            ),
        )

        with pytest.raises(RegionArtifactInputError, match="wire budget"):
            session._compile_direct_transfers(
                transfers,
                operation_kind=operation_kind,
            )

        record = session._diagnostic_allocation_records[0]
        assert record.slot_geometry is None
        assert session._next_rpc_generation == 1
        assert not record.data_rpc_used

    def test_wire_measurement_uses_complete_request_and_exact_client_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="complete")
        transfer = _transfer(
            tensor,
            offset_bytes=16,
            byte_length=16,
            engine_key=b"complete-request",
        )

        prepared = session._compile_direct_transfers(
            (transfer,),
            operation_kind=RegionSessionOperationKind.GET_INTO,
        )
        assert isinstance(
            prepared.request,
            store_daemon_pb2.BatchGetIntoRegionRequest,
        )
        assert (
            prepared.request.operation_id
            == region_session._WIRE_OPERATION_ID_PLACEHOLDER
        )
        assert prepared.request.pid == os.getpid()
        assert len(prepared.request.selections) == 1
        assert len(prepared.request.target_layout.storages) == 1
        assert len(prepared.request.target_layout.offsets) == 1
        assert prepared.request.target_layout.offsets[0].HasField("slot_index")
        assert prepared.request.target_layout.offsets[0].HasField("slot_generation")

        exact_request_bytes = region_session._request_wire_bytes(prepared.request)
        region_client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=exact_request_bytes,
            max_receive_message_bytes=1 << 20,
        )
        session._compile_direct_transfers(
            (transfer,),
            operation_kind=RegionSessionOperationKind.GET_INTO,
        )
        region_client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=exact_request_bytes - 1,
            max_receive_message_bytes=1 << 20,
        )
        with pytest.raises(RegionArtifactInputError, match="wire budget"):
            session._compile_direct_transfers(
                (transfer,),
                operation_kind=RegionSessionOperationKind.GET_INTO,
            )


class TestOutcomeValidation:
    @staticmethod
    def _outcome(
        artifact_id: str,
        status: int,
        *,
        message: str = "",
        slot_index: int | None = None,
        slot_generation: int | None = None,
    ) -> store_daemon_pb2.BatchItemOutcome:
        outcome = store_daemon_pb2.BatchItemOutcome(
            artifact_id=artifact_id,
            status=status,
            message=message,
        )
        if slot_index is not None:
            outcome.slot_index = slot_index
        if slot_generation is not None:
            outcome.slot_generation = slot_generation
        return outcome

    def test_accepts_operation_specific_ok_and_miss_in_caller_order(self) -> None:
        expected = (
            region_session._ExpectedOutcome("artifact-a"),
            region_session._ExpectedOutcome("artifact-b"),
        )
        outcomes = (
            self._outcome("artifact-b", store_daemon_pb2.BATCH_ITEM_STATUS_MISS),
            self._outcome("artifact-a", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
        )

        assert region_session._validate_batch_outcomes(
            outcomes,
            expected,
            operation_kind=RegionSessionOperationKind.EXISTS,
        ) == (True, False)
        assert region_session._validate_batch_outcomes(
            outcomes,
            expected,
            operation_kind=RegionSessionOperationKind.GET_INTO,
        ) == (True, False)
        with pytest.raises(region_session._OutcomeValidationError) as error_info:
            region_session._validate_batch_outcomes(
                outcomes,
                expected,
                operation_kind=RegionSessionOperationKind.PUT_FROM,
            )
        assert error_info.value.code is RegionSessionFailureCode.DAEMON_STATUS

    @pytest.mark.parametrize("case", ["missing", "duplicate", "unknown", "empty"])
    def test_rejects_missing_duplicate_unknown_and_badly_correlated_outcomes(
        self,
        case: str,
    ) -> None:
        expected = (
            region_session._ExpectedOutcome("artifact-a"),
            region_session._ExpectedOutcome("artifact-b"),
        )
        outcomes_by_case = {
            "missing": (
                self._outcome("artifact-a", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
            ),
            "duplicate": (
                self._outcome("artifact-a", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
                self._outcome("artifact-a", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
            ),
            "unknown": (
                self._outcome("artifact-b", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
                self._outcome("artifact-c", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
            ),
            "empty": (
                self._outcome("", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
                self._outcome("artifact-b", store_daemon_pb2.BATCH_ITEM_STATUS_OK),
            ),
        }

        with pytest.raises(region_session._OutcomeValidationError) as error_info:
            region_session._validate_batch_outcomes(
                outcomes_by_case[case],
                expected,
                operation_kind=RegionSessionOperationKind.GET_INTO,
            )

        assert error_info.value.code is RegionSessionFailureCode.MALFORMED_RESPONSE

    @pytest.mark.parametrize(
        ("slot_index", "slot_generation"),
        [(None, None), (7, None), (None, 11), (8, 11), (7, 12)],
    )
    def test_requires_exact_echoed_direct_slot_tokens(
        self,
        slot_index: int | None,
        slot_generation: int | None,
    ) -> None:
        expected = (region_session._ExpectedOutcome("artifact-a", 7, 11),)
        outcome = self._outcome(
            "artifact-a",
            store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            slot_index=slot_index,
            slot_generation=slot_generation,
        )

        with pytest.raises(region_session._OutcomeValidationError) as error_info:
            region_session._validate_batch_outcomes(
                (outcome,),
                expected,
                operation_kind=RegionSessionOperationKind.GET_INTO,
            )

        assert error_info.value.code is RegionSessionFailureCode.MALFORMED_RESPONSE

        valid = self._outcome(
            "artifact-a",
            store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            slot_index=7,
            slot_generation=11,
        )
        assert region_session._validate_batch_outcomes(
            (valid,),
            expected,
            operation_kind=RegionSessionOperationKind.GET_INTO,
        ) == (True,)

    @pytest.mark.parametrize(
        "status",
        [
            store_daemon_pb2.BATCH_ITEM_STATUS_UNSPECIFIED,
            store_daemon_pb2.BATCH_ITEM_STATUS_UNAVAILABLE,
            store_daemon_pb2.BATCH_ITEM_STATUS_FAILED_PRECONDITION,
            store_daemon_pb2.BATCH_ITEM_STATUS_INVALID_ARGUMENT,
            store_daemon_pb2.BATCH_ITEM_STATUS_INTERNAL_ERROR,
            999,
        ],
    )
    def test_non_allowlisted_and_unknown_statuses_map_to_daemon_status(
        self,
        status: int,
    ) -> None:
        expected = (region_session._ExpectedOutcome("artifact-a"),)
        outcome = self._outcome(
            "artifact-a",
            status,
            message="region missing and expired",
        )

        with pytest.raises(region_session._OutcomeValidationError) as error_info:
            region_session._validate_batch_outcomes(
                (outcome,),
                expected,
                operation_kind=RegionSessionOperationKind.GET_INTO,
            )

        assert error_info.value.code is RegionSessionFailureCode.DAEMON_STATUS

    def test_region_lost_requires_typed_evidence_not_error_text(self) -> None:
        generic = region_session._OutcomeValidationError(
            RegionSessionFailureCode.DAEMON_STATUS,
            "region lost",
        )
        unrelated = RuntimeError("region lost")
        typed = region_session._StructuredRegionLostError("typed region evidence")

        assert (
            region_session._failure_code_for_exception(generic)
            is RegionSessionFailureCode.DAEMON_STATUS
        )
        assert (
            region_session._failure_code_for_exception(unrelated)
            is RegionSessionFailureCode.INTERNAL
        )
        assert (
            region_session._failure_code_for_exception(typed)
            is RegionSessionFailureCode.REGION_LOST
        )


class TestGeometry:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    def test_compatible_first_use_freezes_geometry_and_monotonic_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="geometry")
        observed_generations: list[int] = []

        def execute(
            prepared: region_session._PreparedDirectTransfer,
        ) -> RegionArtifactTransferResult:
            assert prepared.slot_generation is not None
            observed_generations.append(prepared.slot_generation)
            return _successful_direct_result(prepared)

        monkeypatch.setattr(session, "_execute_get_into", execute)
        first = session.batch_get_into(
            [
                _transfer(
                    tensor,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"slot-0",
                )
            ]
        )
        second = session.batch_get_into(
            [
                _transfer(
                    tensor,
                    offset_bytes=32,
                    byte_length=16,
                    engine_key=b"slot-2",
                )
            ]
        )

        record = session._diagnostic_allocation_records[0]
        assert record.slot_geometry == region_session._RegionSlotGeometry(slot_bytes=16)
        assert observed_generations == [1, 2]
        assert first.operation_id != second.operation_id
        assert first.success_mask == (True,)
        assert second.success_mask == (True,)

    def test_invalid_and_wire_oversized_input_freezes_no_geometry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="invalid")
        record = session._diagnostic_allocation_records[0]
        misaligned = _transfer(
            tensor,
            offset_bytes=8,
            byte_length=16,
            engine_key=b"misaligned",
        )
        with pytest.raises(RegionArtifactInputError, match="aligned"):
            session.batch_get_into((misaligned,))
        assert record.slot_geometry is None

        region_client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=1,
            max_receive_message_bytes=1,
        )
        aligned = _transfer(
            tensor,
            offset_bytes=0,
            byte_length=16,
            engine_key=b"oversized",
        )
        with pytest.raises(RegionArtifactInputError, match="wire budget"):
            session.batch_get_into((aligned,))
        assert record.slot_geometry is None
        assert session._next_rpc_generation == 1

    def test_multi_region_geometry_install_is_all_or_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        first = session.allocate_host_tensor((64,), torch.uint8, name="first")
        second = session.allocate_host_tensor((64,), torch.uint8, name="second")
        first_record, second_record = session._diagnostic_allocation_records
        with session._geometry_lock:
            second_record.slot_geometry = region_session._RegionSlotGeometry(
                slot_bytes=8
            )
        executor_calls = 0

        def execute(
            prepared: region_session._PreparedDirectTransfer,
        ) -> RegionArtifactTransferResult:
            nonlocal executor_calls
            executor_calls += 1
            return _successful_direct_result(prepared)

        monkeypatch.setattr(session, "_execute_put_from", execute)
        transfers = (
            _transfer(
                first,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"first",
            ),
            _transfer(
                second,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"second",
            ),
        )

        with pytest.raises(RegionArtifactInputError, match="incompatible"):
            session.batch_put_from(transfers)

        assert first_record.slot_geometry is None
        assert second_record.slot_geometry == region_session._RegionSlotGeometry(
            slot_bytes=8
        )
        assert executor_calls == 0
        assert session._next_rpc_generation == 1

    def test_concurrent_conflicting_first_use_admits_only_one_geometry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="concurrent")
        barrier = threading.Barrier(2)
        original_admit = session._admit_direct_transfer
        executor_calls: list[int] = []

        def synchronized_admit(
            prepared: region_session._PreparedDirectTransfer,
        ) -> tuple[str, int]:
            barrier.wait(timeout=5.0)
            return original_admit(prepared)

        def execute(
            prepared: region_session._PreparedDirectTransfer,
        ) -> RegionArtifactTransferResult:
            assert prepared.slot_generation is not None
            executor_calls.append(prepared.slot_generation)
            return _successful_direct_result(prepared)

        monkeypatch.setattr(session, "_admit_direct_transfer", synchronized_admit)
        monkeypatch.setattr(session, "_execute_get_into", execute)
        transfers = (
            _transfer(
                tensor,
                offset_bytes=0,
                byte_length=8,
                engine_key=b"eight-byte-geometry",
            ),
            _transfer(
                tensor,
                offset_bytes=16,
                byte_length=16,
                engine_key=b"sixteen-byte-geometry",
            ),
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(session.batch_get_into, (transfer,))
                for transfer in transfers
            )
            outcomes: list[RegionArtifactTransferResult | BaseException] = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=5.0))
                except BaseException as exc:
                    outcomes.append(exc)

        assert (
            sum(isinstance(item, RegionArtifactTransferResult) for item in outcomes)
            == 1
        )
        assert sum(isinstance(item, RegionArtifactInputError) for item in outcomes) == 1
        assert len(executor_calls) == 1
        assert session._next_rpc_generation == 2
        assert session.health is RegionSessionHealth.READY
        record = session._diagnostic_allocation_records[0]
        assert record.slot_geometry in (
            region_session._RegionSlotGeometry(slot_bytes=8),
            region_session._RegionSlotGeometry(slot_bytes=16),
        )


class TestScratchTransfer:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    def test_arenas_are_lazy_direction_specific_fixed_and_process_pinned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
        )
        source = torch.arange(8, dtype=torch.uint8)
        target = torch.zeros(8, dtype=torch.uint8)

        assert not region_client.register_calls
        session.batch_put_from(
            (_transfer(source, offset_bytes=0, byte_length=8, engine_key=b"put-1"),)
        )
        session.batch_put_from(
            (_transfer(source, offset_bytes=0, byte_length=8, engine_key=b"put-2"),)
        )
        assert len(region_client.register_calls) == 1
        session.batch_get_into(
            (_transfer(target, offset_bytes=0, byte_length=8, engine_key=b"get-1"),)
        )
        session.batch_get_into(
            (_transfer(target, offset_bytes=0, byte_length=8, engine_key=b"get-2"),)
        )

        assert len(region_client.register_calls) == 2
        put_registration, get_registration = region_client.register_calls
        assert put_registration["size_bytes"] == 64
        assert get_registration["size_bytes"] == 64
        assert put_registration["host_shared_region_class"] is (
            HostSharedRegionClass.SCRATCH
        )
        assert get_registration["host_shared_region_class"] is (
            HostSharedRegionClass.SCRATCH
        )
        assert str(put_registration["region_name"]).endswith("-scratch_put")
        assert str(get_registration["region_name"]).endswith("-scratch_get")
        records = session._diagnostic_allocation_records
        assert len(records) == 2
        assert all(
            record.lifecycle is region_session._RegionAllocationLifecycle.PROCESS_PINNED
            for record in records
        )
        assert all(record.capacity_bytes == 64 for record in records)
        assert all(record.tensor_root is not None for record in records)
        assert all(record.mapped_region is not None for record in records)
        assert all(record.view_escaped for record in records)
        assert all(record.data_rpc_used for record in records)

        session.terminate_process_session()
        assert not region_client.release_calls
        assert not region_client.unregister_calls
        assert all(record.mapped_region is not None for record in records)

    def test_put_packs_exact_bytes_in_caller_order_and_uses_zero_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
            transfer_timeout_s=7.5,
        )
        first = torch.tensor([90, 1, 2, 3, 91], dtype=torch.uint8)
        second = torch.tensor([92, 4, 5, 93], dtype=torch.uint8)
        transfers = (
            _transfer(first, offset_bytes=1, byte_length=3, engine_key=b"first"),
            _transfer(second, offset_bytes=1, byte_length=2, engine_key=b"second"),
        )
        original_put = region_client.batch_put_if_absent_from_region

        def reversed_put_outcomes(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
            ordinary = original_put(
                items=items,
                source_layout=source_layout,
                pid=pid,
                device_uuid=device_uuid,
                ttl_ms=ttl_ms,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
            response = store_daemon_pb2.BatchPutIfAbsentFromRegionResponse()
            response.outcomes.extend(reversed(ordinary.outcomes))
            return response

        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            reversed_put_outcomes,
        )

        result = session.batch_put_from(transfers)

        assert result.success_mask == (True, True)
        assert result.operation_id
        assert result.pack_elapsed_s >= 0.0
        assert result.copy_elapsed_s == 0.0
        assert result.rpc_elapsed_s >= 0.0
        assert len(region_client.put_region_calls) == 1
        call = region_client.put_region_calls[0]
        assert call.operation_id == result.operation_id
        assert call.timeout_s == 7.5
        assert call.retries == 0
        assert call.ttl_ms is None
        assert [offset.storage_offset for offset in call.source_layout.offsets] == [
            0,
            3,
        ]
        assert [offset.logical_length for offset in call.source_layout.offsets] == [
            3,
            2,
        ]
        region_id = call.source_layout.storages[0].region_ref.region_id
        assert region_client.read_region_bytes(
            region_id,
            offset=0,
            byte_length=5,
        ) == bytes((1, 2, 3, 4, 5))
        metrics = session._diagnostic_scratch_metrics
        assert metrics == (
            region_session._ScratchTransferMetrics(
                direction=RegionSessionOperationKind.PUT_FROM,
                artifact_count=2,
                byte_count=5,
                pack_elapsed_s=result.pack_elapsed_s,
                copy_elapsed_s=0.0,
                rpc_elapsed_s=result.rpc_elapsed_s,
            ),
        )

    def test_get_copies_only_ok_items_after_complete_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
        )
        first_target = torch.full((5,), 70, dtype=torch.uint8)
        second_target = torch.full((4,), 80, dtype=torch.uint8)
        transfers = (
            _transfer(
                first_target,
                offset_bytes=1,
                byte_length=3,
                engine_key=b"first-get",
            ),
            _transfer(
                second_target,
                offset_bytes=1,
                byte_length=2,
                engine_key=b"second-get",
            ),
        )
        original_get = region_client.batch_get_into_region

        def mixed_get_outcomes(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            original_get(
                selections=selections,
                target_layout=target_layout,
                pid=pid,
                device_uuid=device_uuid,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
            region_id = target_layout.storages[0].region_ref.region_id
            region_client.write_region_bytes(region_id, b"abcXY", offset=0)
            response = store_daemon_pb2.BatchGetIntoRegionResponse()
            response.outcomes.add(
                artifact_id=selections[1].artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_MISS,
            )
            response.outcomes.add(
                artifact_id=selections[0].artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            )
            return response

        monkeypatch.setattr(
            region_client,
            "batch_get_into_region",
            mixed_get_outcomes,
        )

        result = session.batch_get_into(transfers)

        assert result.success_mask == (True, False)
        assert first_target.tolist() == [70, 97, 98, 99, 70]
        assert second_target.tolist() == [80, 80, 80, 80]
        assert result.pack_elapsed_s == 0.0
        assert result.copy_elapsed_s >= 0.0
        assert len(region_client.get_region_calls) == 1
        call = region_client.get_region_calls[0]
        assert call.operation_id == result.operation_id
        assert call.timeout_s is None
        assert call.retries == 0
        metrics = session._diagnostic_scratch_metrics
        assert metrics[0].direction is RegionSessionOperationKind.GET_INTO
        assert metrics[0].artifact_count == 2
        assert metrics[0].byte_count == 5
        assert metrics[0].copy_elapsed_s == result.copy_elapsed_s

    def test_malformed_get_copies_nothing_latches_and_never_reuses_arena(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
        )
        first_target = torch.full((3,), 11, dtype=torch.uint8)
        second_target = torch.full((2,), 22, dtype=torch.uint8)
        transfers = (
            _transfer(
                first_target,
                offset_bytes=0,
                byte_length=3,
                engine_key=b"malformed-a",
            ),
            _transfer(
                second_target,
                offset_bytes=0,
                byte_length=2,
                engine_key=b"malformed-b",
            ),
        )
        original_get = region_client.batch_get_into_region

        def malformed_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            original_get(
                selections=selections,
                target_layout=target_layout,
                pid=pid,
                device_uuid=device_uuid,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
            region_id = target_layout.storages[0].region_ref.region_id
            region_client.write_region_bytes(region_id, b"abcde", offset=0)
            response = store_daemon_pb2.BatchGetIntoRegionResponse()
            response.outcomes.add(
                artifact_id=selections[0].artifact_id,
                status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
            )
            return response

        monkeypatch.setattr(region_client, "batch_get_into_region", malformed_get)

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.batch_get_into(transfers)

        assert error_info.value.failure.code is (
            RegionSessionFailureCode.MALFORMED_RESPONSE
        )
        assert first_target.tolist() == [11, 11, 11]
        assert second_target.tolist() == [22, 22]
        assert session.health is RegionSessionHealth.FAILED
        assert len(region_client.register_calls) == 1
        assert len(region_client.get_region_calls) == 1
        with pytest.raises(RegionSessionFailedError) as repeated_error:
            session.batch_get_into(transfers)
        assert repeated_error.value.failure is error_info.value.failure
        assert len(region_client.register_calls) == 1
        assert len(region_client.get_region_calls) == 1
        assert not region_client.release_calls
        assert not region_client.unregister_calls

    def test_overflow_rejects_before_arena_mutation_or_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=16,
        )
        seed = torch.arange(8, dtype=torch.uint8)
        session.batch_put_from(
            (_transfer(seed, offset_bytes=0, byte_length=8, engine_key=b"seed"),)
        )
        record = session._diagnostic_allocation_records[0]
        before = region_client.read_region_bytes(
            record.handle.region_id,
            offset=0,
            byte_length=16,
        )
        first = torch.full((12,), 1, dtype=torch.uint8)
        second = torch.full((12,), 2, dtype=torch.uint8)
        overflowing = (
            _transfer(first, offset_bytes=0, byte_length=12, engine_key=b"large-a"),
            _transfer(second, offset_bytes=0, byte_length=12, engine_key=b"large-b"),
        )

        with pytest.raises(RegionArtifactInputError, match="exceeds scratch"):
            session.batch_put_from(overflowing)

        assert session.health is RegionSessionHealth.READY
        assert len(region_client.register_calls) == 1
        assert len(region_client.put_region_calls) == 1
        assert (
            region_client.read_region_bytes(
                record.handle.region_id,
                offset=0,
                byte_length=16,
            )
            == before
        )

    def test_empty_calls_allocate_no_arena_and_setup_rollback_is_exact(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(monkeypatch, region_client)

        assert session.batch_get_into(()).success_mask == ()
        assert session.batch_put_from(()).success_mask == ()
        assert not region_client.register_calls

        def fail_mapping(file_descriptor: int, byte_length: int) -> NoReturn:
            del file_descriptor, byte_length
            raise OSError("scratch mmap failed")

        monkeypatch.setattr(region_session, "_SHARED_REGION_MAPPER", fail_mapping)
        target = torch.zeros(8, dtype=torch.uint8)
        with pytest.raises(RegionArtifactInputError, match="rolled back"):
            session.batch_get_into(
                (
                    _transfer(
                        target,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"rollback",
                    ),
                )
            )

        assert session.health is RegionSessionHealth.READY
        assert len(region_client.release_calls) == 1
        assert region_client.unregister_calls == ["region:1"]
        assert not region_client.get_region_calls

    def test_same_direction_calls_serialize_for_get_and_put(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
        )
        target_a = torch.zeros(8, dtype=torch.uint8)
        target_b = torch.zeros(8, dtype=torch.uint8)
        source_a = torch.arange(8, dtype=torch.uint8)
        source_b = torch.arange(8, dtype=torch.uint8)

        get_state_lock = threading.Lock()
        get_first_entered = threading.Event()
        get_second_entered = threading.Event()
        release_first_get = threading.Event()
        get_active = 0
        get_max_active = 0
        get_call_count = 0

        def blocking_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            del target_layout, pid, device_uuid, operation_id, timeout_s, retries
            nonlocal get_active, get_max_active, get_call_count
            with get_state_lock:
                get_call_count += 1
                call_index = get_call_count
                get_active += 1
                get_max_active = max(get_max_active, get_active)
            if call_index == 1:
                get_first_entered.set()
                assert release_first_get.wait(timeout=5.0)
            else:
                get_second_entered.set()
            response = store_daemon_pb2.BatchGetIntoRegionResponse()
            for selection in selections:
                response.outcomes.add(
                    artifact_id=selection.artifact_id,
                    status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
                )
            with get_state_lock:
                get_active -= 1
            return response

        monkeypatch.setattr(region_client, "batch_get_into_region", blocking_get)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                session.batch_get_into,
                (
                    _transfer(
                        target_a,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"serialized-get-a",
                    ),
                ),
            )
            assert get_first_entered.wait(timeout=5.0)
            second_future = executor.submit(
                session.batch_get_into,
                (
                    _transfer(
                        target_b,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"serialized-get-b",
                    ),
                ),
            )
            assert not get_second_entered.wait(timeout=0.05)
            release_first_get.set()
            assert first_future.result(timeout=5.0).success_mask == (True,)
            assert second_future.result(timeout=5.0).success_mask == (True,)
        assert get_max_active == 1

        put_state_lock = threading.Lock()
        put_first_entered = threading.Event()
        put_second_entered = threading.Event()
        release_first_put = threading.Event()
        put_active = 0
        put_max_active = 0
        put_call_count = 0

        def blocking_put(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
            del source_layout, pid, device_uuid, ttl_ms, operation_id, timeout_s
            del retries
            nonlocal put_active, put_max_active, put_call_count
            with put_state_lock:
                put_call_count += 1
                call_index = put_call_count
                put_active += 1
                put_max_active = max(put_max_active, put_active)
            if call_index == 1:
                put_first_entered.set()
                assert release_first_put.wait(timeout=5.0)
            else:
                put_second_entered.set()
            response = store_daemon_pb2.BatchPutIfAbsentFromRegionResponse()
            for item in items:
                response.outcomes.add(
                    artifact_id=item.selection.artifact_id,
                    status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
                )
            with put_state_lock:
                put_active -= 1
            return response

        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            blocking_put,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                session.batch_put_from,
                (
                    _transfer(
                        source_a,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"serialized-put-a",
                    ),
                ),
            )
            assert put_first_entered.wait(timeout=5.0)
            second_future = executor.submit(
                session.batch_put_from,
                (
                    _transfer(
                        source_b,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"serialized-put-b",
                    ),
                ),
            )
            assert not put_second_entered.wait(timeout=0.05)
            release_first_put.set()
            assert first_future.result(timeout=5.0).success_mask == (True,)
            assert second_future.result(timeout=5.0).success_mask == (True,)
        assert put_max_active == 1

    def test_get_and_put_direction_locks_allow_overlap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_scratch_session(
            monkeypatch,
            region_client,
            capacity_bytes=64,
        )
        target = torch.zeros(8, dtype=torch.uint8)
        source = torch.arange(8, dtype=torch.uint8)
        session.batch_get_into(
            (
                _transfer(
                    target,
                    offset_bytes=0,
                    byte_length=8,
                    engine_key=b"prewarm-get",
                ),
            )
        )
        session.batch_put_from(
            (
                _transfer(
                    source,
                    offset_bytes=0,
                    byte_length=8,
                    engine_key=b"prewarm-put",
                ),
            )
        )
        overlap_barrier = threading.Barrier(2)

        def overlapping_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            del target_layout, pid, device_uuid, operation_id, timeout_s, retries
            overlap_barrier.wait(timeout=5.0)
            response = store_daemon_pb2.BatchGetIntoRegionResponse()
            for selection in selections:
                response.outcomes.add(
                    artifact_id=selection.artifact_id,
                    status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
                )
            return response

        def overlapping_put(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
            del source_layout, pid, device_uuid, ttl_ms, operation_id, timeout_s
            del retries
            overlap_barrier.wait(timeout=5.0)
            response = store_daemon_pb2.BatchPutIfAbsentFromRegionResponse()
            for item in items:
                response.outcomes.add(
                    artifact_id=item.selection.artifact_id,
                    status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
                )
            return response

        monkeypatch.setattr(region_client, "batch_get_into_region", overlapping_get)
        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            overlapping_put,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            get_future = executor.submit(
                session.batch_get_into,
                (
                    _transfer(
                        target,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"overlap-get",
                    ),
                ),
            )
            put_future = executor.submit(
                session.batch_put_from,
                (
                    _transfer(
                        source,
                        offset_bytes=0,
                        byte_length=8,
                        engine_key=b"overlap-put",
                    ),
                ),
            )
            assert get_future.result(timeout=5.0).success_mask == (True,)
            assert put_future.result(timeout=5.0).success_mask == (True,)


class TestAllocatorDirectTransfer:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    def test_one_region_get_and_put_use_exact_offsets_without_scratch_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="direct")

        def reject_scratch_copy(
            destination_address: int,
            source_address: int,
            byte_length: int,
        ) -> NoReturn:
            del destination_address, source_address, byte_length
            raise AssertionError("direct mode used the scratch copy helper")

        monkeypatch.setattr(region_session, "_HOST_MEMORY_COPY", reject_scratch_copy)
        get_result = session.batch_get_into(
            (
                _transfer(
                    tensor,
                    offset_bytes=16,
                    byte_length=16,
                    engine_key=b"direct-get",
                ),
            )
        )
        put_result = session.batch_put_from(
            (
                _transfer(
                    tensor,
                    offset_bytes=32,
                    byte_length=16,
                    engine_key=b"direct-put",
                ),
            )
        )

        assert get_result.success_mask == (True,)
        assert put_result.success_mask == (True,)
        assert get_result.operation_id != put_result.operation_id
        assert get_result.pack_elapsed_s == get_result.copy_elapsed_s == 0.0
        assert put_result.pack_elapsed_s == put_result.copy_elapsed_s == 0.0
        assert len(region_client.register_calls) == 1
        assert region_client.register_calls[0]["host_shared_region_class"] is (
            HostSharedRegionClass.ALLOCATOR
        )
        assert len(region_client.get_region_calls) == 1
        assert len(region_client.put_region_calls) == 1
        get_call = region_client.get_region_calls[0]
        put_call = region_client.put_region_calls[0]
        assert get_call.timeout_s is None
        assert put_call.timeout_s is None
        assert get_call.retries == put_call.retries == 0
        assert len(get_call.target_layout.storages) == 1
        assert len(put_call.source_layout.storages) == 1
        get_offset = get_call.target_layout.offsets[0]
        put_offset = put_call.source_layout.offsets[0]
        assert (get_offset.storage_offset, get_offset.slot_index) == (16, 1)
        assert (put_offset.storage_offset, put_offset.slot_index) == (32, 2)
        assert get_offset.slot_generation == 1
        assert put_offset.slot_generation == 2
        assert get_call.target_layout.storages[0].region_ref.region_id == "region:1"
        assert put_call.source_layout.storages[0].region_ref.region_id == "region:1"
        assert not any(
            record.handle.host_shared_region_class is HostSharedRegionClass.SCRATCH
            for record in session._diagnostic_allocation_records
        )

    def test_two_and_three_region_batches_use_one_rpc_and_stable_logical_bases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        first = session.allocate_host_tensor((64,), torch.uint8, name="first")
        second = session.allocate_host_tensor((64,), torch.uint8, name="second")
        third = session.allocate_host_tensor((64,), torch.uint8, name="third")
        three_region_transfers = (
            _transfer(
                third,
                offset_bytes=16,
                byte_length=16,
                engine_key=b"third-1",
            ),
            _transfer(
                first,
                offset_bytes=0,
                byte_length=16,
                engine_key=b"first-0",
            ),
            _transfer(
                second,
                offset_bytes=32,
                byte_length=16,
                engine_key=b"second-2",
            ),
            _transfer(
                third,
                offset_bytes=48,
                byte_length=16,
                engine_key=b"third-3",
            ),
        )

        get_result = session.batch_get_into(three_region_transfers)

        assert get_result.success_mask == (True, True, True, True)
        assert len(region_client.get_region_calls) == 1
        get_layout = region_client.get_region_calls[0].target_layout
        assert [storage.storage_id for storage in get_layout.storages] == [
            "storage-0",
            "storage-1",
            "storage-2",
        ]
        assert [storage.region_ref.region_id for storage in get_layout.storages] == [
            "region:3",
            "region:1",
            "region:2",
        ]
        assert [offset.storage_offset for offset in get_layout.offsets] == [
            16,
            64,
            160,
            48,
        ]
        assert [offset.slot_index for offset in get_layout.offsets] == [1, 0, 2, 3]
        assert len({offset.slot_generation for offset in get_layout.offsets}) == 1

        put_result = session.batch_put_from(
            (
                _transfer(
                    second,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"second-put",
                ),
                _transfer(
                    first,
                    offset_bytes=48,
                    byte_length=16,
                    engine_key=b"first-put",
                ),
            )
        )
        assert put_result.success_mask == (True, True)
        assert len(region_client.put_region_calls) == 1
        put_layout = region_client.put_region_calls[0].source_layout
        assert [storage.region_ref.region_id for storage in put_layout.storages] == [
            "region:2",
            "region:1",
        ]
        assert [offset.storage_offset for offset in put_layout.offsets] == [0, 112]

    def test_mixed_keyspaces_restore_get_outcomes_to_caller_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="mixed")
        other_keyspace = ByteArtifactKeyspace(
            namespace="other",
            engine="vllm",
            model_id="model-b",
            model_version="revision-b",
            layout_id="kv-v2",
        )
        original_get = region_client.batch_get_into_region

        def reordered_mixed_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            ordinary = original_get(
                selections=selections,
                target_layout=target_layout,
                pid=pid,
                device_uuid=device_uuid,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )
            region_client.write_region_bytes(
                target_layout.storages[0].region_ref.region_id,
                bytes(range(16)),
                offset=target_layout.offsets[0].storage_offset,
            )
            ordinary.outcomes[1].status = store_daemon_pb2.BATCH_ITEM_STATUS_MISS
            response = store_daemon_pb2.BatchGetIntoRegionResponse()
            response.outcomes.extend(reversed(ordinary.outcomes))
            return response

        monkeypatch.setattr(
            region_client,
            "batch_get_into_region",
            reordered_mixed_get,
        )
        result = session.batch_get_into(
            (
                _transfer(
                    tensor,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"default-keyspace",
                ),
                _transfer(
                    tensor,
                    offset_bytes=16,
                    byte_length=16,
                    engine_key=b"other-keyspace",
                    keyspace=other_keyspace,
                ),
            )
        )

        assert result.success_mask == (True, False)
        assert tensor[:16].tolist() == list(range(16))
        assert tensor[16:32].tolist() == [0] * 16
        assert len(region_client.get_region_calls) == 1

    def test_foreign_crossing_out_of_range_and_overlapping_spans_fail_locally(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="owned")
        foreign = torch.zeros(16, dtype=torch.uint8)
        invalid_batches = (
            (
                _transfer(
                    foreign,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"foreign",
                ),
            ),
            (
                RegionArtifactTransfer(
                    artifact=_artifact(byte_length=8, engine_key=b"crossing"),
                    span=HostMemorySpan.from_address(
                        tensor.data_ptr() + 60,
                        8,
                        owner=tensor,
                    ),
                ),
            ),
            (
                RegionArtifactTransfer(
                    artifact=_artifact(byte_length=8, engine_key=b"outside"),
                    span=HostMemorySpan.from_address(
                        tensor.data_ptr() + 64,
                        8,
                        owner=tensor,
                    ),
                ),
            ),
            (
                _transfer(
                    tensor,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"overlap-a",
                ),
                _transfer(
                    tensor,
                    offset_bytes=8,
                    byte_length=16,
                    engine_key=b"overlap-b",
                ),
            ),
        )

        for invalid in invalid_batches:
            with pytest.raises(RegionArtifactInputError):
                session.batch_get_into(invalid)

        assert session.health is RegionSessionHealth.READY
        assert not region_client.get_region_calls
        assert not region_client.put_region_calls
        assert session._diagnostic_allocation_records[0].slot_geometry is None

    def test_finite_timeout_preserves_zero_retry_and_put_borrow_ends_on_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        _install_fake_attach(monkeypatch, region_client)
        session = RegionBackedArtifactSession.attach(
            _session_options(transfer_timeout_s=4.25)
        )
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="source")
        tensor[:16] = torch.arange(16, dtype=torch.uint8)
        observed_rpc_bytes: list[bytes] = []
        original_put = region_client.batch_put_if_absent_from_region

        def snapshot_put_source(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
            storage = source_layout.storages[0]
            offset = source_layout.offsets[0]
            observed_rpc_bytes.append(
                region_client.read_region_bytes(
                    storage.region_ref.region_id,
                    offset=offset.storage_offset,
                    byte_length=offset.logical_length,
                )
            )
            return original_put(
                items=items,
                source_layout=source_layout,
                pid=pid,
                device_uuid=device_uuid,
                ttl_ms=ttl_ms,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )

        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            snapshot_put_source,
        )
        result = session.batch_put_from(
            (
                _transfer(
                    tensor,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"immutable-after-return",
                ),
            )
        )
        tensor[:16].fill_(255)

        assert result.success_mask == (True,)
        assert observed_rpc_bytes == [bytes(range(16))]
        assert region_client.put_region_calls[0].timeout_s == 4.25
        assert region_client.put_region_calls[0].retries == 0
        assert observed_rpc_bytes[0] != bytes(tensor[:16].tolist())

    def test_failure_and_termination_never_release_allocator_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="retained")
        record = session._diagnostic_allocation_records[0]

        def fail_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> NoReturn:
            del selections, target_layout, pid, device_uuid, operation_id, timeout_s
            del retries
            raise RuntimeError("direct get failed")

        monkeypatch.setattr(region_client, "batch_get_into_region", fail_get)
        with pytest.raises(RegionSessionFailedError):
            session.batch_get_into(
                (
                    _transfer(
                        tensor,
                        offset_bytes=0,
                        byte_length=16,
                        engine_key=b"fatal-direct-get",
                    ),
                )
            )
        session.terminate_process_session()

        assert record.mapped_region is not None
        assert record.tensor_root is tensor
        assert not region_client.release_calls
        assert not region_client.unregister_calls


class TestConcurrentAdmission:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    @staticmethod
    def _assert_short_locks_are_released(
        session: RegionBackedArtifactSession,
    ) -> None:
        for lock in (
            session._state_lock,
            session._region_lock,
            session._geometry_lock,
            session._generation_lock,
        ):
            assert lock.acquire(blocking=False)
            lock.release()

    def test_direct_get_and_put_overlap_with_unique_nonzero_generations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        get_tensor = session.allocate_host_tensor((64,), torch.uint8, name="get")
        put_tensor = session.allocate_host_tensor((64,), torch.uint8, name="put")
        original_get = region_client.batch_get_into_region
        original_put = region_client.batch_put_if_absent_from_region
        rpc_barrier = threading.Barrier(2)
        observed_generations: list[int] = []
        observed_lock = threading.Lock()

        def overlapping_get(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            target_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchGetIntoRegionResponse:
            self._assert_short_locks_are_released(session)
            with observed_lock:
                observed_generations.append(target_layout.offsets[0].slot_generation)
            rpc_barrier.wait(timeout=5.0)
            return original_get(
                selections=selections,
                target_layout=target_layout,
                pid=pid,
                device_uuid=device_uuid,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )

        def overlapping_put(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> store_daemon_pb2.BatchPutIfAbsentFromRegionResponse:
            self._assert_short_locks_are_released(session)
            with observed_lock:
                observed_generations.append(source_layout.offsets[0].slot_generation)
            rpc_barrier.wait(timeout=5.0)
            return original_put(
                items=items,
                source_layout=source_layout,
                pid=pid,
                device_uuid=device_uuid,
                ttl_ms=ttl_ms,
                operation_id=operation_id,
                timeout_s=timeout_s,
                retries=retries,
            )

        monkeypatch.setattr(region_client, "batch_get_into_region", overlapping_get)
        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            overlapping_put,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            get_future = executor.submit(
                session.batch_get_into,
                (
                    _transfer(
                        get_tensor,
                        offset_bytes=0,
                        byte_length=16,
                        engine_key=b"concurrent-get",
                    ),
                ),
            )
            put_future = executor.submit(
                session.batch_put_from,
                (
                    _transfer(
                        put_tensor,
                        offset_bytes=16,
                        byte_length=16,
                        engine_key=b"concurrent-put",
                    ),
                ),
            )
            get_result = get_future.result(timeout=5.0)
            put_result = put_future.result(timeout=5.0)

        assert get_result.success_mask == put_result.success_mask == (True,)
        assert get_result.operation_id != put_result.operation_id
        assert len(observed_generations) == 2
        assert len(set(observed_generations)) == 2
        assert all(generation > 0 for generation in observed_generations)
        assert session._diagnostic_in_flight_count == 0

    def test_failure_prevents_later_admission_and_discards_concurrent_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        get_tensor = session.allocate_host_tensor((64,), torch.uint8, name="get")
        put_tensor = session.allocate_host_tensor((64,), torch.uint8, name="put")
        successful_response_ready = threading.Event()
        allow_success_postcheck = threading.Event()
        original_execute_get = session._execute_get_into

        def gated_success(
            prepared: region_session._PreparedDirectTransfer,
        ) -> RegionArtifactTransferResult:
            result = original_execute_get(prepared)
            successful_response_ready.set()
            assert allow_success_postcheck.wait(timeout=5.0)
            return result

        def fail_put(
            *,
            items: Sequence[store_daemon_pb2.BatchPutIfAbsentFromRegionItem],
            source_layout: store_daemon_pb2.TargetLayout,
            pid: int,
            device_uuid: str,
            ttl_ms: int | None = None,
            operation_id: str | None = None,
            timeout_s: float | None = 600.0,
            retries: int = 1,
        ) -> NoReturn:
            del items, source_layout, pid, device_uuid, ttl_ms, operation_id
            del timeout_s, retries
            raise RuntimeError("concurrent direct put failed")

        monkeypatch.setattr(session, "_execute_get_into", gated_success)
        monkeypatch.setattr(
            region_client,
            "batch_put_if_absent_from_region",
            fail_put,
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            get_future = executor.submit(
                session.batch_get_into,
                (
                    _transfer(
                        get_tensor,
                        offset_bytes=0,
                        byte_length=16,
                        engine_key=b"success-to-discard",
                    ),
                ),
            )
            assert successful_response_ready.wait(timeout=5.0)
            with pytest.raises(RegionSessionFailedError) as put_error:
                session.batch_put_from(
                    (
                        _transfer(
                            put_tensor,
                            offset_bytes=0,
                            byte_length=16,
                            engine_key=b"failure-to-latch",
                        ),
                    )
                )
            allow_success_postcheck.set()
            with pytest.raises(RegionSessionFailedError) as get_error:
                get_future.result(timeout=5.0)

        assert get_error.value.failure is put_error.value.failure
        assert get_error.value.failure.operation_kind is (
            RegionSessionOperationKind.PUT_FROM
        )
        admitted_get_calls = len(region_client.get_region_calls)
        with pytest.raises(RegionSessionFailedError) as later_error:
            session.batch_get_into(
                (
                    _transfer(
                        get_tensor,
                        offset_bytes=16,
                        byte_length=16,
                        engine_key=b"must-not-admit",
                    ),
                )
            )
        assert later_error.value.failure is put_error.value.failure
        assert len(region_client.get_region_calls) == admitted_get_calls
        assert session._diagnostic_in_flight_count == 0

    def test_generation_counter_wrap_fails_closed_before_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="wrap")
        session._next_rpc_generation = region_session._MAX_UINT64

        last_valid = session.batch_get_into(
            (
                _transfer(
                    tensor,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"last-valid-generation",
                ),
            )
        )
        assert last_valid.success_mask == (True,)
        assert (
            region_client.get_region_calls[0].target_layout.offsets[0].slot_generation
            == region_session._MAX_UINT64
        )

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.batch_put_from(
                (
                    _transfer(
                        tensor,
                        offset_bytes=16,
                        byte_length=16,
                        engine_key=b"wrapped-generation",
                    ),
                )
            )

        assert error_info.value.failure.code is RegionSessionFailureCode.INTERNAL
        assert error_info.value.failure.operation_kind is (
            RegionSessionOperationKind.PUT_FROM
        )
        assert session.health is RegionSessionHealth.FAILED
        assert not region_client.put_region_calls
        assert session._diagnostic_in_flight_count == 0


class TestSessionState:
    def test_failed_state_wins_over_terminated_state_and_invalid_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        failure = _failure("first fatal failure")
        session._latch_failure(failure)
        session.terminate_process_session()

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.batch_exists([object()])  # type: ignore[list-item]

        assert error_info.value.failure is failure

    def test_terminated_state_wins_over_invalid_input_on_healthy_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        session.terminate_process_session()

        with pytest.raises(RegionSessionTerminatedError):
            session.batch_get_into([object()])  # type: ignore[list-item]

    def test_concurrent_failure_between_gates_prevents_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        validation_started = threading.Event()
        continue_validation = threading.Event()
        rpc_called = threading.Event()

        def prepare(
            artifacts: tuple[ByteArtifactSpec, ...],
        ) -> tuple[region_session._CompiledArtifact, ...]:
            validation_started.set()
            assert continue_validation.wait(timeout=5.0)
            return region_session._lower_artifacts(artifacts)

        def execute(
            partition: region_session._ExistsPartition,
            operation_id: str,
        ) -> tuple[tuple[bool, ...], float]:
            del operation_id
            rpc_called.set()
            return tuple(True for _ in partition.compiled_artifacts), 0.0

        monkeypatch.setattr(session, "_prepare_exists", prepare)
        monkeypatch.setattr(session, "_execute_exists", execute)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(session.batch_exists, [_artifact()])
            assert validation_started.wait(timeout=5.0)
            failure = _failure("another call failed")
            session._latch_failure(failure)
            continue_validation.set()
            with pytest.raises(RegionSessionFailedError) as error_info:
                future.result(timeout=5.0)

        assert error_info.value.failure is failure
        assert not rpc_called.is_set()
        assert session._diagnostic_in_flight_count == 0

    def test_concurrent_failure_wins_before_returning_local_input_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        failure = _failure("failure during validation")

        def fail_validation(
            artifacts: tuple[ByteArtifactSpec, ...],
        ) -> tuple[ByteArtifactSpec, ...]:
            del artifacts
            session._latch_failure(failure)
            raise RegionArtifactInputError("invalid caller input")

        monkeypatch.setattr(session, "_prepare_exists", fail_validation)

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.batch_exists([_artifact()])

        assert error_info.value.failure is failure

    def test_batch_takes_an_immutable_input_sequence_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        original = [_artifact()]
        observed_snapshot: tuple[ByteArtifactSpec, ...] | None = None

        def prepare(
            artifacts: tuple[ByteArtifactSpec, ...],
        ) -> tuple[region_session._CompiledArtifact, ...]:
            nonlocal observed_snapshot
            observed_snapshot = artifacts
            original.clear()
            return region_session._lower_artifacts(artifacts)

        def execute(
            partition: region_session._ExistsPartition,
            operation_id: str,
        ) -> tuple[tuple[bool, ...], float]:
            del operation_id
            return tuple(True for _ in partition.compiled_artifacts), 0.0

        monkeypatch.setattr(session, "_prepare_exists", prepare)
        monkeypatch.setattr(session, "_execute_exists", execute)

        result = session.batch_exists(original)

        assert original == []
        assert isinstance(observed_snapshot, tuple)
        assert observed_snapshot == (_artifact(),)
        assert result.existence_mask == (True,)

    def test_empty_methods_return_exact_results_without_rpc_admission(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())

        exists_result = session.batch_exists(())
        get_result = session.batch_get_into(())
        put_result = session.batch_put_from(())

        assert exists_result == RegionArtifactExistsResult(
            existence_mask=(),
            rpc_elapsed_s=0.0,
        )
        expected_transfer_result = RegionArtifactTransferResult(
            success_mask=(),
            operation_id=None,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
        )
        assert get_result == expected_transfer_result
        assert put_result == expected_transfer_result
        assert session._diagnostic_in_flight_count == 0
        assert client.config_calls == 1

    def test_empty_methods_still_reject_failed_or_terminated_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        failed_session = RegionBackedArtifactSession.attach(_session_options())
        failure = _failure("empty calls must fail")
        failed_session._latch_failure(failure)

        with pytest.raises(RegionSessionFailedError):
            failed_session.batch_exists(())
        with pytest.raises(RegionSessionFailedError):
            failed_session.batch_get_into(())
        with pytest.raises(RegionSessionFailedError):
            failed_session.batch_put_from(())

        failed_session.terminate_process_session()
        with pytest.raises(RegionSessionFailedError):
            failed_session.batch_exists(())

    def test_healthy_terminated_session_rejects_empty_calls_and_reattach(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        factory_addresses, _ = _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        session.terminate_process_session()

        with pytest.raises(RegionSessionTerminatedError):
            session.batch_exists(())
        with pytest.raises(RegionSessionTerminatedError):
            session.batch_get_into(())
        with pytest.raises(RegionSessionTerminatedError):
            session.batch_put_from(())
        with pytest.raises(RegionSessionTerminatedError):
            RegionBackedArtifactSession.attach(_session_options())

        assert factory_addresses == ["127.0.0.1:8073"]

    def test_termination_is_idempotent_and_never_releases_shared_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())

        def fail_release(server_address: str) -> NoReturn:
            raise AssertionError(f"unexpected client release for {server_address}")

        monkeypatch.setattr(daemon_ctl, "release_daemon_client", fail_release)
        session.terminate_process_session()
        session.terminate_process_session()

        assert session.lifecycle_state is RegionSessionLifecycleState.TERMINATED
        assert session.health is RegionSessionHealth.READY
        assert client.close_calls == 0

    def test_first_failure_is_retained_atomically(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        failures = tuple(
            _failure(f"failure-{index}", operation_id=f"operation-{index}")
            for index in range(8)
        )
        start = threading.Barrier(len(failures))

        def latch_after_barrier(failure: RegionSessionFailure) -> RegionSessionFailure:
            start.wait()
            return session._latch_failure(failure)

        with ThreadPoolExecutor(max_workers=len(failures)) as executor:
            retained = tuple(executor.map(latch_after_barrier, failures))

        assert session.failure is retained[0]
        assert all(failure is retained[0] for failure in retained)
        assert session.health is RegionSessionHealth.FAILED

    def test_termination_does_not_interrupt_admitted_operation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        rpc_started = threading.Event()
        finish_rpc = threading.Event()

        def prepare(
            artifacts: tuple[ByteArtifactSpec, ...],
        ) -> tuple[region_session._CompiledArtifact, ...]:
            return region_session._lower_artifacts(artifacts)

        def execute(
            partition: region_session._ExistsPartition,
            operation_id: str,
        ) -> tuple[tuple[bool, ...], float]:
            del operation_id
            rpc_started.set()
            assert finish_rpc.wait(timeout=5.0)
            return tuple(True for _ in partition.compiled_artifacts), 0.0

        monkeypatch.setattr(session, "_prepare_exists", prepare)
        monkeypatch.setattr(session, "_execute_exists", execute)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(session.batch_exists, [_artifact()])
            assert rpc_started.wait(timeout=5.0)
            assert session._diagnostic_in_flight_count == 1
            session.terminate_process_session()
            finish_rpc.set()
            result = future.result(timeout=5.0)

        assert result.existence_mask == (True,)
        assert session._diagnostic_in_flight_count == 0
        assert session.lifecycle_state is RegionSessionLifecycleState.TERMINATED


class _InjectedGrpcError(grpc.RpcError):
    def __init__(self, status_code: grpc.StatusCode) -> None:
        super().__init__()
        self._status_code = status_code

    def code(self) -> grpc.StatusCode:
        return self._status_code


class TestFailureContract:
    @pytest.fixture
    def region_client(self) -> Iterator[_FakeRegionDaemonClient]:
        client = _FakeRegionDaemonClient()
        yield client
        client.close_test_file_descriptors()

    @pytest.mark.parametrize(
        ("cause", "expected_code"),
        [
            *(
                (_InjectedGrpcError(status_code), RegionSessionFailureCode.TRANSPORT)
                for status_code in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.UNKNOWN,
                )
            ),
            (
                region_session._OutcomeValidationError(
                    RegionSessionFailureCode.DAEMON_STATUS,
                    "non-allowlisted status",
                ),
                RegionSessionFailureCode.DAEMON_STATUS,
            ),
            (
                region_session._OutcomeValidationError(
                    RegionSessionFailureCode.MALFORMED_RESPONSE,
                    "bad token",
                ),
                RegionSessionFailureCode.MALFORMED_RESPONSE,
            ),
            (
                region_session._StructuredRegionLostError("typed loss"),
                RegionSessionFailureCode.REGION_LOST,
            ),
            (RuntimeError("unexpected"), RegionSessionFailureCode.INTERNAL),
        ],
    )
    def test_stable_operational_failure_classification(
        self,
        cause: BaseException,
        expected_code: RegionSessionFailureCode,
    ) -> None:
        assert region_session._failure_code_for_exception(cause) is expected_code

    def test_region_setup_has_its_dedicated_classification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        region_client.registration_error = RuntimeError("ambiguous registration")

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.allocate_host_tensor((64,), torch.uint8, name="setup-failure")

        failure = error_info.value.failure
        assert failure.code is RegionSessionFailureCode.REGION_SETUP
        assert failure.operation_kind is RegionSessionOperationKind.ALLOCATE
        assert failure.operation_id
        assert failure.occurred_at.tzinfo is timezone.utc

    def test_latched_failure_rejects_every_method_without_rpc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        region_client: _FakeRegionDaemonClient,
    ) -> None:
        session = _attach_allocator_session(monkeypatch, region_client)
        tensor = session.allocate_host_tensor((64,), torch.uint8, name="before-failure")
        transfer = _transfer(
            tensor,
            offset_bytes=0,
            byte_length=16,
            engine_key=b"after-failure",
        )
        failure = _failure("sticky failure", operation_id="first-fatal-operation")
        session._latch_failure(failure)
        rpc_counts = (
            len(region_client.register_calls),
            len(region_client.exists_calls),
            len(region_client.get_region_calls),
            len(region_client.put_region_calls),
        )

        calls = (
            lambda: session.allocate_host_tensor((16,), torch.uint8, name="late"),
            lambda: session.batch_exists((transfer.artifact,)),
            lambda: session.batch_get_into((transfer,)),
            lambda: session.batch_put_from((transfer,)),
        )
        for call in calls:
            with pytest.raises(RegionSessionFailedError) as error_info:
                call()
            assert error_info.value.failure is failure

        assert rpc_counts == (
            len(region_client.register_calls),
            len(region_client.exists_calls),
            len(region_client.get_region_calls),
            len(region_client.put_region_calls),
        )

    def test_fatal_exists_partition_retains_that_partition_operation_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        artifacts = tuple(
            _artifact(engine_key=f"partition-{index}".encode()) for index in range(3)
        )
        compiled = region_session._lower_artifacts(artifacts)
        one_item_bytes = region_session._request_wire_bytes(
            region_session._build_exists_request(compiled[:1])
        )
        client.message_limits = daemon_ctl._GrpcMessageLimits(
            max_send_message_bytes=one_item_bytes,
            max_receive_message_bytes=1 << 20,
        )
        operation_ids: list[str] = []

        def fail_second_partition(
            *,
            selections: Sequence[common_pb2.ArtifactSelection],
            timeout_s: float,
            operation_id: str | None = None,
        ) -> store_daemon_pb2.BatchExistsResponse:
            del timeout_s
            assert operation_id is not None
            operation_ids.append(operation_id)
            if len(operation_ids) == 2:
                raise _InjectedGrpcError(grpc.StatusCode.UNAVAILABLE)
            response = store_daemon_pb2.BatchExistsResponse()
            for selection in selections:
                response.outcomes.add(
                    artifact_id=selection.artifact_id,
                    status=store_daemon_pb2.BATCH_ITEM_STATUS_OK,
                )
            return response

        monkeypatch.setattr(client, "batch_exists", fail_second_partition)

        with pytest.raises(RegionSessionFailedError) as error_info:
            session.batch_exists(artifacts)

        assert len(operation_ids) == 2
        assert error_info.value.failure.code is RegionSessionFailureCode.TRANSPORT
        assert error_info.value.failure.operation_id == operation_ids[1]
        assert session._diagnostic_in_flight_count == 0

    def test_first_fatal_exception_logs_one_traceback_and_derivatives_are_quiet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())

        def fail_exists(
            partition: region_session._ExistsPartition,
            operation_id: str,
        ) -> NoReturn:
            del partition, operation_id
            raise RuntimeError("injected fatal exception")

        monkeypatch.setattr(session, "_execute_exists", fail_exists)
        with caplog.at_level("ERROR", logger=region_session.__name__):
            with pytest.raises(RegionSessionFailedError) as error_info:
                session.batch_exists((_artifact(),))
            with pytest.raises(RegionSessionFailedError):
                session.batch_exists((_artifact(engine_key=b"later"),))

        assert error_info.value.failure.code is RegionSessionFailureCode.INTERNAL
        fatal_records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("region-backed artifact Session failed:")
        ]
        assert len(fatal_records) == 1
        assert fatal_records[0].exc_info is not None


class TestTerminationContract:
    def test_helper_cleanup_waits_for_admitted_work_and_runs_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeDaemonClient(_server_config())
        _install_fake_attach(monkeypatch, client)
        session = RegionBackedArtifactSession.attach(_session_options())
        cleaned: list[str] = []
        session._register_owned_helper_cleanup(lambda: cleaned.append("helper"))
        session._final_rpc_admission()

        session.terminate_process_session()
        session.terminate_process_session()
        assert cleaned == []

        assert session._complete_rpc_admission() is None
        assert cleaned == ["helper"]
        session.terminate_process_session()
        assert cleaned == ["helper"]

    def test_termination_retains_all_process_pinned_roots_and_authorities(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeRegionDaemonClient()
        try:
            session = _attach_allocator_session(monkeypatch, client)
            tensor = session.allocate_host_tensor((64,), torch.uint8, name="retained")
            record = session._diagnostic_allocation_records[0]
            roots = (
                session._client,
                record.attachment,
                record.file_descriptor,
                record.mapped_region,
                record.storage_root,
                record.tensor_root,
            )

            session.terminate_process_session()

            assert roots == (
                session._client,
                record.attachment,
                record.file_descriptor,
                record.mapped_region,
                record.storage_root,
                record.tensor_root,
            )
            assert record.tensor_root is tensor
            assert not client.release_calls
            assert not client.unregister_calls
            assert client.close_calls == 0
        finally:
            client.close_test_file_descriptors()


class TestSessionObservability:
    def test_generic_bounded_counters_cover_allocator_transfers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = _FakeRegionDaemonClient()
        try:
            session = _attach_allocator_session(monkeypatch, client)
            first = session.allocate_host_tensor((32,), torch.uint8, name="first")
            second = session.allocate_host_tensor((32,), torch.uint8, name="second")
            transfers = (
                _transfer(
                    first,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"first",
                ),
                _transfer(
                    second,
                    offset_bytes=0,
                    byte_length=16,
                    engine_key=b"second",
                ),
            )

            session.batch_put_from(transfers)
            snapshot = session._diagnostic_observability

            assert snapshot.session_name == "worker-0"
            assert snapshot.owner_pid == os.getpid()
            assert snapshot.endpoint == "127.0.0.1:8073"
            assert snapshot.transfer_mode is RegionTransferMode.ALLOCATOR
            assert snapshot.health is RegionSessionHealth.READY
            assert snapshot.admitted_operation_count == 3
            assert snapshot.completed_operation_count == 3
            assert snapshot.in_flight_operation_count == 0
            assert snapshot.get_transfer_count == 0
            assert snapshot.put_transfer_count == 1
            assert snapshot.artifact_count == 2
            assert snapshot.total_transfer_bytes == 32
            assert snapshot.unique_region_reference_count == 2
            assert snapshot.direct_bytes_submitted == 32
            assert snapshot.scratch_bytes_copied == 0
            assert snapshot.successful_item_count == 2
            assert snapshot.missed_item_count == 0
            assert snapshot.allocated_region_count == 2
            assert snapshot.allocated_region_bytes == 64
            assert snapshot.suppressed_transparent_retry_count == 1
        finally:
            client.close_test_file_descriptors()


class TestPublicDocumentationExamples:
    @pytest.mark.parametrize(
        "transfer_options",
        [
            ScratchTransferOptions(capacity_bytes=4096),
            AllocatorTransferOptions(),
        ],
    )
    def test_readme_style_attach_uses_only_public_store_exports(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transfer_options: ScratchTransferOptions | AllocatorTransferOptions,
    ) -> None:
        client = _FakeRegionDaemonClient()
        try:
            _install_fake_attach(monkeypatch, client)
            options = store_api.RegionBackedArtifactSessionOptions(
                daemon_address="127.0.0.1:8073",
                session_name="runtime-worker-0",
                transfer=transfer_options,
            )
            session = store_api.RegionBackedArtifactSession.attach(options)

            assert session.health is store_api.RegionSessionHealth.READY
            if isinstance(transfer_options, AllocatorTransferOptions):
                source = session.allocate_host_tensor(
                    (64,), torch.uint8, name="readme-region"
                )
            else:
                source = torch.arange(64, dtype=torch.uint8)
            artifact = store_api.ByteArtifactSpec(
                keyspace=_keyspace(),
                engine_key=b"readme-artifact",
                byte_length=16,
            )
            transfer = store_api.RegionArtifactTransfer(
                artifact=artifact,
                span=store_api.HostMemorySpan.from_tensor(
                    source,
                    offset_bytes=0,
                    byte_length=artifact.byte_length,
                ),
            )
            result = session.batch_put_from((transfer,))
            assert result.success_mask == (True,)
        finally:
            client.close_test_file_descriptors()


class TestGrpcMessageLimitsContinuation:
    @pytest.fixture(autouse=True)
    def _reset_message_limit_caches(self) -> Iterator[None]:
        daemon_ctl._grpc_max_send_message_bytes.cache_clear()
        daemon_ctl._grpc_max_receive_message_bytes.cache_clear()
        yield
        daemon_ctl._grpc_max_send_message_bytes.cache_clear()
        daemon_ctl._grpc_max_receive_message_bytes.cache_clear()

    def test_channel_options_and_private_access_share_effective_values(self) -> None:
        client = DaemonCtl("127.0.0.1:65535")
        try:
            limits = client._effective_grpc_message_limits
            options = dict(client._channel_options(limits))

            assert options["grpc.max_send_message_length"] == (
                limits.max_send_message_bytes
            )
            assert options["grpc.max_receive_message_length"] == (
                limits.max_receive_message_bytes
            )
            with pytest.raises(FrozenInstanceError):
                limits.max_send_message_bytes = 1
        finally:
            client.close()

    @pytest.mark.parametrize("invalid_value", ["invalid", "0", "1048575"])
    def test_invalid_environment_values_use_existing_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        invalid_value: str,
    ) -> None:
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_SEND_MESSAGE_BYTES", invalid_value)
        monkeypatch.setenv("TENSORCAST_GRPC_MAX_RECEIVE_MESSAGE_BYTES", invalid_value)
        client = DaemonCtl("127.0.0.1:65535")
        try:
            assert client._effective_grpc_message_limits == (
                daemon_ctl._GrpcMessageLimits(
                    max_send_message_bytes=daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES,
                    max_receive_message_bytes=daemon_ctl._DEFAULT_GRPC_MAX_MESSAGE_BYTES,
                )
            )
        finally:
            client.close()

    def test_region_wrappers_forward_explicit_and_default_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = DaemonCtl("127.0.0.1:65535")
        observed_retries: list[int] = []

        def fake_unary_call(
            method: object,
            request: object,
            *,
            timeout: float | int | None,
            retries: int,
            span: object,
        ) -> object:
            del method, request, timeout, span
            observed_retries.append(retries)
            return object()

        monkeypatch.setattr(client, "_unary_call", fake_unary_call)
        layout = store_daemon_pb2.TargetLayout()
        try:
            client.batch_get_into_region(
                selections=(),
                target_layout=layout,
                pid=1,
                device_uuid="cpu",
                retries=0,
            )
            client.batch_get_into_region(
                selections=(),
                target_layout=layout,
                pid=1,
                device_uuid="cpu",
            )
            client.batch_put_if_absent_from_region(
                items=(),
                source_layout=layout,
                pid=1,
                device_uuid="cpu",
                retries=0,
            )
            client.batch_put_if_absent_from_region(
                items=(),
                source_layout=layout,
                pid=1,
                device_uuid="cpu",
            )

            assert observed_retries == [0, 1, 0, 1]
        finally:
            client.close()

    def test_region_wrappers_forward_an_explicit_no_deadline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = DaemonCtl("127.0.0.1:65535")
        observed_timeouts: list[float | int | None] = []

        def fake_unary_call(
            method: object,
            request: object,
            *,
            timeout: float | int | None,
            retries: int,
            span: object,
        ) -> object:
            del method, request, retries, span
            observed_timeouts.append(timeout)
            return object()

        monkeypatch.setattr(client, "_unary_call", fake_unary_call)
        layout = store_daemon_pb2.TargetLayout()
        try:
            client.batch_get_into_region(
                selections=(),
                target_layout=layout,
                pid=1,
                device_uuid="cpu",
                timeout_s=None,
                retries=0,
            )
            client.batch_put_if_absent_from_region(
                items=(),
                source_layout=layout,
                pid=1,
                device_uuid="cpu",
                timeout_s=None,
                retries=0,
            )

            assert observed_timeouts == [None, None]
        finally:
            client.close()

    def test_region_wrapper_preserves_original_rpc_error_as_cause(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = DaemonCtl("127.0.0.1:65535")
        original = _DirectRpcError()

        def fail_unary_call(
            method: object,
            request: object,
            *,
            timeout: float | int | None,
            retries: int,
            span: object,
        ) -> NoReturn:
            del method, request, timeout, retries, span
            raise original

        monkeypatch.setattr(client, "_unary_call", fail_unary_call)
        try:
            with pytest.raises(RuntimeError) as error_info:
                client.batch_get_into_region(
                    selections=(),
                    target_layout=store_daemon_pb2.TargetLayout(),
                    pid=1,
                    device_uuid="cpu",
                    retries=0,
                )

            assert error_info.value.__cause__ is original
        finally:
            client.close()
