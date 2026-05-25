#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from tensorcast.api._device import device_uuid_for
from tensorcast.api.store.owned_binding_slot import restore_owned_binding_tensors
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.runtime_realization_reference_consumer import (
    ReferenceRuntimeTensorSpec,
    acquire_reference_binding_response,
    build_reference_resolved_spec,
    build_reference_tensor_index_bytes,
    prefetch_reference_binding,
    prefetch_reference_binding_set,
    target_from_reference_cache_record,
    write_reference_resolved_spec_cache_entry,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.types import (
    PrefetchRetentionPolicy,
    RealizationTargetSet,
    RuntimeBindingMemberRef,
    RuntimeTopologyRef,
)
from tests.python.utils.daemon import start_daemon_binary
from tests.python.utils.ports import get_free_port

pytestmark = [pytest.mark.requires_cuda_or_fake, pytest.mark.integration]


_WORKER_SCRIPT = r"""
import json
import os
import sys

import torch

from tensorcast.api.store.owned_binding_slot import restore_owned_binding_tensors
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.runtime_realization_reference_consumer import (
    acquire_reference_binding_response,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import PrefetchHandoff, RealizationTarget

daemon_addr, target_path, prefetched_path = sys.argv[1:4]
target_proto = operation_pb2.ServingBindingTarget()
target_proto.ParseFromString(open(target_path, "rb").read())
prefetched_proto = operation_pb2.PrefetchServingBindingResult()
prefetched_proto.ParseFromString(open(prefetched_path, "rb").read())
target = RealizationTarget.from_proto(target_proto)
prefetched = PrefetchHandoff.from_proto(prefetched_proto)

runtime = StoreRuntimeContext(daemon_addr)
client = DaemonCtl(daemon_addr)
try:
    response = acquire_reference_binding_response(
        client,
        prefetched=prefetched,
        target=target,
        caller_pid=os.getpid(),
    )
    tensors = restore_owned_binding_tensors(
        response=response,
        runtime=runtime,
        device_id=0,
    )
    value = float(tensors["alpha"].detach().cpu().item())
    del tensors
    torch.cuda.synchronize()
    print(json.dumps({"value": value, "lease_token": bool(response.mem_handle.lease_token)}))
finally:
    client.close()
    runtime.close()
"""


_SET_WORKER_SCRIPT = r"""
import json
import os
import sys

import torch

from tensorcast.api.store.owned_binding_slot import restore_owned_binding_tensors
from tensorcast.api.store.runtime import StoreRuntimeContext
from tensorcast.api.store.runtime_realization_reference_consumer import (
    acquire_reference_binding_response,
)
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import PrefetchHandoffSet, RealizationTargetSet

daemon_addr, target_path, prefetched_path = sys.argv[1:4]
target_proto = operation_pb2.ServingBindingSetTarget()
target_proto.ParseFromString(open(target_path, "rb").read())
prefetched_proto = operation_pb2.PrefetchServingBindingSetResult()
prefetched_proto.ParseFromString(open(prefetched_path, "rb").read())
target_set = RealizationTargetSet.from_proto(target_proto)
prefetched_set = PrefetchHandoffSet.from_proto(prefetched_proto)
targets_by_member = {target.member.member_id: target for target in target_set.members}

runtime = StoreRuntimeContext(daemon_addr)
client = DaemonCtl(daemon_addr)
try:
    values = {}
    lease_tokens = {}
    for prefetched in prefetched_set.members:
        target = targets_by_member[prefetched.member.member_id]
        response = acquire_reference_binding_response(
            client,
            prefetched=prefetched,
            target=target,
            caller_pid=os.getpid(),
        )
        tensors = restore_owned_binding_tensors(
            response=response,
            runtime=runtime,
            device_id=0,
        )
        values[prefetched.member.member_id] = float(tensors["alpha"].detach().cpu().item())
        lease_tokens[prefetched.member.member_id] = bool(response.mem_handle.lease_token)
        del tensors
        torch.cuda.synchronize()
    print(json.dumps({"values": values, "lease_tokens": lease_tokens}, sort_keys=True))
finally:
    client.close()
    runtime.close()
"""


def _skip_without_real_cuda() -> None:
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        pytest.skip("real CUDA IPC E2E requires the real CUDA backend")
    if not torch.cuda.is_available():
        pytest.skip("real CUDA IPC E2E requires a CUDA-capable host")


def _write_single_float_artifact(artifact_dir: Path, value: float) -> None:
    tensor = ReferenceRuntimeTensorSpec(
        name="alpha",
        size_bytes=4,
        dtype="torch.float32",
        shape=(1,),
        stride=(1,),
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tensor_index.json").write_bytes(
        build_reference_tensor_index_bytes(tensor)
    )
    (artifact_dir / "tensor.data_0").write_bytes(struct.pack("<f", value))


def _acquire_and_release(
    client: DaemonCtl,
    runtime: StoreRuntimeContext,
    *,
    prefetched,
    target,
) -> None:
    response = acquire_reference_binding_response(
        client,
        prefetched=prefetched,
        target=target,
    )
    token = bytes(response.mem_handle.lease_token)
    assert token
    tensors = restore_owned_binding_tensors(
        response=cast(Any, response),
        runtime=runtime,
        device_id=0,
    )
    assert "alpha" in tensors
    del tensors
    torch.cuda.synchronize()


def _wait_until_acquire_fails(
    client: DaemonCtl,
    runtime: StoreRuntimeContext,
    *,
    prefetched,
    target,
) -> None:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            response = acquire_reference_binding_response(
                client,
                prefetched=prefetched,
                target=target,
                timeout_s=2.0,
            )
        except Exception:  # noqa: BLE001
            return
        else:
            token = bytes(response.mem_handle.lease_token)
            assert token
            tensors = restore_owned_binding_tensors(
                response=cast(Any, response),
                runtime=runtime,
                device_id=0,
            )
            del tensors
            torch.cuda.synchronize()
            time.sleep(0.25)
    raise AssertionError("acquire did not fail after idle TTL")


def test_prefetch_serving_binding_real_cuda_worker_read_and_release(tmp_path) -> None:
    _skip_without_real_cuda()

    listen_addr = f"127.0.0.1:{get_free_port()}"
    storage_path = tmp_path / "daemon-storage"
    source_root = tmp_path / "public-source-root"
    artifact_dir = source_root / "model"
    expected_value = 3.25
    _write_single_float_artifact(artifact_dir, expected_value)

    proc = start_daemon_binary(
        listen_addr,
        storage_path,
        serving_prefetch_enabled=True,
        serving_prefetch_default_idle_ttl_after_last_release="1s",
        public_disk_source_root=source_root,
    )
    client = DaemonCtl(listen_addr)
    runtime = StoreRuntimeContext(listen_addr)
    try:
        source = client.resolve_public_disk_source(
            path=str(artifact_dir),
            verify_checksums=True,
        ).source
        assert source.artifact_id.startswith("msa1:")
        assert source.canonical_index_bytes == build_reference_tensor_index_bytes(
            ReferenceRuntimeTensorSpec(
                name="alpha",
                size_bytes=4,
                dtype="torch.float32",
                shape=(1,),
                stride=(1,),
            )
        )

        resolved = build_reference_resolved_spec(
            source_artifact_id=source.artifact_id,
            artifact_selection_digest=source.trusted_content_artifact_id
            or source.artifact_id,
            device_uuid=device_uuid_for(0),
            tensor=ReferenceRuntimeTensorSpec(
                name="alpha",
                size_bytes=4,
                dtype="torch.float32",
                shape=(1,),
                stride=(1,),
            ),
        )
        record = write_reference_resolved_spec_cache_entry(
            tmp_path / "resolved-spec-cache",
            resolved_spec=resolved,
        )
        target = target_from_reference_cache_record(record, device_uuid=device_uuid_for(0))
        prefetched = prefetch_reference_binding(
            client,
            source_artifact_id=source.artifact_id,
            target=target,
            retention_policy=PrefetchRetentionPolicy(
                expire_if_unacquired_after_ms=10_000,
                idle_ttl_after_last_release_ms=500,
                materialization_timeout_ms=30_000,
                allow_acquire_after_creator_exit=True,
            ),
        )

        target_path = tmp_path / "target.pb"
        prefetched_path = tmp_path / "prefetched.pb"
        target_path.write_bytes(target.to_proto().SerializeToString(deterministic=True))
        prefetched_path.write_bytes(
            prefetched.to_proto().SerializeToString(deterministic=True)
        )
        worker = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(_WORKER_SCRIPT),
                listen_addr,
                str(target_path),
                str(prefetched_path),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=True,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
        worker_result = json.loads(worker.stdout.strip().splitlines()[-1])
        assert worker_result["lease_token"] is True
        assert math.isclose(worker_result["value"], expected_value, rel_tol=0.0)

        _acquire_and_release(
            client,
            runtime,
            prefetched=prefetched,
            target=target,
        )
        _wait_until_acquire_fails(
            client,
            runtime,
            prefetched=prefetched,
            target=target,
        )
    finally:
        runtime.close()
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_prefetch_serving_binding_set_real_cuda_worker_reads_members(tmp_path) -> None:
    _skip_without_real_cuda()

    listen_addr = f"127.0.0.1:{get_free_port()}"
    storage_path = tmp_path / "daemon-storage"
    source_root = tmp_path / "public-source-root"
    artifact_dir = source_root / "model"
    expected_value = 7.5
    tensor = ReferenceRuntimeTensorSpec(
        name="alpha",
        size_bytes=4,
        dtype="torch.float32",
        shape=(1,),
        stride=(1,),
    )
    _write_single_float_artifact(artifact_dir, expected_value)

    proc = start_daemon_binary(
        listen_addr,
        storage_path,
        serving_prefetch_enabled=True,
        serving_prefetch_default_idle_ttl_after_last_release="2s",
        public_disk_source_root=source_root,
    )
    client = DaemonCtl(listen_addr)
    try:
        source = client.resolve_public_disk_source(
            path=str(artifact_dir),
            verify_checksums=True,
        ).source
        selection_digest = source.trusted_content_artifact_id or source.artifact_id
        members = []
        cache_root = tmp_path / "resolved-spec-cache"
        topology = RuntimeTopologyRef(
            schema_topology_digest="vllm-tp2-schema",
            admission_topology_digest="vllm-tp2-admission",
            logical_topology_ref="vllm://parallelism?tp=2&pp=1&dp=1",
        )
        for index in range(2):
            member = RuntimeBindingMemberRef(
                member_id=f"dp0:pp0:tp{index}",
                member_index=index,
                member_count=2,
                group_id="vllm-tp-group",
            )
            resolved = build_reference_resolved_spec(
                source_artifact_id=source.artifact_id,
                artifact_selection_digest=selection_digest,
                device_uuid=device_uuid_for(0),
                tensor=tensor,
                runtime="vllm",
                topology=topology,
                member=member,
                binding_layout_id=f"reference-layout-{index}",
            )
            record = write_reference_resolved_spec_cache_entry(
                cache_root / member.member_id,
                resolved_spec=resolved,
            )
            members.append(
                target_from_reference_cache_record(
                    record,
                    device_uuid=device_uuid_for(0),
                )
            )
        target_set = RealizationTargetSet(
            runtime="vllm",
            source=members[0].source,
            topology=topology,
            group_id="vllm-tp-group",
            members=tuple(members),
        )
        assert target_set.topology.logical_topology_ref == "vllm://parallelism?tp=2&pp=1&dp=1"
        prefetched = prefetch_reference_binding_set(
            client,
            source_artifact_id=source.artifact_id,
            target=target_set,
            retention_policy=PrefetchRetentionPolicy(
                expire_if_unacquired_after_ms=10_000,
                idle_ttl_after_last_release_ms=1_000,
                materialization_timeout_ms=30_000,
                allow_acquire_after_creator_exit=True,
            ),
        )
        assert prefetched.partial is False
        assert len(prefetched.members) == 2
        assert {member.member.member_id for member in prefetched.members} == {
            "dp0:pp0:tp0",
            "dp0:pp0:tp1",
        }

        target_path = tmp_path / "target-set.pb"
        prefetched_path = tmp_path / "prefetched-set.pb"
        target_path.write_bytes(target_set.to_proto().SerializeToString(deterministic=True))
        prefetched_path.write_bytes(
            prefetched.to_proto().SerializeToString(deterministic=True)
        )
        worker = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(_SET_WORKER_SCRIPT),
                listen_addr,
                str(target_path),
                str(prefetched_path),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=True,
            text=True,
            capture_output=True,
            timeout=45.0,
        )
        worker_result = json.loads(worker.stdout.strip().splitlines()[-1])
        assert worker_result["lease_tokens"] == {
            "dp0:pp0:tp0": True,
            "dp0:pp0:tp1": True,
        }
        assert set(worker_result["values"]) == {"dp0:pp0:tp0", "dp0:pp0:tp1"}
        for value in worker_result["values"].values():
            assert math.isclose(value, expected_value, rel_tol=0.0)
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
