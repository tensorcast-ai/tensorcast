#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import importlib
import json
import time
import types
from typing import Any, Iterator, Mapping

import pytest
import torch

import tensorcast.api.store as store_mod
from tensorcast.api import _region_cache as region_cache
from tensorcast.api import context as tc_context
from tensorcast.api._config import PlanType
from tensorcast.api.store import ArtifactError, Store
from tensorcast.api.store.binding import Binding, BindingUpdateEpoch
from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.types import VramRegionHandle

artifact_mod = importlib.import_module("tensorcast.api.store.artifact")
inplace_slot_mod = importlib.import_module("tensorcast.api.store.inplace_slot")
owned_binding_slot_mod = importlib.import_module(
    "tensorcast.api.store.owned_binding_slot"
)


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available - binding tests require CUDA tensors")


def _make_index_bytes() -> bytes:
    size_bytes = 16
    elem_bytes = 4
    beta_storage_offset = size_bytes // elem_bytes
    index = {
        "alpha": [0, size_bytes, [4], [1], "torch.float32", 0],
        "beta": [
            size_bytes,
            size_bytes,
            [4],
            [1],
            "torch.float32",
            beta_storage_offset,
        ],
    }
    return json.dumps(index, separators=(",", ":"), sort_keys=True).encode("utf-8")


class FakeBindingClient:
    def __init__(self, index_bytes: bytes) -> None:
        self._index_bytes = index_bytes
        self.materialize_calls: list[dict[str, Any]] = []
        self.create_binding_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []
        self.begin_update_calls: list[dict[str, Any]] = []
        self.seal_calls: list[dict[str, Any]] = []
        self.refill_calls: list[dict[str, Any]] = []
        self.close_calls: list[str] = []
        self.publish_calls: list[dict[str, Any]] = []
        self.retire_calls: list[dict[str, Any]] = []
        self.register_calls: list[dict[str, Any]] = []
        self.unregister_calls: list[str] = []
        self.swap_key_calls: list[dict[str, Any]] = []
        self.keepalive_calls: list[tuple[str, int, int]] = []
        self.submit_contribution_calls: list[dict[str, Any]] = []
        self.refill_failures = 0
        self.omit_current_value_on_seal = False
        self.refill_error: ArtifactError | None = None
        self._token_counter = 0
        self._region_counter = 0
        self._key_state: dict[str, tuple[str, int]] = {}
        self._binding_counter = 0
        self._binding_selections: dict[str, common_pb2.ArtifactSelection] = {}
        self._binding_layout_ids: dict[str, str] = {}
        self._binding_generations: dict[str, int] = {}
        self._binding_value_counter = 0

    def _make_binding_value(
        self,
        *,
        binding_id: str,
        selection: common_pb2.ArtifactSelection | None,
    ) -> store_daemon_pb2.BindingValue:
        self._binding_value_counter += 1
        generation = self._binding_generations.get(binding_id, 0) + 1
        self._binding_generations[binding_id] = generation
        value = store_daemon_pb2.BindingValue(
            binding_id=binding_id,
            binding_layout_id=self._binding_layout_ids.get(binding_id, ""),
            binding_value_id=f"value-{self._binding_value_counter}",
            seal_generation=generation,
            is_artifact_backed=selection is not None,
        )
        if selection is not None:
            value.source_artifact_id = str(selection.artifact_id)
            value.selection.CopyFrom(selection)
        return value

    def get_artifact_index_by_id(self, artifact_id: str) -> bytes:
        return self._index_bytes

    def register_vram_region(
        self,
        *,
        device_id: int,
        size_bytes: int,
        ttl_ms: int,
        cuda_ipc_handle: bytes,
        region_name: str | None = None,
    ) -> VramRegionHandle:
        self._region_counter += 1
        self.register_calls.append(
            {
                "device_id": device_id,
                "size_bytes": size_bytes,
                "ttl_ms": ttl_ms,
                "handle": cuda_ipc_handle,
                "region_name": region_name,
            }
        )
        return VramRegionHandle(
            region_id=f"region:binding:{self._region_counter}",
            ttl_ms=ttl_ms,
        )

    def unregister_vram_region(
        self, region_id: str, *, force: bool | None = None
    ) -> bool:
        self.unregister_calls.append(region_id)
        return True

    def materialize_into_target_v2(self, **kwargs: Any) -> Any:
        self.materialize_calls.append(kwargs)
        self._token_counter += 1
        token = f"token-{self._token_counter}".encode("utf-8")
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        return types.SimpleNamespace(
            status=store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED,
            target_publication_token=token,
            resolved_selection=selection,
        )

    def create_binding(self, **kwargs: Any) -> Any:
        self.create_binding_calls.append(kwargs)
        self._binding_counter += 1
        binding_id = f"binding-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        current_value = None
        if kwargs.get("initial_selection") is not None:
            selection = common_pb2.ArtifactSelection()
            selection.CopyFrom(kwargs["initial_selection"])
            self._binding_selections[binding_id] = selection
            current_value = self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            )
        return types.SimpleNamespace(
            binding_id=binding_id,
            target_index_bytes=bytes(kwargs["target_index_bytes"]),
            current_value=current_value,
        )

    def publish_target_replica(self, **kwargs: Any) -> Any:
        self.publish_calls.append(kwargs)
        return types.SimpleNamespace(lease_id="lease-1", replica_id="replica-1")

    def create_owned_binding(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        self._token_counter += 1
        self._binding_counter += 1
        binding_id = f"binding-{self._binding_counter}"
        self._binding_layout_ids[binding_id] = str(kwargs["binding_layout_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["source_selection"])
        if kwargs.get("copy_plan") is not None and kwargs["copy_plan"].entries:
            selection.ClearField("view_spec")
            selection.ClearField("tensor_names")
            selection.ClearField("view_subset_hash")
            selection.view_id = str(kwargs["target_layout"].view_id)
            selection.logical_layout_hash = bytes(
                kwargs["target_layout"].logical_layout_hash
            )
            selection.selection_hash = b"mapped-selection"
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            binding_id=binding_id,
            artifact_id=str(selection.artifact_id),
            target_index_bytes=bytes(kwargs["target_index_bytes"]),
            resolved_selection=selection,
            target_publication_token=f"token-{self._token_counter}".encode("utf-8"),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def commit_binding_artifact(self, **kwargs: Any) -> Any:
        self.commit_calls.append(kwargs)
        binding_id = str(kwargs["binding_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(kwargs["selection"])
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            )
        )

    def begin_binding_update(self, **kwargs: Any) -> Any:
        self.begin_update_calls.append(kwargs)
        return types.SimpleNamespace(update_epoch=f"bue:{kwargs['binding_id']}:1")

    def seal_binding(self, **kwargs: Any) -> Any:
        self.seal_calls.append(kwargs)
        if self.omit_current_value_on_seal:
            return types.SimpleNamespace()
        return types.SimpleNamespace(
            current_value=self._make_binding_value(
                binding_id=str(kwargs["binding_id"]),
                selection=None,
            )
        )

    def refill_owned_binding(self, **kwargs: Any) -> Any:
        self.refill_calls.append(kwargs)
        if self.refill_error is not None:
            raise self.refill_error
        if self.refill_failures > 0:
            self.refill_failures -= 1
            raise ArtifactError(
                "refill failed after bytes changed",
                status_code="DATA_LOSS",
                retryable=False,
            )
        self._token_counter += 1
        binding_id = str(kwargs["binding_id"])
        selection = common_pb2.ArtifactSelection()
        selection.CopyFrom(self._binding_selections[binding_id])
        selection.artifact_id = str(kwargs["artifact_id"])
        self._binding_selections[binding_id] = selection
        return types.SimpleNamespace(
            artifact_id=str(selection.artifact_id),
            resolved_selection=selection,
            target_publication_token=f"token-{self._token_counter}".encode("utf-8"),
            current_value=self._make_binding_value(
                binding_id=binding_id,
                selection=selection,
            ),
        )

    def close_owned_binding(self, *, binding_id: str, **_kwargs: Any) -> Any:
        self.close_calls.append(str(binding_id))
        return types.SimpleNamespace(closed=True)

    def retire_published_replica(self, **kwargs: Any) -> Any:
        self.retire_calls.append(kwargs)
        return types.SimpleNamespace(drained=True, removed=True)

    def keep_alive_registered_artifact(
        self, registration_id: str, ttl_ms: int, epoch: int
    ) -> bool:
        self.keepalive_calls.append((registration_id, ttl_ms, epoch))
        return True

    def swap_key_mapping(
        self,
        *,
        key: str,
        new_artifact_id: str,
        expected_artifact_id: str | None = None,
        expected_generation: int | None = None,
        operation_id: str | None = None,
        timeout_s: float = 10.0,
    ) -> Any:
        self.swap_key_calls.append(
            {
                "key": key,
                "new_artifact_id": new_artifact_id,
                "expected_artifact_id": expected_artifact_id,
                "expected_generation": expected_generation,
                "operation_id": operation_id,
                "timeout_s": timeout_s,
            }
        )
        current_id, generation = self._key_state.get(key, ("", 0))
        if expected_artifact_id and expected_artifact_id != current_id:
            return types.SimpleNamespace(
                ok=False, artifact_id=current_id, generation=generation
            )
        if expected_generation is not None and expected_generation != generation:
            return types.SimpleNamespace(
                ok=False, artifact_id=current_id, generation=generation
            )
        if current_id == new_artifact_id:
            return types.SimpleNamespace(
                ok=True, artifact_id=current_id, generation=generation
            )
        generation += 1
        self._key_state[key] = (new_artifact_id, generation)
        return types.SimpleNamespace(
            ok=True, artifact_id=new_artifact_id, generation=generation
        )

    def submit_binding_contribution(self, **kwargs: Any) -> Any:
        self.submit_contribution_calls.append(dict(kwargs))
        return types.SimpleNamespace(
            accepted=True,
            already_exists=False,
            lease_id="lease-contrib-1",
            lease_generation=1,
            state="accepted",
        )


class FakeRuntime:
    _DEFAULT_LEASE_TTL_MS = 600_000

    def __init__(self, client: FakeBindingClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cache: dict[str, ArtifactCacheEntry] = {}

    def ensure_client(self) -> FakeBindingClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None

    def get_artifact_index_cached(self, artifact_id: str) -> ArtifactCacheEntry | None:
        return self._cache.get(artifact_id)

    def cache_artifact_index(self, entry: ArtifactCacheEntry) -> None:
        self._cache[entry.artifact_id] = entry

    def invalidate_artifact(self, *_args: object, **_kwargs: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _clear_region_cache() -> Iterator[None]:
    yield
    for device_id in list(region_cache._REGIONS_BY_DEVICE.keys()):
        for rec in list(region_cache._REGIONS_BY_DEVICE[device_id]):
            region_cache.unregister_region(rec.region_id)


def _cache_index(runtime: FakeRuntime, artifact_id: str, index_bytes: bytes) -> None:
    parsed = canonical_index_from_bytes(index_bytes)
    entry = ArtifactCacheEntry(
        artifact_id=artifact_id,
        canonical_index_bytes=index_bytes,
        parsed_index=parsed,
        generation=1,
        expires_at=time.monotonic(),
    )
    runtime.cache_artifact_index(entry)


def _setup_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Store, FakeRuntime, FakeBindingClient]:
    _skip_if_no_cuda()
    index_bytes = _make_index_bytes()
    client = FakeBindingClient(index_bytes)
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    _cache_index(runtime, "artifact-1", index_bytes)
    _cache_index(runtime, "artifact-2", index_bytes)
    monkeypatch.setattr(
        store_mod, "get_cuda_memory_handle", lambda *args, **kwargs: b"fake-handle"
    )
    monkeypatch.setattr(
        store_mod,
        "get_cuda_memory_handle_with_offset",
        lambda *args, **kwargs: (b"fake-handle", 0),
    )
    monkeypatch.setattr(artifact_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(inplace_slot_mod, "device_uuid_for", lambda device_id: "gpu-0")
    monkeypatch.setattr(
        artifact_mod,
        "restore_owned_binding_tensors",
        lambda *, response, runtime, device_id: {
            entry.name: torch.empty_strided(
                size=tuple(int(v) for v in entry.shape),
                stride=tuple(int(v) for v in entry.stride),
                dtype=entry.dtype,
                device=torch.device("cuda", int(device_id)),
            )
            for entry in canonical_index_from_bytes(
                bytes(response.target_index_bytes)
            ).entries
        },
    )
    monkeypatch.setattr(
        store_mod,
        "restore_owned_binding_tensors",
        lambda *, response, runtime, device_id: {
            entry.name: torch.empty_strided(
                size=tuple(int(v) for v in entry.shape),
                stride=tuple(int(v) for v in entry.stride),
                dtype=entry.dtype,
                device=torch.device("cuda", int(device_id)),
            )
            for entry in canonical_index_from_bytes(
                bytes(response.target_index_bytes)
            ).entries
        },
    )
    monkeypatch.setattr(
        owned_binding_slot_mod, "device_uuid_for", lambda device_id: "gpu-0"
    )
    return store, runtime, client


def test_binding_swap_preserves_data_ptr(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    binding.swap(artifact2)

    assert binding.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert binding.tensors["beta"].data_ptr() == ptrs["beta"]


def test_binding_view_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    slices = {"alpha": (slice(0, 2),)}
    view = artifact1.view(slices=slices)
    binding = view.bind(device="cuda:0", packing="byte_space")
    selection_before = binding.selection

    binding.swap(artifact2)

    selection_after = binding.selection
    assert selection_before.view_id == selection_after.view_id
    assert selection_before.selection_hash == selection_after.selection_hash


def test_binding_publishability_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    with pytest.raises(ArtifactError) as excinfo:
        _ = artifact.bind(device="cuda:0", packing="append", publish=True)
    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_binding_activation_cas(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    artifact1 = store.artifact(artifact_id="artifact-1")
    artifact2 = store.artifact(artifact_id="artifact-2")

    binding = artifact1.bind(device="cuda:0", packing="byte_space")
    binding.swap(
        artifact2,
        activate_key="model:latest",
        expected_active_artifact_id="",
    )
    assert client.swap_key_calls

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(
            artifact1,
            activate_key="model:latest",
            expected_active_artifact_id="artifact-1",
        )
    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_create_binding_layout_seeded_then_seal_local_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    binding = artifact.bind(device="cuda:0", packing="byte_space")
    layout = binding.layout

    created = store.create_binding(layout, ownership="daemon", device="cuda:0")

    assert created.current_value is None
    assert created.artifact_id is None
    assert created.binding_layout_id == layout.binding_layout_id

    update_epoch = created.begin_update()
    assert update_epoch.binding_id == created.binding_id
    sealed = created.seal_current(update_epoch=update_epoch)

    assert sealed.is_artifact_backed is False
    assert created.current_value is not None
    assert created.artifact_id is None
    with pytest.raises(ArtifactError) as excinfo:
        created.publish_replica()
    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert client.begin_update_calls
    assert client.seal_calls


def test_binding_begin_update_clears_current_value_and_seal_preserves_ptrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    binding = artifact.bind(device="cuda:0", packing="byte_space")
    ptrs = {name: tensor.data_ptr() for name, tensor in binding.tensors.items()}

    update_epoch = binding.begin_update()

    assert binding.current_value is None
    assert binding.artifact_id is None

    sealed = binding.seal_current(update_epoch=update_epoch)

    assert sealed.is_artifact_backed is False
    assert binding.current_value is not None
    assert binding.tensors["alpha"].data_ptr() == ptrs["alpha"]
    assert binding.tensors["beta"].data_ptr() == ptrs["beta"]


def test_binding_rejects_wrong_binding_update_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    binding = store.artifact(artifact_id="artifact-1").bind(
        device="cuda:0",
        packing="byte_space",
    )
    correct_epoch = binding.begin_update()

    with pytest.raises(ArtifactError) as excinfo:
        binding.seal_current(
            update_epoch=BindingUpdateEpoch(
                binding_id="binding-other",
                update_epoch="bue:binding-other:1",
            )
        )
    assert excinfo.value.status_code == "FAILED_PRECONDITION"

    sealed = binding.seal_current(update_epoch=correct_epoch)
    assert sealed.binding_id == binding.binding_id


def test_binding_failed_refill_clears_current_value_and_marks_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    binding = store.artifact(artifact_id="artifact-1").bind(
        device="cuda:0",
        packing="byte_space",
    )
    client.refill_failures = 1

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(store.artifact(artifact_id="artifact-2"))

    assert excinfo.value.status_code == "DATA_LOSS"
    assert binding.current_value is None
    assert binding.artifact_id is None
    assert binding.selection is None
    assert binding._slot.dirty is True


def test_binding_missing_seal_current_value_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    layout = artifact.bind(device="cuda:0", packing="byte_space").layout
    created = store.create_binding(layout, ownership="daemon", device="cuda:0")
    update_epoch = created.begin_update()
    client.omit_current_value_on_seal = True

    with pytest.raises(ArtifactError) as excinfo:
        created.seal_current(update_epoch=update_epoch)

    assert excinfo.value.status_code == "DATA_LOSS"
    assert created.current_value is None
    assert created.artifact_id is None
    assert created.selection is None
    assert created._slot.dirty is True


def test_daemon_owned_swap_precondition_failure_keeps_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    binding = store.artifact(artifact_id="artifact-1").bind(
        device="cuda:0",
        packing="byte_space",
    )
    previous_value = binding.current_value
    assert previous_value is not None
    client.refill_error = ArtifactError(
        "binding value has live assembly contributions: assembly-1",
        status_code="FAILED_PRECONDITION",
        retryable=False,
    )

    with pytest.raises(ArtifactError) as excinfo:
        binding.swap(store.artifact(artifact_id="artifact-2"))

    assert excinfo.value.status_code == "FAILED_PRECONDITION"
    assert binding.current_value is not None
    assert binding.current_value.binding_value_id == previous_value.binding_value_id
    assert binding._slot.dirty is False


def test_create_client_binding_rejects_dtype_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    layout = (
        store.artifact(artifact_id="artifact-1")
        .bind(
            device="cuda:0",
            packing="byte_space",
        )
        .layout
    )
    target_tensors = {
        "alpha": torch.empty((4,), dtype=torch.float16, device="cuda:0"),
        "beta": torch.empty((4,), dtype=torch.float32, device="cuda:0"),
    }

    with pytest.raises(ArtifactError) as excinfo:
        store.create_binding(
            layout,
            ownership="client",
            target_tensors=target_tensors,
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_create_client_binding_rejects_storage_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    layout = (
        store.artifact(artifact_id="artifact-1")
        .bind(
            device="cuda:0",
            packing="byte_space",
        )
        .layout
    )
    base = torch.empty((9,), dtype=torch.float32, device="cuda:0")
    target_tensors = {
        "alpha": base[1:5],
        "beta": base[5:9],
    }

    with pytest.raises(ArtifactError) as excinfo:
        store.create_binding(
            layout,
            ownership="client",
            target_tensors=target_tensors,
        )

    assert excinfo.value.status_code == "FAILED_PRECONDITION"


def test_sealed_binding_value_contributes_piece_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    binding = artifact.view(slices={"alpha": (slice(1, 3),)}).bind(
        device="cuda:0",
        packing="byte_space",
    )
    initial_selection = binding.selection
    assert initial_selection is not None
    expected_view_id = str(initial_selection.view_id)
    captured: dict[str, object] = {}

    def _fake_perform_registration(
        tensors: Mapping[str, torch.Tensor], **kwargs: object
    ) -> object:
        del tensors
        captured.update(kwargs)
        view_registration = kwargs.get("view_registration")
        canonical_ranges = (
            tuple(view_registration.canonical_ranges)
            if view_registration is not None
            else ()
        )
        return types.SimpleNamespace(
            registration_result=types.SimpleNamespace(
                view_id=expected_view_id,
                canonical_ranges=canonical_ranges,
            )
        )

    monkeypatch.setattr(
        store._registration,
        "_perform_registration",
        _fake_perform_registration,
    )
    sealed = binding.seal_current(update_epoch=binding.begin_update())
    attempt = store_mod.AssemblyAttemptRef(
        assembly_id="cgid:assembly-piece",
        layout_id="layout-piece",
        contribution_contract_hash="bafkcontract-piece",
        coordinator_operation_id="sealop-piece",
        coordinator_generation=1,
        expected_view_ids=(expected_view_id,),
    )

    result = sealed.contribute_to_assembly(attempt=attempt)

    assert result.contribution_kind == "piece_partial"
    assert result.view_id == expected_view_id
    assert result.coverage_plan_hash.startswith("bcp1:")
    assert captured["artifact_id"] == "cgid:assembly-piece"
    assert captured["plan"] is PlanType.VRAM_COALESCED
    submit_call = client.submit_contribution_calls[-1]
    assert submit_call["assembly_id"] == "cgid:assembly-piece"
    assert submit_call["view_id"] == expected_view_id
    assert (
        submit_call["contribution_kind"]
        == store_daemon_pb2.BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL
    )


def test_subset_binding_exposes_piece_view_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    binding = store.artifact(artifact_id="artifact-1").subset(["alpha"]).bind(
        device="cuda:0",
        packing="byte_space",
    )
    selection = binding.selection
    assert selection is not None
    assert str(selection.view_id)
    assert selection.HasField("view_spec")
    assert tuple(str(name) for name in selection.tensor_names) == ("alpha",)

    captured: dict[str, object] = {}

    def _fake_perform_registration(
        tensors: Mapping[str, torch.Tensor], **kwargs: object
    ) -> object:
        del tensors
        captured.update(kwargs)
        view_registration = kwargs.get("view_registration")
        canonical_ranges = (
            tuple(view_registration.canonical_ranges)
            if view_registration is not None
            else ()
        )
        return types.SimpleNamespace(
            registration_result=types.SimpleNamespace(
                view_id=str(selection.view_id),
                canonical_ranges=canonical_ranges,
            )
        )

    monkeypatch.setattr(
        store._registration,
        "_perform_registration",
        _fake_perform_registration,
    )

    sealed = binding.seal_current(update_epoch=binding.begin_update())
    attempt = store_mod.AssemblyAttemptRef(
        assembly_id="cgid:assembly-subset-piece",
        layout_id="layout-subset-piece",
        contribution_contract_hash="bafkcontract-subset-piece",
        coordinator_operation_id="sealop-subset-piece",
        coordinator_generation=1,
        expected_view_ids=(str(selection.view_id),),
    )

    result = sealed.contribute_to_assembly(attempt=attempt)

    assert result.contribution_kind == "piece_partial"
    assert result.view_id == str(selection.view_id)
    view_registration = captured["view_registration"]
    assert view_registration is not None
    assert "alpha" in view_registration.view_options.spec.tensors


def test_sealed_binding_value_contributes_canonical_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _runtime, client = _setup_store(monkeypatch)
    binding = store.artifact(artifact_id="artifact-1").bind(
        device="cuda:0",
        packing="byte_space",
    )
    captured: dict[str, object] = {}

    def _fake_perform_registration(
        tensors: Mapping[str, torch.Tensor], **kwargs: object
    ) -> object:
        del tensors
        captured.update(kwargs)
        return types.SimpleNamespace(
            registration_result=types.SimpleNamespace(
                view_id=None,
                canonical_ranges=(),
            )
        )

    monkeypatch.setattr(
        store._registration,
        "_perform_registration",
        _fake_perform_registration,
    )
    sealed = binding.seal_current(update_epoch=binding.begin_update())
    attempt = store_mod.AssemblyAttemptRef(
        assembly_id="cgid:assembly-canonical",
        layout_id="layout-canonical",
        contribution_contract_hash="bafkcontract-canonical",
        coordinator_operation_id="sealop-canonical",
        coordinator_generation=1,
        expected_view_ids=(),
    )

    result = sealed.contribute_to_assembly(attempt=attempt)

    assert result.contribution_kind == "canonical_full"
    assert result.view_id is None
    assert result.coverage_plan_hash.startswith("bcp1:")
    assert captured["artifact_id"] == "cgid:assembly-canonical"
    assert captured["plan"] is PlanType.VRAM_COALESCED
    submit_call = client.submit_contribution_calls[-1]
    assert "view_id" not in submit_call or submit_call["view_id"] is None
    assert (
        submit_call["contribution_kind"]
        == store_daemon_pb2.BINDING_CONTRIBUTION_KIND_CANONICAL_FULL
    )


class _FakeBindingRuntimeClient:
    def keep_alive_registered_artifact(
        self,
        registration_id: str,
        ttl_ms: int,
        epoch: int,
    ) -> bool:
        return True


class _FakeBindingRuntime:
    def __init__(self) -> None:
        self.closed = False
        self._client = _FakeBindingRuntimeClient()

    def ensure_client(self) -> _FakeBindingRuntimeClient:
        return self._client

    def invalidate_artifact(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakeBindingSlot:
    def __init__(self) -> None:
        self._runtime = _FakeBindingRuntime()
        self.tensors: dict[str, object] = {}
        self.binding_id = "binding-1"
        self.binding_layout_id = "layout-1"
        self.layout = types.SimpleNamespace(binding_layout_id="layout-1")
        self.artifact_id = "artifact-1"
        self.selection = common_pb2.ArtifactSelection(artifact_id="artifact-1")
        self.current_value_metadata = types.SimpleNamespace(
            binding_id=self.binding_id,
            binding_layout_id=self.binding_layout_id,
            binding_value_id="value-1",
            seal_generation=1,
            source_artifact_id="artifact-1",
            selection=self.selection,
            is_artifact_backed=True,
        )
        self.published_lease_id: str | None = None
        self.swap_calls: list[dict[str, object]] = []

    def swap(self, artifact: object, **kwargs: object) -> None:
        self.swap_calls.append(kwargs)
        self.artifact_id = str(artifact)
        self.selection = common_pb2.ArtifactSelection(artifact_id=self.artifact_id)
        self.current_value_metadata = types.SimpleNamespace(
            binding_id=self.binding_id,
            binding_layout_id=self.binding_layout_id,
            binding_value_id="value-2",
            seal_generation=2,
            source_artifact_id=self.artifact_id,
            selection=self.selection,
            is_artifact_backed=True,
        )

    def publish_replica(self, *, ttl_ms: int = 0, ctx: object | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeWaitEvent:
    def __init__(self, state: dict[str, bool]) -> None:
        self._state = state

    def synchronize(self) -> None:
        self._state["synced"] = True


class _FakeWaitEventSlot(_FakeBindingSlot):
    def __init__(self, state: dict[str, bool]) -> None:
        super().__init__()
        self._state = state

    def begin_update(self, **_kwargs: object) -> object:
        assert self._state["synced"] is True
        return BindingUpdateEpoch(
            binding_id=self.binding_id,
            update_epoch=f"bue:{self.binding_id}:1",
        )

    def seal_current(self, **_kwargs: object) -> None:
        assert self._state["synced"] is True
        self.current_value_metadata = types.SimpleNamespace(
            binding_id=self.binding_id,
            binding_layout_id=self.binding_layout_id,
            binding_value_id="value-local",
            seal_generation=3,
            source_artifact_id=None,
            selection=None,
            is_artifact_backed=False,
        )
        self.artifact_id = None
        self.selection = None


def test_binding_wait_events_are_synchronized_before_transitions() -> None:
    state = {"synced": False}
    slot = _FakeWaitEventSlot(state)
    binding = Binding(slot)

    epoch = binding.begin_update(wait_events=[_FakeWaitEvent(state)])

    assert isinstance(epoch, BindingUpdateEpoch)
    state["synced"] = False
    sealed = binding.seal_current(
        update_epoch=epoch,
        wait_events=[_FakeWaitEvent(state)],
    )
    assert sealed.is_artifact_backed is False


def test_binding_wait_events_reject_invalid_objects() -> None:
    slot = _FakeWaitEventSlot({"synced": False})
    binding = Binding(slot)

    with pytest.raises(ArtifactError) as excinfo:
        binding.begin_update(wait_events=[object()])
    assert excinfo.value.status_code == "INVALID_ARGUMENT"


def test_binding_swap_encodes_transport_group_tags_into_operation_id() -> None:
    slot = _FakeBindingSlot()
    binding = Binding(slot)
    ctx = tc_context(
        tags={
            "tc.transport.group.kind": "tp_version",
            "tc.transport.group.id": "case-a1:v2",
            "tc.transport.group.total_parts": 16,
            "tc.transport.group.part_id": "rx1:r0",
            "tc.transport.group.priority": 0,
            "tc.transport.group.epoch": 0,
            "tc.transport.request_id": "case-a1:v2:rx1:r0",
        }
    )

    binding.swap("artifact-2", ctx=ctx)

    assert len(slot.swap_calls) == 1
    operation_id = str(slot.swap_calls[0].get("operation_id", ""))
    assert "#tcg:" in operation_id
    assert "kind=tp_version" in operation_id
    assert "gid=case-a1:v2" in operation_id
    assert "tot=16" in operation_id
    assert "part=rx1:r0" in operation_id
    assert "rid=case-a1:v2:rx1:r0" in operation_id


def test_bind_does_not_delegate_to_bind_into(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _runtime, _client = _setup_store(monkeypatch)
    artifact = store.artifact(artifact_id="artifact-1")
    monkeypatch.setattr(
        artifact_mod.Artifact,
        "bind_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bind_into called")
        ),
    )

    binding = artifact.bind(device="cuda:0")

    assert binding.artifact_id == "artifact-1"
