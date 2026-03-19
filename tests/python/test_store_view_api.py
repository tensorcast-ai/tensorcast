#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import contextlib
import json
from types import SimpleNamespace
from typing import Any, Sequence

import grpc
import pytest
import torch

from tensorcast import _C
from tensorcast.api._materialize import MaterializationPayload, TensorPayloadDescriptor
from tensorcast.api._view_ops import (
    NarrowOp,
    ResolvedViewInputs,
    TransposeOp,
    ViewSpecBuildResult,
)
from tensorcast.api.store import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    Store,
    StoreOptions,
)
from tensorcast.api.store import materialization as materialization_mod
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.retry import map_registration_error
from tensorcast.api.store.views import ViewOrchestrator
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def _make_canonical_index() -> CanonicalIndex:
    entry = CanonicalIndexEntry(
        name="weights",
        dtype=torch.float32,
        shape=(256, 512),
        stride=(512, 1),
        storage_offset=0,
        segment_offset=0,
        size_bytes=256 * 512 * 4,
    )
    return CanonicalIndex(entries=(entry,), total_size_bytes=entry.size_bytes, avbs_hash="")


def _make_three_dim_index() -> CanonicalIndex:
    entry = CanonicalIndexEntry(
        name="tensor",
        dtype=torch.float32,
        shape=(2, 3, 4),
        stride=(12, 4, 1),
        storage_offset=0,
        segment_offset=0,
        size_bytes=2 * 3 * 4 * 4,
    )
    return CanonicalIndex(entries=(entry,), total_size_bytes=entry.size_bytes, avbs_hash="")


def _make_payload(
    tensors: dict[str, torch.Tensor],
    *,
    replica_uuid: str = "replica",
    canonical_index_bytes: bytes | None = None,
    view_index_bytes: bytes | None = None,
) -> MaterializationPayload:
    descriptors: list[TensorPayloadDescriptor] = []
    index: dict[str, list[object]] = {}
    offset = 0
    for name, tensor in tensors.items():
        size_bytes = int(tensor.element_size() * tensor.numel())
        index[name] = [
            offset,
            size_bytes,
            list(map(int, tensor.shape)),
            list(map(int, tensor.stride())),
            str(tensor.dtype),
            int(tensor.storage_offset()),
        ]
        descriptors.append(
            TensorPayloadDescriptor(
                name=name,
                dtype=str(tensor.dtype),
                shape=tuple(map(int, tensor.shape)),
                stride=tuple(map(int, tensor.stride())),
                buffer_offset=offset,
                byte_length=size_bytes,
                storage_offset=int(tensor.storage_offset()),
            )
        )
        offset += size_bytes
    canonical_bytes = canonical_index_bytes or json.dumps(index, separators=(",", ":")).encode("utf-8")

    def _iter():
        for desc in descriptors:
            yield desc, tensors[desc.name]

    return MaterializationPayload(
        artifact_id="artifact-123",
        canonical_index_bytes=canonical_bytes,
        descriptors=tuple(descriptors),
        payload_iter=_iter,
        state_dict=dict(tensors),
        replica_uuid=replica_uuid,
        view_index_bytes=view_index_bytes,
    )


class _DummyExecutor:
    def submit(self, fn, *args, **kwargs):
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut


class _DummyRuntime:
    def __init__(self, client: Any | None = None) -> None:
        self._client = client or SimpleNamespace(get_artifact_index_by_id=lambda _: b"{}")
        self.daemon_endpoint = "dummy"
        self.session_id = "sess"
        self.opts = StoreOptions()
        self.retry_policies: dict[str, object] = {}
        self.executor = _DummyExecutor()
        self._key_cache: dict[str, str | None] = {}

    def ensure_client(self) -> Any:
        return self._client

    def operation_span(self, *_args, **_kwargs):
        span = SimpleNamespace(
            add_event=lambda *a, **k: None,
            set_attribute=lambda *a, **k: None,
            set_status=lambda *a, **k: None,
            record_exception=lambda *a, **k: None,
        )
        return contextlib.nullcontext(span)

    def track_future(self, _future: Any) -> None:
        return None

    def resolve_key_mapping_cached(
        self, key: str
    ) -> tuple[str | None, str | None]:
        return self._key_cache.get(key), None

    def cache_key_mapping(self, key: str, artifact_id: str | None) -> None:
        self._key_cache[key] = artifact_id


def _fresh_store() -> Store:
    runtime = _DummyRuntime()
    return Store("dummy:0", runtime=runtime)


class _PlacementRpcError(grpc.RpcError):
    def __init__(self, details: str) -> None:
        self._details = details

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.FAILED_PRECONDITION

    def details(self) -> str:
        return self._details


def test_build_view_spec_narrow_normalizes_length() -> None:
    store = _fresh_store()
    canonical = _make_canonical_index()
    result = store._views._build_view_spec(
        canonical_index=canonical,
        slices={"weights": (slice(32, 96),)},
        transpose=None,
    )
    assert isinstance(result, ViewSpecBuildResult)
    assert result.proto is not None
    assert not result.has_transpose
    assert "weights" in result.proto.tensors
    view_ops = result.proto.tensors["weights"].ops
    assert len(view_ops) == 1
    narrow = view_ops[0].narrow
    assert narrow.dim == 0
    assert narrow.start == 32
    assert narrow.length == 64
    assert result.tensor_ops["weights"][0] == NarrowOp(dim=0, start=32, length=64)


def test_build_view_spec_identity_returns_none() -> None:
    store = _fresh_store()
    canonical = _make_canonical_index()
    result = store._views._build_view_spec(
        canonical_index=canonical,
        slices={"weights": (slice(0, 256),)},
        transpose=None,
    )
    assert result.proto is None
    assert result.is_identity
    assert not result.has_transpose


def test_build_view_spec_transpose_canonicalizes_sequence() -> None:
    store = _fresh_store()
    canonical = _make_three_dim_index()
    result = store._views._build_view_spec(
        canonical_index=canonical,
        slices=None,
        transpose={"tensor": [(0, 1), (0, 2)]},
    )
    assert result.proto is not None
    assert result.has_transpose
    ops = result.proto.tensors["tensor"].ops
    # Canonical ordering should be deterministic
    assert [(op.transpose.dim0, op.transpose.dim1) for op in ops] == [(0, 2), (1, 2)]
    assert len(result.tensor_ops["tensor"]) == 2
    assert all(isinstance(op, TransposeOp) for op in result.tensor_ops["tensor"])


def test_build_view_spec_rejects_non_mapping_inputs() -> None:
    store = _fresh_store()
    canonical = _make_canonical_index()
    with pytest.raises(ArtifactError, match="Slice spec must be a mapping"):
        store._views._build_view_spec(
            canonical_index=canonical,
            slices=[(slice(0, 1),)],
            transpose=None,
        )
    with pytest.raises(ArtifactError, match="Transpose spec must be a mapping"):
        store._views._build_view_spec(
            canonical_index=canonical,
            slices=None,
            transpose=[(0, 1)],
        )


def test_get_view_invokes_perform_with_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()
    fake_spec = common_pb2.ViewSpec()
    fake_spec.tensors["weights"].ops.add().narrow.dim = 0
    fake_spec.tensors["weights"].ops[-1].narrow.start = 0
    fake_spec.tensors["weights"].ops[-1].narrow.length = 16

    build_result = ViewSpecBuildResult(
        proto=fake_spec,
        tensor_ops={"weights": [NarrowOp(dim=0, start=0, length=16)]},
    )

    def fake_resolve_inputs(
        self: ViewOrchestrator,
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        return ResolvedViewInputs.from_build_result(
            artifact_id="artifact-123",
            canonical_index_bytes=b"{}",
            build_result=build_result,
        )

    monkeypatch.setattr(
        store._views,
        "resolve_view_inputs",
        fake_resolve_inputs.__get__(store._views, ViewOrchestrator),
    )

    placement_value = store_daemon_pb2.TransformPlacement.TRANSFORM_PLACEMENT_SERVER
    monkeypatch.setattr(
        store._views,
        "resolve_transform_placement",
        lambda placement, *, has_transpose, for_registration=False: placement_value,
    )
    payload = _make_payload(
        {"weights": torch.zeros(16)},
        canonical_index_bytes=b"{}",
        replica_uuid="replica",
    )
    captured: dict[str, Any] = {}

    def fake_perform(self: MaterializationPipeline, **kwargs: Any) -> tuple[MaterializationPayload, int]:
        captured.update(kwargs)
        return payload, 0

    monkeypatch.setattr(
        store._materialization,
        "_perform_get_with_retry",
        fake_perform.__get__(store._materialization, MaterializationPipeline),
    )

    result = store._materialization.get_view(
        artifact_id="artifact-123",
        slices={"weights": (slice(0, 16),)},
        resolver=store._views.resolve_view_inputs,
    )
    assert set(result.keys()) == {"weights"}
    assert payload.state_dict is not None
    assert result["weights"] is payload.state_dict["weights"]
    assert captured["artifact_id"] == "artifact-123"
    assert captured["view"] is fake_spec
    assert captured["placement"] == placement_value
    assert captured["canonical_index_hint"] == b"{}"


def test_get_view_into_uses_layout_index(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()

    layout_index_json = (
        b'{"weights":[0,16,[16],[1],"torch.float32",0]}'
    )
    payload = _make_payload(
        {"weights": torch.ones(16)},
        canonical_index_bytes=b"{}",
        replica_uuid="replica",
        view_index_bytes=layout_index_json,
    )

    def fake_resolve(
        self: ViewOrchestrator,
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        return ResolvedViewInputs.from_build_result(
            artifact_id="artifact-123",
            canonical_index_bytes=b"{}",
            build_result=ViewSpecBuildResult.identity(),
        )

    monkeypatch.setattr(
        store._views,
        "resolve_view_inputs",
        fake_resolve.__get__(store._views, ViewOrchestrator),
    )

    def fake_perform(self: MaterializationPipeline, **kwargs: Any) -> tuple[MaterializationPayload, int]:
        return payload, 0

    monkeypatch.setattr(
        store._materialization,
        "_perform_get_with_retry",
        fake_perform.__get__(store._materialization, MaterializationPipeline),
    )
    released: dict[str, Any] = {}
    monkeypatch.setattr(
        store._materialization,
        "_release_materialized",
        lambda mat, client: released.setdefault("called", True),
    )

    captured_index: dict[str, Any] = {}

    def fake_validate(
        *,
        canonical_index: CanonicalIndex,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
        required_names: Sequence[str] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        captured_index["shapes"] = [entry.shape for entry in canonical_index.entries]
        return [(target["weights"], source["weights"])]

    monkeypatch.setattr(materialization_mod, "validate_targets", fake_validate)

    target = {"weights": torch.zeros(16)}
    store._materialization.get_view_into(
        target,
        artifact_id="artifact-123",
        view_id=None,
        slices={"weights": (slice(0, 16),)},
        resolver=store._views.resolve_view_inputs,
    )
    assert torch.allclose(target["weights"], torch.ones(16))
    assert captured_index["shapes"] == [(16,)]
    assert released.get("called") is True


def test_compute_view_registration_plan_binding_narrow() -> None:
    canonical_index = b'{"weights":[0,16,[4],[1],"torch.float32",0]}'
    ops = {"weights": [{"type": "narrow", "dim": 0, "start": 1, "length": 2}]}
    plan = _C.compute_view_registration_plan(canonical_index, ops)
    forward = plan["forward"]
    assert forward["view_size_bytes"] == 8
    chunks = plan["write_chunks"]
    assert len(chunks) == 1
    first = chunks[0]
    assert first["canonical_offset"] == 4
    assert first["length"] == 8


def test_register_view_client_placement_builds_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()

    canonical_index = b'{"weights":[0,16,[4],[1],"torch.float32",0]}'
    view_spec = common_pb2.ViewSpec()
    narrow = view_spec.tensors["weights"].ops.add().narrow
    narrow.dim = 0
    narrow.start = 1
    narrow.length = 2

    def fake_resolve(
        self: ViewOrchestrator,
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        build_result = ViewSpecBuildResult(
            proto=view_spec,
            tensor_ops={"weights": [NarrowOp(dim=0, start=1, length=2)]},
        )
        return ResolvedViewInputs.from_build_result(
            artifact_id="artifact-123",
            canonical_index_bytes=canonical_index,
            build_result=build_result,
        )

    monkeypatch.setattr(
        store._views,
        "resolve_view_inputs",
        fake_resolve.__get__(store._views, ViewOrchestrator),
    )

    captured: dict[str, Any] = {}

    def fake_perform(self, upload_tensors, **kwargs):
        captured["tensors"] = upload_tensors
        captured["view_registration"] = kwargs.get("view_registration")
        return SimpleNamespace()

    monkeypatch.setattr(
        store._registration,
        "_perform_registration",
        fake_perform.__get__(store._registration, store._registration.__class__),
    )

    view_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)
    result = store.register_view(
        tensors={"weights": view_tensor},
        artifact_id="artifact-123",
        placement="CLIENT",
    )
    assert isinstance(result, SimpleNamespace)
    canonical = captured["tensors"]["weights"]
    assert torch.allclose(canonical, torch.tensor([0.0, 10.0, 20.0, 0.0]))
    assert canonical.dtype == torch.float32
    view_reg = captured["view_registration"]
    assert view_reg.placement == store_daemon_pb2.TRANSFORM_PLACEMENT_CLIENT


def test_register_view_piece_registration_kind_builds_piece_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _fresh_store()

    canonical_index = b'{"weights":[0,16,[4],[1],"torch.float32",0]}'
    view_spec = common_pb2.ViewSpec()
    narrow = view_spec.tensors["weights"].ops.add().narrow
    narrow.dim = 0
    narrow.start = 1
    narrow.length = 2

    def fake_resolve(
        self: ViewOrchestrator,
        *,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        del artifact_id, key, slices, transpose, view_id
        build_result = ViewSpecBuildResult(
            proto=view_spec,
            tensor_ops={"weights": [NarrowOp(dim=0, start=1, length=2)]},
        )
        return ResolvedViewInputs.from_build_result(
            artifact_id="artifact-123",
            canonical_index_bytes=canonical_index,
            build_result=build_result,
        )

    monkeypatch.setattr(
        store._views,
        "resolve_view_inputs",
        fake_resolve.__get__(store._views, ViewOrchestrator),
    )

    captured: dict[str, Any] = {}

    def fake_perform(self, upload_tensors, **kwargs):
        del self, upload_tensors
        captured["view_registration"] = kwargs.get("view_registration")
        return SimpleNamespace()

    monkeypatch.setattr(
        store._registration,
        "_perform_registration",
        fake_perform.__get__(store._registration, store._registration.__class__),
    )

    result = store.register_view(
        tensors={"weights": torch.tensor([10.0, 20.0], dtype=torch.float32)},
        artifact_id="artifact-123",
        registration_kind="piece",
    )

    assert isinstance(result, SimpleNamespace)
    view_reg = captured["view_registration"]
    assert view_reg.registration_kind == store_daemon_pb2.VIEW_REGISTRATION_KIND_PIECE


def test_register_view_server_placement_error_guidance() -> None:
    _fresh_store()
    failure = _PlacementRpcError(
        "SERVER placement for view registration requires GPU transpose support on device 0; "
        "retry with placement=CLIENT (mock device missing)"
    )

    mapped = map_registration_error(failure)
    assert isinstance(mapped, ArtifactError)
    assert mapped.status_code == "FAILED_PRECONDITION"
    assert mapped.retryable is False
    assert "placement='CLIENT'" in str(mapped)
