#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import contextlib
import threading
from typing import Any
from types import SimpleNamespace

import grpc
import pytest
import torch

from tensorcast import _C
from tensorcast.api.store import (
    CanonicalIndex,
    CanonicalIndexEntry,
    MaterializedArtifact,
    Store,
    StoreOptions,
    ArtifactError,
)
from tensorcast.api._view_ops import (
    NarrowOp,
    ResolvedViewInputs,
    TransposeOp,
    ViewSpecBuildResult,
)
from tensorcast.proto.daemon.v1 import store_daemon_pb2


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


def _fresh_store() -> Store:
    store = object.__new__(Store)
    store._opts = StoreOptions()
    store._retry_policies = {}
    store._daemon_endpoint = "dummy"
    store._session_id = "sess"
    store._pending_futures = set()
    store._key_cache = {}
    store._key_cache_lock = threading.RLock()
    store._metadata_lock = threading.RLock()
    store._metadata = {}
    return store


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
    result = Store._build_view_spec(
        store,
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
    result = Store._build_view_spec(
        store,
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
    result = Store._build_view_spec(
        store,
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
        Store._build_view_spec(
            store,
            canonical_index=canonical,
            slices=[(slice(0, 1),)],
            transpose=None,
        )
    with pytest.raises(ArtifactError, match="Transpose spec must be a mapping"):
        Store._build_view_spec(
            store,
            canonical_index=canonical,
            slices=None,
            transpose=[(0, 1)],
        )


def test_get_view_invokes_perform_with_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()
    fake_client = object()
    monkeypatch.setattr(Store, "_ensure_client", lambda self: fake_client)
    fake_spec = store_daemon_pb2.ViewSpec()
    fake_spec.tensors["weights"].ops.add().narrow.dim = 0
    fake_spec.tensors["weights"].ops[-1].narrow.start = 0
    fake_spec.tensors["weights"].ops[-1].narrow.length = 16

    build_result = ViewSpecBuildResult(
        proto=fake_spec,
        tensor_ops={"weights": [NarrowOp(dim=0, start=0, length=16)]},
    )

    def fake_resolve(
        self,
        *,
        client: Any,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> ResolvedViewInputs:
        assert client is fake_client
        return ResolvedViewInputs.from_build_result(
            artifact_id="artifact-123",
            canonical_index_bytes=b"{}",
            build_result=build_result,
        )

    monkeypatch.setattr(Store, "_resolve_view_inputs", fake_resolve)

    fake_span = SimpleNamespace(
        add_event=lambda *args, **kwargs: None,
        set_attribute=lambda *args, **kwargs: None,
        set_status=lambda *args, **kwargs: None,
        record_exception=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        Store,
        "_operation_span",
        lambda self, *args, **kwargs: contextlib.nullcontext(fake_span),
    )
    monkeypatch.setattr(
        store,
        "_operation_span",
        lambda *args, **kwargs: contextlib.nullcontext(fake_span),
    )
    placement_value = store_daemon_pb2.TransformPlacement.TRANSFORM_PLACEMENT_SERVER
    monkeypatch.setattr(
        Store,
        "_resolve_transform_placement",
        lambda self, placement, has_transpose: placement_value,
    )
    materialized = MaterializedArtifact(
        artifact_id="artifact-123",
        state_dict={"weights": torch.zeros(16)},
        canonical_index_bytes=b"{}",
        replica_uuid="replica",
    )
    captured: dict[str, Any] = {}

    def fake_perform(self, **kwargs: Any) -> tuple[MaterializedArtifact, int]:
        captured.update(kwargs)
        return materialized, 0

    monkeypatch.setattr(Store, "_perform_get_with_retry", fake_perform)

    result = Store.get_view(store, artifact_id="artifact-123", slices={"weights": (slice(0, 16),)})
    assert result is materialized.state_dict
    assert captured["artifact_id"] == "artifact-123"
    assert captured["view"] is fake_spec
    assert captured["placement"] == placement_value
    assert captured["canonical_index_hint"] == b"{}"


def test_get_view_into_uses_layout_index(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()
    fake_client = object()
    monkeypatch.setattr(Store, "_ensure_client", lambda self: fake_client)

    layout_index_json = (
        b'{"weights":[0,16,[16],[1],"torch.float32",0]}'
    )
    materialized = MaterializedArtifact(
        artifact_id="artifact-123",
        state_dict={"weights": torch.ones(16)},
        canonical_index_bytes=b"{}",
        replica_uuid="replica",
        view_index_bytes=layout_index_json,
    )

    def fake_resolve(
        self,
        *,
        client: Any,
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

    monkeypatch.setattr(Store, "_resolve_view_inputs", fake_resolve)

    def fake_perform(self, **kwargs: Any) -> tuple[MaterializedArtifact, int]:
        return materialized, 0

    monkeypatch.setattr(Store, "_perform_get_with_retry", fake_perform)
    released: dict[str, Any] = {}
    monkeypatch.setattr(
        Store,
        "_release_materialized",
        lambda self, mat, client: released.setdefault("called", True),
    )

    captured_index: dict[str, Any] = {}

    def fake_validate(
        self,
        *,
        canonical_index: CanonicalIndex,
        target: dict[str, torch.Tensor],
        source: dict[str, torch.Tensor],
        device_id: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        captured_index["shapes"] = [entry.shape for entry in canonical_index.entries]
        return [(target["weights"], source["weights"])]

    monkeypatch.setattr(Store, "_validate_targets", fake_validate)

    target = {"weights": torch.zeros(16)}
    Store.get_view_into(store, target, artifact_id="artifact-123", view_id=None, slices={"weights": (slice(0, 16),)})
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
    fake_client = object()
    monkeypatch.setattr(Store, "_ensure_client", lambda self: fake_client)

    canonical_index = b'{"weights":[0,16,[4],[1],"torch.float32",0]}'
    view_spec = store_daemon_pb2.ViewSpec()
    narrow = view_spec.tensors["weights"].ops.add().narrow
    narrow.dim = 0
    narrow.start = 1
    narrow.length = 2

    def fake_resolve(
        self,
        *,
        client: Any,
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

    monkeypatch.setattr(Store, "_resolve_view_inputs", fake_resolve)

    captured: dict[str, Any] = {}

    def fake_perform(self, upload_tensors, **kwargs):
        captured["tensors"] = upload_tensors
        captured["view_registration"] = kwargs.get("view_registration")
        return SimpleNamespace()

    monkeypatch.setattr(Store, "_perform_registration", fake_perform)

    view_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)
    result = Store.register_view(
        store,
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


def test_register_view_server_placement_error_guidance() -> None:
    store = _fresh_store()
    failure = _PlacementRpcError(
        "SERVER placement for view registration requires GPU transpose support on device 0; "
        "retry with placement=CLIENT (mock device missing)"
    )

    mapped = store._map_registration_error(failure)
    assert isinstance(mapped, ArtifactError)
    assert mapped.status_code == "FAILED_PRECONDITION"
    assert mapped.retryable is False
    assert "placement='CLIENT'" in str(mapped)
