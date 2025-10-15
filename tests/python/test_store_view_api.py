#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import threading
from typing import Any, cast

import pytest
import torch

from tensorcast.api.store import (
    CanonicalIndex,
    CanonicalIndexEntry,
    MaterializedArtifact,
    Store,
    StoreOptions,
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
    return store


def test_build_view_spec_narrow_normalizes_length() -> None:
    store = _fresh_store()
    canonical = _make_canonical_index()
    view_spec, has_transpose = Store._build_view_spec(
        store,
        canonical_index=canonical,
        slices={"weights": (slice(32, 96),)},
        transpose=None,
    )
    assert view_spec is not None
    assert not has_transpose
    assert "weights" in view_spec.tensors
    ops = view_spec.tensors["weights"].ops
    assert len(ops) == 1
    narrow = ops[0].narrow
    assert narrow.dim == 0
    assert narrow.start == 32
    assert narrow.length == 64


def test_build_view_spec_identity_returns_none() -> None:
    store = _fresh_store()
    canonical = _make_canonical_index()
    view_spec, has_transpose = Store._build_view_spec(
        store,
        canonical_index=canonical,
        slices={"weights": (slice(0, 256),)},
        transpose=None,
    )
    assert view_spec is None
    assert not has_transpose


def test_build_view_spec_transpose_canonicalizes_sequence() -> None:
    store = _fresh_store()
    canonical = _make_three_dim_index()
    view_spec, has_transpose = Store._build_view_spec(
        store,
        canonical_index=canonical,
        slices=None,
        transpose={"tensor": [(0, 1), (0, 2)]},
    )
    assert view_spec is not None
    assert has_transpose
    ops = view_spec.tensors["tensor"].ops
    # Canonical ordering should be deterministic
    assert [(op.transpose.dim0, op.transpose.dim1) for op in ops] == [(0, 2), (1, 2)]


def test_get_view_invokes_perform_with_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store()
    fake_client = object()
    monkeypatch.setattr(Store, "_ensure_client", lambda self: fake_client)
    fake_spec = store_daemon_pb2.ViewSpec()
    fake_spec.tensors["weights"].ops.add().narrow.dim = 0
    fake_spec.tensors["weights"].ops[-1].narrow.start = 0
    fake_spec.tensors["weights"].ops[-1].narrow.length = 16

    def fake_resolve(
        self,
        *,
        client: Any,
        artifact_id: str | None,
        key: str | None,
        slices: Any,
        transpose: Any,
        view_id: str | None,
    ) -> tuple[str, bytes | None, store_daemon_pb2.ViewSpec | None, bool, str | None]:
        assert client is fake_client
        return "artifact-123", b"{}", fake_spec, False, None

    monkeypatch.setattr(Store, "_resolve_view_inputs", fake_resolve)
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
    ) -> tuple[str, bytes | None, store_daemon_pb2.ViewSpec | None, bool, str | None]:
        return "artifact-123", b"{}", None, False, None

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
