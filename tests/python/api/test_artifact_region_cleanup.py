#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

region_lifecycle = importlib.import_module(
    "tensorcast.api.store.target_region_lifecycle"
)


def test_target_region_registration_release_uses_realization_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Store:
        def __init__(self) -> None:
            self.register_calls: list[tuple[int, int, int, int]] = []
            self.unregister_calls: list[tuple[str, bool | None]] = []

        def register_vram_region(
            self,
            *,
            device_id: int,
            base_ptr: int,
            size_bytes: int,
            ttl_ms: int,
        ) -> SimpleNamespace:
            self.register_calls.append((device_id, base_ptr, size_bytes, ttl_ms))
            return SimpleNamespace(region_id=f"region:{base_ptr}")

        def unregister_vram_region(
            self, region_id: str, *, force: bool | None = None
        ) -> bool:
            self.unregister_calls.append((str(region_id), force))
            return True

    monkeypatch.setattr(
        region_lifecycle,
        "collect_storage_bases",
        lambda target: {101: 64, 202: 128},
    )
    store = _Store()
    registration = region_lifecycle.register_store_target_regions_for_realization(
        store=store,
        target_tensors={"a": object()},
        device_id=3,
        ttl_ms=0,
        context="test.register",
    )

    assert registration.region_ids == ("region:101", "region:202")
    assert registration.release_policy == ("unregister_target_region",)
    registration.release(context="test.release")
    registration.release(context="test.release_again")

    assert store.register_calls == [(3, 101, 64, 0), (3, 202, 128, 0)]
    assert store.unregister_calls == [("region:101", None), ("region:202", None)]
    assert registration.release_contract is not None
    assert registration.release_contract.release_policy == ("unregister_target_region",)


def test_release_target_region_ids_for_realization_uses_contract() -> None:
    calls: list[tuple[str, bool | None]] = []

    def _unregister(region_id: str, *, force: bool | None = None) -> bool:
        calls.append((str(region_id), force))
        return True

    registration = region_lifecycle.release_target_region_ids_for_realization(
        unregister_region=_unregister,
        region_ids=("region-a", "region-a", "region-b"),
        context="test.release_ids",
    )

    assert calls == [("region-a", None), ("region-b", None)]
    assert registration.region_ids == ("region-a", "region-b")
    assert registration.release_policy == ("unregister_target_region",)
    assert registration.release_contract is not None
    assert registration.release_contract.released is True


def test_target_region_release_contract_swallows_force_failure() -> None:
    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None]] = []

        def unregister_vram_region(
            self, region_id: str, *, force: bool | None = None
        ) -> bool:
            self.calls.append((str(region_id), force))
            if force is None:
                raise RuntimeError("region has active references")
            raise RuntimeError("force unregister not supported")

    store = _Store()
    registration = region_lifecycle.TargetRegionRegistration(
        unregister_region=store.unregister_vram_region,
        region_ids=("region-2",),
        envelope=region_lifecycle.envelope_for_target_region_registration(
            total_bytes=4
        ),
    )
    registration.release(context="test")

    assert store.calls == [("region-2", None), ("region-2", True)]
    assert registration.release_contract is not None
    assert registration.release_contract.release_policy == ("unregister_target_region",)


def test_target_region_release_contract_does_not_force_other_errors() -> None:
    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None]] = []

        def unregister_vram_region(
            self, region_id: str, *, force: bool | None = None
        ) -> bool:
            self.calls.append((str(region_id), force))
            raise RuntimeError("rpc timeout")

    store = _Store()
    registration = region_lifecycle.TargetRegionRegistration(
        unregister_region=store.unregister_vram_region,
        region_ids=("region-3",),
        envelope=region_lifecycle.envelope_for_target_region_registration(
            total_bytes=4
        ),
    )
    registration.release(context="test")

    assert store.calls == [("region-3", None)]
    assert registration.release_contract is not None
    assert registration.release_contract.release_policy == ("unregister_target_region",)
