#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib

artifact_mod = importlib.import_module("tensorcast.api.store.artifact")


def test_cleanup_region_uses_force_on_active_references() -> None:
    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None]] = []

        def unregister_vram_region(
            self, region_id: str, *, force: bool | None = None
        ) -> bool:
            self.calls.append((str(region_id), force))
            if force is None:
                raise RuntimeError("region has active references")
            return True

    store = _Store()
    artifact_mod._cleanup_region_ids_best_effort(
        store=store, region_ids=("region-1",), context="test"
    )
    assert store.calls == [("region-1", None), ("region-1", True)]


def test_cleanup_region_swallow_force_failure_on_active_references() -> None:
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
    artifact_mod._cleanup_region_ids_best_effort(
        store=store, region_ids=("region-2",), context="test"
    )
    assert store.calls == [("region-2", None), ("region-2", True)]


def test_cleanup_region_does_not_force_for_other_errors() -> None:
    class _Store:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None]] = []

        def unregister_vram_region(
            self, region_id: str, *, force: bool | None = None
        ) -> bool:
            self.calls.append((str(region_id), force))
            raise RuntimeError("rpc timeout")

    store = _Store()
    artifact_mod._cleanup_region_ids_best_effort(
        store=store, region_ids=("region-3",), context="test"
    )
    assert store.calls == [("region-3", None)]
