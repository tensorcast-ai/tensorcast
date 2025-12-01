#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import base64
import threading
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import torch

from tensorcast.api.store.cache import ArtifactCacheEntry
from tensorcast.api.store.common import canonical_index_from_bytes
from tensorcast.api.store.materialization import MaterializationPipeline
from tensorcast.api.store.retry import map_materialization_error
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    FallbackOptions,
)

if TYPE_CHECKING:
    from tensorcast.api.store import Store
    from tensorcast.api.store.runtime import StoreRuntimeContext


@dataclass(frozen=True, slots=True)
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    artifact_id: str
    tensor_names: tuple[str, ...]
    tensor_metas: Mapping[str, TensorMeta]
    total_bytes: int
    generation: int | None


def _fallback_to_dict(fallback: FallbackOptions | None) -> dict[str, object] | None:
    if fallback is None:
        return None
    return {
        "disk_path": fallback.disk_path,
        "prefer_disk": bool(fallback.prefer_disk),
        "allow_p2p": bool(fallback.allow_p2p),
        "verify_checksums": bool(fallback.verify_checksums),
    }


def _fallback_from_dict(data: Mapping[str, object] | None) -> FallbackOptions | None:
    if data is None:
        return None
    disk_path_value = data.get("disk_path")
    disk_path = disk_path_value if isinstance(disk_path_value, str) else None
    return FallbackOptions(
        disk_path=disk_path,
        prefer_disk=bool(data.get("prefer_disk", False)),
        allow_p2p=bool(data.get("allow_p2p", True)),
        verify_checksums=bool(data.get("verify_checksums", True)),
    )


def _meta_from_entry(entry: CanonicalIndexEntry) -> TensorMeta:
    return TensorMeta(
        name=entry.name,
        shape=tuple(entry.shape),
        stride=tuple(entry.stride),
        dtype=entry.dtype,
        storage_offset=int(entry.storage_offset),
        size_bytes=int(entry.size_bytes),
    )


class Artifact:
    """Lazy handle to a TensorCast artifact."""

    def __init__(
        self,
        *,
        store_ref: weakref.ReferenceType["Store"],
        artifact_id: str | None = None,
        key: str | None = None,
        disk_path: str | None = None,
        fallback: FallbackOptions | None = None,
        canonical_index_bytes: bytes | None = None,
        canonical_index: CanonicalIndex | None = None,
        generation: int | None = None,
    ) -> None:
        identifiers = [bool(artifact_id), bool(key), bool(disk_path)]
        if sum(identifiers) != 1:
            raise ArtifactError(
                "Exactly one of artifact_id, key, or disk_path is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        self._artifact_id = artifact_id
        self._key_hint = key
        self._disk_path_hint = disk_path
        self._fallback = fallback
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._tensor_metas: dict[str, TensorMeta] | None = None
        if canonical_index is not None:
            self._tensor_metas = {
                entry.name: _meta_from_entry(entry) for entry in canonical_index.entries
            }
        self._generation = generation
        self._store_ref = store_ref
        self._lock = threading.RLock()
        self._released = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def artifact_id(self) -> str:
        return self._ensure_identified()

    @property
    def key(self) -> str | None:
        return self._key_hint

    @property
    def tensor_names(self) -> tuple[str, ...]:
        canonical_index = self._ensure_metadata()
        return tuple(entry.name for entry in canonical_index.entries)

    def tensor_meta(self, name: str) -> TensorMeta:
        canonical_index = self._ensure_metadata()
        metas = self._tensor_metas or {}
        if name in metas:
            return metas[name]
        for entry in canonical_index.entries:
            if entry.name == name:
                meta = _meta_from_entry(entry)
                metas[name] = meta
                self._tensor_metas = metas
                return meta
        raise ArtifactError(
            f"Tensor '{name}' not found in artifact",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    def describe(self) -> ArtifactDescriptor:
        canonical_index = self._ensure_metadata()
        metas = self._tensor_metas or {
            entry.name: _meta_from_entry(entry) for entry in canonical_index.entries
        }
        self._tensor_metas = metas
        return ArtifactDescriptor(
            artifact_id=self._ensure_identified(),
            tensor_names=tuple(entry.name for entry in canonical_index.entries),
            tensor_metas=dict(metas),
            total_bytes=int(canonical_index.total_size_bytes),
            generation=self._generation,
        )

    def tensor_dict(
        self,
        *,
        device: torch.device | str,
        names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        canonical_index = self._ensure_metadata()
        requested_names = self._validate_tensor_names(canonical_index, names)
        store, runtime, pipeline = self._require_components()
        payload, _ = pipeline.materialize_subset(
            artifact_id=self._artifact_id,
            key=None,
            device=device,
            fallback=self._fallback,
            tensor_names=requested_names,
            canonical_index_hint=self._canonical_index_bytes,
            disk_path_hint=self._disk_path_hint,
        )
        try:
            self._update_metadata_from_payload(payload, runtime)
            state = pipeline._payload_state_dict(payload)
            if requested_names is None:
                return state
            return {name: state[name] for name in requested_names}
        finally:
            pipeline._release_materialized(payload, runtime.ensure_client())

    def tensor(
        self,
        name: str,
        *,
        device: torch.device | str,
        cache: bool = True,  # cache retained for compatibility, no-op in v2
    ) -> torch.Tensor:
        _ = cache  # cache parameter is reserved for future use
        result = self.tensor_dict(device=device, names=[name])
        if name not in result:
            raise ArtifactError(
                f"Tensor '{name}' missing from materialized payload",
                status_code="NOT_FOUND",
                retryable=False,
            )
        return result[name]

    def tensor_dict_into(
        self,
        target: dict[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        canonical_index = self._ensure_metadata()
        _ = self._validate_tensor_names(canonical_index, None)
        _, _, pipeline = self._require_components()
        pipeline.get_into(
            target,
            artifact_id=self._artifact_id,
            key=None,
            device=device,
            fallback=self._fallback,
        )

    def with_fallback(self, fallback: FallbackOptions) -> Artifact:
        clone = Artifact(
            store_ref=self._store_ref,
            artifact_id=self._artifact_id,
            key=self._key_hint,
            disk_path=self._disk_path_hint,
            fallback=fallback,
            canonical_index_bytes=self._canonical_index_bytes,
            canonical_index=self._canonical_index,
            generation=self._generation,
        )
        clone._released = self._released
        clone._tensor_metas = dict(self._tensor_metas or {})
        return clone

    def exists(self) -> bool:
        if self._canonical_index is not None:
            return True
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        try:
            artifact_id = self._ensure_identified()
        except ArtifactError as exc:
            if exc.status_code == "NOT_FOUND":
                return False
            raise
        cached = runtime.get_artifact_index_cached(artifact_id)
        if cached:
            self._hydrate_from_cache_entry(cached)
            return True
        try:
            canonical_index_bytes = runtime.ensure_client().get_artifact_index_by_id(
                artifact_id
            )
        except Exception as exc:  # noqa: BLE001
            error = map_materialization_error(exc)
            if error.status_code == "NOT_FOUND":
                runtime.invalidate_artifact(
                    artifact_id, key=self._key_hint, reason="exists_not_found"
                )
                return False
            raise error from exc
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        self._set_metadata(
            canonical_index_bytes,
            canonical_index,
            generation=self._generation,
            disk_path=self._disk_path_hint,
        )
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                parsed_index=canonical_index,
                generation=self._generation,
                disk_path=self._disk_path_hint,
                expires_at=time.monotonic(),
            )
        )
        return True

    @property
    def is_valid(self) -> bool:
        store = self._store_ref() if self._store_ref is not None else None
        return not self._released and store is not None and not store.closed

    def release(self) -> None:
        with self._lock:
            self._released = True

    def to_dict(self) -> dict[str, object]:
        encoded_index: str | None = None
        if self._canonical_index_bytes:
            encoded_index = base64.b64encode(self._canonical_index_bytes).decode(
                "utf-8"
            )
        return {
            "artifact_id": self._artifact_id,
            "key": self._key_hint,
            "fallback": _fallback_to_dict(self._fallback),
            "canonical_index": encoded_index,
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object], store: "Store") -> Artifact:
        artifact_id = data.get("artifact_id")
        key_hint = data.get("key")
        fallback_dict = data.get("fallback")
        canonical_blob = data.get("canonical_index")
        generation = data.get("generation")
        canonical_index_bytes: bytes | None = None
        canonical_index: CanonicalIndex | None = None
        if canonical_blob:
            try:
                canonical_index_bytes = base64.b64decode(
                    str(canonical_blob).encode("utf-8")
                )
                canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            except Exception as exc:  # noqa: BLE001
                raise ArtifactError(
                    "Failed to decode serialized artifact metadata",
                    status_code="DATA_LOSS",
                    retryable=False,
                ) from exc
        fallback = None
        if isinstance(fallback_dict, Mapping):
            fallback = _fallback_from_dict(cast(Mapping[str, object], fallback_dict))
        generation_value = (
            int(generation) if isinstance(generation, (int, float)) else None
        )
        return cls(
            store_ref=weakref.ref(store),
            artifact_id=str(artifact_id) if artifact_id else None,
            key=str(key_hint) if key_hint else None,
            fallback=fallback,
            canonical_index_bytes=canonical_index_bytes,
            canonical_index=canonical_index,
            generation=generation_value,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_components(
        self,
    ) -> tuple["Store", StoreRuntimeContext, MaterializationPipeline]:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or store.closed or self._released:
            raise ArtifactError(
                "Artifact handle is released or store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return store, store._runtime, store._materialization

    def _runtime_if_available(self) -> StoreRuntimeContext | None:
        store = self._store_ref() if self._store_ref is not None else None
        if store is None or store.closed:
            return None
        return store._runtime

    def _ensure_identified(self) -> str:
        if self._artifact_id:
            return self._artifact_id
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        with self._lock:
            if self._artifact_id:
                return self._artifact_id
            if self._key_hint:
                artifact_id, disk_path = runtime.resolve_key_mapping_cached(
                    key=self._key_hint
                )
                if disk_path and not self._disk_path_hint:
                    self._disk_path_hint = disk_path
                if not artifact_id:
                    raise ArtifactError(
                        f"Artifact key '{self._key_hint}' is not mapped",
                        status_code="NOT_FOUND",
                        retryable=False,
                    )
                self._artifact_id = artifact_id
                return artifact_id
            if self._disk_path_hint:
                raise ArtifactError(
                    "Disk-backed artifact handles are not supported yet",
                    status_code="UNIMPLEMENTED",
                    retryable=False,
                )
            raise ArtifactError(
                "Artifact handle missing identity",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )

    def _ensure_metadata(self) -> CanonicalIndex:
        if (
            self._canonical_index is not None
            and self._canonical_index_bytes is not None
        ):
            return self._canonical_index
        runtime = self._runtime_if_available()
        if runtime is None:
            raise ArtifactError(
                "Store is closed",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        artifact_id = self._ensure_identified()
        with self._lock:
            if (
                self._canonical_index is not None
                and self._canonical_index_bytes is not None
            ):
                return self._canonical_index
            cached = runtime.get_artifact_index_cached(artifact_id)
            if cached:
                self._hydrate_from_cache_entry(cached)
                assert self._canonical_index is not None
                return self._canonical_index
            try:
                canonical_index_bytes = (
                    runtime.ensure_client().get_artifact_index_by_id(artifact_id)
                )
            except Exception as exc:  # noqa: BLE001
                raise map_materialization_error(exc) from exc
            canonical_index = canonical_index_from_bytes(canonical_index_bytes)
            self._set_metadata(
                canonical_index_bytes,
                canonical_index,
                generation=self._generation,
                disk_path=self._disk_path_hint,
            )
            runtime.cache_artifact_index(
                ArtifactCacheEntry(
                    artifact_id=artifact_id,
                    canonical_index_bytes=canonical_index_bytes,
                    parsed_index=canonical_index,
                    generation=self._generation,
                    disk_path=self._disk_path_hint,
                    expires_at=time.monotonic(),
                )
            )
            return canonical_index

    def _hydrate_from_cache_entry(self, entry: ArtifactCacheEntry) -> None:
        self._set_metadata(
            entry.canonical_index_bytes,
            entry.parsed_index,
            generation=entry.generation,
            disk_path=entry.disk_path,
        )

    def _set_metadata(
        self,
        canonical_index_bytes: bytes,
        canonical_index: CanonicalIndex,
        *,
        generation: int | None,
        disk_path: str | None,
    ) -> None:
        self._canonical_index_bytes = canonical_index_bytes
        self._canonical_index = canonical_index
        self._generation = generation
        if disk_path and not self._disk_path_hint:
            self._disk_path_hint = disk_path
        self._tensor_metas = {
            entry.name: _meta_from_entry(entry) for entry in canonical_index.entries
        }

    def _update_metadata_from_payload(
        self, payload, runtime: StoreRuntimeContext
    ) -> None:
        try:
            canonical_index_bytes = payload.canonical_index_bytes
        except Exception:  # noqa: BLE001
            return
        if not canonical_index_bytes:
            return
        canonical_index = canonical_index_from_bytes(canonical_index_bytes)
        artifact_id = self._artifact_id
        if not artifact_id:
            return
        with self._lock:
            if self._canonical_index is None:
                self._set_metadata(
                    canonical_index_bytes,
                    canonical_index,
                    generation=getattr(payload, "generation", None),
                    disk_path=getattr(payload, "disk_path", None),
                )
        runtime.cache_artifact_index(
            ArtifactCacheEntry(
                artifact_id=artifact_id,
                canonical_index_bytes=canonical_index_bytes,
                parsed_index=canonical_index,
                generation=getattr(payload, "generation", None),
                disk_path=getattr(payload, "disk_path", None),
                expires_at=time.monotonic(),
            )
        )

    @staticmethod
    def _validate_tensor_names(
        canonical_index: CanonicalIndex, names: Sequence[str] | None
    ) -> tuple[str, ...] | None:
        if names is None:
            return None
        available = {entry.name for entry in canonical_index.entries}
        ordered: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name in seen:
                continue
            if name not in available:
                raise ArtifactError(
                    f"Tensor '{name}' not found in artifact",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            seen.add(name)
            ordered.append(name)
        return tuple(ordered)


__all__ = ["Artifact", "ArtifactDescriptor", "TensorMeta"]
