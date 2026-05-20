#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tensorcast.types import (
    BlobRef,
    ServingBindingResolvedSpecCacheEntry,
    ServingTopologyRef,
)

_MANIFEST_PRODUCER = "tensorcast.serving_binding_spec_cache"
_MANIFEST_PRODUCER_VERSION = 1


def _validate_runtime(runtime: str) -> None:
    if not str(runtime).strip():
        raise ValueError("serving runtime must not be empty")


class ServingBindingSpecCacheRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    entry: ServingBindingResolvedSpecCacheEntry
    blobs: Mapping[str, bytes]


class ServingBindingSpecCacheGroupIndex(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    group_cache_key_digest: str
    runtime: str
    topology: ServingTopologyRef
    group_id: str
    member_cache_key_digests: Mapping[str, str]

    def canonical_group_key_json(self) -> str:
        payload = {
            "schema_version": int(self.schema_version),
            "runtime": self.runtime,
            "topology": self.topology.model_dump(mode="json", exclude_none=True),
            "group_id": self.group_id,
            "member_cache_key_digests": dict(
                sorted(self.member_cache_key_digests.items())
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def computed_group_cache_key_digest(self) -> str:
        return hashlib.sha256(
            self.canonical_group_key_json().encode("utf-8")
        ).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def serving_binding_spec_cache_root(cache_root: str | os.PathLike[str]) -> Path:
    return Path(cache_root) / "serving_binding_specs" / "v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_blob_path(path: str) -> Path:
    blob_path = Path(path)
    if blob_path.is_absolute() or ".." in blob_path.parts:
        raise ValueError("blob path must be relative and must not contain '..'")
    if not blob_path.parts:
        raise ValueError("blob path must not be empty")
    return blob_path


def _validate_blob_ref(name: str, blob_ref: BlobRef, data: bytes) -> None:
    _safe_relative_blob_path(blob_ref.path)
    if int(blob_ref.size_bytes) != len(data):
        raise ValueError(f"blob {name!r} size does not match BlobRef")
    if blob_ref.sha256 != _sha256_bytes(data):
        raise ValueError(f"blob {name!r} sha256 does not match BlobRef")


def _validate_entry(
    entry: ServingBindingResolvedSpecCacheEntry, blobs: Mapping[str, bytes]
) -> None:
    _validate_runtime(entry.runtime)
    if entry.cache_key_digest != entry.computed_cache_key_digest():
        raise ValueError("cache_key_digest does not match canonical key")
    if entry.spec_digest != entry.computed_spec_digest():
        raise ValueError("spec_digest does not match canonical spec core")
    missing_blobs = set(entry.blob_refs) - set(blobs)
    if missing_blobs:
        missing = ", ".join(sorted(missing_blobs))
        raise ValueError(f"missing blob payload(s): {missing}")
    for name, blob_ref in entry.blob_refs.items():
        _validate_blob_ref(name, blob_ref, blobs[name])


@contextlib.contextmanager
def _key_lock(root: Path, cache_key_digest: str) -> Iterator[None]:
    lock_dir = root / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{cache_key_digest}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _spec_dir(root: Path, spec_digest: str) -> Path:
    return root / "specs" / "sha256" / spec_digest


def _key_path(root: Path, cache_key_digest: str) -> Path:
    return root / "keys" / "sha256" / f"{cache_key_digest}.json"


def _group_path(root: Path, group_cache_key_digest: str) -> Path:
    return root / "groups" / "sha256" / f"{group_cache_key_digest}.json"


def _validate_group_index(index: ServingBindingSpecCacheGroupIndex) -> None:
    if int(index.schema_version) != 1:
        raise ValueError("unsupported group index schema_version")
    if index.group_cache_key_digest != index.computed_group_cache_key_digest():
        raise ValueError("group_cache_key_digest does not match canonical group key")
    _validate_runtime(index.runtime)
    if not index.group_id:
        raise ValueError("group_id must not be empty")
    if not index.member_cache_key_digests:
        raise ValueError("member_cache_key_digests must not be empty")
    if len(set(index.member_cache_key_digests.values())) != len(
        index.member_cache_key_digests
    ):
        raise ValueError("member cache key digests must be distinct")


def _manifest_payload(entry: ServingBindingResolvedSpecCacheEntry) -> dict[str, object]:
    return {
        "schema_version": 1,
        "producer": _MANIFEST_PRODUCER,
        "producer_version": _MANIFEST_PRODUCER_VERSION,
        "entry": entry.model_dump(mode="json", exclude_none=True),
    }


def write_resolved_spec_cache_entry(
    cache_root: str | os.PathLike[str],
    *,
    entry: ServingBindingResolvedSpecCacheEntry,
    blobs: Mapping[str, bytes],
) -> None:
    root = serving_binding_spec_cache_root(cache_root)
    _validate_entry(entry, blobs)
    root.mkdir(parents=True, exist_ok=True)

    with _key_lock(root, entry.cache_key_digest):
        key_path = _key_path(root, entry.cache_key_digest)
        if key_path.exists():
            existing = read_resolved_spec_cache_entry(
                cache_root, entry.cache_key_digest
            )
            if existing.entry != entry:
                raise ValueError("cache key already maps to a different resolved spec")
            return

        spec_dir = _spec_dir(root, entry.spec_digest)
        if spec_dir.exists():
            read_record = _read_record_from_spec_dir(spec_dir=spec_dir)
            if read_record.entry != entry:
                raise ValueError(
                    "spec digest already maps to a different resolved spec"
                )
        else:
            tmp_root = root / "tmp"
            tmp_root.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(
                tempfile.mkdtemp(prefix=f"{entry.cache_key_digest}.", dir=tmp_root)
            )
            tmp_spec_dir = tmp_dir / "spec"
            tmp_spec_dir.mkdir()
            try:
                for name, blob_ref in entry.blob_refs.items():
                    blob_path = tmp_spec_dir / _safe_relative_blob_path(blob_ref.path)
                    blob_path.parent.mkdir(parents=True, exist_ok=True)
                    blob_path.write_bytes(blobs[name])
                manifest_path = tmp_spec_dir / "manifest.json"
                manifest_path.write_bytes(
                    canonical_json_bytes(_manifest_payload(entry))
                )
                read_record = _read_record_from_spec_dir(spec_dir=tmp_spec_dir)
                _validate_entry(read_record.entry, read_record.blobs)
                spec_dir.parent.mkdir(parents=True, exist_ok=True)
                os.rename(tmp_spec_dir, spec_dir)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_tmp = key_path.with_name(f".{key_path.name}.{os.getpid()}.tmp")
        key_payload = {
            "schema_version": 1,
            "cache_key_digest": entry.cache_key_digest,
            "spec_digest": entry.spec_digest,
            "entry": entry.model_dump(mode="json", exclude_none=True),
        }
        key_tmp.write_bytes(canonical_json_bytes(key_payload))
        os.replace(key_tmp, key_path)


def write_resolved_spec_cache_group_index(
    cache_root: str | os.PathLike[str],
    *,
    index: ServingBindingSpecCacheGroupIndex,
) -> None:
    root = serving_binding_spec_cache_root(cache_root)
    _validate_group_index(index)
    root.mkdir(parents=True, exist_ok=True)

    with _key_lock(root, f"group.{index.group_cache_key_digest}"):
        for member_id, cache_key_digest in index.member_cache_key_digests.items():
            record = read_resolved_spec_cache_entry(cache_root, cache_key_digest)
            if record.entry.member.member_id != member_id:
                raise ValueError("group index member id does not match cache entry")
            if record.entry.runtime != index.runtime:
                raise ValueError("group index member runtime mismatch")
            if record.entry.topology != index.topology:
                raise ValueError("group index member topology mismatch")
        group_path = _group_path(root, index.group_cache_key_digest)
        group_path.parent.mkdir(parents=True, exist_ok=True)
        group_tmp = group_path.with_name(f".{group_path.name}.{os.getpid()}.tmp")
        payload = {
            "schema_version": 1,
            "producer": _MANIFEST_PRODUCER,
            "producer_version": _MANIFEST_PRODUCER_VERSION,
            "index": index.model_dump(mode="json", exclude_none=True),
        }
        group_tmp.write_bytes(canonical_json_bytes(payload))
        os.replace(group_tmp, group_path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing cache file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"cache file must contain a JSON object: {path}")
    return payload


def _read_record_from_spec_dir(*, spec_dir: Path) -> ServingBindingSpecCacheRecord:
    manifest = _read_json(spec_dir / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported spec manifest schema_version")
    if manifest.get("producer") != _MANIFEST_PRODUCER:
        raise ValueError("unsupported spec manifest producer")
    if manifest.get("producer_version") != _MANIFEST_PRODUCER_VERSION:
        raise ValueError("unsupported spec manifest producer_version")
    entry_payload = manifest.get("entry")
    if not isinstance(entry_payload, dict):
        raise ValueError("spec manifest missing entry")
    entry = ServingBindingResolvedSpecCacheEntry.model_validate(entry_payload)
    blobs: dict[str, bytes] = {}
    for name, blob_ref in entry.blob_refs.items():
        blob_path = spec_dir / _safe_relative_blob_path(blob_ref.path)
        blobs[name] = blob_path.read_bytes()
    _validate_entry(entry, blobs)
    return ServingBindingSpecCacheRecord(entry=entry, blobs=blobs)


def read_resolved_spec_cache_entry(
    cache_root: str | os.PathLike[str],
    cache_key_digest: str,
) -> ServingBindingSpecCacheRecord:
    root = serving_binding_spec_cache_root(cache_root)
    key_payload = _read_json(_key_path(root, cache_key_digest))
    if key_payload.get("schema_version") != 1:
        raise ValueError("unsupported cache key schema_version")
    if key_payload.get("cache_key_digest") != cache_key_digest:
        raise ValueError("cache key digest mismatch")
    spec_digest = key_payload.get("spec_digest")
    if not isinstance(spec_digest, str) or not spec_digest:
        raise ValueError("cache key missing spec_digest")
    entry_payload = key_payload.get("entry")
    if not isinstance(entry_payload, dict):
        raise ValueError("cache key missing entry")
    key_entry = ServingBindingResolvedSpecCacheEntry.model_validate(entry_payload)
    if key_entry.cache_key_digest != cache_key_digest:
        raise ValueError("cache key entry digest mismatch")
    if key_entry.spec_digest != spec_digest:
        raise ValueError("cache key entry spec_digest mismatch")
    record = _read_record_from_spec_dir(spec_dir=_spec_dir(root, spec_digest))
    if record.entry != key_entry:
        raise ValueError("spec manifest entry does not match cache key entry")
    return record


def read_matching_resolved_spec_cache_entry(
    cache_root: str | os.PathLike[str],
    *,
    expected_entry: ServingBindingResolvedSpecCacheEntry,
) -> ServingBindingSpecCacheRecord:
    if expected_entry.cache_key_digest != expected_entry.computed_cache_key_digest():
        raise ValueError("expected cache_key_digest does not match canonical key")
    if expected_entry.spec_digest != expected_entry.computed_spec_digest():
        raise ValueError("expected spec_digest does not match canonical spec core")

    record = read_resolved_spec_cache_entry(cache_root, expected_entry.cache_key_digest)
    if record.entry != expected_entry:
        raise ValueError("cached resolved spec does not match expected entry")
    return record


def read_resolved_spec_cache_group_index(
    cache_root: str | os.PathLike[str],
    group_cache_key_digest: str,
) -> ServingBindingSpecCacheGroupIndex:
    root = serving_binding_spec_cache_root(cache_root)
    payload = _read_json(_group_path(root, group_cache_key_digest))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported group index schema_version")
    if payload.get("producer") != _MANIFEST_PRODUCER:
        raise ValueError("unsupported group index producer")
    if payload.get("producer_version") != _MANIFEST_PRODUCER_VERSION:
        raise ValueError("unsupported group index producer_version")
    index_payload = payload.get("index")
    if not isinstance(index_payload, dict):
        raise ValueError("group index missing index")
    index = ServingBindingSpecCacheGroupIndex.model_validate(index_payload)
    if index.group_cache_key_digest != group_cache_key_digest:
        raise ValueError("group cache key digest mismatch")
    _validate_group_index(index)
    for member_id, cache_key_digest in index.member_cache_key_digests.items():
        record = read_resolved_spec_cache_entry(cache_root, cache_key_digest)
        if record.entry.member.member_id != member_id:
            raise ValueError("group index member id does not match cache entry")
        if record.entry.runtime != index.runtime:
            raise ValueError("group index member runtime mismatch")
        if record.entry.topology != index.topology:
            raise ValueError("group index member topology mismatch")
    return index
