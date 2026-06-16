#  Copyright (c) 2026, TensorCast Team.

"""Artifact runtime source catalog primitives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from tensorcast._c_ext import build_canonical_index_from_safetensors
from tensorcast.api.store.common import (
    canonical_index_to_bytes,
    dtype_from_string,
)
from tensorcast.api.store.types import ArtifactError, CanonicalIndex
from tensorcast.artifact_runtime.errors import SourceSubjectError
from tensorcast.common.identity import ArtifactIdKind, validate_artifact_id
from tensorcast.types import PublicDiskSourceHandle

_SOURCE_CATALOG_FINGERPRINT_VERSION = "tensorcast-source-catalog-v1"
SOURCE_CATALOG_SCHEMA_VERSION = 1
_DTYPE_CACHE: dict[str, torch.dtype] = {}


@dataclass(frozen=True)
class SourceTensorMeta:
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


@dataclass(frozen=True)
class SourceFileEntry:
    path: Path
    logical_name: str
    size_bytes: int
    digest: str | None = None


@dataclass(frozen=True)
class SourceManifest:
    canonical_index_bytes: bytes
    selected_files: tuple[SourceFileEntry, ...]
    schema_version: int = SOURCE_CATALOG_SCHEMA_VERSION


@dataclass(frozen=True)
class SourceCatalog:
    ordered_names: tuple[str, ...]
    meta_by_name: Mapping[str, SourceTensorMeta]
    selected_files: tuple[SourceFileEntry, ...]
    source_artifact_ref: str
    canonical_index_hash: str
    metadata_fingerprint: str
    canonical_index_bytes: bytes
    schema_version: int = SOURCE_CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_artifact_ref",
            resolve_source_artifact_ref(self.source_artifact_ref),
        )


@dataclass(frozen=True)
class SourceSubject:
    """Framework-facing source subject with a durable source artifact root."""

    artifact_ref: str
    subject: Any
    source_kind: str = "opaque"
    metadata_fingerprint: str | None = None

    def broadcast_payload(self) -> dict[str, Any]:
        if self.source_kind == "public_disk":
            subject_payload = _public_disk_source_payload(self.subject)
        else:
            subject_payload = self.subject
        return {
            "kind": self.source_kind,
            "artifact_ref": self.artifact_ref,
            "subject": subject_payload,
            "metadata_fingerprint": self.metadata_fingerprint,
        }

    def profile_fields(self) -> dict[str, Any]:
        source = self.subject
        fields: dict[str, Any] = {
            "artifact_ref": self.artifact_ref,
            "source_kind": self.source_kind,
        }
        if self.metadata_fingerprint is not None:
            fields["metadata_fingerprint"] = self.metadata_fingerprint
        canonical_index = getattr(source, "canonical_index_bytes", None)
        if canonical_index is not None:
            fields["canonical_index_bytes"] = len(canonical_index)
        source_index = getattr(source, "source_index_bytes", None)
        if source_index is not None:
            fields["source_index_bytes"] = len(bytes(source_index or b""))
        for name in ("format_kind", "metadata_capability"):
            value = getattr(source, name, None)
            if value is not None:
                fields[name] = str(value or "")
        return fields


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _optional_text(value: Any) -> str | None:
    return _optional_str(value)


def _optional_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    data = bytes(value)
    return data or None


def _enum_wire_value(value: Any) -> str | int | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, (str, int)):
        return enum_value
    return str(enum_value)


def _public_disk_source_payload(source: Any) -> dict[str, Any]:
    return {
        "path": str(getattr(source, "path", "") or ""),
        "canonical_index_bytes": bytes(source.canonical_index_bytes),
        "artifact_id": str(getattr(source, "artifact_id", "") or ""),
        "generation": int(getattr(source, "generation", 0) or 0),
        "verify_checksums": bool(getattr(source, "verify_checksums", True)),
        "trusted_content_artifact_id": _optional_str(
            getattr(source, "trusted_content_artifact_id", None)
        ),
        "source_index_bytes": _optional_bytes(
            getattr(source, "source_index_bytes", None)
        ),
        "format_kind": _enum_wire_value(getattr(source, "format_kind", None)),
        "metadata_capability": _enum_wire_value(
            getattr(source, "metadata_capability", None)
        ),
        "resolution_strategy": _enum_wire_value(
            getattr(source, "resolution_strategy", None)
        ),
        "validation_mode": _enum_wire_value(getattr(source, "validation_mode", None)),
        "policy_id": _optional_str(getattr(source, "policy_id", None)),
        "exact_size_bytes": int(getattr(source, "exact_size_bytes", 0) or 0),
    }


def _source_subject_from_handle(source: Any) -> SourceSubject:
    artifact_ref = str(getattr(source, "artifact_id", "") or "")
    if not artifact_ref:
        raise RuntimeError("TensorCast source subject is missing a source artifact_id")
    return SourceSubject(
        artifact_ref=artifact_ref,
        subject=source,
        source_kind="public_disk",
    )


def resolve_source_subject(
    path: str,
    *,
    verify_checksums: bool,
) -> SourceSubject:
    from tensorcast.api.store import resolve_public_disk_source

    return _source_subject_from_handle(
        resolve_public_disk_source(
            path,
            verify_checksums=verify_checksums,
        )
    )


def source_subject_from_broadcast_payload(payload: Mapping[str, Any]) -> SourceSubject:
    payload_dict = dict(payload)
    if "kind" not in payload_dict:
        raise SourceSubjectError(
            "TensorCast source subject broadcast payload is missing kind"
        )
    kind = str(payload_dict.get("kind") or "")
    artifact_ref = str(payload_dict.get("artifact_ref") or "")
    if not artifact_ref:
        raise SourceSubjectError(
            "TensorCast source subject broadcast payload is missing artifact_ref"
        )
    source: Any
    if kind == "public_disk":
        subject_payload = payload_dict.get("subject")
        if not isinstance(subject_payload, Mapping):
            raise SourceSubjectError(
                "TensorCast public_disk source subject payload must be a mapping"
            )
        source = PublicDiskSourceHandle(**dict(subject_payload))
    else:
        source = payload_dict.get("subject")
    return SourceSubject(
        artifact_ref=artifact_ref,
        subject=source,
        source_kind=kind,
        metadata_fingerprint=_optional_text(payload_dict.get("metadata_fingerprint")),
    )


def source_subject_broadcast_payload(subject: SourceSubject) -> dict[str, Any]:
    return subject.broadcast_payload()


def is_public_disk_source_subject(subject: Any) -> bool:
    return isinstance(subject, PublicDiskSourceHandle)


def source_catalog_from_selected_safetensors(
    directory: Path | str,
    *,
    selected_files: Sequence[str | os.PathLike[str]],
    source_artifact_ref: str,
) -> SourceCatalog:
    source_ref = resolve_source_artifact_ref(source_artifact_ref)
    root = Path(directory).expanduser().resolve()
    if not selected_files:
        raise ValueError(
            "source_catalog_from_selected_safetensors requires selected_files"
        )
    entries = tuple(_selected_file_entry(root, selected) for selected in selected_files)
    _validate_unique_selected_files(entries)
    canonical_index_bytes = _build_selected_canonical_index_bytes(entries)
    return source_catalog_from_canonical_index_bytes(
        canonical_index_bytes,
        source_artifact_ref=source_ref,
        selected_files=entries,
    )


def source_catalog_from_all_safetensors_dir(
    directory: Path | str,
    *,
    source_artifact_ref: str,
) -> SourceCatalog:
    """Offline/admin helper that intentionally scans every safetensors file."""

    root = Path(directory).expanduser().resolve()
    selected = tuple(str(path) for path in sorted(root.glob("*.safetensors")))
    if not selected:
        raise ValueError(f"No safetensors files found in {root}")
    return source_catalog_from_selected_safetensors(
        root,
        selected_files=selected,
        source_artifact_ref=source_artifact_ref,
    )


def source_catalog_from_manifest(
    manifest: SourceManifest,
    *,
    source_artifact_ref: str,
) -> SourceCatalog:
    if int(manifest.schema_version) != SOURCE_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            "SourceManifest schema_version mismatch: "
            f"expected={SOURCE_CATALOG_SCHEMA_VERSION}, "
            f"actual={manifest.schema_version}"
        )
    source_ref = resolve_source_artifact_ref(source_artifact_ref)
    canonical_index_bytes = bytes(manifest.canonical_index_bytes)
    return source_catalog_from_canonical_index_bytes(
        canonical_index_bytes,
        source_artifact_ref=source_ref,
        selected_files=manifest.selected_files,
    )


def source_catalog_from_canonical_index_bytes(
    canonical_index_bytes: bytes,
    *,
    source_artifact_ref: str,
    selected_files: Sequence[SourceFileEntry] = (),
) -> SourceCatalog:
    """Build SourceCatalog directly from canonical-index JSON bytes.

    The local-ready recipe path only needs source tensor metadata and canonical
    identity. Avoiding the intermediate CanonicalIndex dataclasses keeps the
    daemon-attested manifest fast path artifact-centered while reducing
    per-rank Python object churn.
    """

    source_ref = resolve_source_artifact_ref(source_artifact_ref)
    index_bytes = bytes(canonical_index_bytes)
    try:
        raw = json.loads(index_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            "Failed to parse canonical index JSON",
            status_code="DATA_LOSS",
            retryable=False,
        ) from exc
    if not isinstance(raw, dict):
        raise ArtifactError(
            "Canonical index JSON must be an object",
            status_code="DATA_LOSS",
            retryable=False,
        )

    ordered_names: list[str] = []
    meta_by_name: dict[str, SourceTensorMeta] = {}
    for name, meta in raw.items():
        if not isinstance(meta, (list, tuple)) or len(meta) != 6:
            raise ArtifactError(
                f"Invalid canonical index entry for '{name}'",
                status_code="DATA_LOSS",
                retryable=False,
            )
        _, _, shape, stride, dtype_str, storage_offset = meta
        dtype_key = str(dtype_str)
        dtype = _DTYPE_CACHE.get(dtype_key)
        if dtype is None:
            dtype = dtype_from_string(dtype_key)
            _DTYPE_CACHE[dtype_key] = dtype
        tensor_name = str(name)
        ordered_names.append(tensor_name)
        meta_by_name[tensor_name] = SourceTensorMeta(
            dtype=dtype,
            shape=tuple(int(dim) for dim in shape),
            stride=tuple(int(dim) for dim in stride),
            storage_offset=int(storage_offset),
        )

    ordered_names_tuple = tuple(ordered_names)
    meta_proxy = MappingProxyType(meta_by_name)
    canonical_index_hash = hashlib.sha256(index_bytes).hexdigest()
    return SourceCatalog(
        ordered_names=ordered_names_tuple,
        meta_by_name=meta_proxy,
        selected_files=tuple(selected_files),
        source_artifact_ref=source_ref,
        canonical_index_hash=canonical_index_hash,
        metadata_fingerprint=compute_source_metadata_fingerprint(
            ordered_names=ordered_names_tuple,
            meta_by_name=meta_proxy,
            canonical_index_hash=canonical_index_hash,
            selected_files=selected_files,
        ),
        canonical_index_bytes=index_bytes,
    )


def source_catalog_from_canonical_index(
    index: CanonicalIndex,
    *,
    source_artifact_ref: str,
    selected_files: Sequence[SourceFileEntry] = (),
    canonical_index_bytes: bytes | None = None,
) -> SourceCatalog:
    source_ref = resolve_source_artifact_ref(source_artifact_ref)
    index_bytes = (
        bytes(canonical_index_bytes)
        if canonical_index_bytes is not None
        else canonical_index_to_bytes(index)
    )
    ordered_names = tuple(str(entry.name) for entry in index.entries)
    meta_by_name = MappingProxyType(
        {
            str(entry.name): SourceTensorMeta(
                dtype=entry.dtype,
                shape=tuple(int(dim) for dim in entry.shape),
                stride=tuple(int(dim) for dim in entry.stride),
                storage_offset=int(entry.storage_offset),
            )
            for entry in index.entries
        }
    )
    canonical_index_hash = hashlib.sha256(index_bytes).hexdigest()
    return SourceCatalog(
        ordered_names=ordered_names,
        meta_by_name=meta_by_name,
        selected_files=tuple(selected_files),
        source_artifact_ref=source_ref,
        canonical_index_hash=canonical_index_hash,
        metadata_fingerprint=compute_source_metadata_fingerprint(
            ordered_names=ordered_names,
            meta_by_name=meta_by_name,
            canonical_index_hash=canonical_index_hash,
            selected_files=selected_files,
        ),
        canonical_index_bytes=index_bytes,
    )


def compute_source_metadata_fingerprint(
    *,
    ordered_names: Sequence[str],
    meta_by_name: Mapping[str, SourceTensorMeta],
    canonical_index_hash: str = "",
    selected_files: Sequence[SourceFileEntry] = (),
) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_CATALOG_FINGERPRINT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(ordered_names)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(canonical_index_hash or "").encode("utf-8"))
    digest.update(b"\n")
    for file_entry in selected_files:
        digest.update(file_entry.logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(file_entry.size_bytes)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file_entry.digest or "").encode("utf-8"))
        digest.update(b"\n")
    if canonical_index_hash:
        return digest.hexdigest()
    for name in ordered_names:
        meta = meta_by_name[str(name)]
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(meta.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(",".join(str(int(dim)) for dim in meta.shape).encode("utf-8"))
        digest.update(b"\0")
        digest.update(",".join(str(int(dim)) for dim in meta.stride).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(meta.storage_offset)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _selected_file_entry(
    root: Path, selected: str | os.PathLike[str]
) -> SourceFileEntry:
    raw = Path(selected).expanduser()
    path = raw if raw.is_absolute() else root / raw
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Selected safetensors file not found: {path}")
    if resolved.suffix != ".safetensors":
        raise ValueError(f"Selected source file is not a safetensors file: {path}")
    try:
        logical_name = str(resolved.relative_to(root))
    except ValueError:
        logical_name = resolved.name
    return SourceFileEntry(
        path=resolved,
        logical_name=logical_name,
        size_bytes=int(resolved.stat().st_size),
        digest=None,
    )


def _validate_unique_selected_files(entries: Sequence[SourceFileEntry]) -> None:
    seen: set[Path] = set()
    for entry in entries:
        if entry.path in seen:
            raise ValueError(f"Duplicate selected source file: {entry.path}")
        seen.add(entry.path)


def _build_selected_canonical_index_bytes(
    entries: Sequence[SourceFileEntry],
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="tensorcast-selected-safetensors-") as tmp:
        tmp_dir = Path(tmp)
        for idx, entry in enumerate(entries):
            target = tmp_dir / f"{idx:08d}-{entry.path.name}"
            try:
                target.symlink_to(entry.path)
            except OSError:
                shutil.copy2(entry.path, target)
        return build_canonical_index_from_safetensors(str(tmp_dir))


def resolve_source_artifact_ref(source_artifact_ref: str) -> str:
    normalized_ref = str(source_artifact_ref).strip()
    if not normalized_ref:
        raise ValueError(
            "SourceCatalog requires a real source artifact identity "
            "(real imported source artifact required)"
        )
    try:
        artifact_kind = validate_artifact_id(normalized_ref)
    except ValueError as exc:
        raise ValueError(
            "SourceCatalog requires a real source artifact identity, "
            "not a synthetic disk/key/path ref "
            "(real imported source artifact required)"
        ) from exc
    if artifact_kind not in {ArtifactIdKind.MI2, ArtifactIdKind.MSA1}:
        raise ValueError(
            "SourceCatalog requires a real source artifact identity "
            "(mi2 content identity or msa1 daemon-attested mounted source required)"
        )
    return normalized_ref


__all__ = [
    "SOURCE_CATALOG_SCHEMA_VERSION",
    "SourceCatalog",
    "SourceFileEntry",
    "SourceManifest",
    "SourceSubject",
    "SourceTensorMeta",
    "compute_source_metadata_fingerprint",
    "is_public_disk_source_subject",
    "resolve_source_artifact_ref",
    "resolve_source_subject",
    "source_catalog_from_all_safetensors_dir",
    "source_catalog_from_canonical_index",
    "source_catalog_from_canonical_index_bytes",
    "source_catalog_from_manifest",
    "source_catalog_from_selected_safetensors",
    "source_subject_broadcast_payload",
    "source_subject_from_broadcast_payload",
]
