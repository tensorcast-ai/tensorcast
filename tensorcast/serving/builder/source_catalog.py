#  Copyright (c) 2026, TensorCast Team.

"""Selected-file source catalog primitives for serving builders."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from tensorcast._c_ext import build_canonical_index_from_safetensors
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.types import CanonicalIndex

_SOURCE_CATALOG_FINGERPRINT_VERSION = "tensorcast-source-catalog-v1"


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


@dataclass(frozen=True)
class SourceCatalog:
    ordered_names: tuple[str, ...]
    meta_by_name: Mapping[str, SourceTensorMeta]
    selected_files: tuple[SourceFileEntry, ...]
    source_artifact_ref: str
    canonical_index_hash: str
    metadata_fingerprint: str
    canonical_index_bytes: bytes


def source_catalog_from_selected_safetensors(
    directory: Path | str,
    *,
    selected_files: Sequence[str | os.PathLike[str]],
    source_artifact_ref: str,
) -> SourceCatalog:
    root = Path(directory).expanduser().resolve()
    if not selected_files:
        raise ValueError(
            "source_catalog_from_selected_safetensors requires selected_files"
        )
    entries = tuple(_selected_file_entry(root, selected) for selected in selected_files)
    _validate_unique_selected_files(entries)
    canonical_index_bytes = _build_selected_canonical_index_bytes(entries)
    return source_catalog_from_canonical_index(
        canonical_index_from_bytes(canonical_index_bytes),
        source_artifact_ref=source_artifact_ref,
        selected_files=entries,
        canonical_index_bytes=canonical_index_bytes,
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
    canonical_index_bytes = bytes(manifest.canonical_index_bytes)
    return source_catalog_from_canonical_index(
        canonical_index_from_bytes(canonical_index_bytes),
        source_artifact_ref=source_artifact_ref,
        selected_files=manifest.selected_files,
        canonical_index_bytes=canonical_index_bytes,
    )


def source_catalog_from_canonical_index(
    index: CanonicalIndex,
    *,
    source_artifact_ref: str,
    selected_files: Sequence[SourceFileEntry] = (),
    canonical_index_bytes: bytes | None = None,
) -> SourceCatalog:
    source_ref = _resolve_source_artifact_ref(source_artifact_ref)
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


def _resolve_source_artifact_ref(source_artifact_ref: str) -> str:
    normalized_ref = str(source_artifact_ref).strip()
    if not normalized_ref:
        raise ValueError("SourceCatalog requires a real source artifact identity")
    if normalized_ref.startswith(("disk:", "key:")):
        raise ValueError(
            "SourceCatalog requires a real source artifact identity, "
            "not a synthetic disk/key ref"
        )
    return normalized_ref


__all__ = [
    "SourceCatalog",
    "SourceFileEntry",
    "SourceManifest",
    "SourceTensorMeta",
    "compute_source_metadata_fingerprint",
    "source_catalog_from_all_safetensors_dir",
    "source_catalog_from_canonical_index",
    "source_catalog_from_manifest",
    "source_catalog_from_selected_safetensors",
]
