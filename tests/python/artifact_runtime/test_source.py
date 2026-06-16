#  Copyright (c) 2026, TensorCast Team.

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensorcast.api.store.common import canonical_index_to_bytes
from tensorcast.api.store.types import CanonicalIndex, CanonicalIndexEntry
from tensorcast.api.store.types import ArtifactError
from tensorcast.artifact_runtime.source import (
    SOURCE_CATALOG_SCHEMA_VERSION,
    SourceCatalog,
    SourceManifest,
    SourceTensorMeta,
    compute_source_metadata_fingerprint,
    resolve_source_artifact_ref,
    source_catalog_from_all_safetensors_dir,
    source_catalog_from_canonical_index,
    source_catalog_from_canonical_index_bytes,
    source_catalog_from_manifest,
    source_catalog_from_selected_safetensors,
)


def test_source_catalog_from_selected_safetensors_ignores_unselected_files(
    tmp_path: Path,
) -> None:
    save_file(
        {"a": torch.arange(4, dtype=torch.float32)}, str(tmp_path / "a.safetensors")
    )
    save_file(
        {"z": torch.arange(4, dtype=torch.float32)}, str(tmp_path / "z.safetensors")
    )

    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("a",)
    assert catalog.schema_version == SOURCE_CATALOG_SCHEMA_VERSION
    assert tuple(catalog.meta_by_name) == ("a",)
    assert catalog.meta_by_name["a"].dtype == torch.float32
    assert catalog.meta_by_name["a"].shape == (4,)
    assert catalog.meta_by_name["a"].stride == (1,)
    assert catalog.meta_by_name["a"].storage_offset == 0
    assert catalog.selected_files[0].logical_name == "a.safetensors"
    assert catalog.source_artifact_ref == "mi2:test:source"


def test_source_catalog_requires_selected_files_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    with pytest.raises(ValueError, match="requires selected_files"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=[],
            source_artifact_ref="mi2:test:source",
        )

    with pytest.raises(ValueError, match="Duplicate selected source file"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=["a.safetensors", "a.safetensors"],
            source_artifact_ref="mi2:test:source",
        )


def test_source_catalog_preserves_selected_file_order(tmp_path: Path) -> None:
    save_file({"b": torch.arange(1)}, str(tmp_path / "b.safetensors"))
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["b.safetensors", "a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    assert tuple(entry.logical_name for entry in catalog.selected_files) == (
        "b.safetensors",
        "a.safetensors",
    )
    assert catalog.ordered_names == ("a", "b")


def test_source_catalog_manifest_round_trips_canonical_identity(tmp_path: Path) -> None:
    save_file(
        {"a": torch.arange(4, dtype=torch.float16)}, str(tmp_path / "a.safetensors")
    )
    catalog = source_catalog_from_selected_safetensors(
        tmp_path,
        selected_files=["a.safetensors"],
        source_artifact_ref="mi2:test:source",
    )

    roundtrip = source_catalog_from_manifest(
        SourceManifest(
            canonical_index_bytes=catalog.canonical_index_bytes,
            selected_files=catalog.selected_files,
        ),
        source_artifact_ref="mi2:test:source",
    )

    assert roundtrip.ordered_names == catalog.ordered_names
    assert roundtrip.schema_version == SOURCE_CATALOG_SCHEMA_VERSION
    assert roundtrip.canonical_index_hash == catalog.canonical_index_hash
    assert roundtrip.metadata_fingerprint == catalog.metadata_fingerprint

    with pytest.raises(ValueError, match="schema_version"):
        source_catalog_from_manifest(
            SourceManifest(
                canonical_index_bytes=catalog.canonical_index_bytes,
                selected_files=catalog.selected_files,
                schema_version=999,
            ),
            source_artifact_ref="mi2:test:source",
        )


def test_source_catalog_from_canonical_index_preserves_stride_storage_offset() -> None:
    index = CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="view",
                dtype=torch.float32,
                shape=(2, 2),
                stride=(4, 1),
                storage_offset=3,
                segment_offset=0,
                size_bytes=16,
            ),
        ),
        total_size_bytes=16,
        avbs_hash="",
    )

    catalog = source_catalog_from_canonical_index(
        index,
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("view",)
    assert catalog.meta_by_name["view"].stride == (4, 1)
    assert catalog.meta_by_name["view"].storage_offset == 3


def test_source_catalog_from_canonical_index_bytes_matches_index_path() -> None:
    index = CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="a",
                dtype=torch.bfloat16,
                shape=(2, 4),
                stride=(4, 1),
                storage_offset=0,
                segment_offset=0,
                size_bytes=16,
            ),
            CanonicalIndexEntry(
                name="b",
                dtype=torch.float32,
                shape=(3,),
                stride=(1,),
                storage_offset=2,
                segment_offset=16,
                size_bytes=12,
            ),
        ),
        total_size_bytes=28,
        avbs_hash="",
    )
    index_bytes = canonical_index_to_bytes(index)

    from_index = source_catalog_from_canonical_index(
        index,
        source_artifact_ref="mi2:test:source",
        canonical_index_bytes=index_bytes,
    )
    from_bytes = source_catalog_from_canonical_index_bytes(
        index_bytes,
        source_artifact_ref="mi2:test:source",
    )

    assert from_bytes.ordered_names == from_index.ordered_names
    assert from_bytes.canonical_index_hash == from_index.canonical_index_hash
    assert from_bytes.metadata_fingerprint == from_index.metadata_fingerprint
    assert from_bytes.canonical_index_bytes == index_bytes
    assert from_bytes.meta_by_name["a"] == from_index.meta_by_name["a"]
    assert from_bytes.meta_by_name["b"] == from_index.meta_by_name["b"]


def test_source_catalog_from_canonical_index_bytes_rejects_invalid_json() -> None:
    with pytest.raises(ArtifactError, match="Failed to parse canonical index JSON"):
        source_catalog_from_canonical_index_bytes(
            b"{not-json",
            source_artifact_ref="mi2:test:source",
        )

    with pytest.raises(ArtifactError, match="Canonical index JSON must be an object"):
        source_catalog_from_canonical_index_bytes(
            b"[]",
            source_artifact_ref="mi2:test:source",
        )


def test_source_metadata_fingerprint_uses_canonical_hash_as_identity() -> None:
    base_meta = {
        "w": SourceTensorMeta(
            dtype=torch.float16,
            shape=(4,),
            stride=(1,),
            storage_offset=0,
        )
    }
    changed_meta = {
        "w": SourceTensorMeta(
            dtype=torch.float32,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
        )
    }

    assert compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=base_meta,
        canonical_index_hash="index-hash",
    ) == compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=changed_meta,
        canonical_index_hash="index-hash",
    )
    assert compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=base_meta,
        canonical_index_hash="index-hash-a",
    ) != compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=base_meta,
        canonical_index_hash="index-hash-b",
    )
    assert compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=base_meta,
    ) != compute_source_metadata_fingerprint(
        ordered_names=("w",),
        meta_by_name=changed_meta,
    )


def test_source_catalog_direct_construction_requires_real_source_identity() -> None:
    catalog = SourceCatalog(
        ordered_names=("w",),
        meta_by_name={
            "w": SourceTensorMeta(
                dtype=torch.float32,
                shape=(1,),
                stride=(1,),
                storage_offset=0,
            )
        },
        selected_files=(),
        source_artifact_ref=" msa1:test:source ",
        canonical_index_hash="hash",
        metadata_fingerprint="fingerprint",
        canonical_index_bytes=b"index",
    )

    assert catalog.source_artifact_ref == "msa1:test:source"

    with pytest.raises(ValueError, match="real source artifact identity"):
        SourceCatalog(
            ordered_names=(),
            meta_by_name={},
            selected_files=(),
            source_artifact_ref="disk:/tmp/model",
            canonical_index_hash="hash",
            metadata_fingerprint="fingerprint",
            canonical_index_bytes=b"index",
        )


def test_source_catalog_all_safetensors_dir_is_offline_helper(tmp_path: Path) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))
    save_file({"b": torch.arange(1)}, str(tmp_path / "b.safetensors"))

    catalog = source_catalog_from_all_safetensors_dir(
        tmp_path,
        source_artifact_ref="mi2:test:source",
    )

    assert catalog.ordered_names == ("a", "b")


def test_source_catalog_rejects_synthetic_source_identity(tmp_path: Path) -> None:
    save_file({"a": torch.arange(1)}, str(tmp_path / "a.safetensors"))

    with pytest.raises(ValueError, match="real source artifact identity"):
        source_catalog_from_selected_safetensors(
            tmp_path,
            selected_files=["a.safetensors"],
            source_artifact_ref="disk:/tmp/model",
        )


def test_resolve_source_artifact_ref_is_public_contract() -> None:
    assert resolve_source_artifact_ref(" mi2:test:source ") == "mi2:test:source"
    assert resolve_source_artifact_ref(" msa1:test:source ") == "msa1:test:source"

    with pytest.raises(ValueError, match="real source artifact identity"):
        resolve_source_artifact_ref("key:synthetic")
    with pytest.raises(ValueError, match="real source artifact identity"):
        resolve_source_artifact_ref("/tmp/not-an-artifact")
    with pytest.raises(ValueError, match="real source artifact identity"):
        resolve_source_artifact_ref(
            "cgid:byte_artifact~ns~engine~b64u.bW9kZWw~b64u.djE~layout~b64u.a2V5"
        )
