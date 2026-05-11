#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from tensorcast.api.store.serving_binding_spec_cache import (
    ServingBindingSpecCacheGroupIndex,
    read_matching_resolved_spec_cache_entry,
    read_resolved_spec_cache_entry,
    read_resolved_spec_cache_group_index,
    serving_binding_spec_cache_root,
    write_resolved_spec_cache_entry,
    write_resolved_spec_cache_group_index,
)
from tensorcast.types import (
    BlobRef,
    ServingBindingMemberRef,
    ServingBindingResolvedSpecCacheEntry,
    ServingBindingSourceRef,
    ServingBindingSourceReuseDecision,
    ServingTopologyRef,
)


def _entry(
    *,
    blob: bytes = b"layout-bytes",
    member_index: int = 0,
    member_count: int = 1,
) -> ServingBindingResolvedSpecCacheEntry:
    topology = ServingTopologyRef(schema_topology_digest="topology-schema")
    member = ServingBindingMemberRef(
        member_id=f"member-{member_index}",
        member_index=member_index,
        member_count=member_count,
        group_id="group-1",
    )
    source = ServingBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:checkpoint",
        source_schema_hash="source-schema",
    )
    blob_ref = BlobRef(
        path="target_layout.bin",
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
    )
    draft = ServingBindingResolvedSpecCacheEntry(
        schema_version=1,
        cache_key_digest="placeholder",
        spec_digest="placeholder",
        runtime="vllm",
        source=source,
        source_reuse=ServingBindingSourceReuseDecision(
            mode="checkpoint_to_serving",
            representation_contract_hash="repr-contract",
        ),
        topology=topology,
        member=member,
        source_schema_hash="source-schema",
        model_config_digest="model-config",
        serving_build_digest="serving-build",
        binding_layout_id="layout-1",
        target_layout_hash="target-layout-hash",
        tensor_schema_hash="tensor-schema",
        blob_refs={"target_layout": blob_ref},
    )
    with_cache_key = draft.model_copy(
        update={"cache_key_digest": draft.computed_cache_key_digest()}
    )
    return with_cache_key.model_copy(
        update={"spec_digest": with_cache_key.computed_spec_digest()}
    )


def _with_recomputed_spec_digest(
    entry: ServingBindingResolvedSpecCacheEntry,
) -> ServingBindingResolvedSpecCacheEntry:
    return entry.model_copy(update={"spec_digest": entry.computed_spec_digest()})


def _with_recomputed_digests(
    entry: ServingBindingResolvedSpecCacheEntry,
) -> ServingBindingResolvedSpecCacheEntry:
    with_cache_key = entry.model_copy(
        update={"cache_key_digest": entry.computed_cache_key_digest()}
    )
    return _with_recomputed_spec_digest(with_cache_key)


def test_resolved_spec_cache_roundtrip(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)

    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    record = read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)

    assert record.entry == entry
    assert record.blobs == {"target_layout": blob}


def test_first_cold_start_cache_write_publishes_readable_entry(tmp_path) -> None:
    blob = b"cold-start-compiled-layout"
    entry = _entry(blob=blob)

    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )

    record = read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)
    spec_dir = (
        serving_binding_spec_cache_root(tmp_path)
        / "specs"
        / "sha256"
        / entry.spec_digest
    )
    assert record.entry == entry
    assert (spec_dir / "manifest.json").is_file()
    assert (spec_dir / "target_layout.bin").read_bytes() == blob


def test_resolved_spec_cache_steady_state_reuses_exact_cache_entry(
    tmp_path,
) -> None:
    blob = b"steady-state-layout"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )

    record = read_matching_resolved_spec_cache_entry(tmp_path, expected_entry=entry)

    assert record.entry == entry
    assert record.blobs == {"target_layout": blob}


def test_resolved_spec_cache_matching_read_rejects_worker_mismatch(
    tmp_path,
) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    expected_different_worker_spec = _with_recomputed_spec_digest(
        entry.model_copy(update={"binding_layout_id": "worker-layout-2"})
    )

    with pytest.raises(ValueError, match="does not match expected"):
        read_matching_resolved_spec_cache_entry(
            tmp_path, expected_entry=expected_different_worker_spec
        )


def test_resolved_spec_cache_rejects_bad_cache_key_digest(tmp_path) -> None:
    entry = _entry().model_copy(update={"cache_key_digest": "bad"})

    with pytest.raises(ValueError, match="cache_key_digest"):
        write_resolved_spec_cache_entry(
            tmp_path,
            entry=entry,
            blobs={"target_layout": b"layout-bytes"},
        )


def test_resolved_spec_cache_rejects_bad_spec_digest(tmp_path) -> None:
    entry = _entry().model_copy(update={"spec_digest": "bad"})

    with pytest.raises(ValueError, match="spec_digest"):
        write_resolved_spec_cache_entry(
            tmp_path,
            entry=entry,
            blobs={"target_layout": b"layout-bytes"},
        )


def test_resolved_spec_cache_rejects_blob_hash_mismatch(tmp_path) -> None:
    entry = _entry(blob=b"expected")

    with pytest.raises(ValueError, match="sha256"):
        write_resolved_spec_cache_entry(
            tmp_path,
            entry=entry,
            blobs={"target_layout": b"actual!!"},
        )


def test_resolved_spec_cache_rejects_unsafe_blob_path(tmp_path) -> None:
    blob = b"x"
    unsafe_ref = BlobRef(
        path="../escape.bin",
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
    )
    draft = _entry(blob=blob).model_copy(update={"blob_refs": {"unsafe": unsafe_ref}})
    with_cache_key = draft.model_copy(
        update={"cache_key_digest": draft.computed_cache_key_digest()}
    )
    entry = with_cache_key.model_copy(
        update={"spec_digest": with_cache_key.computed_spec_digest()}
    )

    with pytest.raises(ValueError, match="relative"):
        write_resolved_spec_cache_entry(tmp_path, entry=entry, blobs={"unsafe": blob})


def test_resolved_spec_cache_rejects_unsupported_key_schema_version(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    key_path = (
        serving_binding_spec_cache_root(tmp_path)
        / "keys"
        / "sha256"
        / f"{entry.cache_key_digest}.json"
    )
    payload = json.loads(key_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    key_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)


def test_resolved_spec_cache_rejects_unsupported_manifest_schema_version(
    tmp_path,
) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    manifest_path = (
        serving_binding_spec_cache_root(tmp_path)
        / "specs"
        / "sha256"
        / entry.spec_digest
        / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)


def test_resolved_spec_cache_rejects_unsupported_manifest_producer_version(
    tmp_path,
) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    manifest_path = (
        serving_binding_spec_cache_root(tmp_path)
        / "specs"
        / "sha256"
        / entry.spec_digest
        / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["producer_version"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="producer_version"):
        read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)


def test_resolved_spec_cache_rejects_unsupported_runtime(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _with_recomputed_digests(
        _entry(blob=blob).model_copy(update={"runtime": "unknown-runtime"})
    )

    with pytest.raises(ValueError, match="unsupported serving runtime"):
        write_resolved_spec_cache_entry(
            tmp_path,
            entry=entry,
            blobs={"target_layout": blob},
        )


def test_resolved_spec_cache_cleans_tmp_after_publish(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)

    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )

    tmp_dir = serving_binding_spec_cache_root(tmp_path) / "tmp"
    assert tmp_dir.exists()
    assert list(tmp_dir.iterdir()) == []


def test_resolved_spec_cache_concurrent_writers_are_idempotent(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    errors: list[BaseException] = []

    def write_once() -> None:
        try:
            write_resolved_spec_cache_entry(
                tmp_path,
                entry=entry,
                blobs={"target_layout": blob},
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write_once) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    record = read_resolved_spec_cache_entry(tmp_path, entry.cache_key_digest)
    assert record.entry == entry


def test_resolved_spec_cache_rejects_same_key_different_spec(tmp_path) -> None:
    blob = b"layout-bytes"
    entry = _entry(blob=blob)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    different_spec = _with_recomputed_spec_digest(
        entry.model_copy(update={"binding_layout_id": "layout-2"})
    )

    with pytest.raises(ValueError, match="different resolved spec"):
        write_resolved_spec_cache_entry(
            tmp_path,
            entry=different_spec,
            blobs={"target_layout": blob},
        )


def test_resolved_spec_cache_group_index_roundtrip(tmp_path) -> None:
    blob_0 = b"layout-member-0"
    blob_1 = b"layout-member-1"
    entry_0 = _entry(blob=blob_0, member_index=0, member_count=2)
    entry_1 = _entry(blob=blob_1, member_index=1, member_count=2)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry_0,
        blobs={"target_layout": blob_0},
    )
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry_1,
        blobs={"target_layout": blob_1},
    )
    draft = ServingBindingSpecCacheGroupIndex(
        group_cache_key_digest="placeholder",
        runtime="vllm",
        topology=entry_0.topology,
        group_id="group-1",
        member_cache_key_digests={
            entry_0.member.member_id: entry_0.cache_key_digest,
            entry_1.member.member_id: entry_1.cache_key_digest,
        },
    )
    index = draft.model_copy(
        update={"group_cache_key_digest": draft.computed_group_cache_key_digest()}
    )

    write_resolved_spec_cache_group_index(tmp_path, index=index)
    read_index = read_resolved_spec_cache_group_index(
        tmp_path, index.group_cache_key_digest
    )

    assert read_index == index
    assert set(read_index.member_cache_key_digests) == {"member-0", "member-1"}


def test_resolved_spec_cache_group_index_rejects_member_mismatch(tmp_path) -> None:
    blob = b"layout-member-0"
    entry = _entry(blob=blob, member_index=0, member_count=2)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    draft = ServingBindingSpecCacheGroupIndex(
        group_cache_key_digest="placeholder",
        runtime="vllm",
        topology=entry.topology,
        group_id="group-1",
        member_cache_key_digests={"member-1": entry.cache_key_digest},
    )
    index = draft.model_copy(
        update={"group_cache_key_digest": draft.computed_group_cache_key_digest()}
    )

    with pytest.raises(ValueError, match="member id"):
        write_resolved_spec_cache_group_index(tmp_path, index=index)


def test_resolved_spec_cache_group_lookup_validates_member_cache(tmp_path) -> None:
    blob = b"layout-member-0"
    entry = _entry(blob=blob, member_index=0, member_count=1)
    write_resolved_spec_cache_entry(
        tmp_path,
        entry=entry,
        blobs={"target_layout": blob},
    )
    draft = ServingBindingSpecCacheGroupIndex(
        group_cache_key_digest="placeholder",
        runtime="vllm",
        topology=entry.topology,
        group_id="group-1",
        member_cache_key_digests={entry.member.member_id: entry.cache_key_digest},
    )
    index = draft.model_copy(
        update={"group_cache_key_digest": draft.computed_group_cache_key_digest()}
    )
    write_resolved_spec_cache_group_index(tmp_path, index=index)
    key_path = (
        serving_binding_spec_cache_root(tmp_path)
        / "keys"
        / "sha256"
        / f"{entry.cache_key_digest}.json"
    )
    payload = json.loads(key_path.read_text(encoding="utf-8"))
    payload["entry"]["runtime"] = "other-runtime"
    key_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported serving runtime|match cache key"):
        read_resolved_spec_cache_group_index(tmp_path, index.group_cache_key_digest)
