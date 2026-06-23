#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tensorcast.api._config import GetArtifactOptions
from tensorcast.api.context import CallContext
from tensorcast.api.operation import OperationRefMetadata
from tensorcast.api.store import artifact as artifact_by_ref
from tensorcast.api.store import from_disk
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.retained_realization import (
    retained_realization_claim_extra_from_handoff,
)
from tensorcast.types import (
    PrefetchHandoffSet,
    PrefetchRetentionPolicy,
    RealizationTarget,
    RealizationTargetSet,
    RuntimeBindingSourceRef,
)

SourceMode = Literal["artifact_ref", "source_path"]

TARGET_PLAN_MANIFEST_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST"
TARGET_PLAN_MANIFEST_WRITE_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_WRITE"
TARGET_PLAN_MANIFEST_JSON_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_JSON"
TARGET_PLAN_MANIFEST_B64_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_B64"
TARGET_PLAN_MANIFEST_SHA256_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_SHA256"
TARGET_PLAN_MANIFEST_CACHE_DIR_ENV = (
    "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_CACHE_DIR"
)
TARGET_PLAN_MANIFEST_CACHE_WRITE_DIR_ENV = (
    "TENSORCAST_LOCAL_READY_TARGET_PLAN_MANIFEST_CACHE_WRITE_DIR"
)
RETAINED_MANIFEST_WRITE_ENV = "TENSORCAST_RETAINED_BINDING_MANIFEST_WRITE"
SOURCE_PATH_FILTER_ENV = "TENSORCAST_LOCAL_READY_TARGET_PLAN_SOURCE_PATH"
PREWARM_SOURCE_MODE_ENV = "TENSORCAST_LOCAL_READY_PREWARM_SOURCE_MODE"
PREWARM_TIMEOUT_S_ENV = "TENSORCAST_LOCAL_READY_PREWARM_TIMEOUT_S"
PREWARM_RETENTION_TTL_MS_ENV = "TENSORCAST_LOCAL_READY_PREWARM_RETENTION_TTL_MS"
WRITE_MATERIALIZING_RECORDS_ENV = "TENSORCAST_LOCAL_READY_WRITE_MATERIALIZING_RECORDS"
MATERIALIZING_RECORD_TIMEOUT_S_ENV = (
    "TENSORCAST_LOCAL_READY_MATERIALIZING_RECORD_TIMEOUT_S"
)
MATERIALIZING_READY_WRITE_ENV = "TENSORCAST_LOCAL_READY_MATERIALIZING_READY_WRITE"
DAEMON_ADDRESS_ENV = "TENSORCAST_LOCAL_READY_DAEMON_ADDRESS"


@dataclass(frozen=True, slots=True)
class LocalReadyTargetPlanIntent:
    record: dict[str, Any]
    source_path: str
    intent_key: str
    source_artifact_ref: str | None
    target_device: str
    target: RealizationTarget
    materialization_options: GetArtifactOptions | None
    retention: PrefetchRetentionPolicy | None


@dataclass(frozen=True, slots=True)
class LocalReadyPrewarmResult:
    intent: LocalReadyTargetPlanIntent
    operation_id: str | None
    operation_ref: dict[str, str] | None
    handoff: Any | None
    retained_record: dict[str, Any] | None
    retained_manifest_path: Path | None


@dataclass(frozen=True, slots=True)
class _IssuedLocalReadyPrewarm:
    intent: LocalReadyTargetPlanIntent
    operation_id: str | None
    operation_ref: dict[str, str] | None
    operation: Any


@dataclass(frozen=True, slots=True)
class _PreparedLocalReadyPrewarm:
    intent: LocalReadyTargetPlanIntent
    artifact: Any


@dataclass(frozen=True, slots=True)
class LocalReadyMaterializingSummary:
    expected_records: int
    retained_records: int
    ready: bool
    readiness_counts: dict[str, int]
    retained_manifest_path: Path | None
    source_paths: tuple[str, ...]
    elapsed_sec: float

    def to_dict(self) -> dict[str, Any]:
        event = (
            "materializing_records_ready"
            if self.ready
            else "materializing_records_unready"
        )
        return {
            "schema_version": 1,
            "producer": "tensorcast.artifact_runtime.local_ready_prewarm",
            "event": event,
            "ready": self.ready,
            "expected_records": self.expected_records,
            "retained_records": self.retained_records,
            "readiness": dict(self.readiness_counts),
            "retained_manifest": (
                str(self.retained_manifest_path)
                if self.retained_manifest_path is not None
                else None
            ),
            "source_paths": list(self.source_paths),
            "elapsed_sec": self.elapsed_sec,
            "written_at_ms": int(time.time() * 1000),
        }


def _canonical_source_path(source_path: str) -> str:
    return str(Path(source_path).expanduser())


def _json_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _collective_group_rankless_options(value: Any) -> Any:
    dumped = _json_model_dump(value)
    if not isinstance(dumped, dict):
        return dumped
    normalized = dict(dumped)
    execution_topology = normalized.get("execution_topology")
    if isinstance(execution_topology, dict):
        execution_topology = dict(execution_topology)
        collective_group = execution_topology.get("collective_group")
        if isinstance(collective_group, dict):
            collective_group = dict(collective_group)
            collective_group.pop("rank", None)
            execution_topology["collective_group"] = collective_group
        normalized["execution_topology"] = execution_topology
    return normalized


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _split_env_values(raw: str | None) -> list[str]:
    return [
        entry.strip() for entry in str(raw or "").split(os.pathsep) if entry.strip()
    ]


def _normalize_sha256_digest(value: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid target-plan manifest sha256: {value!r}")
    return digest


def target_plan_manifest_cache_path(
    cache_dir: str | os.PathLike[str],
    sha256_digest: str,
) -> Path:
    digest = _normalize_sha256_digest(sha256_digest)
    root = Path(cache_dir).expanduser()
    sharded = root / digest[:2] / f"{digest}.json"
    if sharded.exists():
        return sharded
    return root / f"{digest}.json"


def _target_plan_manifest_cache_write_path(
    cache_dir: str | os.PathLike[str],
    sha256_digest: str,
) -> Path:
    digest = _normalize_sha256_digest(sha256_digest)
    root = Path(cache_dir).expanduser()
    return root / digest[:2] / f"{digest}.json"


def _source_path_cache_index_key(source_path: str) -> str:
    return _stable_hash(
        {
            "schema_version": 1,
            "index_kind": "local_ready_target_plan_manifest_by_source_path",
            "source_path": _canonical_source_path(source_path),
        }
    )


def target_plan_manifest_cache_source_index_path(
    cache_dir: str | os.PathLike[str],
    source_path: str,
) -> Path:
    index_key = _source_path_cache_index_key(source_path)
    return (
        Path(cache_dir).expanduser()
        / "index"
        / "source_path"
        / (index_key[:2])
        / f"{index_key}.json"
    )


def _expected_records_from_manifest_records(records: Sequence[dict[str, Any]]) -> int:
    expected = 0
    for record in records:
        member = record.get("expected_member") or record.get("member") or {}
        if isinstance(member, dict):
            with suppress(Exception):
                expected = max(expected, int(member.get("member_count") or 0))
        topology = record.get("topology") or {}
        if isinstance(topology, dict):
            with suppress(Exception):
                expected = max(expected, int(topology.get("member_count") or 0))
    return expected


def write_target_plan_manifest_cache(
    cache_dir: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    source_path: str | None = None,
    producer: str = "tensorcast.artifact_runtime.local_ready_prewarm",
) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        default=str,
    ).encode("utf-8")
    digest = _sha256_bytes(encoded)
    cache_path = _target_plan_manifest_cache_write_path(cache_dir, digest)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or _sha256_bytes(cache_path.read_bytes()) != digest:
        tmp_path = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
        tmp_path.write_bytes(encoded)
        os.replace(tmp_path, cache_path)

    source_records = (
        _records_from_manifest_payload(payload, source_path=source_path)
        if source_path is not None
        else []
    )
    expected_records = _expected_records_from_manifest_records(source_records)
    ready = bool(expected_records > 0 and len(source_records) >= expected_records)
    summary = {
        "schema_version": 1,
        "producer": producer,
        "index_kind": "local_ready_target_plan_manifest_cache",
        "manifest_sha256": digest,
        "manifest_path": str(cache_path),
        "record_count": len(source_records),
        "expected_records": expected_records,
        "ready": ready,
        "updated_at_ms": int(time.time() * 1000),
    }
    if source_path is not None:
        summary["source_path"] = _canonical_source_path(source_path)
        index_path = target_plan_manifest_cache_source_index_path(
            cache_dir, source_path
        )
        _atomic_write_json(index_path, summary)
    return summary


def update_target_plan_manifest_cache_record(
    cache_dir: str | os.PathLike[str],
    *,
    source_path: str,
    record: dict[str, Any],
    producer: str = "tensorcast.artifact_runtime.local_ready_prewarm",
) -> dict[str, Any]:
    intent_key = str(record.get("intent_key") or "")
    if not intent_key:
        raise ValueError("target-plan manifest record requires intent_key")

    import fcntl

    source_key = _canonical_source_path(source_path)
    index_path = target_plan_manifest_cache_source_index_path(cache_dir, source_path)
    work_path = index_path.with_name(index_path.stem + ".manifest.json")
    lock_path = index_path.with_name(index_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = json.loads(work_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        grouped = payload.get("records_by_source_path")
        if not isinstance(grouped, dict):
            grouped = {}
        existing_records = grouped.get(source_key)
        if not isinstance(existing_records, list):
            existing_records = []
        records = [
            dict(candidate)
            for candidate in existing_records
            if isinstance(candidate, dict) and candidate.get("intent_key") != intent_key
        ]
        records.append(dict(record))
        grouped[source_key] = records
        payload.update(
            {
                "schema_version": 1,
                "producer": producer,
                "intent_kind": "local_ready_target_plan",
                "updated_at_ms": int(time.time() * 1000),
                "records_by_source_path": grouped,
            }
        )
        _atomic_write_json(work_path, payload)
        summary = write_target_plan_manifest_cache(
            cache_dir,
            payload,
            source_path=source_path,
            producer=producer,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return summary


def _source_index_sha256_from_cache(
    cache_dir: str | os.PathLike[str],
    source_path: str,
) -> str:
    index_path = target_plan_manifest_cache_source_index_path(cache_dir, source_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target-plan manifest cache index is not an object")
    if payload.get("ready") is not True:
        raise ValueError(
            "target-plan manifest cache index is not ready: "
            f"path={index_path} record_count={payload.get('record_count')} "
            f"expected_records={payload.get('expected_records')}"
        )
    return _normalize_sha256_digest(str(payload.get("manifest_sha256") or ""))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def initialize_runtime_from_env() -> None:
    daemon_address = os.environ.get(DAEMON_ADDRESS_ENV)
    if not daemon_address:
        return
    from tensorcast import startup

    if startup.is_initialized():
        return
    startup.init(mode="connect", address=daemon_address)


def _operation_ref_record(operation: Any) -> dict[str, str] | None:
    metadata = getattr(operation, "operation_ref_metadata", None)
    if callable(metadata):
        metadata = metadata()
    if isinstance(metadata, OperationRefMetadata):
        return metadata.to_dict()
    if isinstance(metadata, dict):
        return OperationRefMetadata.from_dict(metadata).to_dict()

    proto_fn = getattr(operation, "operation_ref_proto", None)
    if callable(proto_fn):
        proto = proto_fn()
        return OperationRefMetadata.from_proto(proto).to_dict()

    operation_id = str(getattr(operation, "operation_id", "") or "")
    if operation_id:
        return {"operation_id": operation_id}
    return None


def operation_ref_from_retained_manifest_record(
    record: Mapping[str, Any],
) -> OperationRefMetadata | None:
    operation_ref = record.get("operation_ref")
    if isinstance(operation_ref, Mapping):
        return OperationRefMetadata.from_dict(operation_ref)
    operation_id = str(record.get("operation_id") or "")
    if operation_id:
        return OperationRefMetadata(operation_id=operation_id)
    return None


def wait_retained_manifest_record_operation_handoff(
    record: Mapping[str, Any],
    *,
    runtime: Any,
    timeout_s: float | None = None,
    ctx: CallContext | None = None,
) -> Any | None:
    operation_ref = operation_ref_from_retained_manifest_record(record)
    if operation_ref is None:
        return None
    from tensorcast.api.operation import DaemonGlobalStoreOperation
    from tensorcast.api.store.artifact import (
        _serving_prefetch_result_from_operation_response,
    )

    if hasattr(runtime, "ensure_initialized"):
        runtime.ensure_initialized()
    operation_runtime = runtime
    if not callable(getattr(operation_runtime, "ensure_client", None)):
        from tensorcast.api.store.runtime import get_context as get_store_context

        operation_runtime = get_store_context()
    try:
        runtime_ref = weakref.ref(operation_runtime)
    except TypeError:

        def runtime_ref() -> Any:
            return operation_runtime

    operation = DaemonGlobalStoreOperation(
        operation_id=operation_ref.operation_id,
        runtime_ref=runtime_ref,
        ctx=ctx,
        context={
            "operation_kind": operation_ref.kind or "prefetch_serving_binding",
            "target_artifact_id": operation_ref.target_artifact_id or "",
        },
        result_factory=_serving_prefetch_result_from_operation_response,
        operation_ref=operation_ref.to_proto(),
    )
    return operation.result(timeout_s=timeout_s)


def _encode_target_proto_fields(target: RealizationTarget) -> dict[str, Any]:
    proto = target.to_proto()
    proto_bytes = proto.SerializeToString(deterministic=True)
    return {
        "target_proto_type": str(
            getattr(getattr(proto, "DESCRIPTOR", None), "full_name", "") or ""
        ),
        "target_proto_encoding": "base64",
        "target_proto_b64": base64.b64encode(proto_bytes).decode("ascii"),
        "target_proto_sha256": hashlib.sha256(proto_bytes).hexdigest(),
    }


def retained_local_ready_cache_key_for_intent(
    intent: LocalReadyTargetPlanIntent,
) -> str:
    target = intent.target
    record = intent.record
    payload = {
        "schema_version": 1,
        "source_path": _canonical_source_path(intent.source_path),
        "model_hash": str(record.get("model_hash") or ""),
        "target_device": intent.target_device,
        "device_uuid": record.get("device_uuid") or target.device_uuid,
        "member": _json_model_dump(target.member),
        "target_layout_hash": str(
            record.get("target_layout_hash")
            or target.resolved_layout.target_layout_hash
        ),
        "tensor_schema_hash": str(
            record.get("tensor_schema_hash")
            or target.resolved_layout.tensor_schema_hash
        ),
        "serving_build_digest": str(
            record.get("serving_build_digest") or target.runtime_build_digest
        ),
        "source_selection_digest": str(
            record.get("source_selection_digest")
            or target.source.artifact_selection_digest
        ),
    }
    return _stable_hash(payload)


def _retained_manifest_identity_key(record: dict[str, Any]) -> str:
    payload = {
        "schema_version": 1,
        "cache_identity_version": record.get("cache_identity_version"),
        "source_path": _canonical_source_path(str(record.get("source_path") or "")),
        "model_hash": str(record.get("model_hash") or ""),
        "target_device": str(record.get("target_device") or ""),
        "expected_member": record.get("expected_member"),
        "framework_identity": record.get("framework_identity") or {},
        "runtime_config_digest": str(record.get("runtime_config_digest") or ""),
    }
    return _stable_hash(payload)


def _decode_target(record: dict[str, Any]) -> RealizationTarget:
    encoding = str(record.get("target_proto_encoding") or "")
    if encoding != "base64":
        raise ValueError("target_proto_encoding must be base64")
    raw_b64 = str(record.get("target_proto_b64") or "")
    if not raw_b64:
        raise ValueError("target_proto_b64 is required")
    target_bytes = base64.b64decode(raw_b64, validate=True)
    expected_sha = str(record.get("target_proto_sha256") or "")
    actual_sha = hashlib.sha256(target_bytes).hexdigest()
    if expected_sha and expected_sha != actual_sha:
        raise ValueError("target_proto_sha256 does not match target_proto_b64")
    proto = operation_pb2.ServingBindingTarget()
    proto.ParseFromString(target_bytes)
    return RealizationTarget.from_proto(proto)


def _decode_options(record: dict[str, Any]) -> GetArtifactOptions | None:
    raw = record.get("materialization_options")
    if raw is None:
        return None
    if isinstance(raw, GetArtifactOptions):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("materialization_options must be an object")
    return GetArtifactOptions.model_validate(raw)


def _retention_policy_from_ttl_ms(ttl_ms: int) -> PrefetchRetentionPolicy:
    if ttl_ms <= 0:
        raise ValueError("retention_ttl_ms must be positive")
    return PrefetchRetentionPolicy(
        expire_if_unacquired_after_ms=ttl_ms,
        idle_ttl_after_last_release_ms=ttl_ms,
        materialization_timeout_ms=ttl_ms,
        allow_acquire_after_creator_exit=True,
    )


def _decode_retention(record: dict[str, Any]) -> PrefetchRetentionPolicy | None:
    ttl = record.get("retention_ttl_ms")
    if ttl is None:
        return None
    return _retention_policy_from_ttl_ms(int(ttl))


def _intent_with_retention_ttl_ms(
    intent: LocalReadyTargetPlanIntent,
    retention_ttl_ms: int | None,
) -> LocalReadyTargetPlanIntent:
    if retention_ttl_ms is None:
        return intent
    return LocalReadyTargetPlanIntent(
        record=dict(intent.record),
        source_path=intent.source_path,
        intent_key=intent.intent_key,
        source_artifact_ref=intent.source_artifact_ref,
        target_device=intent.target_device,
        target=intent.target,
        materialization_options=intent.materialization_options,
        retention=_retention_policy_from_ttl_ms(int(retention_ttl_ms)),
    )


def _member_identity(member: Any) -> tuple[str, int, int, str]:
    return (
        str(getattr(member, "member_id", "") or ""),
        int(getattr(member, "member_index", 0) or 0),
        int(getattr(member, "member_count", 0) or 0),
        str(getattr(member, "group_id", "") or ""),
    )


def _materializing_key(
    *,
    operation_id: str | None,
    intent: LocalReadyTargetPlanIntent,
) -> str:
    if operation_id:
        return f"{operation_id}:{intent.intent_key}"
    return intent.intent_key


def _handoff_from_prefetch_result(
    intent: LocalReadyTargetPlanIntent,
    result: Any,
) -> Any:
    if not isinstance(result, PrefetchHandoffSet):
        return result
    expected = _member_identity(intent.target.member)
    for member_handoff in result.members:
        if _member_identity(member_handoff.member) == expected:
            return member_handoff
    failed = [
        failure
        for failure in result.member_failures
        if _member_identity(failure.member) == expected
    ]
    if failed:
        failure = failed[0]
        raise RuntimeError(
            "TensorCast retained target-set prewarm failed for "
            f"member={expected[0]} phase={failure.phase or ''} "
            f"code={failure.code}: {failure.message}"
        )
    raise RuntimeError(
        "TensorCast retained target-set prewarm did not return handoff for "
        f"member={expected[0]} index={expected[1]}"
    )


def _target_set_source_identity(source: RuntimeBindingSourceRef) -> dict[str, Any]:
    return {
        "source_kind": source.source_kind,
        "artifact_selection_digest": source.artifact_selection_digest,
        "source_artifact_ref": source.source_artifact_ref,
        "source_schema_hash": source.source_schema_hash,
        "runtime_build_digest": source.runtime_build_digest,
        "tensor_schema_hash": source.tensor_schema_hash,
        "members": _json_model_dump(source.members),
    }


def _target_set_group_key(intent: LocalReadyTargetPlanIntent) -> str | None:
    target = intent.target
    member = target.member
    group_id = str(getattr(member, "group_id", "") or "")
    if not group_id:
        return None
    return _stable_hash(
        {
            "schema_version": 1,
            "runtime": target.runtime,
            "source": _target_set_source_identity(target.source),
            "group_id": group_id,
            "model_config_digest": target.model_config_digest,
            "runtime_build_digest": target.runtime_build_digest,
            "materialization_options": _collective_group_rankless_options(
                intent.materialization_options
            ),
            "retention": _json_model_dump(intent.retention),
        }
    )


def _complete_target_set_items(
    items: Sequence[_PreparedLocalReadyPrewarm],
) -> list[_PreparedLocalReadyPrewarm] | None:
    if len(items) <= 1:
        return None
    indexed: dict[int, _PreparedLocalReadyPrewarm] = {}
    member_count = 0
    for item in items:
        member = item.intent.target.member
        current_count = int(getattr(member, "member_count", 0) or 0)
        if current_count <= 1:
            return None
        if member_count == 0:
            member_count = current_count
        elif current_count != member_count:
            return None
        member_index = int(getattr(member, "member_index", -1))
        if member_index < 0 or member_index >= member_count:
            return None
        if member_index in indexed:
            return None
        indexed[member_index] = item
    if len(indexed) != member_count:
        return None
    return [indexed[index] for index in range(member_count)]


def _prefetch_groups(
    items: Sequence[_PreparedLocalReadyPrewarm],
) -> list[list[_PreparedLocalReadyPrewarm]]:
    keyed: dict[str, list[_PreparedLocalReadyPrewarm]] = {}
    singles: list[_PreparedLocalReadyPrewarm] = []
    for item in items:
        key = _target_set_group_key(item.intent)
        if key is None:
            singles.append(item)
            continue
        keyed.setdefault(key, []).append(item)

    groups: list[list[_PreparedLocalReadyPrewarm]] = []
    groups.extend([single] for single in singles)
    for candidates in keyed.values():
        complete = _complete_target_set_items(candidates)
        if complete is None:
            groups.extend([candidate] for candidate in candidates)
        else:
            groups.append(complete)
    return groups


def decode_local_ready_target_plan_record(
    record: dict[str, Any],
) -> LocalReadyTargetPlanIntent:
    if int(record.get("schema_version") or 0) != 1:
        raise ValueError("local-ready target-plan record schema_version must be 1")
    raw_source_path = str(record.get("source_path") or "")
    if not raw_source_path:
        raise ValueError("source_path is required")
    source_path = _canonical_source_path(raw_source_path)
    intent_key = str(record.get("intent_key") or "")
    if not intent_key:
        raise ValueError("intent_key is required")
    target = _decode_target(record)
    target_device = str(record.get("target_device") or target.device)
    source_artifact_ref = str(record.get("source_artifact_ref") or "") or None
    if source_artifact_ref is None:
        source_artifact_ref = target.source.source_artifact_ref
    return LocalReadyTargetPlanIntent(
        record=dict(record),
        source_path=source_path,
        intent_key=intent_key,
        source_artifact_ref=source_artifact_ref,
        target_device=target_device,
        target=target,
        materialization_options=_decode_options(record),
        retention=_decode_retention(record),
    )


def _rebound_intent_key(
    *,
    base_intent_key: str,
    source_artifact_ref: str,
    source_selection_digest: str,
    target_proto_sha256: str,
) -> str:
    return _stable_hash(
        {
            "schema_version": 1,
            "base_intent_key": base_intent_key,
            "source_artifact_ref": source_artifact_ref,
            "source_selection_digest": source_selection_digest,
            "target_proto_sha256": target_proto_sha256,
        }
    )


def _target_with_source(
    target: RealizationTarget,
    *,
    source: RuntimeBindingSourceRef,
) -> RealizationTarget:
    resolved_layout = target.resolved_layout.model_copy(update={"source": source})
    return target.model_copy(
        update={
            "source": source,
            "resolved_layout": resolved_layout,
        }
    )


def rebind_intent_to_artifact_selection(
    intent: LocalReadyTargetPlanIntent,
    artifact: Any,
) -> LocalReadyTargetPlanIntent:
    resolve = getattr(artifact, "_resolve_realization_selection", None)
    if not callable(resolve):
        raise ValueError(
            "source_path prewarm requires an artifact capable of resolving "
            "realization selection"
        )
    selection = resolve()
    source_artifact_ref = str(getattr(selection, "artifact_id", "") or "")
    source_selection_digest = str(
        getattr(selection, "source_selection_digest", "") or ""
    )
    if not source_artifact_ref or not source_selection_digest:
        raise ValueError(
            "source_path prewarm could not resolve source artifact identity"
        )
    source = intent.target.source.model_copy(
        update={
            "source_artifact_ref": source_artifact_ref,
            "artifact_selection_digest": source_selection_digest,
        }
    )
    target = _target_with_source(intent.target, source=source)
    record = dict(intent.record)
    record.update(
        {
            "source_artifact_ref": source_artifact_ref,
            "source_selection_digest": source_selection_digest,
            "source_rebind_mode": "source_path",
            "source_rebound_from_artifact_ref": intent.source_artifact_ref,
        }
    )
    record.update(_encode_target_proto_fields(target))
    intent_key = _rebound_intent_key(
        base_intent_key=intent.intent_key,
        source_artifact_ref=source_artifact_ref,
        source_selection_digest=source_selection_digest,
        target_proto_sha256=str(record["target_proto_sha256"]),
    )
    record["intent_key"] = intent_key
    return LocalReadyTargetPlanIntent(
        record=record,
        source_path=intent.source_path,
        intent_key=intent_key,
        source_artifact_ref=source_artifact_ref,
        target_device=intent.target_device,
        target=target,
        materialization_options=intent.materialization_options,
        retention=intent.retention,
    )


def _records_from_manifest_payload(
    payload: Any,
    *,
    source_path: str | None,
) -> list[dict[str, Any]]:
    candidate_records: list[Any]
    if isinstance(payload, list):
        candidate_records = payload
    elif isinstance(payload, dict) and payload.get("target_proto_b64"):
        candidate_records = [payload]
    elif isinstance(payload, dict):
        grouped = payload.get("records_by_source_path")
        if isinstance(grouped, dict):
            if source_path is None:
                candidate_records = [
                    record
                    for records in grouped.values()
                    if isinstance(records, list)
                    for record in records
                ]
            else:
                canonical = _canonical_source_path(source_path)
                candidates = (
                    grouped.get(canonical)
                    or grouped.get(source_path)
                    or grouped.get(str(Path(source_path).expanduser()))
                )
                candidate_records = candidates if isinstance(candidates, list) else []
        else:
            records = payload.get("records")
            candidate_records = records if isinstance(records, list) else []
    else:
        candidate_records = []
    return [
        dict(record)
        for record in candidate_records
        if isinstance(record, dict) and record.get("target_proto_b64")
    ]


def iter_local_ready_target_plan_intents(
    manifest_paths: str
    | os.PathLike[str]
    | Sequence[str | os.PathLike[str]]
    | None = None,
    *,
    source_path: str | None = None,
    manifest_payloads: Sequence[Any] = (),
    manifest_sha256s: Sequence[str] = (),
) -> Iterable[LocalReadyTargetPlanIntent]:
    paths: Sequence[str | os.PathLike[str]]
    if manifest_paths is None:
        paths = []
    elif isinstance(manifest_paths, (str, os.PathLike)):
        paths = [manifest_paths]
    else:
        paths = manifest_paths
    expected_sha256s = [str(value).strip() for value in manifest_sha256s if value]
    if expected_sha256s and len(expected_sha256s) not in {1, len(paths)}:
        raise ValueError(
            "target-plan manifest sha256 count must be 1 or match manifest "
            f"path count: sha256s={len(expected_sha256s)} paths={len(paths)}"
        )
    for index, manifest_path in enumerate(paths):
        raw_manifest = Path(manifest_path).read_bytes()
        if expected_sha256s:
            expected_sha256 = (
                expected_sha256s[index]
                if len(expected_sha256s) > 1
                else expected_sha256s[0]
            )
            actual_sha256 = _sha256_bytes(raw_manifest)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "target-plan manifest sha256 mismatch: "
                    f"path={manifest_path} expected={expected_sha256} "
                    f"actual={actual_sha256}"
                )
        payload = json.loads(raw_manifest.decode("utf-8"))
        for record in _records_from_manifest_payload(
            payload,
            source_path=source_path,
        ):
            yield decode_local_ready_target_plan_record(record)
    for payload in manifest_payloads:
        for record in _records_from_manifest_payload(
            payload,
            source_path=source_path,
        ):
            yield decode_local_ready_target_plan_record(record)


def retained_manifest_record_from_target_plan_handoff(
    intent: LocalReadyTargetPlanIntent,
    handoff: Any,
    *,
    operation_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    local_serving_ref = getattr(handoff, "local_serving_ref", None)
    if not local_serving_ref:
        raise RuntimeError(
            "TensorCast retained prewarm did not return local_serving_ref"
        )
    target = intent.target
    record = intent.record
    expected_member = _json_model_dump(target.member)
    runtime_config_digest = str(record.get("runtime_config_digest") or "")
    runtime_config_digest_source = "target_plan_manifest"
    if record.get("source_rebind_mode") == "source_path":
        runtime_config_digest = ""
        runtime_config_digest_source = "unbound_source_path_rebind"
    retained_readiness = str(
        getattr(
            getattr(handoff, "readiness", "runtime_local_ready"),
            "value",
            getattr(handoff, "readiness", "runtime_local_ready"),
        )
        or "runtime_local_ready"
    )
    retained_claim_extra = retained_realization_claim_extra_from_handoff(
        handoff=handoff,
        target=target,
        expected_member=target.member,
    )
    operation_ref_record = dict(operation_ref or {})
    return {
        "schema_version": 1,
        "cache_key": retained_local_ready_cache_key_for_intent(intent),
        "retained_readiness": retained_readiness,
        "retained_realization_claim_extra": retained_claim_extra,
        "operation_id": operation_ref_record.get("operation_id"),
        "operation_ref": operation_ref_record or None,
        "local_serving_ref": str(local_serving_ref),
        "expected_member": expected_member,
        "expected_tensor_schema_hash": str(
            record.get("tensor_schema_hash")
            or target.resolved_layout.tensor_schema_hash
        ),
        "expected_serving_build_digest": str(
            record.get("serving_build_digest") or target.runtime_build_digest
        ),
        "expected_target_layout_hash": str(
            record.get("target_layout_hash")
            or target.resolved_layout.target_layout_hash
        ),
        "expected_daemon_id": str(getattr(handoff, "daemon_id", "") or ""),
        "expected_daemon_session_id": str(
            getattr(handoff, "daemon_session_id", "") or ""
        ),
        "serving_artifact_id": getattr(handoff, "serving_artifact_id", None),
        "device_uuid": record.get("device_uuid")
        or target.device_uuid
        or getattr(handoff, "device_uuid", None),
        "reservation_bytes": int(getattr(handoff, "reservation_bytes", 0) or 0),
        "expires_at_ms": getattr(handoff, "expires_at_ms", None),
        "cache_identity_version": 2,
        "source_path": _canonical_source_path(intent.source_path),
        "model_hash": str(record.get("model_hash") or ""),
        "target_device": intent.target_device,
        "framework_identity": dict(record.get("framework_identity") or {}),
        "runtime_config_digest": runtime_config_digest,
        "runtime_config_digest_source": runtime_config_digest_source,
        "source_selection_digest": str(
            record.get("source_selection_digest")
            or target.source.artifact_selection_digest
        ),
        "written_at_ms": int(time.time() * 1000),
    }


def write_retained_manifest_record(
    manifest_path: str | os.PathLike[str],
    *,
    source_path: str,
    record: dict[str, Any],
) -> None:
    cache_key = str(record.get("cache_key") or "")
    local_serving_ref = str(record.get("local_serving_ref") or "")
    if not cache_key or not local_serving_ref:
        raise ValueError(
            "retained manifest record requires cache_key and local_serving_ref"
        )
    record_identity = _retained_manifest_identity_key(record)

    import fcntl

    path = Path(manifest_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    source_key = _canonical_source_path(source_path)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        grouped = payload.get("records_by_source_path")
        if not isinstance(grouped, dict):
            grouped = {}
        existing_records = grouped.get(source_key)
        if not isinstance(existing_records, list):
            existing_records = []
        records = [
            dict(candidate)
            for candidate in existing_records
            if isinstance(candidate, dict)
            and candidate.get("cache_key") != cache_key
            and candidate.get("local_serving_ref") != local_serving_ref
            and _retained_manifest_identity_key(candidate) != record_identity
        ]
        records.append(dict(record))
        grouped[source_key] = records
        payload.update(
            {
                "schema_version": 1,
                "producer": "tensorcast.artifact_runtime.local_ready_prewarm",
                "updated_at_ms": int(time.time() * 1000),
                "records_by_source_path": grouped,
            }
        )
        tmp_path = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_latest_materializing_records(
    issued: Sequence[_IssuedLocalReadyPrewarm],
    *,
    retained_manifest_write: str | os.PathLike[str] | None,
    timeout_s: float | None,
) -> dict[str, dict[str, Any]]:
    if retained_manifest_write is None:
        return {}
    pending = list(issued)
    if not pending:
        return {}
    deadline = (
        time.monotonic() + max(0.0, float(timeout_s))
        if timeout_s is not None and timeout_s >= 0
        else None
    )
    retained_manifest_path = Path(retained_manifest_write).expanduser()
    records_by_key: dict[str, dict[str, Any]] = {}
    sleep_s = 0.02
    while pending:
        progressed = False
        for issued_item in list(pending):
            latest_result = getattr(issued_item.operation, "latest_result", None)
            if not callable(latest_result):
                pending.remove(issued_item)
                progressed = True
                continue
            try:
                handoff = _handoff_from_prefetch_result(
                    issued_item.intent,
                    latest_result(),
                )
            except Exception:  # noqa: BLE001
                handoff = None
            if handoff is None:
                continue
            operation_ref = (
                _operation_ref_record(issued_item.operation)
                or issued_item.operation_ref
            )
            retained_record = retained_manifest_record_from_target_plan_handoff(
                issued_item.intent,
                handoff,
                operation_ref=operation_ref,
            )
            write_retained_manifest_record(
                retained_manifest_path,
                source_path=issued_item.intent.source_path,
                record=retained_record,
            )
            key = _materializing_key(
                operation_id=issued_item.operation_id,
                intent=issued_item.intent,
            )
            records_by_key[key] = retained_record
            pending.remove(issued_item)
            progressed = True
        if not pending:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if not progressed:
            sleep_remaining = (
                sleep_s
                if deadline is None
                else min(sleep_s, max(0.0, deadline - time.monotonic()))
            )
            if sleep_remaining <= 0:
                break
            time.sleep(sleep_remaining)
            sleep_s = min(0.1, sleep_s * 1.5)
    return records_by_key


def _materializing_summary(
    *,
    issued: Sequence[_IssuedLocalReadyPrewarm],
    materializing_records: dict[str, dict[str, Any]],
    retained_manifest_write: str | os.PathLike[str] | None,
    elapsed_sec: float,
) -> LocalReadyMaterializingSummary:
    readiness_counts: dict[str, int] = {}
    source_paths: set[str] = set()
    for record in materializing_records.values():
        readiness = str(record.get("retained_readiness") or "unknown")
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        source_path = str(record.get("source_path") or "")
        if source_path:
            source_paths.add(_canonical_source_path(source_path))
    retained_manifest_path = (
        Path(retained_manifest_write).expanduser()
        if retained_manifest_write is not None
        else None
    )
    expected_records = len(issued)
    retained_records = len(materializing_records)
    return LocalReadyMaterializingSummary(
        expected_records=expected_records,
        retained_records=retained_records,
        ready=(expected_records > 0 and retained_records >= expected_records),
        readiness_counts=readiness_counts,
        retained_manifest_path=retained_manifest_path,
        source_paths=tuple(sorted(source_paths)),
        elapsed_sec=elapsed_sec,
    )


def write_materializing_ready_marker(
    path: str | os.PathLike[str],
    summary: LocalReadyMaterializingSummary,
) -> None:
    _atomic_write_json(Path(path).expanduser(), summary.to_dict())


def write_materializing_failure_marker(
    path: str | os.PathLike[str],
    *,
    error: BaseException,
    phase: str,
    elapsed_sec: float,
    retained_manifest_write: str | os.PathLike[str] | None = None,
    issued: Sequence[_IssuedLocalReadyPrewarm] = (),
    materializing_records: dict[str, dict[str, Any]] | None = None,
) -> None:
    summary = _materializing_summary(
        issued=issued,
        materializing_records=materializing_records or {},
        retained_manifest_write=retained_manifest_write,
        elapsed_sec=elapsed_sec,
    )
    payload = summary.to_dict()
    payload.update(
        {
            "event": "materializing_records_failed",
            "ready": False,
            "phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
        }
    )
    _atomic_write_json(Path(path).expanduser(), payload)


def _artifact_for_intent(
    intent: LocalReadyTargetPlanIntent,
    *,
    source_mode: SourceMode,
) -> Any:
    if source_mode == "source_path":
        return from_disk(intent.source_path)
    if source_mode != "artifact_ref":
        raise ValueError("source_mode must be artifact_ref or source_path")
    if intent.source_artifact_ref:
        return artifact_by_ref(ref=intent.source_artifact_ref)
    return from_disk(intent.source_path)


def _prewarm_retention_ttl_ms_from_env() -> int | None:
    raw = os.environ.get(PREWARM_RETENTION_TTL_MS_ENV)
    if raw is None or not raw.strip():
        return None
    ttl_ms = int(raw)
    if ttl_ms <= 0:
        raise ValueError(f"{PREWARM_RETENTION_TTL_MS_ENV} must be positive")
    return ttl_ms


def prewarm_local_ready_target_plan_manifest(
    manifest_paths: str
    | os.PathLike[str]
    | Sequence[str | os.PathLike[str]]
    | None = None,
    *,
    manifest_payloads: Sequence[Any] = (),
    manifest_sha256s: Sequence[str] = (),
    retained_manifest_write: str | os.PathLike[str] | None = None,
    source_path: str | None = None,
    source_mode: SourceMode = "artifact_ref",
    wait: bool = True,
    timeout_s: float | None = None,
    write_materializing_records: bool = False,
    materializing_record_timeout_s: float | None = 5.0,
    materializing_ready_write: str | os.PathLike[str] | None = None,
    retention_ttl_ms: int | None = None,
    artifact_factory: Callable[[LocalReadyTargetPlanIntent], Any] | None = None,
) -> list[LocalReadyPrewarmResult]:
    started_at = time.perf_counter()
    results: list[LocalReadyPrewarmResult] = []
    issued: list[_IssuedLocalReadyPrewarm] = []
    materializing_records: dict[str, dict[str, Any]] = {}
    marker_ready_written = False
    if materializing_ready_write is not None:
        write_materializing_records = True
    try:
        prepared_items: list[_PreparedLocalReadyPrewarm] = []
        for intent in iter_local_ready_target_plan_intents(
            manifest_paths,
            source_path=source_path,
            manifest_payloads=manifest_payloads,
            manifest_sha256s=manifest_sha256s,
        ):
            intent = _intent_with_retention_ttl_ms(intent, retention_ttl_ms)
            artifact = (
                artifact_factory(intent)
                if artifact_factory is not None
                else _artifact_for_intent(intent, source_mode=source_mode)
            )
            if source_mode == "source_path":
                intent = rebind_intent_to_artifact_selection(intent, artifact)
                intent = _intent_with_retention_ttl_ms(intent, retention_ttl_ms)
            prepared_items.append(
                _PreparedLocalReadyPrewarm(intent=intent, artifact=artifact)
            )

        for group_items in _prefetch_groups(prepared_items):
            first = group_items[0]
            intent = first.intent
            artifact = first.artifact
            target: RealizationTarget | RealizationTargetSet
            options = intent.materialization_options
            retention = intent.retention
            if len(group_items) == 1:
                target = intent.target
                context_key = intent.intent_key
                context_prefix = "local-ready-prewarm"
            else:
                group_id = str(intent.target.member.group_id or "")
                target = RealizationTargetSet(
                    runtime=intent.target.runtime,
                    source=intent.target.source,
                    topology=intent.target.topology,
                    group_id=group_id,
                    members=tuple(item.intent.target for item in group_items),
                )
                context_key = _stable_hash(
                    {
                        "schema_version": 1,
                        "intent_kind": "local_ready_target_set",
                        "group_id": group_id,
                        "intent_keys": [item.intent.intent_key for item in group_items],
                    }
                )
                context_prefix = "local-ready-prewarm-set"
            ctx = CallContext(
                request_id=f"{context_prefix}:{context_key}",
                idempotency_key=f"{context_prefix}:{context_key}",
                deadline_ms=(int(timeout_s * 1000) if timeout_s is not None else None),
            )
            operation = artifact.prefetch(
                target=target,
                readiness="runtime_local_ready",
                retention=retention,
                options=options,
                ctx=ctx,
            )
            operation_id = str(getattr(operation, "operation_id", "") or "") or None
            operation_ref = _operation_ref_record(operation)
            for group_item in group_items:
                item_intent = group_item.intent
                if not wait:
                    results.append(
                        LocalReadyPrewarmResult(
                            intent=item_intent,
                            operation_id=operation_id,
                            operation_ref=operation_ref,
                            handoff=None,
                            retained_record=None,
                            retained_manifest_path=None,
                        )
                    )
                issued.append(
                    _IssuedLocalReadyPrewarm(
                        intent=item_intent,
                        operation_id=operation_id,
                        operation_ref=operation_ref,
                        operation=operation,
                    )
                )
        if write_materializing_records:
            materializing_start = time.perf_counter()
            materializing_records = _write_latest_materializing_records(
                issued,
                retained_manifest_write=retained_manifest_write,
                timeout_s=materializing_record_timeout_s,
            )
            if materializing_ready_write is not None:
                materializing_summary = _materializing_summary(
                    issued=issued,
                    materializing_records=materializing_records,
                    retained_manifest_write=retained_manifest_write,
                    elapsed_sec=time.perf_counter() - materializing_start,
                )
                write_materializing_ready_marker(
                    materializing_ready_write,
                    materializing_summary,
                )
                marker_ready_written = materializing_summary.ready
                if not materializing_summary.ready:
                    raise RuntimeError(
                        "materializing retained records were not ready: "
                        f"expected={materializing_summary.expected_records} "
                        f"retained={materializing_summary.retained_records} "
                        f"readiness={materializing_summary.readiness_counts}"
                    )
            if not wait:
                for index, result in enumerate(results):
                    key = _materializing_key(
                        operation_id=result.operation_id,
                        intent=result.intent,
                    )
                    retained_record = materializing_records.get(key)
                    if retained_record is not None:
                        retained_manifest_path = (
                            Path(retained_manifest_write).expanduser()
                            if retained_manifest_write is not None
                            else None
                        )
                        results[index] = LocalReadyPrewarmResult(
                            intent=result.intent,
                            operation_id=result.operation_id,
                            operation_ref=result.operation_ref,
                            handoff=result.handoff,
                            retained_record=retained_record,
                            retained_manifest_path=retained_manifest_path,
                        )
        if not wait:
            return results
        for issued_item in issued:
            intent = issued_item.intent
            operation_id = issued_item.operation_id
            operation_ref = (
                _operation_ref_record(issued_item.operation)
                or issued_item.operation_ref
            )
            operation = issued_item.operation
            handoff = _handoff_from_prefetch_result(
                intent,
                operation.result(timeout_s=timeout_s),
            )
            retained_record = retained_manifest_record_from_target_plan_handoff(
                intent,
                handoff,
                operation_ref=operation_ref,
            )
            retained_manifest_path = None
            if retained_manifest_write is not None:
                retained_manifest_path = Path(retained_manifest_write).expanduser()
                write_retained_manifest_record(
                    retained_manifest_path,
                    source_path=intent.source_path,
                    record=retained_record,
                )
            results.append(
                LocalReadyPrewarmResult(
                    intent=intent,
                    operation_id=operation_id,
                    operation_ref=operation_ref,
                    handoff=handoff,
                    retained_record=retained_record,
                    retained_manifest_path=retained_manifest_path,
                )
            )
        return results
    except Exception as exc:
        if materializing_ready_write is not None and not marker_ready_written:
            with suppress(Exception):
                write_materializing_failure_marker(
                    materializing_ready_write,
                    error=exc,
                    phase="materializing",
                    elapsed_sec=time.perf_counter() - started_at,
                    retained_manifest_write=retained_manifest_write,
                    issued=issued,
                    materializing_records=materializing_records,
                )
        raise


def _manifest_paths_from_env() -> list[str]:
    raw = (
        os.environ.get(TARGET_PLAN_MANIFEST_ENV)
        or os.environ.get(TARGET_PLAN_MANIFEST_WRITE_ENV)
        or ""
    )
    paths = _split_env_values(raw)
    if paths:
        return paths
    cache_dir = os.environ.get(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV)
    if not cache_dir:
        return []
    sha256s = _manifest_sha256s_from_env()
    return [
        str(target_plan_manifest_cache_path(cache_dir, digest)) for digest in sha256s
    ]


def _manifest_payloads_from_env() -> list[Any]:
    payloads: list[Any] = []
    for env_name, encoding in (
        (TARGET_PLAN_MANIFEST_JSON_ENV, None),
        (TARGET_PLAN_MANIFEST_B64_ENV, "base64"),
    ):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            text = raw
            if encoding == "base64":
                text = base64.b64decode(raw, validate=True).decode("utf-8")
            payloads.append(json.loads(text))
        except Exception as exc:
            raise ValueError(
                f"{env_name} must contain a valid target-plan JSON manifest"
            ) from exc
    return payloads


def _manifest_sha256s_from_env() -> list[str]:
    explicit = [
        _normalize_sha256_digest(entry)
        for entry in _split_env_values(os.environ.get(TARGET_PLAN_MANIFEST_SHA256_ENV))
    ]
    if explicit:
        return explicit
    cache_dir = os.environ.get(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV)
    source_path = os.environ.get(SOURCE_PATH_FILTER_ENV)
    if not cache_dir or not source_path:
        return []
    return [_source_index_sha256_from_cache(cache_dir, source_path)]


def prewarm_local_ready_target_plans_from_env(
    *,
    artifact_factory: Callable[[LocalReadyTargetPlanIntent], Any] | None = None,
) -> list[LocalReadyPrewarmResult]:
    initialize_runtime_from_env()
    manifest_paths = _manifest_paths_from_env()
    manifest_payloads = _manifest_payloads_from_env()
    manifest_sha256s = _manifest_sha256s_from_env()
    if not manifest_paths and not manifest_payloads:
        return []
    timeout_s_raw = os.environ.get(PREWARM_TIMEOUT_S_ENV)
    timeout_s = float(timeout_s_raw) if timeout_s_raw else None
    source_mode = os.environ.get(PREWARM_SOURCE_MODE_ENV, "artifact_ref")
    if source_mode not in {"artifact_ref", "source_path"}:
        raise ValueError(
            f"{PREWARM_SOURCE_MODE_ENV} must be artifact_ref or source_path"
        )
    materializing_timeout_s_raw = os.environ.get(MATERIALIZING_RECORD_TIMEOUT_S_ENV)
    materializing_timeout_s = (
        float(materializing_timeout_s_raw) if materializing_timeout_s_raw else 5.0
    )
    return prewarm_local_ready_target_plan_manifest(
        manifest_paths,
        manifest_payloads=manifest_payloads,
        manifest_sha256s=manifest_sha256s,
        retained_manifest_write=os.environ.get(RETAINED_MANIFEST_WRITE_ENV),
        source_path=os.environ.get(SOURCE_PATH_FILTER_ENV),
        source_mode=source_mode,
        wait=True,
        timeout_s=timeout_s,
        write_materializing_records=_truthy_env(
            os.environ.get(WRITE_MATERIALIZING_RECORDS_ENV)
        )
        or bool(os.environ.get(MATERIALIZING_READY_WRITE_ENV)),
        materializing_record_timeout_s=materializing_timeout_s,
        materializing_ready_write=os.environ.get(MATERIALIZING_READY_WRITE_ENV),
        retention_ttl_ms=_prewarm_retention_ttl_ms_from_env(),
        artifact_factory=artifact_factory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prewarm TensorCast local-ready target-plan intents."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Target-plan manifest path. Defaults to environment manifests.",
    )
    parser.add_argument(
        "--retained-manifest-write",
        default=os.environ.get(RETAINED_MANIFEST_WRITE_ENV),
        help="Retained binding manifest output path.",
    )
    parser.add_argument(
        "--source-path",
        default=os.environ.get(SOURCE_PATH_FILTER_ENV),
        help="Optional source_path filter.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("artifact_ref", "source_path"),
        default=os.environ.get(PREWARM_SOURCE_MODE_ENV, "artifact_ref"),
    )
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument(
        "--write-materializing-records",
        action="store_true",
        default=_truthy_env(os.environ.get(WRITE_MATERIALIZING_RECORDS_ENV)),
        help="Write retained runtime_reserved records before terminal wait.",
    )
    parser.add_argument(
        "--materializing-record-timeout-s",
        type=float,
        default=(
            float(os.environ[MATERIALIZING_RECORD_TIMEOUT_S_ENV])
            if os.environ.get(MATERIALIZING_RECORD_TIMEOUT_S_ENV)
            else 5.0
        ),
        help="Maximum time to wait for early materializing handoffs.",
    )
    parser.add_argument(
        "--materializing-ready-write",
        default=os.environ.get(MATERIALIZING_READY_WRITE_ENV),
        help=(
            "Optional JSON marker path written after materializing retained "
            "records are published and before terminal local-ready wait."
        ),
    )
    parser.add_argument(
        "--retention-ttl-ms",
        type=int,
        default=_prewarm_retention_ttl_ms_from_env(),
        help=(
            "Override retained binding TTL for prewarm materialization. "
            f"Defaults to {PREWARM_RETENTION_TTL_MS_ENV} or the manifest."
        ),
    )
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args(argv)
    initialize_runtime_from_env()

    try:
        manifest_paths = args.manifest or _manifest_paths_from_env()
        manifest_payloads = _manifest_payloads_from_env()
    except ValueError as error:
        if args.materializing_ready_write:
            write_materializing_failure_marker(
                args.materializing_ready_write,
                error=error,
                phase="argument_validation",
                elapsed_sec=0.0,
                retained_manifest_write=args.retained_manifest_write,
            )
        raise SystemExit(str(error)) from error
    if not manifest_paths and not manifest_payloads:
        error = ValueError(
            "no target-plan manifest supplied; set "
            f"{TARGET_PLAN_MANIFEST_ENV}, {TARGET_PLAN_MANIFEST_JSON_ENV}, "
            f"{TARGET_PLAN_MANIFEST_B64_ENV}, or "
            f"{TARGET_PLAN_MANIFEST_CACHE_DIR_ENV} with "
            f"{TARGET_PLAN_MANIFEST_SHA256_ENV}"
        )
        if args.materializing_ready_write:
            write_materializing_failure_marker(
                args.materializing_ready_write,
                error=error,
                phase="argument_validation",
                elapsed_sec=0.0,
                retained_manifest_write=args.retained_manifest_write,
            )
        raise SystemExit(str(error))
    results = prewarm_local_ready_target_plan_manifest(
        manifest_paths,
        manifest_payloads=manifest_payloads,
        manifest_sha256s=_manifest_sha256s_from_env(),
        retained_manifest_write=args.retained_manifest_write,
        source_path=args.source_path,
        source_mode=args.source_mode,
        wait=not args.no_wait,
        timeout_s=args.timeout_s,
        write_materializing_records=(
            args.write_materializing_records or bool(args.materializing_ready_write)
        ),
        materializing_record_timeout_s=args.materializing_record_timeout_s,
        materializing_ready_write=args.materializing_ready_write,
        retention_ttl_ms=args.retention_ttl_ms,
    )
    print(
        json.dumps(
            {
                "prewarm_intents": len(results),
                "retained_records": sum(
                    1 for result in results if result.retained_record is not None
                ),
                "retained_manifest_write": args.retained_manifest_write,
                "write_materializing_records": (
                    args.write_materializing_records
                    or bool(args.materializing_ready_write)
                ),
                "materializing_ready_write": args.materializing_ready_write,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "LocalReadyMaterializingSummary",
    "LocalReadyPrewarmResult",
    "LocalReadyTargetPlanIntent",
    "decode_local_ready_target_plan_record",
    "DAEMON_ADDRESS_ENV",
    "initialize_runtime_from_env",
    "iter_local_ready_target_plan_intents",
    "operation_ref_from_retained_manifest_record",
    "prewarm_local_ready_target_plan_manifest",
    "prewarm_local_ready_target_plans_from_env",
    "rebind_intent_to_artifact_selection",
    "retained_local_ready_cache_key_for_intent",
    "retained_manifest_record_from_target_plan_handoff",
    "target_plan_manifest_cache_path",
    "target_plan_manifest_cache_source_index_path",
    "update_target_plan_manifest_cache_record",
    "write_retained_manifest_record",
    "wait_retained_manifest_record_operation_handoff",
    "write_materializing_failure_marker",
    "write_materializing_ready_marker",
    "write_target_plan_manifest_cache",
    "WRITE_MATERIALIZING_RECORDS_ENV",
    "MATERIALIZING_READY_WRITE_ENV",
    "PREWARM_RETENTION_TTL_MS_ENV",
    "TARGET_PLAN_MANIFEST_B64_ENV",
    "TARGET_PLAN_MANIFEST_CACHE_DIR_ENV",
    "TARGET_PLAN_MANIFEST_CACHE_WRITE_DIR_ENV",
    "TARGET_PLAN_MANIFEST_JSON_ENV",
    "TARGET_PLAN_MANIFEST_SHA256_ENV",
]

if __name__ == "__main__":
    raise SystemExit(main())
