#  Copyright (c) 2026, TensorCast Team.

"""Group version-set realization service."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from typing import Any

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.config.settings import GroupRealizationConfig
from tensorcast.global_store.exceptions import (
    ConflictError,
    DatabaseError,
    NotFoundError,
    ValidationError,
)
from tensorcast.global_store.repositories.group_realization_repository import (
    GroupRealizationConflictError,
    GroupRealizationRepository,
)
from tensorcast.global_store.repositories.group_version_set_repository import (
    GroupVersionSetRepository,
)
from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class GroupRealizationFeatureDisabledError(ValidationError):
    """Raised when group realization is disabled by configuration."""


class GroupRealizationPreconditionError(ValidationError):
    """Raised when a state transition is not currently admissible."""


def _is_group_realization_conflict(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, GroupRealizationConflictError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _stable_hash(payload: dict[str, Any], domain: bytes) -> bytes:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + encoded).digest()


def _deterministic_proto_bytes(message: Any) -> bytes:
    return message.SerializeToString(deterministic=True)


def _kind_proto_to_text(kind: int) -> str:
    if kind == global_store_pb2.GROUP_REALIZATION_KIND_PER_PART_SELECTION:
        return "per_part_selection"
    if kind == global_store_pb2.GROUP_REALIZATION_KIND_SAME_SELECTION:
        return "same_selection"
    raise ValidationError("realization_kind is required")


def _required_part_ids(
    context: global_store_pb2.GroupRealizationContext,
) -> list[str]:
    required = sorted({str(part_id).strip() for part_id in context.required_part_ids})
    if not required:
        raise ValidationError("required_part_ids must be provided")
    if "" in required:
        raise ValidationError("required_part_ids must not contain empty part ids")
    if int(context.total_parts) != len(required):
        raise ValidationError("total_parts must match required_part_ids")
    return required


def _effective_same_selection_byte_space(
    part: global_store_pb2.GroupVersionSetPart,
) -> common_pb2.ByteSpaceRef:
    if part.HasField("requested_byte_space") and (
        part.requested_byte_space.kind != common_pb2.BYTE_SPACE_KIND_UNSPECIFIED
        or bool(part.requested_byte_space.id)
    ):
        return part.requested_byte_space
    if part.selection.view_id:
        return common_pb2.ByteSpaceRef(
            kind=common_pb2.BYTE_SPACE_KIND_VIEW,
            id=part.selection.view_id,
        )
    return common_pb2.ByteSpaceRef(kind=common_pb2.BYTE_SPACE_KIND_CANONICAL)


def _validate_same_selection_parts(
    parts: list[global_store_pb2.GroupVersionSetPart],
) -> common_pb2.ByteSpaceRef:
    if not parts:
        raise ValidationError("same_selection requires at least one part")
    first = parts[0]
    first_selection = _deterministic_proto_bytes(first.selection)
    first_byte_space = _effective_same_selection_byte_space(first)
    first_byte_space_bytes = _deterministic_proto_bytes(first_byte_space)
    for part in parts[1:]:
        if _deterministic_proto_bytes(part.selection) != first_selection:
            raise ValidationError("same_selection requires identical selections")
        if (
            _deterministic_proto_bytes(_effective_same_selection_byte_space(part))
            != first_byte_space_bytes
        ):
            raise ValidationError("same_selection requires identical byte spaces")
    return first_byte_space


class GroupRealizationService:
    """Semantic transaction orchestration over version-set manifests."""

    def __init__(
        self,
        *,
        version_set_repository: GroupVersionSetRepository,
        realization_repository: GroupRealizationRepository,
        key_mapping_repository: KeyMappingRepository,
        config: GroupRealizationConfig,
    ) -> None:
        self._version_sets = version_set_repository
        self._realizations = realization_repository
        self._key_mappings = key_mapping_repository
        self._config = config
        self._waiter_lock = threading.Lock()
        self._waiters_by_transaction: dict[str, int] = {}
        self._terminal_cache: dict[str, dict[str, Any]] = {}
        self._last_expiration_scan_ns = 0

    def _ensure_enabled(self) -> None:
        if not self._config.enabled:
            raise GroupRealizationFeatureDisabledError("group realization is disabled")

    def _observe(
        self,
        *,
        operation: str,
        started_at: float,
        result: str,
    ) -> None:
        gs_metrics.inc_group_realization_event(operation=operation, result=result)
        gs_metrics.observe_group_realization_latency(
            operation=operation,
            duration_seconds=time.monotonic() - started_at,
        )

    def _scan_expired(self, *, force: bool = False) -> int:
        now_ns = time.time_ns()
        interval_ns = int(self._config.expiration_scan_interval_ms) * 1_000_000
        if (
            not force
            and interval_ns > 0
            and now_ns - self._last_expiration_scan_ns < interval_ns
        ):
            return 0
        self._last_expiration_scan_ns = now_ns
        expired = self._realizations.expire_due(
            now_unix_nanos=now_ns,
            batch_limit=self._config.cleanup_batch_limit,
        )
        gs_metrics.inc_group_realization_cleanup(
            operation="expire_due",
            result="expired" if expired else "empty",
            count=max(1, expired),
        )
        if expired:
            gs_metrics.inc_group_realization_control_write(
                operation="expire_due",
                table="group_realization_transactions",
                count=expired,
            )
            gs_metrics.inc_group_realization_control_write(
                operation="expire_due",
                table="group_realization_members",
                count=expired,
            )
        return expired

    def _transaction_deadline_unix_nanos(self, requested: int) -> int | None:
        now_ns = time.time_ns()
        ttl_ms = max(0, int(self._config.transaction_ttl_ms))
        ttl_deadline = now_ns + ttl_ms * 1_000_000 if ttl_ms > 0 else None
        if requested > 0:
            if ttl_deadline is None:
                return int(requested)
            return min(int(requested), ttl_deadline)
        default_ms = max(
            0,
            int(self._config.default_deadline_ms or self._config.transaction_ttl_ms),
        )
        if default_ms <= 0:
            return ttl_deadline
        default_deadline = now_ns + default_ms * 1_000_000
        if ttl_deadline is None:
            return default_deadline
        return min(default_deadline, ttl_deadline)

    def _remember_terminal(self, transaction: dict[str, Any]) -> None:
        if transaction["state"] not in {"published", "aborted", "expired"}:
            return
        with self._waiter_lock:
            self._terminal_cache[transaction["transaction_id"]] = dict(transaction)
            max_entries = max(1, int(self._config.max_active_transactions))
            while len(self._terminal_cache) > max_entries:
                oldest = next(iter(self._terminal_cache))
                self._terminal_cache.pop(oldest, None)

    def _get_terminal_cached(self, *, transaction_id: str) -> dict[str, Any] | None:
        with self._waiter_lock:
            cached = self._terminal_cache.get(transaction_id)
            return None if cached is None else dict(cached)

    def _enter_waiter(self, *, transaction_id: str) -> None:
        with self._waiter_lock:
            active = int(self._waiters_by_transaction.get(transaction_id, 0))
            if active >= self._config.max_waiters_per_transaction:
                raise GroupRealizationPreconditionError(
                    "too many waiters for group realization transaction"
                )
            self._waiters_by_transaction[transaction_id] = active + 1
            total = sum(self._waiters_by_transaction.values())
        gs_metrics.set_group_realization_active_waiters(count=total)

    def _leave_waiter(self, *, transaction_id: str) -> None:
        with self._waiter_lock:
            active = int(self._waiters_by_transaction.get(transaction_id, 0))
            if active <= 1:
                self._waiters_by_transaction.pop(transaction_id, None)
            else:
                self._waiters_by_transaction[transaction_id] = active - 1
            total = sum(self._waiters_by_transaction.values())
        gs_metrics.set_group_realization_active_waiters(count=total)

    def register_version_set(
        self,
        *,
        realization_kind: int,
        parts: list[global_store_pb2.GroupVersionSetPart],
        namespace: str | None = None,
        key: str | None = None,
        key_generation: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        started_at = time.monotonic()
        result = "ok"
        self._ensure_enabled()
        try:
            kind = _kind_proto_to_text(realization_kind)
            if len(parts) > self._config.max_parts_per_version_set:
                raise ValidationError("version set has too many parts")
            if kind == "same_selection":
                first = parts[0]
                byte_space = _validate_same_selection_parts(parts)
                required = [part.part_id for part in parts]
                return self._version_sets.create_or_get_same_selection(
                    selection=first.selection,
                    requested_byte_space=byte_space,
                    required_part_ids=required,
                    namespace=namespace or None,
                    key=key or None,
                    key_generation=key_generation,
                )
            return self._version_sets.register_per_part_selection(
                parts=[
                    {
                        "part_id": part.part_id,
                        "selection": part.selection,
                        "requested_byte_space": part.requested_byte_space,
                        "part_metadata_json": part.part_metadata_json or None,
                    }
                    for part in parts
                ],
                namespace=namespace or None,
                key=key or None,
                key_generation=key_generation,
            )
        except Exception:
            result = "error"
            raise
        finally:
            self._observe(
                operation="register_version_set",
                started_at=started_at,
                result=result,
            )

    def _resolve_key_target(
        self,
        *,
        key_reference: global_store_pb2.KeyVersionReference,
        required_part_ids: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int | None]:
        alias = (key_reference.alias or "current").strip().lower()
        if alias not in {"", "current", "latest"}:
            raise ValidationError("only current/latest key aliases are supported")
        namespace = key_reference.namespace.strip()
        key = key_reference.key.strip()
        if not key:
            raise ValidationError("key_reference.key is required")
        target = self._key_mappings.get_current_target(key=key, namespace=namespace)
        if target is None:
            raise NotFoundError("key target not found")
        generation = int(target["generation"])
        if (
            key_reference.HasField("expected_generation")
            and int(key_reference.expected_generation) != generation
        ):
            raise GroupRealizationPreconditionError("key generation mismatch")
        target_kind = str(target["target_kind"])
        if target_kind == "group_version_set":
            version_set_id = str(target["group_version_set_id"] or "")
            resolved = self._version_sets.get(version_set_id=version_set_id)
            if resolved is None:
                raise NotFoundError("group version set target not found")
            return resolved[0], resolved[1], generation
        if target_kind != "artifact_selection":
            raise ValidationError(f"unsupported key target kind: {target_kind}")
        selection = common_pb2.ArtifactSelection(
            artifact_id=str(target["artifact_id"] or ""),
            view_id=str(target["view_id"] or ""),
        )
        if target.get("selection_hash"):
            selection.selection_hash = bytes(target["selection_hash"])
        version_set, parts = self._version_sets.create_or_get_same_selection(
            selection=selection,
            required_part_ids=required_part_ids,
            namespace=namespace or None,
            key=key,
            key_generation=generation,
        )
        if not target.get("selection_hash") and parts:
            self._key_mappings.persist_artifact_selection_hash(
                key=key,
                namespace=namespace,
                generation=generation,
                selection_hash=bytes(parts[0]["selection_hash"]),
            )
        return version_set, parts, generation

    def _resolve_version(
        self,
        *,
        version: global_store_pb2.VersionReference,
        required_part_ids: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int | None]:
        value = version.WhichOneof("value")
        if value == "explicit_selection":
            version_set, parts = self._version_sets.create_or_get_same_selection(
                selection=version.explicit_selection,
                required_part_ids=required_part_ids,
            )
            return version_set, parts, None
        if value == "explicit_version_set":
            ref = version.explicit_version_set
            resolved = self._version_sets.get(version_set_id=ref.version_set_id)
            if resolved is None:
                raise NotFoundError("group version set not found")
            version_set, parts = resolved
            if ref.manifest_hash and bytes(ref.manifest_hash) != bytes(
                version_set["manifest_hash"]
            ):
                raise ConflictError("group version set manifest hash mismatch")
            return version_set, parts, None
        if value == "key_reference":
            return self._resolve_key_target(
                key_reference=version.key_reference,
                required_part_ids=required_part_ids,
            )
        raise ValidationError("version reference is required")

    def _transaction_fingerprint(
        self,
        *,
        context: global_store_pb2.GroupRealizationContext,
        required_part_ids: list[str],
        version_set: dict[str, Any],
        key_generation: int | None,
    ) -> bytes:
        return _stable_hash(
            {
                "group_kind": context.group_kind,
                "group_id": context.group_id,
                "epoch": int(context.epoch),
                "total_parts": int(context.total_parts),
                "required_part_ids": required_part_ids,
                "version_set_id": version_set["version_set_id"],
                "manifest_hash": bytes(version_set["manifest_hash"]).hex(),
                "realization_kind": version_set["realization_kind"],
                "key_generation": key_generation,
            },
            b"tensorcast.group_realization_transaction.v1",
        )

    def begin_or_join(
        self,
        *,
        request: global_store_pb2.BeginOrJoinGroupRealizationRequest,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        self._ensure_enabled()
        self._scan_expired(force=True)
        context = request.context
        if not context.group_kind.strip() or not context.group_id.strip():
            raise ValidationError("group_kind and group_id are required")
        if not context.part_id.strip():
            raise ValidationError("part_id is required")
        required_part_ids = _required_part_ids(context)
        if len(required_part_ids) > self._config.max_total_parts:
            raise ValidationError("group has too many required parts")
        if context.part_id not in required_part_ids:
            raise ValidationError("part_id is not in required_part_ids")
        existing = self._realizations.get_by_slot(
            group_kind=context.group_kind.strip(),
            group_id=context.group_id.strip(),
            epoch=int(context.epoch),
        )
        if existing is not None:
            if existing["state"] in {"aborted", "expired"}:
                self._observe(
                    operation="begin_or_join",
                    started_at=started_at,
                    result="terminal_conflict",
                )
                raise GroupRealizationPreconditionError(
                    "semantic slot already has a terminal transaction"
                )
            if required_part_ids != list(existing["required_part_ids"]) or int(
                context.total_parts
            ) != int(existing["total_parts"]):
                raise ConflictError("semantic slot already froze another part set")
            if request.transaction_fingerprint and bytes(
                request.transaction_fingerprint
            ) != bytes(existing["transaction_fingerprint"]):
                raise ConflictError("transaction_fingerprint does not match request")
            resolved = self._version_sets.get(version_set_id=existing["version_set_id"])
            if resolved is None:
                raise NotFoundError("frozen group version set not found")
            version_set, parts = resolved
            value = request.version.WhichOneof("value")
            if value == "explicit_version_set":
                requested_id = request.version.explicit_version_set.version_set_id
                if requested_id and requested_id != version_set["version_set_id"]:
                    raise ConflictError(
                        "semantic slot already froze another version set"
                    )
            elif value == "explicit_selection":
                requested_version_set, _ = (
                    self._version_sets.create_or_get_same_selection(
                        selection=request.version.explicit_selection,
                        required_part_ids=required_part_ids,
                    )
                )
                if (
                    requested_version_set["version_set_id"]
                    != version_set["version_set_id"]
                ):
                    raise ConflictError("semantic slot already froze another selection")
            elif value == "key_reference":
                key_reference = request.version.key_reference
                namespace = key_reference.namespace.strip()
                key = key_reference.key.strip()
                if key != str(existing.get("key") or "") or namespace != str(
                    existing.get("namespace") or ""
                ):
                    raise ConflictError("semantic slot already froze another key")
                if key_reference.HasField("expected_generation"):
                    expected_generation = int(key_reference.expected_generation)
                    if expected_generation != int(existing.get("key_generation") or 0):
                        raise GroupRealizationPreconditionError(
                            "key generation mismatch"
                        )
            part_by_id = {part["part_id"]: part for part in parts}
            part = part_by_id.get(context.part_id)
            if part is None:
                raise ConflictError("frozen version set is missing requested part")
            try:
                transaction = self._realizations.begin_or_join(
                    transaction_id=existing["transaction_id"],
                    group_kind=existing["group_kind"],
                    group_id=existing["group_id"],
                    epoch=int(existing["epoch"]),
                    version_set_id=version_set["version_set_id"],
                    realization_kind=version_set["realization_kind"],
                    transaction_fingerprint=existing["transaction_fingerprint"],
                    required_part_ids=existing["required_part_ids"],
                    total_parts=int(existing["total_parts"]),
                    part=part,
                    deadline_unix_nanos=existing["deadline_unix_nanos"],
                    namespace=existing.get("namespace"),
                    key=existing.get("key"),
                    key_generation=existing.get("key_generation"),
                    manifest_hash=existing.get("manifest_hash"),
                    daemon_id=request.daemon_id,
                    daemon_session_id=request.daemon_session_id or None,
                    worker_id=request.worker_id or None,
                )
            except GroupRealizationConflictError as exc:
                gs_metrics.inc_group_realization_db_conflict_retry(
                    operation="begin_or_join",
                    result="semantic_conflict",
                )
                raise ConflictError(str(exc)) from exc
            except DatabaseError as exc:
                if _is_group_realization_conflict(exc):
                    gs_metrics.inc_group_realization_db_conflict_retry(
                        operation="begin_or_join",
                        result="db_conflict",
                    )
                    raise ConflictError(str(exc)) from exc
                raise
            gs_metrics.inc_group_realization_control_write(
                operation="begin_or_join",
                table="group_realization_members",
            )
            result = {
                "transaction": transaction,
                "version_set": version_set,
                "part": part,
                "transaction_fingerprint": existing["transaction_fingerprint"],
                "key_generation": existing.get("key_generation"),
            }
            self._observe(
                operation="begin_or_join",
                started_at=started_at,
                result="replay",
            )
            return result
        version_set, parts, key_generation = self._resolve_version(
            version=request.version,
            required_part_ids=required_part_ids,
        )
        part_by_id = {part["part_id"]: part for part in parts}
        missing = [
            part_id for part_id in required_part_ids if part_id not in part_by_id
        ]
        if missing:
            raise ConflictError("version set missing required part ids")
        part = part_by_id[context.part_id]
        fingerprint = self._transaction_fingerprint(
            context=context,
            required_part_ids=required_part_ids,
            version_set=version_set,
            key_generation=key_generation,
        )
        if request.transaction_fingerprint and bytes(
            request.transaction_fingerprint
        ) != bytes(fingerprint):
            raise ConflictError("transaction_fingerprint does not match request")
        transaction_id = f"grt_{fingerprint.hex()}"
        try:
            transaction = self._realizations.begin_or_join(
                transaction_id=transaction_id,
                group_kind=context.group_kind.strip(),
                group_id=context.group_id.strip(),
                epoch=int(context.epoch),
                version_set_id=version_set["version_set_id"],
                realization_kind=version_set["realization_kind"],
                transaction_fingerprint=fingerprint,
                required_part_ids=required_part_ids,
                total_parts=int(context.total_parts),
                part=part,
                deadline_unix_nanos=self._transaction_deadline_unix_nanos(
                    int(request.deadline_unix_nanos or 0)
                ),
                namespace=version_set.get("namespace"),
                key=version_set.get("key"),
                key_generation=key_generation,
                manifest_hash=version_set["manifest_hash"],
                daemon_id=request.daemon_id,
                daemon_session_id=request.daemon_session_id or None,
                worker_id=request.worker_id or None,
            )
        except GroupRealizationConflictError as exc:
            gs_metrics.inc_group_realization_db_conflict_retry(
                operation="begin_or_join",
                result="semantic_conflict",
            )
            raise ConflictError(str(exc)) from exc
        except DatabaseError as exc:
            if _is_group_realization_conflict(exc):
                gs_metrics.inc_group_realization_db_conflict_retry(
                    operation="begin_or_join",
                    result="db_conflict",
                )
                raise ConflictError(str(exc)) from exc
            raise
        gs_metrics.inc_group_realization_control_write(
            operation="begin_or_join",
            table="group_realization_transactions",
        )
        gs_metrics.inc_group_realization_control_write(
            operation="begin_or_join",
            table="group_realization_members",
        )
        result = {
            "transaction": transaction,
            "version_set": version_set,
            "part": part,
            "transaction_fingerprint": fingerprint,
            "key_generation": key_generation,
        }
        self._observe(
            operation="begin_or_join",
            started_at=started_at,
            result="ok",
        )
        return result

    def member_fingerprint(
        self,
        *,
        transaction: dict[str, Any],
        member: dict[str, Any],
        request: global_store_pb2.ReportGroupRealizationPreparedRequest,
    ) -> bytes:
        staged = request.staged_value
        return _stable_hash(
            {
                "transaction_id": transaction["transaction_id"],
                "version_set_id": transaction["version_set_id"],
                "part_id": request.part_id,
                "selection_hash": bytes(member["selection_hash"]).hex(),
                "daemon_id": request.daemon_id or staged.daemon_id,
                "daemon_session_id": request.daemon_session_id
                or staged.daemon_session_id,
                "worker_id": request.worker_id,
                "materialization_attempt_id": request.materialization_attempt_id,
                "staged_binding_id": staged.binding_id,
                "staged_binding_value_id": staged.binding_value_id,
                "staging_token": staged.staging_token,
                "staging_epoch": int(staged.staging_epoch),
                "source_replica_id": request.source_replica_id,
                "source_export_generation": int(request.source_export_generation),
                "child_transport_request_id": request.child_transport_request_id,
            },
            b"tensorcast.group_realization_member.v1",
        )

    def report_prepared(
        self,
        *,
        request: global_store_pb2.ReportGroupRealizationPreparedRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], bytes]:
        started_at = time.monotonic()
        result = "ok"
        self._ensure_enabled()
        self._scan_expired(force=True)
        if (
            not request.staged_value.binding_id
            or not request.staged_value.binding_value_id
            or not request.staged_value.staging_token
        ):
            self._observe(
                operation="report_prepared",
                started_at=started_at,
                result="invalid",
            )
            raise ValidationError("prepared report requires a staged value ref")
        diagnostic = self._realizations.get_diagnostic(
            transaction_id=request.transaction_id
        )
        if diagnostic is None:
            raise NotFoundError("transaction not found")
        transaction = diagnostic["transaction"]
        member = next(
            (
                item
                for item in diagnostic["members"]
                if item["part_id"] == request.part_id
            ),
            None,
        )
        if member is None:
            raise NotFoundError("transaction member not found")
        member_was_prepared = member["state"] in {"prepared", "published"}
        fingerprint = self.member_fingerprint(
            transaction=transaction,
            member=member,
            request=request,
        )
        if request.member_fingerprint and bytes(request.member_fingerprint) != bytes(
            fingerprint
        ):
            raise ConflictError("member_fingerprint does not match prepared report")
        auto_publish_when_ready = (
            self._config.publish_authority_mode == "AUTO_WHEN_READY"
        )
        try:
            updated_txn, updated_member = self._realizations.report_prepared(
                transaction_id=request.transaction_id,
                part_id=request.part_id,
                member_fingerprint=fingerprint,
                daemon_id=request.daemon_id or request.staged_value.daemon_id,
                daemon_session_id=request.daemon_session_id
                or request.staged_value.daemon_session_id
                or None,
                worker_id=request.worker_id or None,
                materialization_attempt_id=request.materialization_attempt_id or None,
                staged_binding_id=request.staged_value.binding_id or None,
                staged_binding_value_id=request.staged_value.binding_value_id or None,
                staging_token=request.staged_value.staging_token or None,
                staging_epoch=int(request.staged_value.staging_epoch)
                if request.staged_value.staging_epoch
                else None,
                expected_previous_seal_generation=int(
                    request.expected_previous_seal_generation
                )
                if request.expected_previous_seal_generation
                else None,
                prepared_value_hash=bytes(request.prepared_value_hash)
                if request.prepared_value_hash
                else None,
                source_replica_id=request.source_replica_id or None,
                source_export_generation=int(request.source_export_generation),
                child_transport_request_id=request.child_transport_request_id or None,
                auto_publish_when_ready=auto_publish_when_ready,
            )
        except GroupRealizationConflictError as exc:
            result = "conflict"
            gs_metrics.inc_group_realization_db_conflict_retry(
                operation="report_prepared",
                result="semantic_conflict",
            )
            raise ConflictError(str(exc)) from exc
        except DatabaseError as exc:
            if _is_group_realization_conflict(exc):
                result = "conflict"
                gs_metrics.inc_group_realization_db_conflict_retry(
                    operation="report_prepared",
                    result="db_conflict",
                )
                raise ConflictError(str(exc)) from exc
            if "not ready to publish" in str(exc):
                result = "invalid"
                raise ValidationError(str(exc)) from exc
            result = "error"
            raise
        if not member_was_prepared:
            gs_metrics.inc_group_realization_control_write(
                operation="report_prepared",
                table="group_realization_members",
            )
            gs_metrics.inc_group_realization_control_write(
                operation="report_prepared",
                table="group_realization_transactions",
            )
        if (
            auto_publish_when_ready
            and updated_txn["state"] in {"published", "aborted"}
            and not member_was_prepared
        ):
            if (
                updated_txn["state"] == "aborted"
                and updated_txn.get("failure_code") == "source_visibility_stale"
            ):
                result = "conflict"
                self._remember_terminal(updated_txn)
                raise ConflictError(
                    "source visibility fence failed for group realization"
                )
            gs_metrics.inc_group_realization_control_write(
                operation="auto_publish",
                table="group_realization_members",
            )
            gs_metrics.inc_group_realization_control_write(
                operation="auto_publish",
                table="group_realization_transactions",
            )
            self._remember_terminal(updated_txn)
        self._observe(
            operation="report_prepared",
            started_at=started_at,
            result=result,
        )
        return updated_txn, updated_member, fingerprint

    def publish(
        self, *, transaction_id: str, require_ready_to_publish: bool
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        result = "ok"
        self._ensure_enabled()
        self._scan_expired(force=True)
        try:
            transaction = self._realizations.publish(
                transaction_id=transaction_id,
                require_ready_to_publish=require_ready_to_publish,
            )
            if (
                transaction["state"] == "aborted"
                and transaction.get("failure_code") == "source_visibility_stale"
            ):
                result = "conflict"
                self._remember_terminal(transaction)
                raise ConflictError(
                    "source visibility fence failed for group realization"
                )
            self._remember_terminal(transaction)
            gs_metrics.inc_group_realization_control_write(
                operation="publish",
                table="group_realization_members",
            )
            gs_metrics.inc_group_realization_control_write(
                operation="publish",
                table="group_realization_transactions",
            )
            return transaction
        except KeyError as exc:
            result = "not_found"
            raise NotFoundError("transaction not found") from exc
        except GroupRealizationConflictError as exc:
            result = "conflict"
            gs_metrics.inc_group_realization_db_conflict_retry(
                operation="publish",
                result="semantic_conflict",
            )
            raise ConflictError(str(exc)) from exc
        except DatabaseError as exc:
            if _is_group_realization_conflict(exc):
                result = "conflict"
                gs_metrics.inc_group_realization_db_conflict_retry(
                    operation="publish",
                    result="db_conflict",
                )
                raise ConflictError(str(exc)) from exc
            if "not ready to publish" in str(exc):
                result = "invalid"
                raise ValidationError(str(exc)) from exc
            result = "error"
            raise
        except Exception:
            result = "error"
            raise
        finally:
            self._observe(operation="publish", started_at=started_at, result=result)

    def abort(
        self,
        *,
        transaction_id: str,
        failure_code: str,
        failure_detail: str,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        result = "ok"
        self._ensure_enabled()
        try:
            transaction = self._realizations.abort(
                transaction_id=transaction_id,
                failure_code=failure_code,
                failure_detail=failure_detail,
            )
            self._remember_terminal(transaction)
            gs_metrics.inc_group_realization_control_write(
                operation="abort",
                table="group_realization_members",
            )
            gs_metrics.inc_group_realization_control_write(
                operation="abort",
                table="group_realization_transactions",
            )
            return transaction
        except KeyError as exc:
            result = "not_found"
            raise NotFoundError("transaction not found") from exc
        except Exception:
            result = "error"
            raise
        finally:
            self._observe(operation="abort", started_at=started_at, result=result)

    def wait_published(
        self,
        *,
        transaction_id: str,
        deadline_unix_nanos: int | None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        result = "ok"
        self._ensure_enabled()
        cached = self._get_terminal_cached(transaction_id=transaction_id)
        if cached is not None:
            self._observe(
                operation="wait_published",
                started_at=started_at,
                result="terminal_cache",
            )
            return cached
        self._enter_waiter(transaction_id=transaction_id)
        try:
            min_interval = max(0.001, self._config.min_wait_poll_interval_ms / 1000.0)
            max_interval = max(
                min_interval, self._config.max_wait_poll_interval_ms / 1000.0
            )
            interval = min_interval
            while True:
                self._scan_expired(force=True)
                transaction = self._realizations.get(transaction_id=transaction_id)
                if transaction is None:
                    result = "not_found"
                    raise NotFoundError("transaction not found")
                if transaction["state"] in {"published", "aborted", "expired"}:
                    self._remember_terminal(transaction)
                    gs_metrics.inc_group_realization_waiter_poll(result="terminal")
                    return transaction
                if (
                    deadline_unix_nanos is not None
                    and time.time_ns() >= deadline_unix_nanos
                ):
                    result = "timeout"
                    gs_metrics.inc_group_realization_waiter_poll(result="timeout")
                    return transaction
                gs_metrics.inc_group_realization_waiter_poll(result="poll")
                sleep_for = interval * random.uniform(0.8, 1.2)
                if deadline_unix_nanos is not None:
                    remaining = max(0.0, (deadline_unix_nanos - time.time_ns()) / 1e9)
                    sleep_for = min(sleep_for, remaining)
                if sleep_for <= 0:
                    result = "timeout"
                    return transaction
                time.sleep(sleep_for)
                interval = min(max_interval, interval * 2.0)
        except Exception:
            if result == "ok":
                result = "error"
            raise
        finally:
            self._leave_waiter(transaction_id=transaction_id)
            self._observe(
                operation="wait_published",
                started_at=started_at,
                result=result,
            )

    def get_diagnostic(self, *, transaction_id: str) -> dict[str, Any] | None:
        self._ensure_enabled()
        self._scan_expired(force=True)
        diagnostic = self._realizations.get_diagnostic(transaction_id=transaction_id)
        if diagnostic is None:
            gs_metrics.inc_group_realization_counter_reconciliation(result="not_found")
            return None
        counters = self._realizations.reconcile_counters(transaction_id=transaction_id)
        if counters is None:
            gs_metrics.inc_group_realization_counter_reconciliation(result="not_found")
            return diagnostic
        transaction = diagnostic["transaction"]
        result = (
            "match"
            if int(transaction["prepared_count"]) == counters["prepared"]
            and int(transaction["failed_count"]) == counters["failed"]
            and int(transaction["published_count"]) == counters["published"]
            else "mismatch"
        )
        gs_metrics.inc_group_realization_counter_reconciliation(result=result)
        return diagnostic
