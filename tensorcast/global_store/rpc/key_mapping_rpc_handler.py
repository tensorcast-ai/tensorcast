#  Copyright (c) 2025-2026, TensorCast Team.

"""Key mapping RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

import grpc

from tensorcast.global_store.repositories.key_mapping_repository import (
    KeyMappingRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2


class KeyMappingRpcHandler:
    """Owns key mapping gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        key_mapping_repository: KeyMappingRepository,
        logger,
    ) -> None:
        self._key_mapping_repository = key_mapping_repository
        self._logger = logger

    def upsert_key_mapping(
        self,
        request: global_store_pb2.UpsertKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertKeyMappingResponse:
        """Create or update a key mapping with conflict checks."""
        try:
            key = request.key.strip()
            artifact_id = request.artifact_id.strip()
            if not key or not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key and artifact_id are required")
                return global_store_pb2.UpsertKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            existing = self._key_mapping_repository.get(key)
            if existing and existing.get("artifact_id") != artifact_id:
                return global_store_pb2.UpsertKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR,
                    conflict_reason=f"key already mapped to {existing.get('artifact_id')}",
                )

            ttl_seconds = None
            if request.HasField("ttl"):
                ttl_seconds = int(
                    request.ttl.seconds + (request.ttl.nanos // 1_000_000_000)
                )

            self._key_mapping_repository.upsert(
                key=key,
                artifact_id=artifact_id,
                replica_uuid=(request.replica_uuid or None),
                daemon_address=(request.daemon_address or None),
                ttl_seconds=ttl_seconds,
            )
            return global_store_pb2.UpsertKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in UpsertKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpsertKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def resolve_key_mapping(
        self,
        request: global_store_pb2.ResolveKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ResolveKeyMappingResponse:
        try:
            key = request.key.strip()
            if not key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key is required")
                return global_store_pb2.ResolveKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            row = self._key_mapping_repository.get(key)
            if not row:
                return global_store_pb2.ResolveKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            cache_ttl_seconds = 30
            kind = (row.get("kind") or "IMMUTABLE").upper()
            if kind == "ALIAS":
                cache_ttl_seconds = 0
            else:
                ttl_seconds = row.get("ttl_seconds")
                if ttl_seconds is not None:
                    cache_ttl_seconds = max(0, int(ttl_seconds))
            return global_store_pb2.ResolveKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=row.get("artifact_id", ""),
                replica_uuid=row.get("replica_uuid", "") or "",
                daemon_address=row.get("daemon_address", "") or "",
                generation=int(row.get("generation", 0) or 0),
                cache_ttl_seconds=int(cache_ttl_seconds),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in ResolveKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ResolveKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def swap_key_mapping(
        self,
        request: global_store_pb2.SwapKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SwapKeyMappingResponse:
        try:
            key = request.key.strip()
            new_artifact_id = request.new_artifact_id.strip()
            if not key or not new_artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key and new_artifact_id are required")
                return global_store_pb2.SwapKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            expected_artifact_id = (
                request.expected_artifact_id.strip()
                if request.expected_artifact_id
                else None
            )
            expected_generation = (
                int(request.expected_generation)
                if request.HasField("expected_generation")
                else None
            )
            result = self._key_mapping_repository.swap(
                key=key,
                new_artifact_id=new_artifact_id,
                expected_artifact_id=expected_artifact_id,
                expected_generation=expected_generation,
            )
            if not result.get("ok"):
                return global_store_pb2.SwapKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR,
                    artifact_id=result.get("artifact_id", "") or "",
                    generation=int(result.get("generation", 0) or 0),
                )
            return global_store_pb2.SwapKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                artifact_id=result.get("artifact_id", "") or "",
                generation=int(result.get("generation", 0) or 0),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in SwapKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.SwapKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def revoke_key_mapping(
        self,
        request: global_store_pb2.RevokeKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RevokeKeyMappingResponse:
        try:
            key = request.key.strip()
            if not key:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("key is required")
                return global_store_pb2.RevokeKeyMappingResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            ok = self._key_mapping_repository.delete(key)
            return global_store_pb2.RevokeKeyMappingResponse(
                status=(
                    global_store_pb2.Status.STATUS_OK
                    if ok
                    else global_store_pb2.Status.STATUS_NOT_FOUND
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Error in RevokeKeyMapping")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.RevokeKeyMappingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
