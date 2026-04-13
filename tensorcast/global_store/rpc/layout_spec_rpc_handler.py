#  Copyright (c) 2025-2026, TensorCast Team.

"""Layout spec RPC handler."""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import grpc

from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.layout_spec_repository import (
    LayoutSpecRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.layout.v1 import layout_pb2


class LayoutSpecRpcHandler:
    """Owns layout spec put/get behavior and validation."""

    def __init__(
        self,
        *,
        artifact_indices: ArtifactIndexRepository,
        layout_spec_repository: LayoutSpecRepository,
        multibase_sha256_to_hex: Callable[[str], str | None],
        sha256_digest_to_multibase: Callable[[bytes], str | None],
        logger,
    ) -> None:
        self._artifact_indices = artifact_indices
        self._layout_spec_repository = layout_spec_repository
        self._multibase_sha256_to_hex = multibase_sha256_to_hex
        self._sha256_digest_to_multibase = sha256_digest_to_multibase
        self._logger = logger

    def _tensor_names_for_index_multihash(
        self,
        *,
        index_multihash: str,
        canonical_index_data: bytes | None = None,
    ) -> set[str]:
        index_key = self._multibase_sha256_to_hex(index_multihash)
        if not index_key:
            raise ValidationError("invalid index_multihash")
        data = canonical_index_data
        if data is not None and hashlib.sha256(data).hexdigest() != index_key:
            raise ValidationError(
                "canonical_index_data digest does not match layout.index_multihash"
            )
        if data is None:
            data = self._artifact_indices.get(index_key)
        if data is None:
            raise ValidationError("canonical index bytes missing for index_multihash")
        try:
            decoded = json.loads(bytes(data).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValidationError("failed to decode canonical index bytes") from exc
        if not isinstance(decoded, dict):
            raise ValidationError("canonical index must be a JSON object")
        return {name for name in decoded if isinstance(name, str)}

    @staticmethod
    def _canonicalize_layout_spec(
        *,
        layout: layout_pb2.LayoutSpec,
        tensor_names: set[str],
    ) -> layout_pb2.LayoutSpec:
        if int(layout.layout_schema_version) != 1:
            raise ValidationError("layout_schema_version must be 1")
        if not layout.index_multihash:
            raise ValidationError("layout.index_multihash is required")

        out = layout_pb2.LayoutSpec()
        out.CopyFrom(layout)

        deduped = sorted(set(out.expected_view_ids))
        del out.expected_view_ids[:]
        out.expected_view_ids.extend(deduped)

        has_replicated = False
        for tensor_name, policy in out.tensors.items():
            if tensor_name not in tensor_names:
                raise ValidationError("layout references unknown tensor_name")
            if policy.overlap_mode == layout_pb2.OVERLAP_MODE_UNSPECIFIED:
                policy.overlap_mode = layout_pb2.OVERLAP_MODE_DISJOINT
            if policy.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL:
                has_replicated = True

        if has_replicated:
            if not out.proof_schema_version:
                raise ValidationError(
                    "layout.proof_schema_version is required when using REPLICATE_EQUAL"
                )
        elif out.proof_schema_version:
            raise ValidationError(
                "layout.proof_schema_version must be empty unless REPLICATE_EQUAL is used"
            )
        return out

    def put_layout_spec(
        self,
        request: global_store_pb2.PutLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PutLayoutSpecResponse:
        try:
            if not request.HasField("layout"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("layout is required")
                return global_store_pb2.PutLayoutSpecResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )
            canonical_index_data = (
                bytes(request.canonical_index_data)
                if request.canonical_index_data
                else None
            )
            tensor_names = self._tensor_names_for_index_multihash(
                index_multihash=request.layout.index_multihash,
                canonical_index_data=canonical_index_data,
            )
            canonical = self._canonicalize_layout_spec(
                layout=request.layout,
                tensor_names=tensor_names,
            )
            payload = canonical.SerializeToString(deterministic=True)
            digest = hashlib.sha256(payload).digest()
            layout_id = self._sha256_digest_to_multibase(digest)
            if not layout_id:
                raise ValidationError("failed to compute layout_id")

            with self._layout_spec_repository.transaction() as cursor:
                if canonical_index_data is not None:
                    self._artifact_indices.upsert_index(
                        index_data=canonical_index_data,
                        encoding="json",
                        schema_version="v3",
                        cursor=cursor,
                    )
                self._layout_spec_repository.put(
                    layout_id=layout_id,
                    index_multihash=canonical.index_multihash,
                    layout_proto=payload,
                    layout_json=(request.layout_json or None),
                    cursor=cursor,
                )

            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_OK,
                layout_id=layout_id,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except (ValueError, DatabaseError) as exc:
            message = str(exc)
            if "layout_id collision" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("PutLayoutSpec failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.PutLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def get_layout_spec(
        self,
        request: global_store_pb2.GetLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetLayoutSpecResponse:
        layout_id = request.layout_id
        if not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("layout_id is required")
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._layout_spec_repository.get(layout_id=layout_id)
            if row is None:
                return global_store_pb2.GetLayoutSpecResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            layout = layout_pb2.LayoutSpec()
            layout.ParseFromString(row["layout_proto"])
            record = layout_pb2.LayoutSpecRecord(layout_id=layout_id, layout=layout)
            if row.get("layout_json"):
                record.layout_json = str(row["layout_json"])
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_OK,
                record=record,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetLayoutSpec failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetLayoutSpecResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
