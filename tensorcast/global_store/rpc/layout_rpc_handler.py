#  Copyright (c) 2025-2026, TensorCast Team.

"""Layout v2 RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.repositories.artifact_index_repository import (
    ArtifactIndexRepository,
)
from tensorcast.global_store.repositories.artifact_layout_attachment_repository import (
    ArtifactLayoutAttachmentRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.assembly_layout_binding_repository import (
    AssemblyLayoutBindingRepository,
)
from tensorcast.global_store.repositories.assembly_runtime_policy_repository import (
    AssemblyRuntimePolicyRepository,
)
from tensorcast.global_store.repositories.layout_spec_repository import (
    LayoutSpecRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.layout.v1 import layout_pb2


class LayoutRpcHandler:
    """Owns layout v2 gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        connection,
        artifact_indices: ArtifactIndexRepository,
        artifact_repository: ArtifactRepository,
        layout_spec_repository: LayoutSpecRepository,
        assembly_layout_binding_repository: AssemblyLayoutBindingRepository,
        artifact_layout_attachment_repository: ArtifactLayoutAttachmentRepository,
        assembly_runtime_policy_repository: AssemblyRuntimePolicyRepository,
        multibase_sha256_to_hex: Callable[[str], str | None],
        sha256_digest_to_multibase: Callable[[bytes], str | None],
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._connection = connection
        self._artifact_indices = artifact_indices
        self._artifact_repository = artifact_repository
        self._layout_spec_repository = layout_spec_repository
        self._assembly_layout_binding_repository = assembly_layout_binding_repository
        self._artifact_layout_attachment_repository = (
            artifact_layout_attachment_repository
        )
        self._assembly_runtime_policy_repository = assembly_runtime_policy_repository
        self._multibase_sha256_to_hex = multibase_sha256_to_hex
        self._sha256_digest_to_multibase = sha256_digest_to_multibase
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    def _tensor_names_for_index_multihash(self, *, index_multihash: str) -> set[str]:
        index_key = self._multibase_sha256_to_hex(index_multihash)
        if not index_key:
            raise ValidationError("invalid index_multihash")
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
            tensor_names = self._tensor_names_for_index_multihash(
                index_multihash=request.layout.index_multihash
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

    def get_assembly_layout_binding(
        self,
        request: global_store_pb2.GetAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyLayoutBindingResponse:
        assembly_id = request.assembly_id
        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_layout_binding_repository.get(assembly_id=assembly_id)
            if row is None:
                return global_store_pb2.GetAssemblyLayoutBindingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            binding = global_store_pb2.AssemblyLayoutBinding(
                assembly_id=str(row["assembly_id"]),
                layout_id=str(row["layout_id"]),
                binding_version=int(row["binding_version"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                binding.updated_at.CopyFrom(ts)
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=binding,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyLayoutBinding failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def _assembly_has_any_cross_view_overlap(self, *, assembly_id: str) -> bool:
        rows = self._connection.execute(
            """
            SELECT view_id, range_offset, range_length
            FROM view_coverage_ranges
            WHERE artifact_id = ?
            ORDER BY range_offset ASC
            """,
            [assembly_id],
        ).fetchall()
        max_end = -1
        max_view: str | None = None
        for row in rows:
            view_id = str(row[0])
            start = int(row[1])
            end = start + int(row[2])
            if start < max_end and max_view is not None and view_id != max_view:
                return True
            if end > max_end:
                max_end = end
                max_view = view_id
        return False

    def update_assembly_layout_binding(
        self,
        request: global_store_pb2.UpdateAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyLayoutBindingResponse:
        assembly_id = request.assembly_id
        layout_id = request.layout_id
        if not assembly_id or not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and layout_id are required")
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        try:
            expected_version = int(request.expected_binding_version)
            layout_row = self._layout_spec_repository.get(layout_id=layout_id)
            if layout_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("layout_id not found")
                return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )

            artifact_row = self._artifact_repository.get(assembly_id)
            if expected_version == 0 and (
                not artifact_row or not artifact_row.get("index_multihash")
            ):
                self._connection.execute(
                    """
                    INSERT INTO artifacts (
                        artifact_id,
                        index_multihash,
                        data_multihash,
                        schema_version,
                        encoding,
                        hash_params_json,
                        id_kind
                    ) VALUES (?, ?, NULL, 'v3', 'json', NULL, 'CGID')
                    ON CONFLICT (artifact_id) DO NOTHING
                    """,
                    [assembly_id, str(layout_row.get("index_multihash"))],
                )
                artifact_row = self._artifact_repository.get(assembly_id)
            if not artifact_row or not artifact_row.get("index_multihash"):
                raise ValidationError("canonical index not recorded for assembly_id")
            if str(artifact_row.get("index_multihash")) != str(
                layout_row.get("index_multihash")
            ):
                raise ValidationError(
                    "layout.index_multihash does not match assembly index_multihash"
                )

            existing = self._assembly_layout_binding_repository.get(
                assembly_id=assembly_id
            )
            if existing is not None:
                old_layout_row = self._layout_spec_repository.get(
                    layout_id=str(existing["layout_id"])
                )
                old_has_rep = False
                new_has_rep = False
                if old_layout_row is not None:
                    old_spec = layout_pb2.LayoutSpec()
                    old_spec.ParseFromString(old_layout_row["layout_proto"])
                    old_has_rep = any(
                        p.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
                        for p in old_spec.tensors.values()
                    )
                new_spec = layout_pb2.LayoutSpec()
                new_spec.ParseFromString(layout_row["layout_proto"])
                new_has_rep = any(
                    p.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL
                    for p in new_spec.tensors.values()
                )
                if (
                    old_has_rep
                    and not new_has_rep
                    and self._assembly_has_any_cross_view_overlap(
                        assembly_id=assembly_id
                    )
                ):
                    raise ValueError("cannot tighten to DISJOINT while overlaps exist")

            with self._assembly_layout_binding_repository.transaction() as cursor:
                updated = self._assembly_layout_binding_repository.update(
                    assembly_id=assembly_id,
                    layout_id=layout_id,
                    expected_binding_version=expected_version,
                    cursor=cursor,
                )

            binding = global_store_pb2.AssemblyLayoutBinding(
                assembly_id=str(updated["assembly_id"]),
                layout_id=str(updated["layout_id"]),
                binding_version=int(updated["binding_version"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(updated.get("updated_at"))
            )
            if ts is not None:
                binding.updated_at.CopyFrom(ts)
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_OK,
                binding=binding,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateAssemblyLayoutBinding failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyLayoutBindingResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def attach_layout_to_artifact(
        self,
        request: global_store_pb2.AttachLayoutToArtifactRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AttachLayoutToArtifactResponse:
        mi2_id = request.mi2_id
        layout_id = request.layout_id
        if not mi2_id or not layout_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id and layout_id are required")
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            artifact_row = self._artifact_repository.get(mi2_id)
            if not artifact_row or not artifact_row.get("index_multihash"):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("artifact not found")
                return global_store_pb2.AttachLayoutToArtifactResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            layout_row = self._layout_spec_repository.get(layout_id=layout_id)
            if layout_row is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("layout not found")
                return global_store_pb2.AttachLayoutToArtifactResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            if str(layout_row.get("index_multihash")) != str(
                artifact_row.get("index_multihash")
            ):
                raise ValidationError(
                    "layout.index_multihash does not match artifact index_multihash"
                )
            with self._artifact_layout_attachment_repository.transaction() as cursor:
                self._artifact_layout_attachment_repository.attach(
                    mi2_id=mi2_id,
                    layout_id=layout_id,
                    cursor=cursor,
                )
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("AttachLayoutToArtifact failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.AttachLayoutToArtifactResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_artifact_layouts(
        self,
        request: global_store_pb2.ListArtifactLayoutsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactLayoutsResponse:
        mi2_id = request.mi2_id
        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            layout_ids = self._artifact_layout_attachment_repository.list_by_artifact(
                mi2_id=mi2_id
            )
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                layout_ids=layout_ids,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("ListArtifactLayouts failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListArtifactLayoutsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def get_assembly_runtime_policy(
        self,
        request: global_store_pb2.GetAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            row = self._assembly_runtime_policy_repository.get(assembly_id=assembly_id)
            if row is None:
                return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                    status=global_store_pb2.Status.STATUS_NOT_FOUND
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("GetAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def update_assembly_runtime_policy(
        self,
        request: global_store_pb2.UpdateAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyRuntimePolicyResponse:
        assembly_id = request.assembly_id
        if not assembly_id or not request.policy_json:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id and policy_json are required")
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            expected = int(request.expected_policy_version)
            with self._assembly_runtime_policy_repository.transaction() as cursor:
                row = self._assembly_runtime_policy_repository.update(
                    assembly_id=assembly_id,
                    policy_json=str(request.policy_json),
                    expected_policy_version=expected,
                    cursor=cursor,
                )
            policy = global_store_pb2.AssemblyRuntimePolicy(
                assembly_id=str(row["assembly_id"]),
                policy_version=int(row["policy_version"]),
                policy_json=str(row["policy_json"]),
            )
            ts = self._datetime_to_timestamp(
                self._coerce_db_datetime(row.get("updated_at"))
            )
            if ts is not None:
                policy.updated_at.CopyFrom(ts)
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_OK,
                policy=policy,
            )
        except (ValueError, DatabaseError) as exc:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("UpdateAssemblyRuntimePolicy failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateAssemblyRuntimePolicyResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
