#  Copyright (c) 2025-2026, TensorCast Team.

"""Layout binding and attachment RPC handler."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.repositories.artifact_layout_attachment_repository import (
    ArtifactLayoutAttachmentRepository,
)
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.assembly_layout_binding_repository import (
    AssemblyLayoutBindingRepository,
)
from tensorcast.global_store.repositories.layout_spec_repository import (
    LayoutSpecRepository,
)
from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.layout.v1 import layout_pb2


class LayoutBindingRpcHandler:
    """Owns assembly layout binding and artifact attachment behavior."""

    def __init__(
        self,
        *,
        connection,
        artifact_repository: ArtifactRepository,
        layout_spec_repository: LayoutSpecRepository,
        assembly_layout_binding_repository: AssemblyLayoutBindingRepository,
        artifact_layout_attachment_repository: ArtifactLayoutAttachmentRepository,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._connection = connection
        self._artifact_repository = artifact_repository
        self._layout_spec_repository = layout_spec_repository
        self._assembly_layout_binding_repository = assembly_layout_binding_repository
        self._artifact_layout_attachment_repository = (
            artifact_layout_attachment_repository
        )
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

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
