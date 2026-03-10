#  Copyright (c) 2025-2026, TensorCast Team.

"""Artifact query RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional, cast

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.common.identity import ArtifactIdKind
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Replica
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.services.artifact_service import ArtifactService
from tensorcast.global_store.services.view_state_service import ViewStateService
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class ArtifactQueryRpcHandler:
    """Owns artifact metadata and replica query behavior."""

    def __init__(
        self,
        *,
        artifact_service: ArtifactService,
        artifact_repository: ArtifactRepository,
        view_state_service: ViewStateService,
        replica_to_memory_info: Callable[[Replica], common_pb2.MemoryInfo],
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        coerce_db_datetime: Callable[[object], datetime | None],
        logger,
    ) -> None:
        self._artifact_service = artifact_service
        self._artifact_repository = artifact_repository
        self._view_state_service = view_state_service
        self._replica_to_memory_info = replica_to_memory_info
        self._datetime_to_timestamp = datetime_to_timestamp
        self._coerce_db_datetime = coerce_db_datetime
        self._logger = logger

    @staticmethod
    def _build_hash_space_ref(
        *, space_kind: str, space_id: str
    ) -> common_pb2.HashSpaceRef:
        if space_kind == "C":
            return common_pb2.HashSpaceRef(
                byte_space=common_pb2.ByteSpaceRef(
                    kind=common_pb2.BYTE_SPACE_KIND_CANONICAL
                ),
                canonical_index_multihash=space_id,
            )
        if space_kind == "V":
            return common_pb2.HashSpaceRef(
                byte_space=common_pb2.ByteSpaceRef(
                    kind=common_pb2.BYTE_SPACE_KIND_VIEW, id=space_id
                )
            )
        raise ValidationError(f"Unsupported space_kind: {space_kind}")

    @staticmethod
    def _add_missing_leaf_ranges(
        detail: global_store_pb2.PartialLeafCoverageDetail,
        leaf_indices: list[int],
    ) -> None:
        """Append missing leaf indices as compact ranges."""
        if not leaf_indices:
            return
        sorted_unique = sorted(set(leaf_indices))
        start = sorted_unique[0]
        prev = start
        count = 1
        for idx in sorted_unique[1:]:
            if idx == prev + 1:
                count += 1
            else:
                detail.missing_leaf_ranges.append(
                    global_store_pb2.LeafIndexRange(start=int(start), count=int(count))
                )
                start = idx
                count = 1
            prev = idx
        detail.missing_leaf_ranges.append(
            global_store_pb2.LeafIndexRange(start=int(start), count=int(count))
        )

    def get_artifact_info_by_id(
        self,
        request: global_store_pb2.GetArtifactInfoByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactInfoByIdResponse:
        """Content-addressed query by artifact_id (mi2:...)."""
        artifact_id = request.artifact_id
        set_span_attributes({"tc.artifact.id": artifact_id})
        try:
            if not artifact_id:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("artifact_id is required")
                return global_store_pb2.GetArtifactInfoByIdResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            artifact_row: Optional[dict[str, object]] = self._artifact_repository.get(
                artifact_id
            )

            include_replicas = True
            if request.HasField("include_replicas"):
                include_replicas = request.include_replicas.value

            include_leaves = request.include_leaves
            include_view_meta = request.include_view_meta

            space_kind: Optional[str] = None  # 'C' or 'V'
            space_id: Optional[str] = None

            requested_byte_space: common_pb2.ByteSpaceRef | None = (
                request.requested_byte_space
                if request.HasField("requested_byte_space")
                else None
            )

            if (include_leaves or include_view_meta) and requested_byte_space is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    "requested_byte_space required when requesting metadata or leaves"
                )
                return global_store_pb2.GetArtifactInfoByIdResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

            if requested_byte_space is not None:
                if requested_byte_space.kind == common_pb2.BYTE_SPACE_KIND_CANONICAL:
                    space_kind = "C"
                    if not artifact_row or not artifact_row.get("index_multihash"):
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details("canonical index not recorded")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_NOT_FOUND
                        )
                    space_id = cast(str, artifact_row["index_multihash"])
                elif requested_byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    view_id = requested_byte_space.id
                    if not view_id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("requested_byte_space VIEW requires id")
                        return global_store_pb2.GetArtifactInfoByIdResponse(
                            status=global_store_pb2.Status.STATUS_ERROR
                        )
                    space_kind = "V"
                    space_id = view_id
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("unsupported requested_byte_space kind")
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

            available_replicas: list[common_pb2.MemoryInfo] = []
            if include_replicas:
                replica_view_id = space_id if space_kind == "V" else None
                replicas = self._artifact_service.get_artifact_replicas(
                    artifact_id, view_id=replica_view_id
                )
                available_replicas = [self._replica_to_memory_info(r) for r in replicas]

            view_meta_msg: Optional[global_store_pb2.ViewMeta] = None
            leaves_proto: list[global_store_pb2.Leaf] = []
            partial_byte_details: list[global_store_pb2.PartialCoverageDetail] = []
            partial_leaf_details: list[global_store_pb2.PartialLeafCoverageDetail] = []
            leaf_filter: Optional[list[int]] = None
            view_missing = False
            partial_leaf_miss = False

            view_row: Optional[dict[str, object]] = None
            if space_kind == "V" and (include_leaves or include_view_meta):
                view_row = self._view_state_service.get_view(
                    artifact_id=artifact_id, view_id=space_id or ""
                )
                if view_row is None:
                    view_missing = True
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    context.set_details("view metadata not found")

            if include_view_meta:
                if space_kind != "V":
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "view metadata is only available for view byte space"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )
                if view_row is not None:
                    view_size_value = cast(int, view_row["view_size"])
                    view_meta_msg = global_store_pb2.ViewMeta(
                        view_spec_json=str(view_row["view_spec_json"]),
                        view_size=int(view_size_value),
                    )
                    view_data_hash = view_row.get("view_data_hash")
                    if view_data_hash:
                        view_meta_msg.view_data_hash = str(view_data_hash)
                    verified_at = self._coerce_db_datetime(view_row.get("verified_at"))
                    proto_ts = self._datetime_to_timestamp(verified_at)
                    if proto_ts is not None:
                        view_meta_msg.verified_at.CopyFrom(proto_ts)

            if include_leaves:
                if space_kind is None or space_id is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "space selection required when requesting leaves"
                    )
                    return global_store_pb2.GetArtifactInfoByIdResponse(
                        status=global_store_pb2.Status.STATUS_ERROR
                    )

                leaf_filter = list(request.leaf_idxs) if request.leaf_idxs else None
                if space_kind == "V" and view_row is None:
                    partial_leaf_miss = True
                    detail = global_store_pb2.PartialLeafCoverageDetail(
                        hash_space=self._build_hash_space_ref(
                            space_kind="V", space_id=space_id or ""
                        ),
                    )
                    if leaf_filter:
                        self._add_missing_leaf_ranges(detail, leaf_filter)
                    partial_leaf_details.append(detail)
                else:
                    leaf_rows = self._view_state_service.get_leaves(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        leaf_idxs=leaf_filter,
                    )
                    leaves_proto = [
                        global_store_pb2.Leaf(leaf_idx=idx, digest=digest)
                        for idx, digest in leaf_rows
                    ]
                    if leaf_filter and len(leaves_proto) != len(set(leaf_filter)):
                        partial_leaf_miss = True
                        context.set_code(grpc.StatusCode.NOT_FOUND)
                        context.set_details(
                            "requested leaf digests not fully available"
                        )
                        detail = global_store_pb2.PartialLeafCoverageDetail(
                            hash_space=self._build_hash_space_ref(
                                space_kind=space_kind, space_id=space_id or ""
                            ),
                        )
                        existing = {leaf.leaf_idx for leaf in leaves_proto}
                        missing = sorted(
                            idx for idx in set(leaf_filter) if idx not in existing
                        )
                        self._add_missing_leaf_ranges(detail, missing)
                        partial_leaf_details.append(detail)

            has_payload = False
            if include_replicas and available_replicas:
                has_payload = True
            if include_leaves and leaves_proto:
                has_payload = True
            if include_view_meta and view_meta_msg is not None:
                has_payload = True

            need_not_found = (
                view_missing
                or partial_leaf_miss
                or (
                    not has_payload
                    and (include_replicas or include_leaves or include_view_meta)
                )
            )
            status = (
                global_store_pb2.Status.STATUS_NOT_FOUND
                if need_not_found
                else global_store_pb2.Status.STATUS_OK
            )

            descriptor_pb: Optional[common_pb2.ArtifactDescriptor] = None
            if artifact_row is not None:
                id_kind_value = str(artifact_row.get("id_kind") or "").upper()
                descriptor_pb = common_pb2.ArtifactDescriptor(
                    artifact_id=artifact_id,
                    index_multihash=str(artifact_row.get("index_multihash") or ""),
                    data_multihash=str(artifact_row.get("data_multihash") or ""),
                    schema_version=str(artifact_row.get("schema_version") or ""),
                    encoding=str(artifact_row.get("encoding") or ""),
                    total_size=0,
                )
                if id_kind_value == ArtifactIdKind.CGID.value:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_CGID
                    )
                elif id_kind_value == ArtifactIdKind.MI2.value:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2
                    )
                else:
                    descriptor_pb.id_kind = (
                        common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_UNSPECIFIED
                    )

            response = global_store_pb2.GetArtifactInfoByIdResponse(status=status)
            if include_replicas:
                response.replicas.extend(available_replicas)
            if include_leaves:
                response.leaves.extend(leaves_proto)
            if view_meta_msg is not None:
                response.view_meta.CopyFrom(view_meta_msg)
            if partial_byte_details:
                response.partial_coverage.extend(partial_byte_details)
            if partial_leaf_details:
                response.partial_leaf_coverage.extend(partial_leaf_details)
            if descriptor_pb is not None:
                response.descriptor.CopyFrom(descriptor_pb)
            return response

        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Error getting artifact info by id for %s",
                artifact_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.GetArtifactInfoByIdResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
