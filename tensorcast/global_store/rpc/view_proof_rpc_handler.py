#  Copyright (c) 2025-2026, TensorCast Team.

"""View state and proof RPC handler extracted from Global Store gRPC servicer."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.config.settings import GlobalStoreConfig
from tensorcast.global_store.exceptions import DatabaseError, ValidationError
from tensorcast.global_store.repositories.artifact_repository import ArtifactRepository
from tensorcast.global_store.repositories.proof_repository import ProofRepository
from tensorcast.global_store.repositories.view_coverage_repository import (
    ViewCoverageRepository,
)
from tensorcast.global_store.repositories.view_repository import ViewRepository
from tensorcast.global_store.services.view_state_service import (
    LeafWritePayload,
    PieceProofDigestPayload,
    ViewStateService,
    ViewUpsertPayload,
)
from tensorcast.observability.otel import set_span_attributes
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class ViewProofRpcHandler:
    """Owns view-state/proof gRPC behavior and error mapping."""

    def __init__(
        self,
        *,
        config: GlobalStoreConfig,
        artifact_repository: ArtifactRepository,
        view_repository: ViewRepository,
        view_coverage_repository: ViewCoverageRepository,
        proof_repository: ProofRepository,
        view_state_service: ViewStateService,
        timestamp_to_datetime: Callable,
        datetime_to_timestamp: Callable[
            [datetime | None], timestamp_pb2.Timestamp | None
        ],
        get_tensor_intervals_for_artifact_id: Callable[..., dict[str, tuple[int, int]]],
        logger,
    ) -> None:
        self._config = config
        self._artifact_repository = artifact_repository
        self._view_repository = view_repository
        self._view_coverage_repository = view_coverage_repository
        self._proof_repository = proof_repository
        self._view_state_service = view_state_service
        self._timestamp_to_datetime = timestamp_to_datetime
        self._datetime_to_timestamp = datetime_to_timestamp
        self._get_tensor_intervals_for_artifact_id = (
            get_tensor_intervals_for_artifact_id
        )
        self._logger = logger

    @staticmethod
    def _digest_conflict_grid_from_message(message: str) -> str:
        if "leaves conflict" in message:
            return "leaves"
        if "piece_proof_digests conflict" in message:
            return "piece_proof_digests"
        if "assembly_proof_commitments conflict" in message:
            return "assembly_proof_commitments"
        if "tensor_proof_commitments conflict" in message:
            return "tensor_proof_commitments"
        return "unknown"

    @staticmethod
    def _is_failed_precondition_message(message: str) -> bool:
        return any(
            key in message
            for key in (
                "coverage metadata missing",
                "overlapping canonical coverage",
                "canonical range mismatch",
                "view_data_hash conflict",
                "conflict",
                "proof",
            )
        )

    def update_artifact_view_state(
        self,
        request: global_store_pb2.UpdateArtifactViewStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateArtifactViewStateResponse:
        """Upsert view metadata, leaf digests, and proof digests."""
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        leaf_count = len(request.leaf_writes)
        proof_count = len(request.proof_digests)
        has_digest_write = (leaf_count + proof_count) > 0

        if has_digest_write:
            limits = self._config.limits.digest_writes
            digest_bytes = sum(len(leaf.digest) for leaf in request.leaf_writes) + sum(
                len(digest.digest) for digest in request.proof_digests
            )
            total = leaf_count + proof_count
            too_large = (
                leaf_count > limits.max_leaf_writes_per_request
                or proof_count > limits.max_proof_digests_per_request
                or total > limits.max_total_digests_per_request
                or digest_bytes > limits.max_digest_bytes_per_request
            )
            if too_large:
                gs_metrics.inc_digest_request_rejected(reason="too_large")
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(
                    "digest write request exceeds configured limits "
                    f"(leaf_writes={leaf_count}, proof_digests={proof_count}, total_digests={total}, "
                    f"digest_bytes={digest_bytes})"
                )
                return global_store_pb2.UpdateArtifactViewStateResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        try:
            view_payload = None
            if request.HasField("view"):
                view = request.view
                if not view.view_id:
                    raise ValidationError("view.view_id is required")
                if view.view_size <= 0:
                    raise ValidationError("view.view_size must be positive")
                verified_at = (
                    self._timestamp_to_datetime(view.verified_at)
                    if view.HasField("verified_at")
                    else None
                )
                view_data_hash = view.view_data_hash or None
                canonical_size = None
                canonical_covered = None
                canonical_ranges: list[tuple[int, int]] = []
                if view.HasField("canonical_coverage"):
                    canonical_size = int(view.canonical_coverage.total_bytes)
                    canonical_covered = int(view.canonical_coverage.covered_bytes)
                if view.canonical_ranges:
                    canonical_ranges = [
                        (int(rng.off), int(rng.len)) for rng in view.canonical_ranges
                    ]
                view_payload = ViewUpsertPayload(
                    artifact_id=artifact_id,
                    view_id=view.view_id,
                    view_spec_json=view.view_spec_json,
                    view_size=view.view_size,
                    view_data_hash=view_data_hash,
                    verified_at=verified_at,
                    canonical_size_bytes=canonical_size,
                    canonical_bytes_covered=canonical_covered,
                    canonical_ranges=canonical_ranges or None,
                )

            leaf_payloads: list[LeafWritePayload] = []
            for leaf in request.leaf_writes:
                if not leaf.digest:
                    raise ValidationError("leaf.digest is required")
                if len(leaf.digest) != 32:
                    raise ValidationError("leaf.digest must be 32 bytes (raw sha256)")
                if not leaf.HasField("hash_space"):
                    raise ValidationError("leaf.hash_space must be set")
                hash_space = leaf.hash_space
                if not hash_space.HasField("byte_space"):
                    raise ValidationError("leaf.hash_space.byte_space must be set")
                byte_space = hash_space.byte_space

                if byte_space.kind == common_pb2.BYTE_SPACE_KIND_CANONICAL:
                    if not hash_space.canonical_index_multihash:
                        raise ValidationError(
                            "leaf.hash_space.canonical_index_multihash is required for CANONICAL"
                        )
                    space_kind = "C"
                    space_id = hash_space.canonical_index_multihash
                elif byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW:
                    if not byte_space.id:
                        raise ValidationError(
                            "leaf.hash_space.byte_space VIEW requires id"
                        )
                    space_kind = "V"
                    space_id = byte_space.id
                else:
                    raise ValidationError(
                        f"Unsupported leaf.hash_space.byte_space kind: {byte_space.kind}"
                    )

                leaf_payloads.append(
                    LeafWritePayload(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        leaf_idx=leaf.leaf_idx,
                        digest=bytes(leaf.digest),
                    )
                )

            proof_payloads: list[PieceProofDigestPayload] = []
            for digest in request.proof_digests:
                if not digest.view_id:
                    raise ValidationError("proof_digests.view_id is required")
                if not digest.tensor_name:
                    raise ValidationError("proof_digests.tensor_name is required")
                if not digest.proof_schema_version:
                    raise ValidationError(
                        "proof_digests.proof_schema_version is required"
                    )
                if not digest.digest:
                    raise ValidationError("proof_digests.digest is required")
                proof_payloads.append(
                    PieceProofDigestPayload(
                        artifact_id=artifact_id,
                        view_id=digest.view_id,
                        tensor_name=digest.tensor_name,
                        proof_schema_version=digest.proof_schema_version,
                        proof_chunk_idx=int(digest.proof_chunk_idx),
                        digest=bytes(digest.digest),
                    )
                )

            tensor_intervals = None
            if (
                view_payload is not None
                and artifact_id.startswith("cgid:")
                and view_payload.canonical_ranges
            ):
                tensor_intervals = self._get_tensor_intervals_for_artifact_id(
                    artifact_id=artifact_id
                )

            self._view_state_service.update_view_state(
                view=view_payload,
                leaf_writes=leaf_payloads,
                proof_digests=proof_payloads,
                tensor_intervals=tensor_intervals,
            )
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_OK
            )
        except ValidationError as exc:
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="invalid")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    gs_metrics.inc_digest_conflict(
                        grid=self._digest_conflict_grid_from_message(message)
                    )
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if self._is_failed_precondition_message(message):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(message)
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    gs_metrics.inc_digest_conflict(
                        grid=self._digest_conflict_grid_from_message(message)
                    )
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if self._is_failed_precondition_message(message):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "Failed to update artifact view state for artifact_id=%s",
                artifact_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.UpdateArtifactViewStateResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def write_tensor_proof_commitments(
        self,
        request: global_store_pb2.WriteTensorProofCommitmentsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WriteTensorProofCommitmentsResponse:
        mi2_id = request.mi2_id
        set_span_attributes({"tc.artifact.id": mi2_id})

        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id.startswith("mi2:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id must be a content-addressed mi2 id")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.proof_schema_version:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("proof_schema_version is required")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        artifact_row = self._artifact_repository.get(mi2_id)
        if artifact_row is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("artifact not found")
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )

        proof_count = len(request.commitments)
        has_digest_write = proof_count > 0
        if has_digest_write:
            limits = self._config.limits.digest_writes
            digest_bytes = sum(
                len(commitment.digest) for commitment in request.commitments
            )
            too_large = (
                proof_count > limits.max_proof_digests_per_request
                or proof_count > limits.max_total_digests_per_request
                or digest_bytes > limits.max_digest_bytes_per_request
            )
            if too_large:
                gs_metrics.inc_digest_request_rejected(reason="too_large")
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details(
                    "digest write request exceeds configured limits "
                    f"(commitments={proof_count}, digest_bytes={digest_bytes})"
                )
                return global_store_pb2.WriteTensorProofCommitmentsResponse(
                    status=global_store_pb2.Status.STATUS_ERROR
                )

        try:
            inserted = 0
            with self._proof_repository.transaction() as cursor:
                for entry in request.commitments:
                    if not entry.tensor_name:
                        raise ValidationError("commitments.tensor_name is required")
                    if not entry.digest:
                        raise ValidationError("commitments.digest is required")
                    if len(entry.digest) != 32:
                        raise ValidationError(
                            "commitments.digest must be 32 bytes (raw sha256)"
                        )
                    if self._proof_repository.upsert_tensor_proof_commitment(
                        mi2_id=mi2_id,
                        tensor_name=entry.tensor_name,
                        proof_schema_version=request.proof_schema_version,
                        proof_chunk_idx=int(entry.proof_chunk_idx),
                        digest=bytes(entry.digest),
                        cursor=cursor,
                    ):
                        inserted += 1

            gs_metrics.inc_digest_entries_written(
                grid="tensor_proof_commitments",
                count=inserted,
            )
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                inserted=int(inserted),
            )
        except ValidationError as exc:
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="invalid")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    gs_metrics.inc_digest_conflict(
                        grid=(
                            "tensor_proof_commitments"
                            if "tensor_proof_commitments conflict" in message
                            else "unknown"
                        )
                    )
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if "conflict" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(message)
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            message = str(exc)
            if has_digest_write:
                if "conflict" in message:
                    gs_metrics.inc_digest_request_rejected(reason="conflict")
                    gs_metrics.inc_digest_conflict(
                        grid=(
                            "tensor_proof_commitments"
                            if "tensor_proof_commitments conflict" in message
                            else "unknown"
                        )
                    )
                else:
                    gs_metrics.inc_digest_request_rejected(reason="invalid")
            if "conflict" in message:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            else:
                context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(message)
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "WriteTensorProofCommitments failed for mi2_id=%s", mi2_id
            )
            if has_digest_write:
                gs_metrics.inc_digest_request_rejected(reason="internal")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.WriteTensorProofCommitmentsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def check_proof_commitments_match(
        self,
        request: global_store_pb2.CheckProofCommitmentsMatchRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CheckProofCommitmentsMatchResponse:
        assembly_id = request.assembly_id
        mi2_id = request.mi2_id
        set_span_attributes(
            {"tc.artifact.id": mi2_id, "tc.artifact.assembly_id": assembly_id}
        )

        if not assembly_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not assembly_id.startswith("cgid:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("assembly_id must be a cgid id")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not mi2_id.startswith("mi2:"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mi2_id must be a content-addressed mi2 id")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.proof_schema_version:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("proof_schema_version is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        if not request.tensor_names:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("tensor_names is required")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

        if self._artifact_repository.get(assembly_id) is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("assembly not found")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )
        if self._artifact_repository.get(mi2_id) is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("artifact not found")
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_NOT_FOUND
            )

        try:
            match = self._proof_repository.commitments_match(
                assembly_id=assembly_id,
                mi2_id=mi2_id,
                proof_schema_version=request.proof_schema_version,
                tensor_names=list(request.tensor_names),
            )
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_OK,
                match=match,
            )
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except DatabaseError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "CheckProofCommitmentsMatch failed for assembly_id=%s mi2_id=%s",
                assembly_id,
                mi2_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.CheckProofCommitmentsMatchResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )

    def list_views(
        self,
        request: global_store_pb2.ListViewsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListViewsResponse:
        """List view metadata for an artifact."""
        artifact_id = request.artifact_id
        if not artifact_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("artifact_id is required")
            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
        try:
            page_size = 100
            page_token = ""
            if request.HasField("pagination"):
                if request.pagination.page_size:
                    page_size = int(request.pagination.page_size)
                if request.pagination.page_token:
                    page_token = request.pagination.page_token
            offset = int(page_token) if page_token else 0

            rows, total = self._view_repository.list_by_artifact(
                artifact_id=artifact_id,
                limit=page_size,
                offset=offset,
            )

            views: list[global_store_pb2.ViewInfo] = []
            for row in rows:
                item = global_store_pb2.ViewInfo(
                    view_id=str(row["view_id"]),
                    view_spec_json=str(row["view_spec_json"]),
                    view_size=int(row["view_size"]),
                )
                if row.get("view_data_hash"):
                    item.view_data_hash = str(row["view_data_hash"])
                if row.get("verified_at"):
                    ts = self._datetime_to_timestamp(row["verified_at"])
                    if ts is not None:
                        item.verified_at.CopyFrom(ts)
                if (
                    row.get("canonical_size_bytes") is not None
                    or row.get("canonical_bytes_covered") is not None
                ):
                    coverage = global_store_pb2.CanonicalCoverage()
                    if row.get("canonical_size_bytes") is not None:
                        coverage.total_bytes = int(row["canonical_size_bytes"])
                    if row.get("canonical_bytes_covered") is not None:
                        coverage.covered_bytes = int(row["canonical_bytes_covered"])
                    item.canonical_coverage.CopyFrom(coverage)

                ranges = self._view_coverage_repository.get_ranges(
                    artifact_id=artifact_id,
                    view_id=str(row["view_id"]),
                )
                for off, length in ranges:
                    item.canonical_ranges.add(off=off, len=length)

                views.append(item)

            next_token = ""
            if offset + page_size < total:
                next_token = str(offset + page_size)
            page_info = common_pb2.PageInfo(
                next_page_token=next_token,
                total_size=total,
            )

            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_OK,
                views=views,
                page_info=page_info,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Failed to list views for %s", artifact_id)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return global_store_pb2.ListViewsResponse(
                status=global_store_pb2.Status.STATUS_ERROR
            )
