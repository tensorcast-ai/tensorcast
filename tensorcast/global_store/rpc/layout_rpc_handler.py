#  Copyright (c) 2025-2026, TensorCast Team.

"""Layout v2 RPC facade composed from state-domain handlers."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import grpc
from google.protobuf import timestamp_pb2

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
from tensorcast.global_store.rpc.layout_binding_rpc_handler import (
    LayoutBindingRpcHandler,
)
from tensorcast.global_store.rpc.layout_runtime_policy_rpc_handler import (
    LayoutRuntimePolicyRpcHandler,
)
from tensorcast.global_store.rpc.layout_spec_rpc_handler import LayoutSpecRpcHandler
from tensorcast.proto.global_store.v1 import global_store_pb2


class LayoutRpcHandler:
    """Backward-compatible facade around split layout domain handlers."""

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
        self._spec_rpc_handler = LayoutSpecRpcHandler(
            artifact_indices=artifact_indices,
            layout_spec_repository=layout_spec_repository,
            multibase_sha256_to_hex=multibase_sha256_to_hex,
            sha256_digest_to_multibase=sha256_digest_to_multibase,
            logger=logger,
        )
        self._binding_rpc_handler = LayoutBindingRpcHandler(
            connection=connection,
            artifact_repository=artifact_repository,
            layout_spec_repository=layout_spec_repository,
            assembly_layout_binding_repository=assembly_layout_binding_repository,
            artifact_layout_attachment_repository=artifact_layout_attachment_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )
        self._runtime_policy_rpc_handler = LayoutRuntimePolicyRpcHandler(
            assembly_runtime_policy_repository=assembly_runtime_policy_repository,
            datetime_to_timestamp=datetime_to_timestamp,
            coerce_db_datetime=coerce_db_datetime,
            logger=logger,
        )

    def put_layout_spec(
        self,
        request: global_store_pb2.PutLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PutLayoutSpecResponse:
        return self._spec_rpc_handler.put_layout_spec(request, context)

    def get_layout_spec(
        self,
        request: global_store_pb2.GetLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetLayoutSpecResponse:
        return self._spec_rpc_handler.get_layout_spec(request, context)

    def get_assembly_layout_binding(
        self,
        request: global_store_pb2.GetAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyLayoutBindingResponse:
        return self._binding_rpc_handler.get_assembly_layout_binding(request, context)

    def update_assembly_layout_binding(
        self,
        request: global_store_pb2.UpdateAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyLayoutBindingResponse:
        return self._binding_rpc_handler.update_assembly_layout_binding(
            request, context
        )

    def attach_layout_to_artifact(
        self,
        request: global_store_pb2.AttachLayoutToArtifactRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AttachLayoutToArtifactResponse:
        return self._binding_rpc_handler.attach_layout_to_artifact(request, context)

    def list_artifact_layouts(
        self,
        request: global_store_pb2.ListArtifactLayoutsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactLayoutsResponse:
        return self._binding_rpc_handler.list_artifact_layouts(request, context)

    def get_assembly_runtime_policy(
        self,
        request: global_store_pb2.GetAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyRuntimePolicyResponse:
        return self._runtime_policy_rpc_handler.get_assembly_runtime_policy(
            request, context
        )

    def update_assembly_runtime_policy(
        self,
        request: global_store_pb2.UpdateAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyRuntimePolicyResponse:
        return self._runtime_policy_rpc_handler.update_assembly_runtime_policy(
            request, context
        )
