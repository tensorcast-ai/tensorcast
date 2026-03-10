#  Copyright (c) 2026, TensorCast Team.

"""Composite client stub for split Global Store gRPC services."""

from __future__ import annotations

from typing import Any

import grpc

from tensorcast.proto.global_store.v1 import global_store_pb2_grpc


class GlobalStoreCompositeStub:
    """Facade over split Global Store service stubs."""

    def __init__(self, channel: grpc.Channel | grpc.aio.Channel):
        self.cluster_runtime = global_store_pb2_grpc.ClusterRuntimeServiceStub(channel)
        self.artifact_catalog = global_store_pb2_grpc.ArtifactCatalogServiceStub(
            channel
        )
        self.assembly_view = global_store_pb2_grpc.AssemblyViewServiceStub(channel)
        self.workflow_orchestration = (
            global_store_pb2_grpc.WorkflowOrchestrationServiceStub(channel)
        )
        self.cluster_admin = global_store_pb2_grpc.ClusterAdminServiceStub(channel)
        self._delegates = (
            self.cluster_runtime,
            self.artifact_catalog,
            self.assembly_view,
            self.workflow_orchestration,
            self.cluster_admin,
        )

    def __getattr__(self, name: str) -> Any:
        for stub in self._delegates:
            method = getattr(stub, name, None)
            if method is not None:
                return method
        raise AttributeError(name)


__all__ = ["GlobalStoreCompositeStub"]
