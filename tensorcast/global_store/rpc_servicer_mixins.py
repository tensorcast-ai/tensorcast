#  Copyright (c) 2025-2026, TensorCast Team.

"""Typed gRPC servicer mixins for Global Store RPC domains."""

from __future__ import annotations

from typing import Any

import grpc

from tensorcast.global_store.rpc.artifact_binding_rpc_handler import (
    ArtifactBindingRpcHandler,
)
from tensorcast.global_store.rpc.artifact_index_rpc_handler import (
    ArtifactIndexRpcHandler,
)
from tensorcast.global_store.rpc.artifact_query_rpc_handler import (
    ArtifactQueryRpcHandler,
)
from tensorcast.global_store.rpc.chunk_rpc_handler import ChunkRpcHandler
from tensorcast.global_store.rpc.disk_location_rpc_handler import DiskLocationRpcHandler
from tensorcast.global_store.rpc.instance_rpc_handler import InstanceRpcHandler
from tensorcast.global_store.rpc.key_mapping_rpc_handler import KeyMappingRpcHandler
from tensorcast.global_store.rpc.layout_binding_rpc_handler import (
    LayoutBindingRpcHandler,
)
from tensorcast.global_store.rpc.layout_runtime_policy_rpc_handler import (
    LayoutRuntimePolicyRpcHandler,
)
from tensorcast.global_store.rpc.layout_spec_rpc_handler import LayoutSpecRpcHandler
from tensorcast.global_store.rpc.operation_rpc_handler import OperationRpcHandler
from tensorcast.global_store.rpc.placement_persistence_rpc_handler import (
    PlacementPersistenceRpcHandler,
)
from tensorcast.global_store.rpc.replica_lifecycle_rpc_handler import (
    ReplicaLifecycleRpcHandler,
)
from tensorcast.global_store.rpc.replica_registration_rpc_handler import (
    ReplicaRegistrationRpcHandler,
)
from tensorcast.global_store.rpc.transport_rpc_handler import TransportRpcHandler
from tensorcast.global_store.rpc.view_proof_rpc_handler import ViewProofRpcHandler
from tensorcast.global_store.rpc.worker_rpc_handler import WorkerRpcHandler
from tensorcast.global_store.rpc.worker_state_sync_rpc_handler import (
    WorkerStateSyncRpcHandler,
)


class AssemblyViewRpcServicerMixin:
    """Assembly/view RPC routing."""

    artifact_query_rpc_handler: ArtifactQueryRpcHandler
    view_proof_rpc_handler: ViewProofRpcHandler
    layout_spec_rpc_handler: LayoutSpecRpcHandler
    layout_binding_rpc_handler: LayoutBindingRpcHandler
    layout_runtime_policy_rpc_handler: LayoutRuntimePolicyRpcHandler

    def GetArtifactInfoById(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.artifact_query_rpc_handler.get_artifact_info_by_id(request, context)

    def UpdateArtifactViewState(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.view_proof_rpc_handler.update_artifact_view_state(request, context)

    def WriteTensorProofCommitments(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.view_proof_rpc_handler.write_tensor_proof_commitments(
            request, context
        )

    def CheckProofCommitmentsMatch(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.view_proof_rpc_handler.check_proof_commitments_match(
            request, context
        )

    def ListViews(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.view_proof_rpc_handler.list_views(request, context)

    def PutLayoutSpec(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.layout_spec_rpc_handler.put_layout_spec(request, context)

    def GetLayoutSpec(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.layout_spec_rpc_handler.get_layout_spec(request, context)

    def GetAssemblyLayoutBinding(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.layout_binding_rpc_handler.get_assembly_layout_binding(
            request, context
        )

    def UpdateAssemblyLayoutBinding(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.layout_binding_rpc_handler.update_assembly_layout_binding(
            request, context
        )

    def AttachLayoutToArtifact(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.layout_binding_rpc_handler.attach_layout_to_artifact(
            request, context
        )

    def ListArtifactLayouts(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.layout_binding_rpc_handler.list_artifact_layouts(request, context)

    def GetAssemblyRuntimePolicy(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.layout_runtime_policy_rpc_handler.get_assembly_runtime_policy(
            request, context
        )

    def UpdateAssemblyRuntimePolicy(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.layout_runtime_policy_rpc_handler.update_assembly_runtime_policy(
            request, context
        )


class WorkflowOrchestrationRpcServicerMixin:
    """Workflow and lease RPC routing."""

    operation_rpc_handler: OperationRpcHandler
    placement_persistence_rpc_handler: PlacementPersistenceRpcHandler

    def AcquireOperationLease(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.operation_rpc_handler.acquire_operation_lease(request, context)

    def KeepaliveOperationLease(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.operation_rpc_handler.keepalive_operation_lease(request, context)

    def ReleaseOperationLease(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.operation_rpc_handler.release_operation_lease(request, context)

    def GetOperation(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.operation_rpc_handler.get_operation(request, context)

    def UpdateOperation(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.operation_rpc_handler.update_operation(request, context)

    def PlanPlacement(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.placement_persistence_rpc_handler.plan_placement(request, context)

    def ReportPersistenceStatus(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.placement_persistence_rpc_handler.report_persistence_status(
            request, context
        )


class ArtifactCatalogRpcServicerMixin:
    """Artifact catalog and addressing RPC routing."""

    artifact_binding_rpc_handler: ArtifactBindingRpcHandler
    artifact_index_rpc_handler: ArtifactIndexRpcHandler
    key_mapping_rpc_handler: KeyMappingRpcHandler
    disk_location_rpc_handler: DiskLocationRpcHandler

    def GetArtifactBinding(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.artifact_binding_rpc_handler.get_artifact_binding(request, context)

    def UpsertArtifactBinding(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.artifact_binding_rpc_handler.upsert_artifact_binding(
            request, context
        )

    def GetArtifactIndex(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.artifact_index_rpc_handler.get_artifact_index(request, context)

    def GetArtifactIndexById(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.artifact_index_rpc_handler.get_artifact_index_by_id(
            request, context
        )

    def UpsertKeyMapping(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.key_mapping_rpc_handler.upsert_key_mapping(request, context)

    def ResolveKeyMapping(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.key_mapping_rpc_handler.resolve_key_mapping(request, context)

    def SwapKeyMapping(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.key_mapping_rpc_handler.swap_key_mapping(request, context)

    def RevokeKeyMapping(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.key_mapping_rpc_handler.revoke_key_mapping(request, context)

    def UpsertArtifactDiskLocation(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.disk_location_rpc_handler.upsert_artifact_disk_location(
            request, context
        )

    def ListArtifactDiskLocations(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.disk_location_rpc_handler.list_artifact_disk_locations(
            request, context
        )


class ClusterRuntimeRpcServicerMixin:
    """Cluster runtime state RPC routing."""

    replica_registration_rpc_handler: ReplicaRegistrationRpcHandler
    replica_lifecycle_rpc_handler: ReplicaLifecycleRpcHandler
    transport_rpc_handler: TransportRpcHandler
    worker_rpc_handler: WorkerRpcHandler
    instance_rpc_handler: InstanceRpcHandler
    worker_state_sync_rpc_handler: WorkerStateSyncRpcHandler
    chunk_rpc_handler: ChunkRpcHandler

    def RegisterReplica(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_registration_rpc_handler.register_replica(request, context)

    def UpdateReplica(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_lifecycle_rpc_handler.update_replica(request, context)

    def UnregisterReplica(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_lifecycle_rpc_handler.unregister_replica(request, context)

    def UnregisterReplicaByWorker(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.replica_lifecycle_rpc_handler.unregister_replica_by_worker(
            request, context
        )

    def MarkReplicaUnavailable(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.replica_lifecycle_rpc_handler.mark_replica_unavailable(
            request, context
        )

    def WaitReplicaDrain(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_lifecycle_rpc_handler.wait_replica_drain(request, context)

    def ListReplicasV2(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_lifecycle_rpc_handler.list_replicas_v2(request, context)

    def BatchGetReplicaCounts(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.replica_lifecycle_rpc_handler.batch_get_replica_counts(
            request, context
        )

    def RequestReplicaTransport(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.transport_rpc_handler.request_replica_transport(request, context)

    def CompleteReplicaTransport(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.transport_rpc_handler.complete_replica_transport(request, context)

    def RegisterWorker(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.worker_rpc_handler.register_worker(request, context)

    def WorkerHeartbeat(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.worker_rpc_handler.worker_heartbeat(request, context)

    def UnregisterWorker(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.worker_rpc_handler.unregister_worker(request, context)

    def ListActiveWorkers(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.worker_rpc_handler.list_active_workers(request, context)

    def RegisterInstance(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.instance_rpc_handler.register_instance(request, context)

    def InstanceHeartbeat(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.instance_rpc_handler.instance_heartbeat(request, context)

    def UnregisterInstance(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.instance_rpc_handler.unregister_instance(request, context)

    def ListActiveInstances(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.instance_rpc_handler.list_active_instances(request, context)

    def ReconcileWorkerState(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.worker_state_sync_rpc_handler.reconcile_worker_state(
            request, context
        )

    def QueryChunkLocations(self, request: Any, context: grpc.ServicerContext) -> Any:
        return self.chunk_rpc_handler.query_chunk_locations(request, context)

    def BatchUpdateChunkStates(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        return self.chunk_rpc_handler.batch_update_chunk_states(request, context)
