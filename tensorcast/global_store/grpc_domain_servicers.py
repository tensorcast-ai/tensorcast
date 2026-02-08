#  Copyright (c) 2025-2026, TensorCast Team.

"""Domain-oriented gRPC method mixins for Global Store services."""

from typing import Any

import grpc

from tensorcast.proto.global_store.v1 import global_store_pb2
from tensorcast.proto.operation.v1 import operation_pb2


class AssemblyViewRpcMixin:
    def GetArtifactInfoById(
        self: Any,
        request: global_store_pb2.GetArtifactInfoByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactInfoByIdResponse:
        return self.artifact_query_rpc_handler.get_artifact_info_by_id(request, context)

    def UpdateArtifactViewState(
        self: Any,
        request: global_store_pb2.UpdateArtifactViewStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateArtifactViewStateResponse:
        return self.view_proof_rpc_handler.update_artifact_view_state(request, context)

    def WriteTensorProofCommitments(
        self: Any,
        request: global_store_pb2.WriteTensorProofCommitmentsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WriteTensorProofCommitmentsResponse:
        return self.view_proof_rpc_handler.write_tensor_proof_commitments(
            request, context
        )

    def CheckProofCommitmentsMatch(
        self: Any,
        request: global_store_pb2.CheckProofCommitmentsMatchRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CheckProofCommitmentsMatchResponse:
        return self.view_proof_rpc_handler.check_proof_commitments_match(
            request, context
        )

    def ListViews(
        self: Any,
        request: global_store_pb2.ListViewsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListViewsResponse:
        return self.view_proof_rpc_handler.list_views(request, context)

    def PutLayoutSpec(
        self: Any,
        request: global_store_pb2.PutLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PutLayoutSpecResponse:
        return self.layout_spec_rpc_handler.put_layout_spec(request, context)

    def GetLayoutSpec(
        self: Any,
        request: global_store_pb2.GetLayoutSpecRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetLayoutSpecResponse:
        return self.layout_spec_rpc_handler.get_layout_spec(request, context)

    def GetAssemblyLayoutBinding(
        self: Any,
        request: global_store_pb2.GetAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyLayoutBindingResponse:
        return self.layout_binding_rpc_handler.get_assembly_layout_binding(
            request, context
        )

    def UpdateAssemblyLayoutBinding(
        self: Any,
        request: global_store_pb2.UpdateAssemblyLayoutBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyLayoutBindingResponse:
        return self.layout_binding_rpc_handler.update_assembly_layout_binding(
            request, context
        )

    def AttachLayoutToArtifact(
        self: Any,
        request: global_store_pb2.AttachLayoutToArtifactRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.AttachLayoutToArtifactResponse:
        return self.layout_binding_rpc_handler.attach_layout_to_artifact(
            request, context
        )

    def ListArtifactLayouts(
        self: Any,
        request: global_store_pb2.ListArtifactLayoutsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactLayoutsResponse:
        return self.layout_binding_rpc_handler.list_artifact_layouts(request, context)

    def GetAssemblyRuntimePolicy(
        self: Any,
        request: global_store_pb2.GetAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetAssemblyRuntimePolicyResponse:
        return self.layout_runtime_policy_rpc_handler.get_assembly_runtime_policy(
            request, context
        )

    def UpdateAssemblyRuntimePolicy(
        self: Any,
        request: global_store_pb2.UpdateAssemblyRuntimePolicyRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateAssemblyRuntimePolicyResponse:
        return self.layout_runtime_policy_rpc_handler.update_assembly_runtime_policy(
            request, context
        )


class WorkflowOrchestrationRpcMixin:
    def AcquireOperationLease(
        self: Any,
        request: operation_pb2.AcquireOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.AcquireOperationLeaseResponse:
        return self.operation_rpc_handler.acquire_operation_lease(request, context)

    def KeepaliveOperationLease(
        self: Any,
        request: operation_pb2.KeepaliveOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.KeepaliveOperationLeaseResponse:
        return self.operation_rpc_handler.keepalive_operation_lease(request, context)

    def ReleaseOperationLease(
        self: Any,
        request: operation_pb2.ReleaseOperationLeaseRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.ReleaseOperationLeaseResponse:
        return self.operation_rpc_handler.release_operation_lease(request, context)

    def GetOperation(
        self: Any,
        request: operation_pb2.GetOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.GetOperationResponse:
        return self.operation_rpc_handler.get_operation(request, context)

    def UpdateOperation(
        self: Any,
        request: operation_pb2.UpdateOperationRequest,
        context: grpc.ServicerContext,
    ) -> operation_pb2.UpdateOperationResponse:
        return self.operation_rpc_handler.update_operation(request, context)

    def PlanPlacement(
        self: Any,
        request: global_store_pb2.PlanPlacementRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.PlanPlacementResponse:
        return self.placement_persistence_rpc_handler.plan_placement(request, context)

    def ReportPersistenceStatus(
        self: Any,
        request: global_store_pb2.ReportPersistenceStatusRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ReportPersistenceStatusResponse:
        return self.placement_persistence_rpc_handler.report_persistence_status(
            request, context
        )


class ArtifactCatalogRpcMixin:
    def GetArtifactBinding(
        self: Any,
        request: global_store_pb2.GetArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactBindingResponse:
        return self.artifact_binding_rpc_handler.get_artifact_binding(request, context)

    def UpsertArtifactBinding(
        self: Any,
        request: global_store_pb2.UpsertArtifactBindingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactBindingResponse:
        return self.artifact_binding_rpc_handler.upsert_artifact_binding(
            request, context
        )

    def GetArtifactIndex(
        self: Any,
        request: global_store_pb2.GetArtifactIndexRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexResponse:
        return self.artifact_index_rpc_handler.get_artifact_index(request, context)

    def GetArtifactIndexById(
        self: Any,
        request: global_store_pb2.GetArtifactIndexByIdRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetArtifactIndexByIdResponse:
        return self.artifact_index_rpc_handler.get_artifact_index_by_id(
            request, context
        )

    def UpsertKeyMapping(
        self: Any,
        request: global_store_pb2.UpsertKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertKeyMappingResponse:
        return self.key_mapping_rpc_handler.upsert_key_mapping(request, context)

    def ResolveKeyMapping(
        self: Any,
        request: global_store_pb2.ResolveKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ResolveKeyMappingResponse:
        return self.key_mapping_rpc_handler.resolve_key_mapping(request, context)

    def SwapKeyMapping(
        self: Any,
        request: global_store_pb2.SwapKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SwapKeyMappingResponse:
        return self.key_mapping_rpc_handler.swap_key_mapping(request, context)

    def RevokeKeyMapping(
        self: Any,
        request: global_store_pb2.RevokeKeyMappingRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RevokeKeyMappingResponse:
        return self.key_mapping_rpc_handler.revoke_key_mapping(request, context)

    def UpsertArtifactDiskLocation(
        self: Any,
        request: global_store_pb2.UpsertArtifactDiskLocationRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpsertArtifactDiskLocationResponse:
        return self.disk_location_rpc_handler.upsert_artifact_disk_location(
            request, context
        )

    def ListArtifactDiskLocations(
        self: Any,
        request: global_store_pb2.ListArtifactDiskLocationsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListArtifactDiskLocationsResponse:
        return self.disk_location_rpc_handler.list_artifact_disk_locations(
            request, context
        )


class ClusterRuntimeRpcMixin:
    def RegisterReplica(
        self: Any,
        request: global_store_pb2.RegisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterReplicaResponse:
        return self.replica_registration_rpc_handler.register_replica(request, context)

    def UpdateReplica(
        self: Any,
        request: global_store_pb2.UpdateReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UpdateReplicaResponse:
        return self.replica_lifecycle_rpc_handler.update_replica(request, context)

    def UnregisterReplica(
        self: Any,
        request: global_store_pb2.UnregisterReplicaRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaResponse:
        return self.replica_lifecycle_rpc_handler.unregister_replica(request, context)

    def UnregisterReplicaByWorker(
        self: Any,
        request: global_store_pb2.UnregisterReplicaByWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterReplicaByWorkerResponse:
        return self.replica_lifecycle_rpc_handler.unregister_replica_by_worker(
            request, context
        )

    def MarkReplicaUnavailable(
        self: Any,
        request: global_store_pb2.MarkReplicaUnavailableRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.MarkReplicaUnavailableResponse:
        return self.replica_lifecycle_rpc_handler.mark_replica_unavailable(
            request, context
        )

    def WaitReplicaDrain(
        self: Any,
        request: global_store_pb2.WaitReplicaDrainRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WaitReplicaDrainResponse:
        return self.replica_lifecycle_rpc_handler.wait_replica_drain(request, context)

    def ListReplicasV2(
        self: Any,
        request: global_store_pb2.ListReplicasV2Request,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListReplicasV2Response:
        return self.replica_lifecycle_rpc_handler.list_replicas_v2(request, context)

    def RequestReplicaTransport(
        self: Any,
        request: global_store_pb2.RequestReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestReplicaTransportResponse:
        return self.transport_rpc_handler.request_replica_transport(request, context)

    def CompleteReplicaTransport(
        self: Any,
        request: global_store_pb2.CompleteReplicaTransportRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.CompleteReplicaTransportResponse:
        return self.transport_rpc_handler.complete_replica_transport(request, context)

    def RegisterWorker(
        self: Any,
        request: global_store_pb2.RegisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterWorkerResponse:
        return self.worker_rpc_handler.register_worker(request, context)

    def WorkerHeartbeat(
        self: Any,
        request: global_store_pb2.WorkerHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.WorkerHeartbeatResponse:
        return self.worker_rpc_handler.worker_heartbeat(request, context)

    def UnregisterWorker(
        self: Any,
        request: global_store_pb2.UnregisterWorkerRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterWorkerResponse:
        return self.worker_rpc_handler.unregister_worker(request, context)

    def ListActiveWorkers(
        self: Any,
        request: global_store_pb2.ListActiveWorkersRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveWorkersResponse:
        return self.worker_rpc_handler.list_active_workers(request, context)

    def RegisterInstance(
        self: Any,
        request: global_store_pb2.RegisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RegisterInstanceResponse:
        return self.instance_rpc_handler.register_instance(request, context)

    def InstanceHeartbeat(
        self: Any,
        request: global_store_pb2.InstanceHeartbeatRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.InstanceHeartbeatResponse:
        return self.instance_rpc_handler.instance_heartbeat(request, context)

    def UnregisterInstance(
        self: Any,
        request: global_store_pb2.UnregisterInstanceRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.UnregisterInstanceResponse:
        return self.instance_rpc_handler.unregister_instance(request, context)

    def ListActiveInstances(
        self: Any,
        request: global_store_pb2.ListActiveInstancesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.ListActiveInstancesResponse:
        return self.instance_rpc_handler.list_active_instances(request, context)

    def SynchronizeWorkerState(
        self: Any,
        request: global_store_pb2.SynchronizeWorkerStateRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.SynchronizeWorkerStateResponse:
        return self.worker_state_sync_rpc_handler.synchronize_worker_state(
            request, context
        )

    def RequestFullStateSync(
        self: Any,
        request: global_store_pb2.RequestFullStateSyncRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.RequestFullStateSyncResponse:
        return self.worker_state_sync_rpc_handler.request_full_state_sync(
            request, context
        )

    def QueryChunkLocations(
        self: Any,
        request: global_store_pb2.QueryChunkLocationsRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.QueryChunkLocationsResponse:
        return self.chunk_rpc_handler.query_chunk_locations(request, context)

    def BatchUpdateChunkStates(
        self: Any,
        request: global_store_pb2.BatchUpdateChunkStatesRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.BatchUpdateChunkStatesResponse:
        return self.chunk_rpc_handler.batch_update_chunk_states(request, context)


class ClusterAdminRpcMixin:
    def HealthCheck(
        self: Any,
        request: global_store_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.HealthCheckResponse:
        """Minimal health check endpoint (status + cluster token)."""
        info = getattr(self, "_runtime_info", {}) or {}
        cluster_token = info.get("cluster_token") or self.config.cluster_token
        return global_store_pb2.HealthCheckResponse(
            status=global_store_pb2.Status.STATUS_OK,
            cluster_token=cluster_token or "",
        )

    def GetServerInfo(
        self: Any,
        request: global_store_pb2.GetServerInfoRequest,
        context: grpc.ServicerContext,
    ) -> global_store_pb2.GetServerInfoResponse:
        """Server metadata endpoint for advertised/bind addresses and diagnostics."""
        info = getattr(self, "_runtime_info", {}) or {}
        listen_host = info.get("listen_host") or self.config.listen_host
        listen_port = info.get("listen_port") or self.config.listen_port
        advertise_host = info.get("advertise_host") or self.config.advertise_host
        advertise_port = info.get("advertise_port") or self.config.advertise_port
        metrics_port = info.get("metrics_port") or self.config.metrics_port
        db_file = info.get("db_file") or (
            str(self.config.db_file) if self.config.db_file else ""
        )
        cluster_id = info.get("cluster_id") or self.cluster_id or ""
        version = info.get("version") or ""
        listen_address = (
            f"{listen_host}:{listen_port}" if listen_host and listen_port else ""
        )
        advertise_address = (
            f"{advertise_host}:{advertise_port}"
            if advertise_host and advertise_port
            else ""
        )
        return global_store_pb2.GetServerInfoResponse(
            status=global_store_pb2.Status.STATUS_OK,
            advertise_address=advertise_address,
            advertise_host=advertise_host or "",
            advertise_port=int(advertise_port or 0),
            listen_address=listen_address,
            listen_host=listen_host or "",
            listen_port=int(listen_port or 0),
            metrics_port=int(metrics_port or 0),
            version=version,
            db_file=db_file or "",
            cluster_id=cluster_id,
        )
