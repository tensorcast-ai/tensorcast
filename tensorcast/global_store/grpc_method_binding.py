#  Copyright (c) 2025-2026, TensorCast Team.

"""Dynamic RPC method binding for Global Store gRPC servicer."""

from __future__ import annotations

from typing import Any, Callable

# method_name -> (handler_attribute, handler_method)
_RPC_METHOD_BINDINGS: dict[str, tuple[str, str]] = {
    # AssemblyViewService
    "GetArtifactInfoById": ("artifact_query_rpc_handler", "get_artifact_info_by_id"),
    "UpdateArtifactViewState": (
        "view_proof_rpc_handler",
        "update_artifact_view_state",
    ),
    "WriteTensorProofCommitments": (
        "view_proof_rpc_handler",
        "write_tensor_proof_commitments",
    ),
    "CheckProofCommitmentsMatch": (
        "view_proof_rpc_handler",
        "check_proof_commitments_match",
    ),
    "ListViews": ("view_proof_rpc_handler", "list_views"),
    "PutLayoutSpec": ("layout_spec_rpc_handler", "put_layout_spec"),
    "GetLayoutSpec": ("layout_spec_rpc_handler", "get_layout_spec"),
    "GetAssemblyLayoutBinding": (
        "layout_binding_rpc_handler",
        "get_assembly_layout_binding",
    ),
    "UpdateAssemblyLayoutBinding": (
        "layout_binding_rpc_handler",
        "update_assembly_layout_binding",
    ),
    "AttachLayoutToArtifact": (
        "layout_binding_rpc_handler",
        "attach_layout_to_artifact",
    ),
    "ListArtifactLayouts": ("layout_binding_rpc_handler", "list_artifact_layouts"),
    "GetAssemblyRuntimePolicy": (
        "layout_runtime_policy_rpc_handler",
        "get_assembly_runtime_policy",
    ),
    "UpdateAssemblyRuntimePolicy": (
        "layout_runtime_policy_rpc_handler",
        "update_assembly_runtime_policy",
    ),
    # WorkflowOrchestrationService
    "AcquireOperationLease": ("operation_rpc_handler", "acquire_operation_lease"),
    "KeepaliveOperationLease": ("operation_rpc_handler", "keepalive_operation_lease"),
    "ReleaseOperationLease": ("operation_rpc_handler", "release_operation_lease"),
    "GetOperation": ("operation_rpc_handler", "get_operation"),
    "UpdateOperation": ("operation_rpc_handler", "update_operation"),
    "PlanPlacement": ("placement_persistence_rpc_handler", "plan_placement"),
    "ReportPersistenceStatus": (
        "placement_persistence_rpc_handler",
        "report_persistence_status",
    ),
    # ArtifactCatalogService
    "GetArtifactBinding": ("artifact_binding_rpc_handler", "get_artifact_binding"),
    "UpsertArtifactBinding": (
        "artifact_binding_rpc_handler",
        "upsert_artifact_binding",
    ),
    "GetArtifactIndex": ("artifact_index_rpc_handler", "get_artifact_index"),
    "GetArtifactIndexById": ("artifact_index_rpc_handler", "get_artifact_index_by_id"),
    "UpsertKeyMapping": ("key_mapping_rpc_handler", "upsert_key_mapping"),
    "ResolveKeyMapping": ("key_mapping_rpc_handler", "resolve_key_mapping"),
    "SwapKeyMapping": ("key_mapping_rpc_handler", "swap_key_mapping"),
    "RevokeKeyMapping": ("key_mapping_rpc_handler", "revoke_key_mapping"),
    "UpsertArtifactDiskLocation": (
        "disk_location_rpc_handler",
        "upsert_artifact_disk_location",
    ),
    "ListArtifactDiskLocations": (
        "disk_location_rpc_handler",
        "list_artifact_disk_locations",
    ),
    # ClusterRuntimeService
    "RegisterReplica": ("replica_registration_rpc_handler", "register_replica"),
    "UpdateReplica": ("replica_lifecycle_rpc_handler", "update_replica"),
    "UnregisterReplica": ("replica_lifecycle_rpc_handler", "unregister_replica"),
    "UnregisterReplicaByWorker": (
        "replica_lifecycle_rpc_handler",
        "unregister_replica_by_worker",
    ),
    "MarkReplicaUnavailable": (
        "replica_lifecycle_rpc_handler",
        "mark_replica_unavailable",
    ),
    "WaitReplicaDrain": ("replica_lifecycle_rpc_handler", "wait_replica_drain"),
    "ListReplicasV2": ("replica_lifecycle_rpc_handler", "list_replicas_v2"),
    "RequestReplicaTransport": (
        "transport_rpc_handler",
        "request_replica_transport",
    ),
    "CompleteReplicaTransport": (
        "transport_rpc_handler",
        "complete_replica_transport",
    ),
    "RegisterWorker": ("worker_rpc_handler", "register_worker"),
    "WorkerHeartbeat": ("worker_rpc_handler", "worker_heartbeat"),
    "UnregisterWorker": ("worker_rpc_handler", "unregister_worker"),
    "ListActiveWorkers": ("worker_rpc_handler", "list_active_workers"),
    "RegisterInstance": ("instance_rpc_handler", "register_instance"),
    "InstanceHeartbeat": ("instance_rpc_handler", "instance_heartbeat"),
    "UnregisterInstance": ("instance_rpc_handler", "unregister_instance"),
    "ListActiveInstances": ("instance_rpc_handler", "list_active_instances"),
    "SynchronizeWorkerState": (
        "worker_state_sync_rpc_handler",
        "synchronize_worker_state",
    ),
    "RequestFullStateSync": (
        "worker_state_sync_rpc_handler",
        "request_full_state_sync",
    ),
    "QueryChunkLocations": ("chunk_rpc_handler", "query_chunk_locations"),
    "BatchUpdateChunkStates": ("chunk_rpc_handler", "batch_update_chunk_states"),
}


def _make_delegate(
    servicer: Any, handler_attribute: str, handler_method: str
) -> Callable[[Any, Any], Any]:
    def _delegate(request: Any, context: Any) -> Any:
        return getattr(getattr(servicer, handler_attribute), handler_method)(
            request, context
        )

    return _delegate


def bind_global_store_rpc_methods(servicer: Any) -> None:
    """Bind RPC method names on a servicer instance to handler methods."""
    for rpc_method, (handler_attr, method_name) in _RPC_METHOD_BINDINGS.items():
        setattr(
            servicer, rpc_method, _make_delegate(servicer, handler_attr, method_name)
        )
