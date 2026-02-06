#  Copyright (c) 2025-2026, TensorCast Team.

"""Coverage for runtime dependency rebuilds during servicer state reset."""


def test_reset_state_rebuilds_runtime_handler_dependencies(servicer):
    old_artifact_service = servicer.artifact_query_rpc_handler._artifact_service
    old_placement_service = (
        servicer.placement_persistence_rpc_handler._placement_service
    )
    old_worker_service = servicer.worker_instance_rpc_handler._worker_service
    old_recovery_service = servicer.worker_instance_rpc_handler._recovery_service

    servicer.reset_state()

    assert (
        servicer.artifact_query_rpc_handler._artifact_service
        is servicer.artifact_service
    )
    assert (
        servicer.artifact_query_rpc_handler._artifact_service
        is not old_artifact_service
    )

    assert (
        servicer.placement_persistence_rpc_handler._placement_service
        is servicer.placement_service
    )
    assert (
        servicer.placement_persistence_rpc_handler._placement_service
        is not old_placement_service
    )

    assert (
        servicer.worker_instance_rpc_handler._worker_service is servicer.worker_service
    )
    assert (
        servicer.worker_instance_rpc_handler._worker_service is not old_worker_service
    )
    assert (
        servicer.worker_instance_rpc_handler._recovery_service
        is servicer.recovery_service
    )
    assert (
        servicer.worker_instance_rpc_handler._recovery_service
        is not old_recovery_service
    )
