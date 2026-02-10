#  Copyright (c) 2025-2026, TensorCast Team.

"""Coverage for runtime dependency rebuilds during servicer state reset."""

import threading


def test_reset_state_rebuilds_runtime_handler_dependencies(servicer):
    old_artifact_service = servicer.artifact_query_rpc_handler._artifact_service
    old_placement_service = (
        servicer.placement_persistence_rpc_handler._placement_service
    )
    old_worker_service = servicer.worker_rpc_handler._worker_service
    old_recovery_service = servicer.worker_state_sync_rpc_handler._recovery_service
    old_instance_service = servicer.instance_rpc_handler._instance_service

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

    assert servicer.worker_rpc_handler._worker_service is servicer.worker_service
    assert servicer.worker_rpc_handler._worker_service is not old_worker_service
    assert (
        servicer.worker_state_sync_rpc_handler._recovery_service
        is servicer.recovery_service
    )
    assert (
        servicer.worker_state_sync_rpc_handler._recovery_service
        is not old_recovery_service
    )
    assert servicer.instance_rpc_handler._instance_service is servicer.instance_service
    assert servicer.instance_rpc_handler._instance_service is not old_instance_service


def test_reset_state_does_not_create_duplicate_worker_reducer_threads(servicer):
    thread_prefix = "worker-control-reducer-"
    before = len(
        [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith(thread_prefix) and thread.is_alive()
        ]
    )
    assert before >= 1
    for _ in range(3):
        servicer.reset_state()
    after = len(
        [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith(thread_prefix) and thread.is_alive()
        ]
    )
    assert after == before
