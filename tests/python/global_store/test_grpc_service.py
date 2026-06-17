#  Copyright (c) 2025-2026, TensorCast Team.

"""Tests for Global Store gRPC service interface."""

import base64
import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import grpc
from google.protobuf import duration_pb2, timestamp_pb2, wrappers_pb2

from tensorcast.global_store.config.settings import (
    GlobalStoreConfig,
    get_config,
    set_config,
)
from tensorcast.global_store.exceptions import DatabaseError
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class TestGRPCService:
    """Tests for the gRPC service interface."""

    def test_update_artifact_replica(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test updating a artifact replica"""
        # First register a replica
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="test_artifact",
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterReplica(register_request, test_context)

        # Now update it
        update_request = global_store_pb2.UpdateReplicaRequest(
            artifact_id="test_artifact", replica_id=register_response.replica_id
        )

        update_response = servicer.UpdateReplica(update_request, test_context)

        assert update_response.status == global_store_pb2.Status.STATUS_OK
        assert update_response.artifact_id == "test_artifact"
        assert update_response.replica_id == register_response.replica_id

    def test_update_nonexistent_artifact_replica(self, servicer, test_context):
        """Test updating a artifact replica that doesn't exist"""
        request = global_store_pb2.UpdateReplicaRequest(
            artifact_id="nonexistent_artifact", replica_id=str(uuid.uuid4())
        )

        response = servicer.UpdateReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_update_replica_rejects_msa1(self, servicer, test_context):
        request = global_store_pb2.UpdateReplicaRequest(
            artifact_id="msa1:test-session~policy~partitioned~deadbeef",
            replica_id=str(uuid.uuid4()),
        )

        response = servicer.UpdateReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "daemon-session-local" in (test_context.details or "")

    def test_register_replica_rejects_non_v3_schema(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """schema_version must be v3 for replica registration."""
        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:index:data",
            mem_info=memory_info,
            max_concurrency=5,
            worker_id=registered_worker,
            schema_version="v2",
        )

        response = servicer.RegisterReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT

    def test_register_replica_rejects_msa1(
        self, servicer, test_context, memory_info, registered_worker
    ):
        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="msa1:test-session~policy~partitioned~deadbeef",
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
        )

        response = servicer.RegisterReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "daemon-session-local" in (test_context.details or "")

    def test_register_replica_idempotent_replay(
        self, servicer, test_context, memory_info, registered_worker, monkeypatch
    ):
        calls = {"count": 0}
        original = servicer.artifact_service.register_replica

        def _wrapped(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(servicer.artifact_service, "register_replica", _wrapped)

        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:replay:index",
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
            client_request_id="register-replay-1",
        )
        first = servicer.RegisterReplica(request, test_context)
        assert first.status == global_store_pb2.Status.STATUS_OK

        test_context.code = None
        test_context.details = None
        second = servicer.RegisterReplica(request, test_context)
        assert second.status == global_store_pb2.Status.STATUS_OK
        assert second.replica_id == first.replica_id
        assert calls["count"] == 1

    def test_register_replica_idempotent_payload_mismatch(
        self, servicer, test_context, memory_info, registered_worker
    ):
        first = global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:mismatch:index",
            mem_info=memory_info,
            max_concurrency=5,
            worker_id=registered_worker,
            client_request_id="register-mismatch-1",
        )
        second = global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:mismatch:changed",
            mem_info=memory_info,
            max_concurrency=5,
            worker_id=registered_worker,
            client_request_id="register-mismatch-1",
        )

        first_resp = servicer.RegisterReplica(first, test_context)
        assert first_resp.status == global_store_pb2.Status.STATUS_OK

        test_context.code = None
        test_context.details = None
        second_resp = servicer.RegisterReplica(second, test_context)
        assert second_resp.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION

    def test_register_replica_routes_reducer_by_artifact(
        self, servicer, test_context, memory_info, registered_worker, monkeypatch
    ):
        reducer = servicer.worker_control_reducer
        captured = {"worker_key": ""}
        original_submit = reducer.submit

        def _wrapped(*, worker_key, kind, operation, timeout_s=None):
            captured["worker_key"] = worker_key
            return original_submit(
                worker_key=worker_key,
                kind=kind,
                operation=operation,
                timeout_s=timeout_s,
            )

        monkeypatch.setattr(reducer, "submit", _wrapped)

        artifact_id = "mi2:lane:index"
        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=2,
            worker_id=registered_worker,
        )
        resp = servicer.RegisterReplica(request, test_context)
        assert resp.status == global_store_pb2.Status.STATUS_OK
        assert captured["worker_key"] == f"artifact:{artifact_id}"

    def test_register_replica_tx_conflict_fails_fast(
        self, servicer, test_context, memory_info, registered_worker, monkeypatch
    ):
        calls = {"count": 0}

        def _raise_conflict(*args, **kwargs):
            calls["count"] += 1
            raise DatabaseError("write-write conflict on key artifacts.artifact_id")

        monkeypatch.setattr(
            servicer.artifact_service,
            "register_replica",
            _raise_conflict,
        )
        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="mi2:conflict:index",
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
            client_request_id="register-conflict-1",
        )
        response = servicer.RegisterReplica(request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.ABORTED
        assert calls["count"] == 1

    def test_unregister_artifact_replica(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test unregistering a artifact replica"""
        # First register a replica
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="test_artifact",
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterReplica(register_request, test_context)

        # Now unregister it
        unregister_request = global_store_pb2.UnregisterReplicaRequest(
            artifact_id="test_artifact", replica_id=register_response.replica_id
        )

        unregister_response = servicer.UnregisterReplica(
            unregister_request, test_context
        )

        assert unregister_response.status == global_store_pb2.Status.STATUS_OK

    def test_unregister_nonexistent_artifact_replica(self, servicer, test_context):
        """Test unregistering a artifact replica that doesn't exist"""
        request = global_store_pb2.UnregisterReplicaRequest(
            artifact_id="nonexistent_artifact", replica_id=str(uuid.uuid4())
        )

        response = servicer.UnregisterReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_unregister_replica_rejects_msa1(self, servicer, test_context):
        request = global_store_pb2.UnregisterReplicaRequest(
            artifact_id="msa1:test-session~policy~partitioned~deadbeef",
            replica_id=str(uuid.uuid4()),
        )

        response = servicer.UnregisterReplica(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "daemon-session-local" in (test_context.details or "")

    def test_mark_replica_unavailable_idempotent(
        self, servicer, test_context, memory_info, registered_worker
    ):
        register_response = servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_unavailable",
                mem_info=memory_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        replica_id = register_response.replica_id
        first = servicer.MarkReplicaUnavailable(
            global_store_pb2.MarkReplicaUnavailableRequest(
                artifact_id="test_artifact_unavailable", replica_id=replica_id
            ),
            test_context,
        )
        assert first.status == global_store_pb2.Status.STATUS_OK
        assert first.updated is True

        second = servicer.MarkReplicaUnavailable(
            global_store_pb2.MarkReplicaUnavailableRequest(
                artifact_id="test_artifact_unavailable", replica_id=replica_id
            ),
            test_context,
        )
        assert second.status == global_store_pb2.Status.STATUS_OK
        assert second.updated is True

    def test_mark_replica_unavailable_rejects_msa1(self, servicer, test_context):
        response = servicer.MarkReplicaUnavailable(
            global_store_pb2.MarkReplicaUnavailableRequest(
                artifact_id="msa1:test-session~policy~partitioned~deadbeef",
                replica_id=str(uuid.uuid4()),
            ),
            test_context,
        )

        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert response.updated is False
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "daemon-session-local" in (test_context.details or "")

    def test_wait_replica_drain_timeout_snapshot(
        self, servicer, test_context, memory_info, registered_worker
    ):
        register_response = servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_drain",
                mem_info=memory_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        replica_id = register_response.replica_id

        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )

        transport_response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_drain",
                local_memory_info=memory_info,
                wait_timeout_dur=duration_pb2.Duration(seconds=1),
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-drain-1",
            ),
            test_context,
        )
        assert transport_response.status == global_store_pb2.Status.STATUS_OK

        drain_response = servicer.WaitReplicaDrain(
            global_store_pb2.WaitReplicaDrainRequest(
                replica_id=replica_id, timeout_ms=1
            ),
            test_context,
        )
        assert drain_response.status == global_store_pb2.Status.STATUS_TIMED_OUT
        assert drain_response.drained is False
        assert drain_response.current_requests == 1
        assert drain_response.HasField("oldest_transport_age_ms")

        current = servicer.replica_repository.get_current_requests(
            uuid.UUID(replica_id)
        )
        assert current == 1

        servicer.CompleteReplicaTransport(
            global_store_pb2.CompleteReplicaTransportRequest(
                transport_id=transport_response.transport_id
            ),
            test_context,
        )

    def test_worker_capability_flags_filtering(self, servicer, test_context):
        flags = (1 << global_store_pb2.WORKER_CAPABILITY_FLAG_QUEUE_BROKER_ENABLED) | (
            1 << global_store_pb2.WORKER_CAPABILITY_FLAG_RETENTION_HANDLES_ENABLED
        )
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="cap_worker",
            node_address="192.168.2.10",
            grpc_port=8010,
            p2p_port=8011,
            mem_pool_total_size=1000000000,
            mem_pool_available_size=900000000,
            daemon_id="daemon_cap_worker",
            capability_flags=flags,
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        list_response = servicer.ListActiveWorkers(
            global_store_pb2.ListActiveWorkersRequest(
                required_capability_flags=(
                    1 << global_store_pb2.WORKER_CAPABILITY_FLAG_QUEUE_BROKER_ENABLED
                )
            ),
            test_context,
        )
        assert any(
            w.worker_id == register_response.worker_id for w in list_response.workers
        )

        list_response = servicer.ListActiveWorkers(
            global_store_pb2.ListActiveWorkersRequest(
                required_capability_flags=(
                    1
                    << global_store_pb2.WORKER_CAPABILITY_FLAG_CAPABILITY_TOKENS_V2_ENABLED
                )
            ),
            test_context,
        )
        assert all(
            w.worker_id != register_response.worker_id for w in list_response.workers
        )

    def test_worker_routing_capability_flags_filtering(self, servicer, test_context):
        flags = (
            1 << global_store_pb2.WORKER_CAPABILITY_FLAG_GATEWAY_INGRESS_ENABLED
        ) | (1 << global_store_pb2.WORKER_CAPABILITY_FLAG_SHARD_HOME_ELIGIBLE)
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="routing_cap_worker",
            node_address="192.168.2.30",
            grpc_port=8030,
            p2p_port=8031,
            mem_pool_total_size=1000000000,
            mem_pool_available_size=900000000,
            daemon_id="daemon_routing_cap_worker",
            capability_flags=flags,
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        list_response = servicer.ListActiveWorkers(
            global_store_pb2.ListActiveWorkersRequest(
                required_capability_flags=(
                    1 << global_store_pb2.WORKER_CAPABILITY_FLAG_SHARD_HOME_ELIGIBLE
                )
            ),
            test_context,
        )
        assert any(
            w.worker_id == register_response.worker_id for w in list_response.workers
        )

    def test_instance_capability_flags_filtering(self, servicer, test_context):
        flags = 1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED
        register_request = global_store_pb2.RegisterInstanceRequest(
            instance_id="instance-cap-1",
            daemon_id="daemon-cap-instance",
            engine="test",
            signals_endpoint="ipc://signals",
            capability_flags=flags,
        )
        register_response = servicer.RegisterInstance(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        list_response = servicer.ListActiveInstances(
            global_store_pb2.ListActiveInstancesRequest(
                required_capability_flags=flags
            ),
            test_context,
        )
        assert any(
            inst.instance_id == register_request.instance_id
            for inst in list_response.instances
        )

        list_response = servicer.ListActiveInstances(
            global_store_pb2.ListActiveInstancesRequest(
                required_capability_flags=(
                    1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED
                )
            ),
            test_context,
        )
        assert all(
            inst.instance_id != register_request.instance_id
            for inst in list_response.instances
        )

    def test_worker_capability_flags_clear_on_heartbeat(self, servicer, test_context):
        flags = (1 << global_store_pb2.WORKER_CAPABILITY_FLAG_QUEUE_BROKER_ENABLED) | (
            1 << global_store_pb2.WORKER_CAPABILITY_FLAG_RETENTION_HANDLES_ENABLED
        )
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="cap_worker_clear",
            node_address="192.168.2.20",
            grpc_port=8020,
            p2p_port=8021,
            mem_pool_total_size=1000000000,
            mem_pool_available_size=900000000,
            daemon_id="daemon_cap_worker_clear",
            capability_flags=flags,
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=register_response.worker_id,
                mem_pool_available_size=900000000,
                accepting_new_requests=True,
                state_version=1,
                capability_flags=0,
            ),
            test_context,
        )

        list_response = servicer.ListActiveWorkers(
            global_store_pb2.ListActiveWorkersRequest(required_capability_flags=flags),
            test_context,
        )
        assert all(
            worker.worker_id != register_response.worker_id
            for worker in list_response.workers
        )

    def test_instance_capability_flags_clear_on_heartbeat(self, servicer, test_context):
        flags = 1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED
        register_request = global_store_pb2.RegisterInstanceRequest(
            instance_id="instance-cap-clear-1",
            daemon_id="daemon-cap-instance-clear",
            engine="test",
            signals_endpoint="ipc://signals",
            capability_flags=flags,
        )
        register_response = servicer.RegisterInstance(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        servicer.InstanceHeartbeat(
            global_store_pb2.InstanceHeartbeatRequest(
                instance_id=register_request.instance_id,
                capability_flags=0,
            ),
            test_context,
        )

        list_response = servicer.ListActiveInstances(
            global_store_pb2.ListActiveInstancesRequest(
                required_capability_flags=flags
            ),
            test_context,
        )
        assert all(
            inst.instance_id != register_request.instance_id
            for inst in list_response.instances
        )

    def test_list_replicas(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test listing artifact replicas"""
        # Register multiple replicas
        artifact_ids = ["model1", "model2", "model3"]
        for artifact_id in artifact_ids:
            request = global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=memory_info,
                max_concurrency=10,
                worker_id=registered_worker,
            )
            servicer.RegisterReplica(request, test_context)

        # List all replicas
        list_request = global_store_pb2.ListReplicasV2Request()
        list_response = servicer.ListReplicasV2(list_request, test_context)

        # Convert flat records to dict for assertions
        result = {}
        for rec in list_response.replicas:
            result.setdefault(rec.artifact_id, []).append(rec.memory_info)

        assert len(result) >= 3
        for artifact_id in artifact_ids:
            assert artifact_id in result

    def test_list_replicas_with_filter(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test listing artifact replicas with a filter"""
        # Register a specific artifact
        artifact_id = "filtered_artifact"
        request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(request, test_context)

        # List replicas with filter
        list_request = global_store_pb2.ListReplicasV2Request(artifact_id=artifact_id)
        list_response = servicer.ListReplicasV2(list_request, test_context)

        assert sum(1 for _ in list_response.replicas) == 1
        assert list_response.replicas[0].artifact_id == artifact_id

    def test_batch_get_replica_counts(self, servicer, test_context, memory_info):
        worker_one = servicer.RegisterWorker(
            global_store_pb2.RegisterWorkerRequest(
                node_id="batch-count-node-1",
                node_address="192.168.10.1",
                grpc_port=8101,
                p2p_port=8201,
                mem_pool_total_size=10000000000,
                mem_pool_available_size=9000000000,
                daemon_id="batch-count-daemon-1",
            ),
            test_context,
        ).worker_id
        worker_two = servicer.RegisterWorker(
            global_store_pb2.RegisterWorkerRequest(
                node_id="batch-count-node-2",
                node_address="192.168.10.2",
                grpc_port=8102,
                p2p_port=8202,
                mem_pool_total_size=10000000000,
                mem_pool_available_size=9000000000,
                daemon_id="batch-count-daemon-2",
            ),
            test_context,
        ).worker_id

        artifact_one = "batch-count-artifact-1"
        artifact_two = "batch-count-artifact-2"
        artifact_three = "batch-count-artifact-3"

        mem_info_worker_one = common_pb2.MemoryInfo()
        mem_info_worker_one.CopyFrom(memory_info)
        mem_info_worker_one.node_id = "batch-count-node-1"
        mem_info_worker_one.node_address = "192.168.10.1"
        mem_info_worker_one.node_port = 8201
        mem_info_worker_one.device_id = 0

        mem_info_worker_two = common_pb2.MemoryInfo()
        mem_info_worker_two.CopyFrom(memory_info)
        mem_info_worker_two.node_id = "batch-count-node-2"
        mem_info_worker_two.node_address = "192.168.10.2"
        mem_info_worker_two.node_port = 8202
        mem_info_worker_two.device_id = 1

        register_one = servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_one,
                mem_info=mem_info_worker_one,
                max_concurrency=2,
                worker_id=worker_one,
            ),
            test_context,
        )
        assert register_one.status == global_store_pb2.Status.STATUS_OK

        register_two = servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_two,
                mem_info=mem_info_worker_one,
                max_concurrency=2,
                worker_id=worker_one,
            ),
            test_context,
        )
        assert register_two.status == global_store_pb2.Status.STATUS_OK

        register_three = servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_two,
                mem_info=mem_info_worker_two,
                max_concurrency=2,
                worker_id=worker_two,
            ),
            test_context,
        )
        assert register_three.status == global_store_pb2.Status.STATUS_OK

        mark_unavailable = servicer.MarkReplicaUnavailable(
            global_store_pb2.MarkReplicaUnavailableRequest(
                artifact_id=artifact_two,
                replica_id=register_two.replica_id,
            ),
            test_context,
        )
        assert mark_unavailable.status == global_store_pb2.Status.STATUS_OK
        assert mark_unavailable.updated is True

        response = servicer.BatchGetReplicaCounts(
            global_store_pb2.BatchGetReplicaCountsRequest(
                artifact_ids=[
                    artifact_one,
                    artifact_two,
                    artifact_three,
                    artifact_one,
                ]
            ),
            test_context,
        )
        assert response.status == global_store_pb2.Status.STATUS_OK
        assert [row.artifact_id for row in response.counts] == [
            artifact_one,
            artifact_two,
            artifact_three,
        ]
        counts = {
            row.artifact_id: (int(row.replica_count), int(row.available_count))
            for row in response.counts
        }
        assert counts[artifact_one] == (1, 1)
        assert counts[artifact_two] == (2, 1)
        assert counts[artifact_three] == (0, 0)

    def test_request_artifact_replica_transport(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test requesting a artifact replica transport"""
        # First register a replica
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="test_artifact",
            mem_info=memory_info,
            max_concurrency=3,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(register_request, test_context)

        # Send worker heartbeat to make it available
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=registered_worker,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
            state_version=1,
        )
        servicer.WorkerHeartbeat(heartbeat_request, test_context)

        # Now request transport
        transport_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="test_artifact",
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-request-basic-1",
        )

        transport_response = servicer.RequestReplicaTransport(
            transport_request, test_context
        )

        assert transport_response.status == global_store_pb2.Status.STATUS_OK
        assert transport_response.remote_memory_info is not None
        assert transport_response.transport_id is not None

    def test_request_replica_transport_requires_request_id(
        self, servicer, test_context, memory_info
    ):
        """Transport request must include explicit idempotency request_id."""
        response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_requires_request_id",
                local_memory_info=memory_info,
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
            ),
            test_context,
        )
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "request_id is required" in (test_context.details or "")

    def test_request_replica_transport_rejects_invalid_scheduling_group_fields(
        self, servicer, test_context
    ):
        """Scheduling group requires non-empty group_id/group_kind/part_id."""
        response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_invalid_group_fields",
                request_id="transport-invalid-group-fields-1",
                scheduling_group=global_store_pb2.TransportSchedulingGroup(
                    group_id="",
                    group_kind="tp_rank",
                    total_parts=1,
                    part_id="part-0",
                ),
            ),
            test_context,
        )
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "requires non-empty group_id/group_kind/part_id" in (
            test_context.details or ""
        )

    def test_request_replica_transport_rejects_nonpositive_group_total_parts(
        self, servicer, test_context
    ):
        """Scheduling group total_parts must be > 0."""
        response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_invalid_group_total_parts",
                request_id="transport-invalid-group-total-parts-1",
                scheduling_group=global_store_pb2.TransportSchedulingGroup(
                    group_id="group-a",
                    group_kind="tp_rank",
                    total_parts=0,
                    part_id="part-0",
                ),
            ),
            test_context,
        )
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "scheduling_group.total_parts must be > 0" in (
            test_context.details or ""
        )

    def test_request_replica_transport_request_id_replay(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Duplicate transport request_id should replay the original transport assignment."""
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_request_id_replay",
                mem_info=memory_info,
                max_concurrency=3,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )

        first_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="test_artifact_request_id_replay",
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-request-replay-1",
        )
        first_response = servicer.RequestReplicaTransport(first_request, test_context)
        assert first_response.status == global_store_pb2.Status.STATUS_OK

        second_response = servicer.RequestReplicaTransport(first_request, test_context)
        assert second_response.status == global_store_pb2.Status.STATUS_OK
        assert second_response.transport_id == first_response.transport_id

    def test_request_replica_transport_request_id_payload_mismatch_rejected(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Same request_id with different payload should be rejected."""
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_request_id_conflict",
                mem_info=memory_info,
                max_concurrency=3,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )

        response_a = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_request_id_conflict",
                local_memory_info=memory_info,
                source_node_id="source_node_a",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-request-conflict-1",
            ),
            test_context,
        )
        assert response_a.status == global_store_pb2.Status.STATUS_OK

        response_b = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_request_id_conflict",
                local_memory_info=memory_info,
                source_node_id="source_node_b",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-request-conflict-1",
            ),
            test_context,
        )
        assert response_b.status == global_store_pb2.Status.STATUS_ERROR

    def test_request_replica_transport_normalizes_requester_worker_id(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """requester_worker_id should be trimmed and empty values normalized to None."""
        artifact_id = "test_artifact_requester_worker_id_normalization"
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=memory_info,
                max_concurrency=4,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )

        whitespace_response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id=artifact_id,
                local_memory_info=memory_info,
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-requester-worker-whitespace-1",
                requester_worker_id="   ",
            ),
            test_context,
        )
        assert whitespace_response.status == global_store_pb2.Status.STATUS_OK
        whitespace_row = servicer.transport_repository.find_by_id(
            uuid.UUID(whitespace_response.transport_id)
        )
        assert whitespace_row is not None
        assert whitespace_row.requester_worker_id is None

        trimmed_response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id=artifact_id,
                local_memory_info=memory_info,
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-requester-worker-trimmed-1",
                requester_worker_id="  worker-7  ",
            ),
            test_context,
        )
        assert trimmed_response.status == global_store_pb2.Status.STATUS_OK
        trimmed_row = servicer.transport_repository.find_by_id(
            uuid.UUID(trimmed_response.transport_id)
        )
        assert trimmed_row is not None
        assert trimmed_row.requester_worker_id == "worker-7"

    def test_request_replica_transport_respects_byte_space(
        self, servicer, test_context, registered_worker
    ):
        artifact_id = "artifact-bytespace"

        canonical_info = common_pb2.MemoryInfo(
            node_id=str(uuid.uuid4()),
            node_address="192.168.1.10",
            node_port=8010,
            memory_size=1024,
            memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
            device_id=0,
            byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL, id=""
            ),
        )
        canonical_transport = canonical_info.transport
        canonical_transport.export_state = (
            common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
        )
        canonical_transport.export_generation = 1
        canonical_transport.remote_memory_keys.append("rk0")
        canonical_transport.buffer_sizes.append(canonical_info.memory_size)
        view_id = "view-1"
        view_info = common_pb2.MemoryInfo(
            node_id=str(uuid.uuid4()),
            node_address="192.168.1.11",
            node_port=8011,
            memory_size=256,
            memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
            device_id=0,
            byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW, id=view_id
            ),
        )
        view_transport = view_info.transport
        view_transport.export_state = (
            common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
        )
        view_transport.export_generation = 1
        view_transport.remote_memory_keys.append("rk1")
        view_transport.buffer_sizes.append(view_info.memory_size)

        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=canonical_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=view_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.view_repository.upsert(
            artifact_id=artifact_id,
            view_id=view_id,
            view_spec_json='{"kind":"dense_view"}',
            view_size=256,
            view_data_hash="view-hash-1",
            verified_at=None,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )
        worker = servicer.worker_repository.find_by_worker_id(registered_worker)
        assert worker is not None

        view_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            local_memory_info=canonical_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-bytespace-view-1",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW, id=view_id
            ),
        )
        view_response = servicer.RequestReplicaTransport(view_request, test_context)
        assert view_response.status == global_store_pb2.Status.STATUS_OK
        assert (
            view_response.route_kind
            == global_store_pb2.TransportRouteKind.TRANSPORT_ROUTE_KIND_RESIDENT_VIEW
        )
        assert (
            view_response.remote_memory_info.byte_space.kind
            == common_pb2.BYTE_SPACE_KIND_VIEW
        )
        assert view_response.remote_memory_info.byte_space.id == view_id
        assert view_response.view_transport_metadata.view_id == view_id
        assert view_response.view_transport_metadata.view_size_bytes == 256
        assert view_response.view_transport_metadata.view_data_hash == "view-hash-1"
        assert view_response.source_grpc_port == worker.grpc_port

        canonical_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            local_memory_info=canonical_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-bytespace-canonical-1",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL, id=""
            ),
        )
        canonical_response = servicer.RequestReplicaTransport(
            canonical_request, test_context
        )
        assert canonical_response.status == global_store_pb2.Status.STATUS_OK
        assert (
            canonical_response.route_kind
            == global_store_pb2.TransportRouteKind.TRANSPORT_ROUTE_KIND_CANONICAL
        )
        assert (
            canonical_response.remote_memory_info.byte_space.kind
            == common_pb2.BYTE_SPACE_KIND_CANONICAL
        )
        assert not canonical_response.HasField("view_transport_metadata")
        assert canonical_response.source_grpc_port == worker.grpc_port

    def test_request_replica_transport_routes_canonical_source_for_view_miss(
        self, servicer, test_context, memory_info, registered_worker
    ):
        artifact_id = "mi2:artifact-bytespace-derived"
        view_id = "view-derived-1"
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=memory_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.view_repository.upsert(
            artifact_id=artifact_id,
            view_id=view_id,
            view_spec_json='{"kind":"dense_view"}',
            view_size=2048,
            view_data_hash="derived-hash-1",
            verified_at=None,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )
        worker = servicer.worker_repository.find_by_worker_id(registered_worker)
        assert worker is not None

        request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id=artifact_id,
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-bytespace-derived-1",
            requested_byte_space=common_pb2.ByteSpaceRef(
                kind=common_pb2.BYTE_SPACE_KIND_VIEW, id=view_id
            ),
        )

        response = servicer.RequestReplicaTransport(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_OK
        assert (
            response.route_kind
            == global_store_pb2.TransportRouteKind.TRANSPORT_ROUTE_KIND_DERIVED_VIEW_FROM_CANONICAL
        )
        assert (
            response.remote_memory_info.byte_space.kind
            == common_pb2.BYTE_SPACE_KIND_CANONICAL
        )
        assert response.view_transport_metadata.view_id == view_id
        assert response.view_transport_metadata.view_size_bytes == 2048
        assert response.view_transport_metadata.view_data_hash == "derived-hash-1"
        assert response.source_grpc_port == worker.grpc_port

    def test_canonical_transport_prefers_full_identity_view_replica(
        self, servicer, test_context, registered_worker
    ):
        from tensorcast.api.store.common import canonical_index_from_bytes
        from tensorcast.api.store.view_composer import (
            _build_subset_identity_view_spec_proto,
            compute_index_multihash,
            compute_view_id,
        )

        index_bytes = (
            b'{"alpha":[0,4,[1],[1],"torch.float32",0],'
            b'"beta":[4,4,[1],[1],"torch.float32",0]}'
        )
        canonical_index = canonical_index_from_bytes(index_bytes)
        tensor_names = tuple(entry.name for entry in canonical_index.entries)
        identity_spec = _build_subset_identity_view_spec_proto(
            canonical_index=canonical_index,
            tensor_names=tensor_names,
        )
        assert identity_spec is not None
        identity_view_id = compute_view_id(identity_spec, index_bytes)
        artifact_id = (
            f"mi2:{compute_index_multihash(index_bytes)}:"
            f"{compute_index_multihash(b'payload')}"
        )

        def _memory_info(
            *,
            view_id: str | None,
            remote_key: str,
        ) -> common_pb2.MemoryInfo:
            byte_space = common_pb2.ByteSpaceRef(
                kind=(
                    common_pb2.BYTE_SPACE_KIND_VIEW
                    if view_id
                    else common_pb2.BYTE_SPACE_KIND_CANONICAL
                ),
                id=view_id or "",
            )
            info = common_pb2.MemoryInfo(
                node_id=str(uuid.uuid4()),
                node_address="192.168.8.10",
                node_port=9001 if view_id else 9000,
                memory_size=8,
                memory_type=common_pb2.MemoryType.MEMORY_TYPE_GPU,
                device_id=0,
                byte_space=byte_space,
            )
            info.transport.export_state = (
                common_pb2.ReplicaTransportMetadata.EXPORT_STATE_EXPORTABLE
            )
            info.transport.export_generation = 1
            info.transport.remote_memory_keys.append(remote_key)
            info.transport.buffer_sizes.append(info.memory_size)
            return info

        canonical_info = _memory_info(view_id=None, remote_key="canonical-key")
        view_info = _memory_info(view_id=identity_view_id, remote_key="runtime-key")
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=canonical_info,
                max_concurrency=1,
                worker_id=registered_worker,
                tensor_index_data=index_bytes,
                encoding="json",
                schema_version="v3",
            ),
            test_context,
        )
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id=artifact_id,
                mem_info=view_info,
                max_concurrency=1,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )

        response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id=artifact_id,
                local_memory_info=canonical_info,
                wait_timeout_dur=duration_pb2.Duration(seconds=1),
                source_node_id="consumer",
                source_address="192.168.8.20",
                source_port=9010,
                request_id="transport-canonical-identity-view",
                requested_byte_space=common_pb2.ByteSpaceRef(
                    kind=common_pb2.BYTE_SPACE_KIND_CANONICAL,
                    id="",
                ),
            ),
            test_context,
        )

        assert response.status == global_store_pb2.Status.STATUS_OK
        assert (
            response.remote_memory_info.byte_space.kind
            == common_pb2.BYTE_SPACE_KIND_VIEW
        )
        assert response.remote_memory_info.byte_space.id == identity_view_id
        assert (
            list(response.remote_memory_info.transport.remote_memory_keys)
            == ["runtime-key"]
        )

    def test_get_artifact_index_by_id_with_multibase(
        self, servicer, test_context, memory_info, registered_worker
    ):
        index_bytes = b'{"tensor":[0,4,[1],[1],"float32",0]}'

        def _multibase(d: bytes) -> str:
            digest = hashlib.sha256(d).digest()
            mh = b"\x12\x20" + digest
            encoded = base64.b32encode(mh).decode("ascii").strip("=").lower()
            return "b" + encoded

        index_mh = _multibase(index_bytes)
        data_mh = _multibase(b"payload")
        artifact_id = f"mi2:{index_mh}:{data_mh}"

        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
            tensor_index_data=index_bytes,
            encoding="json",
            schema_version="v3",
        )
        register_response = servicer.RegisterReplica(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        resp = servicer.GetArtifactIndexById(
            global_store_pb2.GetArtifactIndexByIdRequest(artifact_id=artifact_id),
            test_context,
        )

        assert resp.status == global_store_pb2.Status.STATUS_OK
        assert resp.tensor_index_data == index_bytes

    def test_get_artifact_index_by_id_rejects_msa1(self, servicer, test_context):
        resp = servicer.GetArtifactIndexById(
            global_store_pb2.GetArtifactIndexByIdRequest(
                artifact_id="msa1:test-session~policy~partitioned~deadbeef"
            ),
            test_context,
        )

        assert resp.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "daemon-session-local" in (test_context.details or "")

    def test_register_replica_honors_descriptor_binding(
        self, servicer, test_context, memory_info, registered_worker
    ):
        index_bytes = b'{"tensor":[0,16,[4],[1],"float32",0]}'

        def _multibase(d: bytes) -> str:
            digest = hashlib.sha256(d).digest()
            mh = b"\x12\x20" + digest
            encoded = base64.b32encode(mh).decode("ascii").strip("=").lower()
            return "b" + encoded

        index_mh = _multibase(index_bytes)
        artifact_id = "cgid:test-assembly"
        descriptor = common_pb2.ArtifactDescriptor(
            artifact_id=artifact_id,
            index_multihash=index_mh,
            data_multihash="",
            schema_version="v3",
            encoding="json",
            total_size=16,
            id_kind=common_pb2.ARTIFACT_ID_KIND_CGID,
        )

        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
            tensor_index_data=index_bytes,
            encoding="json",
            schema_version="v3",
            descriptor=descriptor,
        )
        register_response = servicer.RegisterReplica(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        resp = servicer.GetArtifactIndexById(
            global_store_pb2.GetArtifactIndexByIdRequest(artifact_id=artifact_id),
            test_context,
        )
        assert resp.status == global_store_pb2.Status.STATUS_OK
        assert resp.tensor_index_data == index_bytes

    def test_request_transport_missing_artifact(
        self, servicer, test_context, memory_info
    ):
        """Requesting transport for a key with no replicas returns NOT_FOUND."""

        transport_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="missing_artifact",
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=0),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-missing-artifact-1",
        )

        transport_response = servicer.RequestReplicaTransport(
            transport_request, test_context
        )

        assert transport_response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_complete_artifact_replica_transport(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test completing a artifact replica transport"""
        # First register a replica
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id="test_artifact",
            mem_info=memory_info,
            max_concurrency=3,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(register_request, test_context)

        # Send worker heartbeat to make it available
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=registered_worker,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
            state_version=1,
        )
        servicer.WorkerHeartbeat(heartbeat_request, test_context)

        # Request transport
        transport_request = global_store_pb2.RequestReplicaTransportRequest(
            artifact_id="test_artifact",
            local_memory_info=memory_info,
            wait_timeout_dur=duration_pb2.Duration(seconds=1),
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
            request_id="transport-complete-basic-1",
        )

        transport_response = servicer.RequestReplicaTransport(
            transport_request, test_context
        )

        # Complete transport
        complete_request = global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport_response.transport_id,
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        )

        complete_response = servicer.CompleteReplicaTransport(
            complete_request, test_context
        )

        assert complete_response.status == global_store_pb2.Status.STATUS_OK

    def test_complete_transport_persists_completion_outcome(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """CompleteReplicaTransport should persist outcome and detail in repository."""
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_completion_outcome",
                mem_info=memory_info,
                max_concurrency=3,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )
        transport_response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_completion_outcome",
                local_memory_info=memory_info,
                wait_timeout_dur=duration_pb2.Duration(seconds=1),
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
                request_id="transport-complete-outcome-1",
            ),
            test_context,
        )
        assert transport_response.status == global_store_pb2.Status.STATUS_OK

        complete_response = servicer.CompleteReplicaTransport(
            global_store_pb2.CompleteReplicaTransportRequest(
                transport_id=transport_response.transport_id,
                outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_FAILED,
                outcome_detail="simulated_transport_failure",
            ),
            test_context,
        )
        assert complete_response.status == global_store_pb2.Status.STATUS_OK

        transport_id = uuid.UUID(transport_response.transport_id)
        transport_row = servicer.transport_repository.find_by_id(transport_id)
        assert transport_row is not None
        assert transport_row.status == "completed"
        assert transport_row.completion_outcome.value == "failed"
        assert transport_row.completion_detail == "simulated_transport_failure"

    def test_complete_nonexistent_transport(self, servicer, test_context):
        """Test completing a transport that doesn't exist"""
        request = global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=str(uuid.uuid4()),
            outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
        )

        response = servicer.CompleteReplicaTransport(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_complete_transport_requires_explicit_outcome(self, servicer, test_context):
        """CompleteReplicaTransport rejects unspecified outcome."""
        request = global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=str(uuid.uuid4())
        )
        response = servicer.CompleteReplicaTransport(request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT
        assert "explicit outcome" in (test_context.details or "")

    def test_query_transport_window_returns_rows(
        self, servicer, test_context, memory_info, registered_worker
    ):
        servicer.RegisterReplica(
            global_store_pb2.RegisterReplicaRequest(
                artifact_id="test_artifact_query_transport_window",
                mem_info=memory_info,
                max_concurrency=3,
                worker_id=registered_worker,
            ),
            test_context,
        )
        servicer.WorkerHeartbeat(
            global_store_pb2.WorkerHeartbeatRequest(
                worker_id=registered_worker,
                mem_pool_available_size=7000000000,
                accepting_new_requests=True,
                state_version=1,
            ),
            test_context,
        )
        transport_response = servicer.RequestReplicaTransport(
            global_store_pb2.RequestReplicaTransportRequest(
                artifact_id="test_artifact_query_transport_window",
                local_memory_info=memory_info,
                wait_timeout_dur=duration_pb2.Duration(seconds=1),
                source_node_id="source_node",
                source_address="192.168.1.2",
                source_port=9000,
                requester_worker_id="receiver-1",
                request_id="query-window-req-1",
                scheduling_group=global_store_pb2.TransportSchedulingGroup(
                    group_id="case-a0:v1",
                    group_kind="group_realization_transport",
                    total_parts=16,
                    part_id="rx1:r0",
                    priority=0,
                    epoch=1,
                ),
            ),
            test_context,
        )
        assert transport_response.status == global_store_pb2.Status.STATUS_OK
        servicer.CompleteReplicaTransport(
            global_store_pb2.CompleteReplicaTransportRequest(
                transport_id=transport_response.transport_id,
                outcome=global_store_pb2.TRANSPORT_COMPLETION_OUTCOME_SUCCESS,
            ),
            test_context,
        )

        start_ts = timestamp_pb2.Timestamp()
        start_ts.FromDatetime(datetime.now(timezone.utc) - timedelta(minutes=5))
        end_ts = timestamp_pb2.Timestamp()
        end_ts.FromDatetime(datetime.now(timezone.utc) + timedelta(minutes=5))
        response = servicer.QueryTransportWindow(
            global_store_pb2.QueryTransportWindowRequest(
                created_at_start=start_ts,
                created_at_end=end_ts,
                limit=1000,
            ),
            test_context,
        )

        assert response.status == global_store_pb2.Status.STATUS_OK
        matched = None
        for row in response.rows:
            if row.transport_id == transport_response.transport_id:
                matched = row
                break
        assert matched is not None
        assert matched.requester_worker_id == "receiver-1"
        assert matched.group_kind == "group_realization_transport"
        assert matched.group_total_parts == 16
        assert matched.HasField("created_at")

    def test_query_transport_window_requires_window_bounds(
        self, servicer, test_context
    ):
        response = servicer.QueryTransportWindow(
            global_store_pb2.QueryTransportWindowRequest(),
            test_context,
        )
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.INVALID_ARGUMENT

    def test_persistence(self, test_context, memory_info, temp_db_file):
        """Test that the database persists data between servicer instances"""
        try:
            try:
                get_config()
            except RuntimeError:
                set_config(GlobalStoreConfig())
            # Create a servicer with the file path - DuckDB will create the file
            servicer1 = GlobalStoreServicer(db_file=temp_db_file)

            # Register a worker first
            worker_request = global_store_pb2.RegisterWorkerRequest(
                node_id="test_node_1",
                node_address="192.168.1.1",
                grpc_port=8001,
                p2p_port=8002,
                mem_pool_total_size=10000000000,
                mem_pool_available_size=8000000000,
                daemon_id="daemon_test_node_1",
            )
            worker_response = servicer1.RegisterWorker(worker_request, test_context)

            # Register a artifact
            register_request = global_store_pb2.RegisterReplicaRequest(
                artifact_id="persistent_artifact",
                mem_info=memory_info,
                max_concurrency=10,
                worker_id=worker_response.worker_id,
            )
            servicer1.RegisterReplica(register_request, test_context)

            # Create a new servicer with the same file
            servicer2 = GlobalStoreServicer(db_file=temp_db_file)

            # List replicas and verify the registered artifact is there
            list_request = global_store_pb2.ListReplicasV2Request(
                artifact_id="persistent_artifact"
            )
            list_response = servicer2.ListReplicasV2(list_request, test_context)

            assert sum(1 for _ in list_response.replicas) == 1
            assert list_response.replicas[0].artifact_id == "persistent_artifact"
        except Exception as e:
            # Clean up is handled by temp_db_file fixture
            raise e

    def test_end_to_end_flow(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """
        End-to-end test that:
        1) Registers a artifact replica
        2) Lists replicas to get the replica information
        3) Verifies the replica exists
        4) Unregisters the replica
        5) Lists replicas again and verifies the replica no longer exists
        """
        # 1. Register a artifact replica
        artifact_id = "e2e_test_artifact"
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterReplica(register_request, test_context)

        assert register_response.status == global_store_pb2.Status.STATUS_OK
        replica_id = register_response.replica_id

        # 2. List replicas to get replica information
        list_request = global_store_pb2.ListReplicasV2Request(artifact_id=artifact_id)
        list_response = servicer.ListReplicasV2(list_request, test_context)

        # 3. Verify the replica exists
        assert sum(1 for _ in list_response.replicas) == 1
        assert list_response.replicas[0].artifact_id == artifact_id
        assert list_response.replicas[0].memory_info.node_id == memory_info.node_id
        assert list_response.replicas[0].memory_info.HasField("transport")
        assert (
            list_response.replicas[0].memory_info.transport.remote_memory_keys[0]
            == memory_info.transport.remote_memory_keys[0]
        )

        # 4. Unregister the replica
        unregister_request = global_store_pb2.UnregisterReplicaRequest(
            artifact_id=artifact_id, replica_id=replica_id
        )
        unregister_response = servicer.UnregisterReplica(
            unregister_request, test_context
        )

        assert unregister_response.status == global_store_pb2.Status.STATUS_OK

        # 5. List replicas again and verify the replica no longer exists
        list_response = servicer.ListReplicasV2(list_request, test_context)

        assert sum(1 for _ in list_response.replicas) == 0

    def test_worker_registration(self, servicer, test_context):
        """Test worker registration functionality"""
        request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_1",
            daemon_id="daemon_test_1",
            node_address="192.168.1.10",
            grpc_port=8001,
            p2p_port=8002,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
        )

        response = servicer.RegisterWorker(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_OK
        assert response.worker_id is not None
        assert len(response.worker_id) > 0
        assert response.heartbeat_interval_ms > 0

    def test_worker_registration_daemon_id_is_stable_identity(
        self, servicer, test_context
    ):
        """Registering with the same daemon_id should upsert the same worker_id."""
        request1 = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_1",
            daemon_id="daemon_stable",
            node_address="192.168.1.10",
            grpc_port=8001,
            p2p_port=8002,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
        )
        response1 = servicer.RegisterWorker(request1, test_context)
        assert response1.status == global_store_pb2.Status.STATUS_OK

        request2 = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_1",
            daemon_id="daemon_stable",
            node_address="192.168.1.11",
            grpc_port=8005,
            p2p_port=8006,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=7000000000,
        )
        response2 = servicer.RegisterWorker(request2, test_context)
        assert response2.status == global_store_pb2.Status.STATUS_OK
        assert response2.worker_id == response1.worker_id

        list_request = global_store_pb2.ListActiveWorkersRequest(
            include_unavailable=True
        )
        list_response = servicer.ListActiveWorkers(list_request, test_context)
        info_by_id = {w.worker_id: w for w in list_response.workers}
        info = info_by_id[response1.worker_id]
        assert info.daemon_id == "daemon_stable"
        assert info.node_address == "192.168.1.11"
        assert info.grpc_port == 8005

    def test_worker_registration_endpoint_takeover_with_new_daemon_id(
        self, servicer, test_context
    ):
        """New daemon registration should replace stale endpoint ownership."""
        request1 = global_store_pb2.RegisterWorkerRequest(
            node_id="node_prev",
            daemon_id="daemon_prev",
            node_address="192.168.1.50",
            grpc_port=8050,
            p2p_port=8051,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
        )
        response1 = servicer.RegisterWorker(request1, test_context)
        assert response1.status == global_store_pb2.Status.STATUS_OK

        request2 = global_store_pb2.RegisterWorkerRequest(
            node_id="node_next",
            daemon_id="daemon_next",
            node_address="192.168.1.50",
            grpc_port=8050,
            p2p_port=8052,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=7000000000,
        )
        response2 = servicer.RegisterWorker(request2, test_context)
        assert response2.status == global_store_pb2.Status.STATUS_OK
        assert response2.worker_id != response1.worker_id

        list_response = servicer.ListActiveWorkers(
            global_store_pb2.ListActiveWorkersRequest(include_unavailable=True),
            test_context,
        )
        info_by_id = {w.worker_id: w for w in list_response.workers}
        assert response1.worker_id not in info_by_id
        assert response2.worker_id in info_by_id
        assert info_by_id[response2.worker_id].daemon_id == "daemon_next"

    def test_worker_heartbeat(self, servicer, test_context):
        """Test worker heartbeat functionality"""
        # First register a worker
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_1",
            node_address="192.168.1.10",
            grpc_port=8001,
            p2p_port=8002,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_test_node_1",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)

        # Send heartbeat
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=register_response.worker_id,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
            state_version=1,
        )

        heartbeat_response = servicer.WorkerHeartbeat(heartbeat_request, test_context)

        assert heartbeat_response.status == global_store_pb2.Status.STATUS_OK

    def test_worker_heartbeat_tx_conflict_fails_fast(
        self, servicer, test_context, monkeypatch
    ):
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_hb_conflict",
            node_address="192.168.1.40",
            grpc_port=8040,
            p2p_port=8041,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_hb_conflict",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        calls = {"count": 0}

        def _raise_conflict(*args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            raise DatabaseError(
                "Transaction failed: TransactionContext Error: Conflict on tuple deletion!"
            )

        monkeypatch.setattr(
            servicer.worker_service,
            "heartbeat",
            _raise_conflict,
        )

        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=register_response.worker_id,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
            state_version=1,
            daemon_id="daemon_hb_conflict",
        )
        heartbeat_response = servicer.WorkerHeartbeat(heartbeat_request, test_context)

        assert heartbeat_response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.ABORTED
        assert calls["count"] == 1

    def test_worker_unregistration(self, servicer, test_context):
        """Test worker unregistration functionality"""
        # First register a worker
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_1",
            node_address="192.168.1.10",
            grpc_port=8001,
            p2p_port=8002,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_test_node_1",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)

        # Unregister worker
        unregister_request = global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id
        )

        unregister_response = servicer.UnregisterWorker(
            unregister_request, test_context
        )

        assert unregister_response.status == global_store_pb2.Status.STATUS_OK

    def test_unregister_worker_idempotent_replay(
        self, servicer, test_context, monkeypatch
    ):
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_unreg_replay",
            node_address="192.168.1.30",
            grpc_port=8030,
            p2p_port=8031,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_unreg_replay",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        calls = {"count": 0}
        original = servicer.worker_service.unregister_worker

        def _wrapped(worker_id):
            calls["count"] += 1
            return original(worker_id)

        monkeypatch.setattr(servicer.worker_service, "unregister_worker", _wrapped)

        request = global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id,
            is_graceful_shutdown=True,
            client_request_id="unregister-replay-1",
        )
        first = servicer.UnregisterWorker(request, test_context)
        assert first.status == global_store_pb2.Status.STATUS_OK

        test_context.code = None
        test_context.details = None
        second = servicer.UnregisterWorker(request, test_context)
        assert second.status == global_store_pb2.Status.STATUS_OK
        assert calls["count"] == 1

    def test_unregister_worker_idempotent_payload_mismatch(
        self, servicer, test_context
    ):
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_unreg_mismatch",
            node_address="192.168.1.31",
            grpc_port=8032,
            p2p_port=8033,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_unreg_mismatch",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        first = global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id,
            is_graceful_shutdown=True,
            client_request_id="unregister-mismatch-1",
        )
        second = global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id,
            is_graceful_shutdown=False,
            client_request_id="unregister-mismatch-1",
        )

        first_resp = servicer.UnregisterWorker(first, test_context)
        assert first_resp.status == global_store_pb2.Status.STATUS_OK

        test_context.code = None
        test_context.details = None
        second_resp = servicer.UnregisterWorker(second, test_context)
        assert second_resp.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION

    def test_unregister_worker_tx_conflict_fails_fast(
        self, servicer, test_context, monkeypatch
    ):
        register_request = global_store_pb2.RegisterWorkerRequest(
            node_id="test_node_unreg_conflict",
            node_address="192.168.1.32",
            grpc_port=8034,
            p2p_port=8035,
            mem_pool_total_size=10000000000,
            mem_pool_available_size=8000000000,
            daemon_id="daemon_unreg_conflict",
        )
        register_response = servicer.RegisterWorker(register_request, test_context)
        assert register_response.status == global_store_pb2.Status.STATUS_OK

        calls = {"count": 0}

        def _raise_conflict(worker_id):
            calls["count"] += 1
            raise DatabaseError("write-write conflict on key workers.worker_id")

        monkeypatch.setattr(
            servicer.worker_service,
            "unregister_worker",
            _raise_conflict,
        )
        request = global_store_pb2.UnregisterWorkerRequest(
            worker_id=register_response.worker_id,
            client_request_id="unregister-conflict-1",
        )
        response = servicer.UnregisterWorker(request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.ABORTED
        assert calls["count"] == 1

    def test_list_active_workers(self, servicer, test_context):
        """Test listing active workers"""
        # Register multiple workers
        worker_records = []
        for i in range(3):
            register_request = global_store_pb2.RegisterWorkerRequest(
                node_id=f"test_node_{i}",
                daemon_id=f"daemon_{i}",
                node_address=f"192.168.1.{10 + i}",
                grpc_port=8001 + i,
                p2p_port=8002 + i,
                mem_pool_total_size=10000000000,
                mem_pool_available_size=8000000000,
            )
            register_response = servicer.RegisterWorker(register_request, test_context)
            worker_records.append((register_response.worker_id, f"daemon_{i}"))

        # List workers
        list_request = global_store_pb2.ListActiveWorkersRequest(
            include_unavailable=True
        )
        list_response = servicer.ListActiveWorkers(list_request, test_context)

        assert len(list_response.workers) >= 3
        info_by_id = {w.worker_id: w for w in list_response.workers}
        for worker_id, daemon_id in worker_records:
            assert worker_id in info_by_id
            assert info_by_id[worker_id].daemon_id == daemon_id

    def test_get_artifact_info(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test getting artifact information"""
        artifact_id = "info_test_artifact"

        # Register a replica
        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(register_request, test_context)

        # Get artifact info
        # New API: GetArtifactInfoById
        info_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id
        )
        info_response = servicer.GetArtifactInfoById(info_request, test_context)

        assert info_response.status == global_store_pb2.Status.STATUS_OK
        assert len(info_response.replicas) == 1
        assert info_response.replicas[0].node_id == memory_info.node_id

    def test_get_nonexistent_artifact_info(self, servicer, test_context):
        """Test getting info for a artifact that doesn't exist"""
        info_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id="nonexistent_artifact"
        )
        info_response = servicer.GetArtifactInfoById(info_request, test_context)

        assert info_response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_update_artifact_view_state_roundtrip(self, servicer, test_context):
        """Variant metadata and leaves round-trip through servicer."""
        artifact_id = "mi2:index_hash:data_hash"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=artifact_id,
            index_multihash="index_hash",
            data_multihash="data_hash",
            schema_version="v3",
            encoding="json",
        )

        ts = timestamp_pb2.Timestamp()
        ts.GetCurrentTime()

        update_request = global_store_pb2.UpdateArtifactViewStateRequest(
            artifact_id=artifact_id,
            view=global_store_pb2.ViewUpsert(
                view_id="view-1",
                view_spec_json="{}",
                view_size=4096,
                view_data_hash="viewhash",
                verified_at=ts,
            ),
            leaf_writes=[
                global_store_pb2.LeafWrite(
                    hash_space=common_pb2.HashSpaceRef(
                        byte_space=common_pb2.ByteSpaceRef(
                            kind=common_pb2.BYTE_SPACE_KIND_VIEW, id="view-1"
                        ),
                    ),
                    leaf_idx=0,
                    digest=b"\x01" * 32,
                ),
                global_store_pb2.LeafWrite(
                    hash_space=common_pb2.HashSpaceRef(
                        byte_space=common_pb2.ByteSpaceRef(
                            kind=common_pb2.BYTE_SPACE_KIND_CANONICAL, id=""
                        ),
                        canonical_index_multihash="index_hash",
                    ),
                    leaf_idx=2,
                    digest=b"\x02" * 32,
                ),
            ],
        )

        update_response = servicer.UpdateArtifactViewState(update_request, test_context)
        assert update_response.status == global_store_pb2.Status.STATUS_OK
        assert test_context.code is None

        test_context.code = None
        view_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_replicas=wrappers_pb2.BoolValue(value=False),
            include_leaves=True,
            include_view_meta=True,
        )
        view_request.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
        view_request.requested_byte_space.id = "view-1"
        view_response = servicer.GetArtifactInfoById(view_request, test_context)

        assert view_response.status == global_store_pb2.Status.STATUS_OK
        assert len(view_response.leaves) == 1
        assert view_response.leaves[0].leaf_idx == 0
        assert view_response.leaves[0].digest == b"\x01" * 32
        assert view_response.view_meta.view_size == 4096
        assert view_response.view_meta.view_data_hash == "viewhash"
        assert view_response.view_meta.HasField("verified_at")
        assert test_context.code is None

        test_context.code = None
        canonical_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_replicas=wrappers_pb2.BoolValue(value=False),
            include_leaves=True,
        )
        canonical_request.requested_byte_space.kind = (
            common_pb2.BYTE_SPACE_KIND_CANONICAL
        )
        canonical_response = servicer.GetArtifactInfoById(
            canonical_request, test_context
        )
        assert canonical_response.status == global_store_pb2.Status.STATUS_OK
        assert len(canonical_response.leaves) == 1
        assert canonical_response.leaves[0].leaf_idx == 2
        assert canonical_response.leaves[0].digest == b"\x02" * 32
        assert test_context.code is None

        test_context.code = None
        partial_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_replicas=wrappers_pb2.BoolValue(value=False),
            include_leaves=True,
            include_view_meta=True,
        )
        partial_request.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
        partial_request.requested_byte_space.id = "view-1"
        partial_request.leaf_idxs.extend([0, 3])
        partial_response = servicer.GetArtifactInfoById(partial_request, test_context)

        assert partial_response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(partial_response.leaves) == 1
        assert partial_response.leaves[0].leaf_idx == 0
        assert len(partial_response.partial_leaf_coverage) == 1
        detail = partial_response.partial_leaf_coverage[0]
        assert detail.hash_space.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
        assert detail.hash_space.byte_space.id == "view-1"
        assert len(detail.missing_leaf_ranges) == 1
        assert detail.missing_leaf_ranges[0].start == 3
        assert detail.missing_leaf_ranges[0].count == 1
        assert test_context.code == grpc.StatusCode.NOT_FOUND

    def test_update_artifact_view_state_accepts_metadata_only_view_residency(
        self, servicer, test_context
    ):
        artifact_id = "mi2:index_hash:view_residency"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=artifact_id,
            index_multihash="index_hash",
            data_multihash="data_hash",
            schema_version="v3",
            encoding="json",
        )

        create_resp = servicer.UpdateArtifactViewState(
            global_store_pb2.UpdateArtifactViewStateRequest(
                artifact_id=artifact_id,
                view=global_store_pb2.ViewUpsert(
                    view_id="view-resident",
                    view_size=8192,
                    view_data_hash="hash-a",
                ),
            ),
            test_context,
        )
        assert create_resp.status == global_store_pb2.Status.STATUS_OK
        assert test_context.code is None

        test_context.code = None
        list_resp = servicer.ListViews(
            global_store_pb2.ListViewsRequest(artifact_id=artifact_id),
            test_context,
        )
        assert list_resp.status == global_store_pb2.Status.STATUS_OK
        assert len(list_resp.views) == 1
        assert list_resp.views[0].view_id == "view-resident"
        assert list_resp.views[0].view_spec_json == ""
        assert list_resp.views[0].view_size == 8192
        assert list_resp.views[0].view_data_hash == "hash-a"
        assert test_context.code is None

        test_context.code = None
        preserve_resp = servicer.UpdateArtifactViewState(
            global_store_pb2.UpdateArtifactViewStateRequest(
                artifact_id=artifact_id,
                view=global_store_pb2.ViewUpsert(
                    view_id="view-resident",
                    view_spec_json="{\"kind\":\"tp\"}",
                    view_size=8192,
                    view_data_hash="hash-a",
                ),
            ),
            test_context,
        )
        assert preserve_resp.status == global_store_pb2.Status.STATUS_OK
        assert test_context.code is None

        test_context.code = None
        servicer.UpdateArtifactViewState(
            global_store_pb2.UpdateArtifactViewStateRequest(
                artifact_id=artifact_id,
                view=global_store_pb2.ViewUpsert(
                    view_id="view-resident",
                    view_size=8192,
                ),
            ),
            test_context,
        )
        assert test_context.code is None

        test_context.code = None
        preserved_list_resp = servicer.ListViews(
            global_store_pb2.ListViewsRequest(artifact_id=artifact_id),
            test_context,
        )
        assert preserved_list_resp.status == global_store_pb2.Status.STATUS_OK
        assert len(preserved_list_resp.views) == 1
        assert preserved_list_resp.views[0].view_spec_json == "{\"kind\":\"tp\"}"
        assert preserved_list_resp.views[0].view_data_hash == "hash-a"
        assert test_context.code is None

    def test_write_tensor_proof_commitments_roundtrip(self, servicer, test_context):
        artifact_id = "mi2:index_hash:data_hash"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=artifact_id,
            index_multihash="index_hash",
            data_multihash="data_hash",
            schema_version="v3",
            encoding="json",
        )

        req = global_store_pb2.WriteTensorProofCommitmentsRequest(
            mi2_id=artifact_id,
            proof_schema_version="v1",
            commitments=[
                global_store_pb2.TensorProofCommitmentWrite(
                    tensor_name="weights",
                    proof_chunk_idx=0,
                    digest=b"\x01" * 32,
                ),
                global_store_pb2.TensorProofCommitmentWrite(
                    tensor_name="weights",
                    proof_chunk_idx=1,
                    digest=b"\x02" * 32,
                ),
            ],
        )

        resp = servicer.WriteTensorProofCommitments(req, test_context)
        assert resp.status == global_store_pb2.Status.STATUS_OK
        assert resp.inserted == 2
        assert test_context.code is None

        test_context.code = None
        resp2 = servicer.WriteTensorProofCommitments(req, test_context)
        assert resp2.status == global_store_pb2.Status.STATUS_OK
        assert resp2.inserted == 0
        assert test_context.code is None

        test_context.code = None
        conflict = global_store_pb2.WriteTensorProofCommitmentsRequest(
            mi2_id=artifact_id,
            proof_schema_version="v1",
            commitments=[
                global_store_pb2.TensorProofCommitmentWrite(
                    tensor_name="weights",
                    proof_chunk_idx=0,
                    digest=b"\x03" * 32,
                ),
            ],
        )
        conflict_resp = servicer.WriteTensorProofCommitments(conflict, test_context)
        assert conflict_resp.status == global_store_pb2.Status.STATUS_ERROR
        assert test_context.code == grpc.StatusCode.FAILED_PRECONDITION

    def test_check_proof_commitments_match(self, servicer, test_context):
        assembly_id = "cgid:assembly-proof"
        mi2_id = "mi2:index:proof"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=assembly_id,
            index_multihash="index",
            data_multihash="data",
            schema_version="v3",
            encoding="json",
        )
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=mi2_id,
            index_multihash="index",
            data_multihash="proof",
            schema_version="v3",
            encoding="json",
        )

        servicer.proof_repository.upsert_assembly_proof_commitment(
            assembly_id=assembly_id,
            tensor_name="weights",
            proof_schema_version="v1",
            proof_chunk_idx=0,
            digest=b"\x01" * 32,
        )
        servicer.proof_repository.upsert_tensor_proof_commitment(
            mi2_id=mi2_id,
            tensor_name="weights",
            proof_schema_version="v1",
            proof_chunk_idx=0,
            digest=b"\x01" * 32,
        )

        req = global_store_pb2.CheckProofCommitmentsMatchRequest(
            assembly_id=assembly_id,
            mi2_id=mi2_id,
            proof_schema_version="v1",
            tensor_names=["weights"],
        )
        test_context.code = None
        resp = servicer.CheckProofCommitmentsMatch(req, test_context)
        assert resp.status == global_store_pb2.Status.STATUS_OK
        assert resp.match is True
        assert test_context.code is None

        servicer.proof_repository.upsert_assembly_proof_commitment(
            assembly_id=assembly_id,
            tensor_name="weights",
            proof_schema_version="v1",
            proof_chunk_idx=1,
            digest=b"\x02" * 32,
        )
        test_context.code = None
        resp2 = servicer.CheckProofCommitmentsMatch(req, test_context)
        assert resp2.status == global_store_pb2.Status.STATUS_OK
        assert resp2.match is False
        assert test_context.code is None

    def test_get_artifact_view_info_not_found(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Missing view should respond with NOT_FOUND."""
        artifact_id = "mi2:index:data"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=artifact_id,
            index_multihash="index",
            data_multihash="data",
            schema_version="v3",
            encoding="json",
        )

        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(register_request, test_context)
        test_context.code = None

        request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_leaves=True,
            include_view_meta=True,
        )
        request.requested_byte_space.kind = common_pb2.BYTE_SPACE_KIND_VIEW
        request.requested_byte_space.id = "missing-view"
        request.leaf_idxs.extend([0, 1])
        response = servicer.GetArtifactInfoById(request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(response.replicas) == 0
        assert len(response.partial_leaf_coverage) == 1
        detail = response.partial_leaf_coverage[0]
        assert detail.hash_space.byte_space.kind == common_pb2.BYTE_SPACE_KIND_VIEW
        assert detail.hash_space.byte_space.id == "missing-view"
        assert len(detail.missing_leaf_ranges) == 1
        assert detail.missing_leaf_ranges[0].start == 0
        assert detail.missing_leaf_ranges[0].count == 2
        assert test_context.code == grpc.StatusCode.NOT_FOUND

    def test_register_replica_with_cgid(
        self, servicer, test_context, memory_info, registered_worker
    ):
        artifact_id = "cgid:test-suite-1"

        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
        )
        response = servicer.RegisterReplica(register_request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_OK
        record = servicer.artifacts_repo.get(artifact_id)
        assert record is not None
        assert record["id_kind"] == "CGID"
        assert record["index_multihash"] is None
        assert record["data_multihash"] is None

    def test_get_artifact_canonical_partial_leaf_coverage(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Canonical leaf queries return partial leaf coverage detail when missing."""
        artifact_id = "mi2:index:data"
        servicer.artifacts_repo.upsert_artifact(
            artifact_id=artifact_id,
            index_multihash="index",
            data_multihash="data",
            schema_version="v3",
            encoding="json",
        )

        register_request = global_store_pb2.RegisterReplicaRequest(
            artifact_id=artifact_id,
            mem_info=memory_info,
            max_concurrency=1,
            worker_id=registered_worker,
        )
        servicer.RegisterReplica(register_request, test_context)

        test_context.code = None
        servicer.UpdateArtifactViewState(
            global_store_pb2.UpdateArtifactViewStateRequest(
                artifact_id=artifact_id,
                leaf_writes=[
                    global_store_pb2.LeafWrite(
                        hash_space=common_pb2.HashSpaceRef(
                            byte_space=common_pb2.ByteSpaceRef(
                                kind=common_pb2.BYTE_SPACE_KIND_CANONICAL, id=""
                            ),
                            canonical_index_multihash="index",
                        ),
                        leaf_idx=0,
                        digest=b"\xaa" * 32,
                    )
                ],
            ),
            test_context,
        )

        canonical_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_leaves=True,
        )
        canonical_request.requested_byte_space.kind = (
            common_pb2.BYTE_SPACE_KIND_CANONICAL
        )
        canonical_request.leaf_idxs.extend([0, 2])
        response = servicer.GetArtifactInfoById(canonical_request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(response.leaves) == 1 and response.leaves[0].leaf_idx == 0
        assert len(response.partial_leaf_coverage) == 1
        detail = response.partial_leaf_coverage[0]
        assert detail.hash_space.byte_space.kind == common_pb2.BYTE_SPACE_KIND_CANONICAL
        assert detail.hash_space.canonical_index_multihash == "index"
        assert len(detail.missing_leaf_ranges) == 1
        assert detail.missing_leaf_ranges[0].start == 2
        assert detail.missing_leaf_ranges[0].count == 1
        assert test_context.code == grpc.StatusCode.NOT_FOUND
