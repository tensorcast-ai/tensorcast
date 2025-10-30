#  Copyright (c) 2025, TensorCast Team.

"""Tests for Global Store gRPC service interface."""

import base64
import hashlib
import uuid

import grpc
from google.protobuf import duration_pb2, timestamp_pb2, wrappers_pb2

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
        register_response = servicer.RegisterReplica(
            register_request, test_context
        )

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
        register_response = servicer.RegisterReplica(
            register_request, test_context
        )

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
        )

        transport_response = servicer.RequestReplicaTransport(
            transport_request, test_context
        )

        assert transport_response.status == global_store_pb2.Status.STATUS_OK
        assert transport_response.remote_memory_info is not None
        assert transport_response.transport_id is not None

    def test_get_artifact_index_by_id_with_multibase(self, servicer, test_context, memory_info, registered_worker):
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
        )

        transport_response = servicer.RequestReplicaTransport(
            transport_request, test_context
        )

        # Complete transport
        complete_request = global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=transport_response.transport_id
        )

        complete_response = servicer.CompleteReplicaTransport(
            complete_request, test_context
        )

        assert complete_response.status == global_store_pb2.Status.STATUS_OK

    def test_complete_nonexistent_transport(self, servicer, test_context):
        """Test completing a transport that doesn't exist"""
        request = global_store_pb2.CompleteReplicaTransportRequest(
            transport_id=str(uuid.uuid4())
        )

        response = servicer.CompleteReplicaTransport(request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND

    def test_persistence(self, test_context, memory_info, temp_db_file):
        """Test that the database persists data between servicer instances"""
        try:
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
        register_response = servicer.RegisterReplica(
            register_request, test_context
        )

        assert register_response.status == global_store_pb2.Status.STATUS_OK
        replica_id = register_response.replica_id

        # 2. List replicas to get replica information
        list_request = global_store_pb2.ListReplicasV2Request(artifact_id=artifact_id)
        list_response = servicer.ListReplicasV2(list_request, test_context)

        # 3. Verify the replica exists
        assert sum(1 for _ in list_response.replicas) == 1
        assert list_response.replicas[0].artifact_id == artifact_id
        assert list_response.replicas[0].memory_info.node_id == memory_info.node_id
        assert (
            list_response.replicas[0].memory_info.remote_memory_keys[0]
            == memory_info.remote_memory_keys[0]
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
        )
        register_response = servicer.RegisterWorker(register_request, test_context)

        # Send heartbeat
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=register_response.worker_id,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
        )

        heartbeat_response = servicer.WorkerHeartbeat(heartbeat_request, test_context)

        assert heartbeat_response.status == global_store_pb2.Status.STATUS_OK

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

    def test_list_active_workers(self, servicer, test_context):
        """Test listing active workers"""
        # Register multiple workers
        worker_ids = []
        for i in range(3):
            register_request = global_store_pb2.RegisterWorkerRequest(
                node_id=f"test_node_{i}",
                node_address=f"192.168.1.{10+i}",
                grpc_port=8001 + i,
                p2p_port=8002 + i,
                mem_pool_total_size=10000000000,
                mem_pool_available_size=8000000000,
            )
            register_response = servicer.RegisterWorker(register_request, test_context)
            worker_ids.append(register_response.worker_id)

        # List workers
        list_request = global_store_pb2.ListActiveWorkersRequest(
            include_unavailable=True
        )
        list_response = servicer.ListActiveWorkers(list_request, test_context)

        assert len(list_response.workers) >= 3
        listed_worker_ids = [w.worker_id for w in list_response.workers]
        for worker_id in worker_ids:
            assert worker_id in listed_worker_ids

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
        info_request = global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=artifact_id)
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
            variant=global_store_pb2.VariantUpsert(
                view_id="view-1",
                view_spec_json="{}",
                view_size=4096,
                view_data_hash="viewhash",
                verified_at=ts,
            ),
            leaf_writes=[
                global_store_pb2.LeafWrite(
                    space_kind=global_store_pb2.BYTE_SPACE_KIND_VARIANT,
                    space_id="view-1",
                    leaf_idx=0,
                    digest=b"\x01" * 32,
                ),
                global_store_pb2.LeafWrite(
                    space_kind=global_store_pb2.BYTE_SPACE_KIND_CANONICAL,
                    space_id="index_hash",
                    leaf_idx=2,
                    digest=b"\x02" * 32,
                ),
            ],
        )

        update_response = servicer.UpdateArtifactViewState(
            update_request, test_context
        )
        assert update_response.status == global_store_pb2.Status.STATUS_OK
        assert test_context.code is None

        test_context.code = None
        view_request = global_store_pb2.GetArtifactInfoByIdRequest(
            artifact_id=artifact_id,
            include_replicas=wrappers_pb2.BoolValue(value=False),
            include_leaves=True,
            include_view_meta=True,
        )
        view_request.view_id = "view-1"
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
            canonical=True,
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
        partial_request.view_id = "view-1"
        partial_request.leaf_idxs.extend([0, 3])
        partial_response = servicer.GetArtifactInfoById(partial_request, test_context)

        assert partial_response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(partial_response.leaves) == 1
        assert partial_response.leaves[0].leaf_idx == 0
        assert len(partial_response.partial_coverage) == 1
        detail = partial_response.partial_coverage[0]
        assert detail.space_kind == global_store_pb2.BYTE_SPACE_KIND_VARIANT
        assert detail.space_id == "view-1"
        assert len(detail.missing_ranges) == 1
        assert detail.missing_ranges[0].off == 3
        assert detail.missing_ranges[0].len == 1
        assert test_context.code == grpc.StatusCode.NOT_FOUND

    def test_get_artifact_view_info_not_found(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Missing variant should respond with NOT_FOUND."""
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
        request.view_id = "missing-view"
        request.leaf_idxs.extend([0, 1])
        response = servicer.GetArtifactInfoById(request, test_context)
        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(response.replicas) == 1
        assert len(response.partial_coverage) == 1
        detail = response.partial_coverage[0]
        assert detail.space_kind == global_store_pb2.BYTE_SPACE_KIND_VARIANT
        assert detail.space_id == "missing-view"
        assert sorted(r.off for r in detail.missing_ranges) == [0, 1]
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

    def test_get_artifact_canonical_partial_coverage(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Canonical leaf queries return partial coverage detail when missing."""
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
                        space_kind=global_store_pb2.BYTE_SPACE_KIND_CANONICAL,
                        space_id="index",
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
            canonical=True,
        )
        canonical_request.leaf_idxs.extend([0, 2])
        response = servicer.GetArtifactInfoById(canonical_request, test_context)

        assert response.status == global_store_pb2.Status.STATUS_NOT_FOUND
        assert len(response.leaves) == 1 and response.leaves[0].leaf_idx == 0
        assert len(response.partial_coverage) == 1
        detail = response.partial_coverage[0]
        assert detail.space_kind == global_store_pb2.BYTE_SPACE_KIND_CANONICAL
        assert detail.space_id == "index"
        assert len(detail.missing_ranges) == 1
        assert detail.missing_ranges[0].off == 2
        assert detail.missing_ranges[0].len == 1
        assert test_context.code == grpc.StatusCode.NOT_FOUND
