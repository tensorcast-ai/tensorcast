#  Copyright (c) 2025, StepCast Team.

"""Tests for Global Store gRPC service interface."""

import uuid

from scstore.global_store.grpc_service import GlobalModelStoreServicer
from scstore.proto import global_store_pb2


class TestGRPCService:
    """Tests for the gRPC service interface."""

    def test_update_model_replica(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test updating a model replica"""
        # First register a replica
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name="test_model",
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterModelReplica(
            register_request, test_context
        )

        # Now update it
        update_request = global_store_pb2.UpdateModelReplicaRequest(
            model_name="test_model", replica_id=register_response.replica_id
        )

        update_response = servicer.UpdateModelReplica(update_request, test_context)

        assert update_response.status == global_store_pb2.Status.OK
        assert update_response.model_name == "test_model"
        assert update_response.replica_id == register_response.replica_id

    def test_update_nonexistent_model_replica(self, servicer, test_context):
        """Test updating a model replica that doesn't exist"""
        request = global_store_pb2.UpdateModelReplicaRequest(
            model_name="nonexistent_model", replica_id=str(uuid.uuid4())
        )

        response = servicer.UpdateModelReplica(request, test_context)

        assert response.status == global_store_pb2.Status.NOT_FOUND

    def test_unregister_model_replica(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test unregistering a model replica"""
        # First register a replica
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name="test_model",
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterModelReplica(
            register_request, test_context
        )

        # Now unregister it
        unregister_request = global_store_pb2.UnregisterModelReplicaRequest(
            model_name="test_model", replica_id=register_response.replica_id
        )

        unregister_response = servicer.UnregisterModelReplica(
            unregister_request, test_context
        )

        assert unregister_response.status == global_store_pb2.Status.OK

    def test_unregister_nonexistent_model_replica(self, servicer, test_context):
        """Test unregistering a model replica that doesn't exist"""
        request = global_store_pb2.UnregisterModelReplicaRequest(
            model_name="nonexistent_model", replica_id=str(uuid.uuid4())
        )

        response = servicer.UnregisterModelReplica(request, test_context)

        assert response.status == global_store_pb2.Status.NOT_FOUND

    def test_list_model_replicas(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test listing model replicas"""
        # Register multiple replicas
        model_names = ["model1", "model2", "model3"]
        for model_name in model_names:
            request = global_store_pb2.RegisterModelReplicaRequest(
                model_name=model_name,
                mem_info=memory_info,
                max_concurrency=10,
                worker_id=registered_worker,
            )
            servicer.RegisterModelReplica(request, test_context)

        # List all replicas
        list_request = global_store_pb2.ListModelReplicasRequest()
        list_response = servicer.ListModelReplicas(list_request, test_context)

        assert len(list_response.model_replicas) >= 3
        for model_name in model_names:
            assert model_name in list_response.model_replicas

    def test_list_model_replicas_with_filter(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test listing model replicas with a filter"""
        # Register a specific model
        model_name = "filtered_model"
        request = global_store_pb2.RegisterModelReplicaRequest(
            model_name=model_name,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        servicer.RegisterModelReplica(request, test_context)

        # List replicas with filter
        list_request = global_store_pb2.ListModelReplicasRequest(model_name=model_name)
        list_response = servicer.ListModelReplicas(list_request, test_context)

        assert len(list_response.model_replicas) == 1
        assert model_name in list_response.model_replicas

    def test_request_model_replica_transport(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test requesting a model replica transport"""
        # First register a replica
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name="test_model",
            mem_info=memory_info,
            max_concurrency=3,
            worker_id=registered_worker,
        )
        servicer.RegisterModelReplica(register_request, test_context)

        # Send worker heartbeat to make it available
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=registered_worker,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
        )
        servicer.WorkerHeartbeat(heartbeat_request, test_context)

        # Now request transport
        transport_request = global_store_pb2.RequestModelReplicaTransportRequest(
            model_name="test_model",
            local_memory_info=memory_info,
            wait_timeout_ms=1000,
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
        )

        transport_response = servicer.RequestModelReplicaTransport(
            transport_request, test_context
        )

        assert transport_response.status == global_store_pb2.Status.OK
        assert transport_response.remote_memory_info is not None
        assert transport_response.transport_id is not None

    def test_complete_model_replica_transport(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test completing a model replica transport"""
        # First register a replica
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name="test_model",
            mem_info=memory_info,
            max_concurrency=3,
            worker_id=registered_worker,
        )
        servicer.RegisterModelReplica(register_request, test_context)

        # Send worker heartbeat to make it available
        heartbeat_request = global_store_pb2.WorkerHeartbeatRequest(
            worker_id=registered_worker,
            mem_pool_available_size=7000000000,
            accepting_new_requests=True,
        )
        servicer.WorkerHeartbeat(heartbeat_request, test_context)

        # Request transport
        transport_request = global_store_pb2.RequestModelReplicaTransportRequest(
            model_name="test_model",
            local_memory_info=memory_info,
            wait_timeout_ms=1000,
            source_node_id="source_node",
            source_address="192.168.1.2",
            source_port=9000,
        )

        transport_response = servicer.RequestModelReplicaTransport(
            transport_request, test_context
        )

        # Complete transport
        complete_request = global_store_pb2.CompleteModelReplicaTransportRequest(
            transport_id=transport_response.transport_id
        )

        complete_response = servicer.CompleteModelReplicaTransport(
            complete_request, test_context
        )

        assert complete_response.status == global_store_pb2.Status.OK

    def test_complete_nonexistent_transport(self, servicer, test_context):
        """Test completing a transport that doesn't exist"""
        request = global_store_pb2.CompleteModelReplicaTransportRequest(
            transport_id=str(uuid.uuid4())
        )

        response = servicer.CompleteModelReplicaTransport(request, test_context)

        assert response.status == global_store_pb2.Status.NOT_FOUND

    def test_persistence(self, test_context, memory_info, temp_db_file):
        """Test that the database persists data between servicer instances"""
        try:
            # Create a servicer with the file path - DuckDB will create the file
            servicer1 = GlobalModelStoreServicer(db_file=temp_db_file)

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

            # Register a model
            register_request = global_store_pb2.RegisterModelReplicaRequest(
                model_name="persistent_model",
                mem_info=memory_info,
                max_concurrency=10,
                worker_id=worker_response.worker_id,
            )
            servicer1.RegisterModelReplica(register_request, test_context)

            # Create a new servicer with the same file
            servicer2 = GlobalModelStoreServicer(db_file=temp_db_file)

            # List models and verify the registered model is there
            list_request = global_store_pb2.ListModelReplicasRequest(
                model_name="persistent_model"
            )
            list_response = servicer2.ListModelReplicas(list_request, test_context)

            assert len(list_response.model_replicas) == 1
            assert "persistent_model" in list_response.model_replicas
        except Exception as e:
            # Clean up is handled by temp_db_file fixture
            raise e

    def test_end_to_end_flow(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """
        End-to-end test that:
        1) Registers a model replica
        2) Lists replicas to get the replica information
        3) Verifies the replica exists
        4) Unregisters the replica
        5) Lists replicas again and verifies the replica no longer exists
        """
        # 1. Register a model replica
        model_name = "e2e_test_model"
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name=model_name,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        register_response = servicer.RegisterModelReplica(
            register_request, test_context
        )

        assert register_response.status == global_store_pb2.Status.OK
        replica_id = register_response.replica_id

        # 2. List replicas to get replica information
        list_request = global_store_pb2.ListModelReplicasRequest(model_name=model_name)
        list_response = servicer.ListModelReplicas(list_request, test_context)

        # 3. Verify the replica exists
        assert model_name in list_response.model_replicas
        assert len(list_response.model_replicas[model_name].list) == 1
        assert (
            list_response.model_replicas[model_name].list[0].node_id
            == memory_info.node_id
        )
        assert (
            list_response.model_replicas[model_name].list[0].remote_memory_keys[0]
            == memory_info.remote_memory_keys[0]
        )

        # 4. Unregister the replica
        unregister_request = global_store_pb2.UnregisterModelReplicaRequest(
            model_name=model_name, replica_id=replica_id
        )
        unregister_response = servicer.UnregisterModelReplica(
            unregister_request, test_context
        )

        assert unregister_response.status == global_store_pb2.Status.OK

        # 5. List replicas again and verify the replica no longer exists
        list_response = servicer.ListModelReplicas(list_request, test_context)

        assert (
            model_name not in list_response.model_replicas
            or len(list_response.model_replicas[model_name].list) == 0
        )

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

        assert response.status == global_store_pb2.Status.OK
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

        assert heartbeat_response.status == global_store_pb2.Status.OK

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

        assert unregister_response.status == global_store_pb2.Status.OK

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

    def test_get_model_info(
        self, servicer, test_context, memory_info, registered_worker
    ):
        """Test getting model information"""
        model_name = "info_test_model"

        # Register a replica
        register_request = global_store_pb2.RegisterModelReplicaRequest(
            model_name=model_name,
            mem_info=memory_info,
            max_concurrency=10,
            worker_id=registered_worker,
        )
        servicer.RegisterModelReplica(register_request, test_context)

        # Get model info
        info_request = global_store_pb2.GetModelInfoRequest(model_name=model_name)
        info_response = servicer.GetModelInfo(info_request, test_context)

        assert info_response.status == global_store_pb2.Status.OK
        assert info_response.model_info.model_name == model_name
        assert len(info_response.model_info.available_replicas) == 1
        assert (
            info_response.model_info.available_replicas[0].node_id
            == memory_info.node_id
        )

    def test_get_nonexistent_model_info(self, servicer, test_context):
        """Test getting info for a model that doesn't exist"""
        info_request = global_store_pb2.GetModelInfoRequest(
            model_name="nonexistent_model"
        )
        info_response = servicer.GetModelInfo(info_request, test_context)

        assert info_response.status == global_store_pb2.Status.NOT_FOUND
        assert len(info_response.model_info.available_replicas) == 0
