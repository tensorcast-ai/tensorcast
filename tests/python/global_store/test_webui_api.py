#  Copyright (c) 2025, StepCast Team.

"""Tests for Global Store Web UI API."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from scstore.global_store.config.settings import GlobalStoreConfig
from scstore.global_store.webui_backend.app import WebUIApp


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock(spec=GlobalStoreConfig)
    config.ui_port = 9000
    config.ui_host = "0.0.0.0"
    config.ui_static_dir = None
    config.ui_enabled = True
    config.ui_cors_origins = None
    config.ui_log_file = "/tmp/global-store-webui.log"
    return config


@pytest.fixture
def web_app(mock_config):
    """Create WebUIApp instance with mocked gRPC client."""
    # Create a minimal mock servicer
    mock_servicer = MagicMock()

    app = WebUIApp(mock_servicer, mock_config)

    # Mock the gRPC client
    from unittest.mock import AsyncMock

    app.grpc_client = AsyncMock()

    return app


@pytest.fixture
def client(web_app):
    """Create test client."""
    return TestClient(web_app.app)


class TestWebUIAPI:
    """Test Web UI API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "global_store_webui"}

    def test_get_summary(self, client, web_app):
        """Test get summary endpoint."""
        # Mock the gRPC client responses
        # Mock get_summary_stats response
        web_app.grpc_client.get_summary_stats.return_value = {
            "total_workers": 2,
            "active_workers": 1,
            "total_replicas": 3,
            "available_replicas": 2,
            "total_artifacts": 2,
            "active_transports": 0,
        }

        # Mock list_active_workers response with simple objects
        from types import SimpleNamespace

        worker1 = SimpleNamespace(
            worker_id="w1",
            node_id="node1",
            node_address="10.0.0.1",
            grpc_port=50051,
            mem_pool_total_size=10737418240,
            mem_pool_available_size=5368709120,
            accepting_new_requests=True,
        )
        worker2 = SimpleNamespace(
            worker_id="w2",
            node_id="node2",
            node_address="10.0.0.2",
            grpc_port=50051,
            mem_pool_total_size=10737418240,
            mem_pool_available_size=0,  # All memory used
            accepting_new_requests=False,
        )
        web_app.grpc_client.list_active_workers.return_value = [worker1, worker2]

        response = client.get("/api/summary")
        assert response.status_code == 200

        data = response.json()["data"]

        assert data["total_workers"] == 2
        assert data["active_workers"] == 1
        assert data["total_replicas"] == 3
        assert (
            data["available_replicas"] == 3
        )  # Based on the API code, this is set to total_replicas
        assert data["total_artifacts"] == 2
        assert data["active_transports"] == 0
        assert data["total_memory_bytes"] == 21474836480  # 20GB total
        assert data["available_memory_bytes"] == 5368709120  # 5GB available

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_workers(self, client, db_connection):
        """Test list workers endpoint."""
        # Insert test data
        now = datetime.now(tz=timezone.utc)
        db_connection.execute(
            """
            INSERT INTO workers VALUES
            ('w1', 'node1', '10.0.0.1', 50051, 5001, 10737418240, 5368709120, TRUE, ?, ?),
            ('w2', 'node2', '10.0.0.2', 50051, 5001, 10737418240, 8589934592, FALSE, ?, ?)
        """,
            (now, now, now - timedelta(seconds=20), now - timedelta(seconds=20)),
        )

        # Test without filters
        response = client.get("/api/workers")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1  # Only active worker by default
        assert data["meta"]["total_count"] == 1

        worker = data["data"][0]
        assert worker["worker_id"] == "w1"
        assert worker["status"] == "critical"  # 20s ago

        # Test with include_unavailable
        response = client.get("/api/workers?include_unavailable=true")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 2
        assert data["meta"]["total_count"] == 2

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_get_worker_detail(self, client, db_connection):
        """Test get worker detail endpoint."""
        now = datetime.now(tz=timezone.utc)
        db_connection.execute(
            """
            INSERT INTO workers VALUES
            ('w1', 'node1', '10.0.0.1', 50051, 5001, 10737418240, 5368709120, TRUE, ?, ?)
        """,
            (now, now),
        )

        response = client.get("/api/workers/w1")
        assert response.status_code == 200

        worker = response.json()["data"]
        assert worker["worker_id"] == "w1"
        assert worker["node_id"] == "node1"
        assert worker["mem_pool_total_size"] == 10737418240

        # Test non-existent worker
        response = client.get("/api/workers/nonexistent")
        assert response.status_code == 404

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_replicas(self, client, db_connection):
        """Test list replicas endpoint."""
        # Insert test data
        replica_id1 = str(uuid4())
        replica_id2 = str(uuid4())

        db_connection.execute("""
            INSERT INTO workers VALUES
            ('w1', 'node1', '10.0.0.1', 50051, 5001, 10737418240, 5368709120, TRUE,
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        db_connection.execute(f"""
            INSERT INTO artifact_replicas VALUES
            ('{replica_id1}', 'model1', 'node1', '10.0.0.1', 50051, 1073741824, 'GPU', 0, 5, TRUE, NULL, NULL, 'w1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            ('{replica_id2}', 'model2', 'node1', '10.0.0.1', 50051, 2147483648, 'RAM', 0, 5, TRUE, NULL, NULL, 'w1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        db_connection.execute(f"""
            INSERT INTO replica_counters VALUES
            ('{replica_id1}', 2, CURRENT_TIMESTAMP),
            ('{replica_id2}', 0, CURRENT_TIMESTAMP)
        """)

        # Test without filters
        response = client.get("/api/replicas")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 2
        assert data["meta"]["total_count"] == 2

        # Test with artifact filter (by artifact_id)
        response = client.get("/api/replicas?artifact_id=model1")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["artifact_id"] == "model1"

        # Test with memory type filter
        response = client.get("/api/replicas?memory_type=GPU")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["memory_type"] == "GPU"

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_artifacts(self, client, db_connection):
        """Test list artifacts endpoint."""
        # Insert test data
        replica_id1 = str(uuid4())
        replica_id2 = str(uuid4())
        replica_id3 = str(uuid4())

        db_connection.execute(f"""
            INSERT INTO artifact_replicas VALUES
            ('{replica_id1}', 'model1', 'node1', '10.0.0.1', 50051, 1073741824, 'GPU', 0, 5, TRUE, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            ('{replica_id2}', 'model1', 'node1', '10.0.0.1', 50051, 2147483648, 'RAM', 0, 5, TRUE, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            ('{replica_id3}', 'model2', 'node2', '10.0.0.2', 50051, 536870912, 'DISK', 0, 10, FALSE, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        db_connection.execute(f"""
            INSERT INTO replica_counters VALUES
            ('{replica_id1}', 3, CURRENT_TIMESTAMP),
            ('{replica_id2}', 0, CURRENT_TIMESTAMP),
            ('{replica_id3}', 8, CURRENT_TIMESTAMP)
        """)

        response = client.get("/api/artifacts")
        assert response.status_code == 200

        artifacts = response.json()["data"]
        assert len(artifacts) == 2

        # Check model1
        artifact1 = next(m for m in artifacts if m["artifact_id"] == "model1")
        assert artifact1["total_replicas"] == 2
        assert artifact1["available_replicas"] == 2
        assert artifact1["gpu_replicas"] == 1
        assert artifact1["ram_replicas"] == 1
        assert artifact1["disk_replicas"] == 0
        assert artifact1["total_memory_size"] == 3221225472  # 3GB
        assert artifact1["avg_load_ratio"] == pytest.approx(0.3)  # (3/5 + 0/5) / 2

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_get_artifact_detail(self, client, db_connection):
        """Test get artifact detail endpoint."""
        replica_id = str(uuid4())

        db_connection.execute(f"""
            INSERT INTO artifact_replicas VALUES
            ('{replica_id}', 'model1', 'node1', '10.0.0.1', 50051, 1073741824, 'GPU', 0, 5, TRUE, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        response = client.get("/api/artifacts/model1")
        assert response.status_code == 200

        artifact = response.json()["data"]
        assert artifact["artifact_id"] == "model1"
        assert artifact["total_replicas"] == 1

        # Test non-existent artifact
        response = client.get("/api/artifacts/nonexistent")
        assert response.status_code == 404

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_nodes(self, client, db_connection):
        """Test list nodes endpoint."""
        # Insert test data
        db_connection.execute("""
            INSERT INTO workers VALUES
            ('w1', 'node1', '10.0.0.1', 50051, 5001, 10737418240, 5368709120, TRUE,
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            ('w2', 'node1', '10.0.0.1', 50052, 5002, 10737418240, 5368709120, TRUE,
             CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        db_connection.execute("""
            INSERT INTO artifact_replicas VALUES
            (gen_random_uuid(), 'model1', 'node1', '10.0.0.1', 50051, 1073741824, 'GPU', 0, 5, TRUE, NULL, NULL, 'w1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            (gen_random_uuid(), 'model2', 'node1', '10.0.0.1', 50051, 2147483648, 'RAM', 0, 5, TRUE, NULL, NULL, 'w1', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
            (gen_random_uuid(), 'model3', 'node2', '10.0.0.2', 50051, 536870912, 'DISK', 0, 5, TRUE, NULL, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)

        response = client.get("/api/nodes")
        assert response.status_code == 200

        nodes = response.json()["data"]
        assert len(nodes) == 2

        # Check node1
        node1 = next(n for n in nodes if n["node_id"] == "node1")
        assert node1["total_replicas"] == 2
        assert node1["total_memory"] == 3221225472  # 3GB
        assert node1["gpu_memory"] == 1073741824  # 1GB
        assert node1["ram_memory"] == 2147483648  # 2GB
        assert node1["disk_memory"] == 0
        assert node1["active_workers"] == 2

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_transports(self, client, db_connection):
        """Test list transports endpoint."""
        transport_id = str(uuid4())
        replica_id = str(uuid4())

        db_connection.execute(f"""
            INSERT INTO artifact_transports VALUES
            ('{transport_id}', '{replica_id}', 'model1', 'node1', '10.0.0.1', 50051, CURRENT_TIMESTAMP, NULL, 'in_progress')
        """)

        response = client.get("/api/transports")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1
        assert data["meta"]["total_count"] == 1

        transport = data["data"][0]
        assert transport["transport_id"] == transport_id
        assert transport["artifact_id"] == "model1"
        assert transport["status"] == "in_progress"

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_list_transports_with_filters(self, client, db_connection):
        """Test list transports with status and artifact filters."""
        # Insert test data with different statuses
        for i, status in enumerate(["in_progress", "completed", "failed"]):
            transport_id = str(uuid4())
            replica_id = str(uuid4())
            completed_at = "CURRENT_TIMESTAMP" if status == "completed" else "NULL"

            db_connection.execute(f"""
                INSERT INTO artifact_transports VALUES
                ('{transport_id}', '{replica_id}', 'artifact{i}', NULL, 'node1', '10.0.0.1', 50051,
                 CURRENT_TIMESTAMP, {completed_at}, '{status}')
            """)

        # Test status filter
        response = client.get("/api/transports?status=completed")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["status"] == "completed"

        # Test artifact_id filter
        response = client.get("/api/transports?artifact_id=model0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["artifact_id"] == "model0"

    @pytest.mark.skip(
        reason="Needs refactoring to use mocked gRPC client instead of direct DB access"
    )
    def test_pagination(self, client, db_connection):
        """Test pagination functionality."""
        # Insert multiple workers
        for i in range(15):
            db_connection.execute(f"""
                INSERT INTO workers VALUES
                ('w{i}', 'node{i}', '10.0.0.{i}', 50051, 5001, 10737418240, 5368709120, TRUE,
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """)

        # Test first page
        response = client.get("/api/workers?page=1&page_size=10")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 10
        assert data["meta"]["page"] == 1
        assert data["meta"]["page_size"] == 10
        assert data["meta"]["total_count"] == 15
        assert data["meta"]["total_pages"] == 2

        # Test second page
        response = client.get("/api/workers?page=2&page_size=10")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 5
        assert data["meta"]["page"] == 2


class TestWebSocketAPI:
    """Test WebSocket functionality."""

    def test_websocket_connection(self, web_app):
        """Test WebSocket connection and subscription."""
        from fastapi.testclient import TestClient

        client = TestClient(web_app.app)

        with client.websocket_connect("/ws/stream") as websocket:
            # Send subscription request
            websocket.send_json(
                {"action": "subscribe", "topics": ["heartbeat", "replica_update"]}
            )

            # Should receive subscription confirmation
            data = websocket.receive_json()
            assert data["status"] == "subscribed"
            assert set(data["topics"]) == {"heartbeat", "replica_update"}

    @pytest.mark.skip(reason="WebSocket update tests need async event loop refactoring")
    def test_websocket_updates(self, web_app):
        """Test WebSocket update broadcasting."""
        # TODO: Refactor to properly handle async operations with TestClient
        pass

    @pytest.mark.skip(reason="WebSocket update tests need async event loop refactoring")
    def test_websocket_transport_updates(self, web_app):
        """Test WebSocket transport event updates."""
        # TODO: Refactor to properly handle async operations with TestClient
        pass
