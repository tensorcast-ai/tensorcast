#  Copyright (c) 2025, StepCast Team.

"""Tests for replica reference tracking data structures."""

import time
import pytest

from scstore.store_daemon.replica_ref import ReplicaKey, ReplicaRefInfo


class TestReplicaKey:
    """Test cases for ReplicaKey."""

    def test_replica_key_creation(self):
        """Test creating a replica key."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        assert key.artifact_id == "model1"
        assert key.device_id == 0

    def test_replica_key_immutable(self):
        """Test that replica key is immutable."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        with pytest.raises(AttributeError):
            key.artifact_id = "model2"  # pyright: ignore[reportAttributeAccessIssue]

    def test_replica_key_equality(self):
        """Test replica key equality."""
        key1 = ReplicaKey(artifact_id="model1", device_id=0)
        key2 = ReplicaKey(artifact_id="model1", device_id=0)
        key3 = ReplicaKey(artifact_id="model2", device_id=0)

        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

    def test_replica_key_str(self):
        """Test string representation."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        assert str(key) == "model1@device_0"


class TestReplicaRefInfo:
    """Test cases for ReplicaRefInfo."""

    def test_replica_ref_info_creation(self):
        """Test creating replica reference info."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024, keep_for_global=True)

        assert info.key == key
        assert info.size_bytes == 1024
        assert info.keep_for_global is True
        assert info.ref_count == 0
        assert len(info.pids) == 0
        assert isinstance(info.last_access_ts, float)

    def test_add_pid(self):
        """Test adding PID references."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        # Add first PID
        assert info.add_pid(1234) is True
        assert info.ref_count == 1
        assert 1234 in info.pids

        # Add same PID again - should not increase ref count
        assert info.add_pid(1234) is False
        assert info.ref_count == 1

        # Add different PID
        assert info.add_pid(5678) is True
        assert info.ref_count == 2
        assert 5678 in info.pids

    def test_remove_pid(self):
        """Test removing PID references."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        # Add PIDs
        info.add_pid(1234)
        info.add_pid(5678)
        assert info.ref_count == 2

        # Remove existing PID
        assert info.remove_pid(1234) is True
        assert info.ref_count == 1
        assert 1234 not in info.pids
        assert 5678 in info.pids

        # Remove non-existent PID
        assert info.remove_pid(9999) is False
        assert info.ref_count == 1

        # Remove last PID
        assert info.remove_pid(5678) is True
        assert info.ref_count == 0
        assert len(info.pids) == 0

    def test_ref_count_never_negative(self):
        """Test that ref count never goes negative."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        # Remove PID when no PIDs exist
        assert info.remove_pid(1234) is False
        assert info.ref_count == 0  # Should stay at 0

    def test_is_evictable(self):
        """Test evictable status."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        # No references - evictable
        assert info.is_evictable() is True

        # Add reference - not evictable
        info.add_pid(1234)
        assert info.is_evictable() is False

        # Remove reference - evictable again
        info.remove_pid(1234)
        assert info.is_evictable() is True

    def test_touch_updates_timestamp(self):
        """Test that touch updates last access timestamp."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        initial_ts = info.last_access_ts
        time.sleep(0.01)  # Small delay to ensure timestamp difference

        info.touch()
        assert info.last_access_ts > initial_ts

    def test_add_pid_updates_timestamp(self):
        """Test that adding PID updates timestamp."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024)

        initial_ts = info.last_access_ts
        time.sleep(0.01)

        info.add_pid(1234)
        assert info.last_access_ts > initial_ts

    def test_str_representation(self):
        """Test string representation."""
        key = ReplicaKey(artifact_id="model1", device_id=0)
        info = ReplicaRefInfo(key=key, size_bytes=1024, keep_for_global=True)
        info.add_pid(1234)
        info.add_pid(5678)

        str_repr = str(info)
        assert "model1@device_0" in str_repr
        assert "refs=2" in str_repr
        assert "pids={1234, 5678}" in str_repr or "pids={5678, 1234}" in str_repr
        assert "keep_global=True" in str_repr
