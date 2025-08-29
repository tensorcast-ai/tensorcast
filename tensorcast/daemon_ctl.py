#  Copyright (c) 2025, TensorCast Team.


import os

import grpc

import tensorcast.proto.store_daemon_pb2 as store_daemon_pb2
import tensorcast.proto.store_daemon_pb2_grpc as store_daemon_pb2_grpc
from tensorcast.logger import init_logger

logger = init_logger(__name__)


def get_host_pid() -> int:
    """Get the host PID of the current process.

    In containerized environments, the PID inside the container may differ
    from the PID on the host. This function attempts to get the host PID
    by reading the NSpid field from /proc/self/status.

    Returns:
        The host PID if available, otherwise falls back to os.getpid()
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    # NSpid shows PIDs in different namespaces
                    # The last value is typically the host PID
                    pids = line.strip().split()[1:]
                    if pids:
                        return int(pids[-1])
    except (IOError, OSError, ValueError) as e:
        logger.warning(f"Failed to get host PID: {e}, falling back to os.getpid()")

    # Fallback to regular PID if host PID cannot be determined
    return os.getpid()


# This is a singleton class that manages the checkpoint
class DaemonCtl:
    def __init__(self, server_address="127.0.0.1:8073"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = store_daemon_pb2_grpc.StoreDaemonStub(self.channel)
        self.checkpoints_in_gpu = {}

        # Check environment variable
        env_use_host_pid = os.environ.get("TENSORCAST_USE_HOST_PID", "").lower()
        self.use_host_pid = env_use_host_pid in ("true", "1", "yes")

        if self.use_host_pid:
            logger.info("DaemonCtl configured to use host PID")

    def __del__(self):
        # TODO: cleanup
        pass

    def unload_from_cpu(self, disk_path):
        request = store_daemon_pb2.UnloadReplicaRequest(
            disk_path=disk_path,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.UnloadReplica(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return False
        else:
            return response

    def load_into_gpu(
        self,
        disk_path: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
    ):
        logger.debug(
            f"load_into_gpu: {disk_path}, {replica_uuid}, wait_for_completion={wait_for_completion}"
        )

        # Choose between host PID and regular PID based on the parameter
        pid = get_host_pid() if self.use_host_pid else os.getpid()
        if self.use_host_pid:
            logger.debug(f"Using host PID: {pid}")
        else:
            logger.debug(f"Using container PID: {pid}")

        request = store_daemon_pb2.MaterializeReplicaRequest(
            pid=pid,
            disk_path=disk_path,
            replica_uuid=replica_uuid,
            device_uuid=device_uuid,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            keep_for_global=False,
        )
        try:
            response = self.stub.MaterializeReplica(request, timeout=60)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                raise RuntimeError(f"Artifact not loaded {e}") from e
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError(
                    f"Local StoreDaemon ({self.server_address}) is not available."
                ) from e
            else:
                raise RuntimeError(f"Error: {e}") from e

        load_status = response.status
        if (
            load_status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_FAILED
        ):
            raise RuntimeError(f"Artifact allocation failed for {disk_path}")

        if not wait_for_completion:
            # In async mode, return both handle and status
            logger.info(
                f"Artifact allocation initiated (async): {disk_path}, {replica_uuid}"
            )

            assert response.mem_handle is not None
            return response.mem_handle.cuda_ipc_handle, load_status

        # In sync mode, wait for confirmation
        logger.info(f"Artifact loaded: {disk_path}, {replica_uuid}")

        # For sync mode with new async backend, we need to confirm
        if (
            response.status
            == store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        ):
            # Wait for asynchronous loading to complete
            success = self.confirm_replica_loaded(disk_path, replica_uuid)
            if not success:
                raise RuntimeError(
                    f"Failed to confirm artifact loading for {disk_path}"
                )

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def confirm_replica_loaded(self, disk_path: str, replica_uuid: str) -> bool:
        logger.info(f"confirm_replica_loaded: {disk_path}, {replica_uuid}")
        request = store_daemon_pb2.ConfirmReplicaRequest(
            disk_path=disk_path,
            replica_uuid=replica_uuid,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            _ = self.stub.ConfirmReplica(request)
            logger.info("Artifact loaded")
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error("Artifact not loaded")
                return False
            else:
                logger.error(f"Error: {e}")
                return False

    def get_server_config(self):
        request = store_daemon_pb2.GetServerConfigRequest()
        try:
            response = self.stub.GetServerConfig(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return None
        else:
            return {
                "chunk_size": response.chunk_size,
                "mem_pool_size": response.mem_pool_size,
            }

    # ------------------------------------------------------------------
    # Memory Artifact registration (outer-layer client API)
    # ------------------------------------------------------------------

    def begin_register_artifact(
        self,
        *,
        artifact_id: str,
        device_id: int,
        total_size_bytes: int,
        enable_p2p: bool = False,
        ttl_ms: int | None = None,
        tensor_index_key: str | None = None,
        tensor_index_data: bytes | None = None,
        encoding: str = "json",
        schema_version: str = "v2",
        timeout_s: float = 30.0,
    ) -> dict:
        """Begin registration of an in-memory tensor dict.

        Returns a dict with keys: registration_id, daemon_ipc_handle, device_id, size_bytes.
        """

        if not artifact_id or device_id < 0 or total_size_bytes <= 0:
            raise ValueError("Invalid arguments for begin_register_artifact")

        if not tensor_index_key and tensor_index_data is None:
            raise ValueError(
                "Either tensor_index_key or tensor_index_data must be provided"
            )

        req = store_daemon_pb2.BeginRegisterArtifactRequest(
            artifact_id=artifact_id,
            device_id=int(device_id),
            total_size=int(total_size_bytes),
            enable_p2p=bool(enable_p2p),
        )

        # Optional TTL presence should be explicit only when provided
        if ttl_ms is not None:
            req.ttl_ms = int(ttl_ms)

        # Oneof index: key vs data
        if tensor_index_data is not None:
            req.tensor_index_data.CopyFrom(
                store_daemon_pb2.TensorIndexData(
                    data=tensor_index_data,
                    schema_version=schema_version,
                    encoding=encoding,
                )
            )
        else:
            req.tensor_index_key = tensor_index_key or ""

        try:
            resp = self.stub.BeginRegisterArtifact(req, timeout=timeout_s)
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError(
                    f"Local StoreDaemon ({self.server_address}) is not available."
                ) from e
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(str(e)) from e
            if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise MemoryError(str(e)) from e
            if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(str(e)) from e
            # if code == grpc.StatusCode.NOT_FOUND:
            # raise FileNotFoundError(str(e)) from e
            raise RuntimeError(f"BeginRegisterArtifact failed: {e}") from e

        return {
            "registration_id": resp.registration_id,
            "daemon_ipc_handle": bytes(resp.daemon_ipc_handle),
            "device_id": int(resp.device_id),
            "size_bytes": int(resp.size),
        }

    def commit_registered_artifact(
        self,
        registration_id: str,
        *,
        timeout_s: float = 30.0,
    ) -> dict:
        """Commit a previously begun tensor dict registration.

        Returns a dict with keys: registration_id, artifact_id, device_id, size_bytes, descriptor.
        """

        if not registration_id:
            raise ValueError("registration_id is required")

        req = store_daemon_pb2.CommitRegisteredArtifactRequest(
            registration_id=registration_id
        )
        try:
            resp = self.stub.CommitRegisteredArtifact(req, timeout=timeout_s)
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError(
                    f"Local StoreDaemon ({self.server_address}) is not available."
                ) from e
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(str(e)) from e
            if code == grpc.StatusCode.NOT_FOUND:
                raise KeyError(str(e)) from e
            if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(str(e)) from e
            raise RuntimeError(f"CommitRegisteredArtifact failed: {e}") from e

        desc = resp.descriptor
        assert desc is not None
        descriptor_dict = {
            "artifact_id": desc.artifact_id,
            "index_multihash": desc.index_multihash,
            "data_multihash": desc.data_multihash,
            "schema_version": desc.schema_version,
            "encoding": desc.encoding,
            "total_size": int(desc.total_size),
        }
        return {
            "registration_id": resp.registration_id,
            "artifact_id": resp.artifact_id,
            "device_id": int(resp.device_id),
            "size_bytes": int(resp.size),
            "descriptor": descriptor_dict,
        }

    def abort_registered_artifact(
        self, registration_id: str, *, timeout_s: float = 15.0
    ) -> bool:
        """Abort a pending tensor dict registration and free allocated memory."""
        if not registration_id:
            raise ValueError("registration_id is required")

        req = store_daemon_pb2.AbortRegisteredArtifactRequest(
            registration_id=registration_id
        )
        try:
            resp = self.stub.AbortRegisteredArtifact(req, timeout=timeout_s)
        except grpc.RpcError as e:
            code = e.code()
            if code == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError(
                    f"Local StoreDaemon ({self.server_address}) is not available."
                ) from e
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(str(e)) from e
            if code == grpc.StatusCode.NOT_FOUND:
                # Treat as already-aborted/missing
                logger.warning(
                    "AbortRegisteredArtifact: registration not found: %s",
                    registration_id,
                )
                return False
            raise RuntimeError(f"AbortRegisteredArtifact failed: {e}") from e

        return bool(resp.ok)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def wait_artifact_verification(
        self,
        artifact_identifier: str,
        replica_uuid: str,
        timeout_ms: int = 30000,
    ) -> store_daemon_pb2.ReplicaVerificationResponse | None:
        """Block until the daemon returns a PASSED/FAILED status or timeout."""

        request = store_daemon_pb2.ReplicaVerificationRequest(
            artifact_id=artifact_identifier,
            replica_uuid=replica_uuid,
            timeout_ms=timeout_ms,
        )

        try:
            response = self.stub.WaitReplicaVerification(
                request, timeout=timeout_ms / 1000 + 5
            )
            return response
        except grpc.RpcError as e:
            logger.error(f"wait_artifact_verification RPC failed: {e}")
            return None
