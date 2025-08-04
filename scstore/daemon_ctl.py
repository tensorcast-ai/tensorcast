#  Copyright (c) 2025, StepCast Team.


import os

import grpc

import scstore.proto.store_daemon_pb2 as store_daemon_pb2
import scstore.proto.store_daemon_pb2_grpc as store_daemon_pb2_grpc
from scstore.logger import init_logger

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
        env_use_host_pid = os.environ.get("SCSTORE_USE_HOST_PID", "").lower()
        self.use_host_pid = env_use_host_pid in ("true", "1", "yes")

        if self.use_host_pid:
            logger.info("DaemonCtl configured to use host PID")

    def __del__(self):
        # TODO: cleanup
        pass

    def unload_from_cpu(self, model_path):
        request = store_daemon_pb2.UnloadModelRequest(
            model_path=model_path,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.UnloadModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return False
        else:
            return response

    def load_into_gpu(
        self,
        model_path: str,
        replica_uuid: str,
        device_uuid: str,
        pinned_allocation_timeout_ms: int = int(30e3),
        wait_for_completion: bool = True,
    ):
        logger.debug(
            f"load_into_gpu: {model_path}, {replica_uuid}, wait_for_completion={wait_for_completion}"
        )

        # Choose between host PID and regular PID based on the parameter
        pid = get_host_pid() if self.use_host_pid else os.getpid()
        if self.use_host_pid:
            logger.debug(f"Using host PID: {pid}")
        else:
            logger.debug(f"Using container PID: {pid}")

        request = store_daemon_pb2.LoadModelRequest(
            pid=pid,
            model_path=model_path,
            replica_uuid=replica_uuid,
            device_uuid=device_uuid,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
            pinned_allocation_timeout_ms=pinned_allocation_timeout_ms,
            keep_for_global=False,
        )
        try:
            response = self.stub.LoadModel(request, timeout=60)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                raise RuntimeError(f"Model not loaded {e}") from e
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                raise RuntimeError(
                    f"Local StoreDaemon ({self.server_address}) is not available."
                ) from e
            else:
                raise RuntimeError(f"Error: {e}") from e

        load_status = response.status
        if load_status == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_FAILED:
            raise RuntimeError(f"Model allocation failed for {model_path}")

        if not wait_for_completion:
            # In async mode, return both handle and status
            logger.info(
                f"Model allocation initiated (async): {model_path}, {replica_uuid}"
            )

            assert response.mem_handle is not None
            return response.mem_handle.cuda_ipc_handle, load_status

        # In sync mode, wait for confirmation
        logger.info(f"Model loaded: {model_path}, {replica_uuid}")

        # For sync mode with new async backend, we need to confirm
        if (
            response.status
            == store_daemon_pb2.LoadModelStatus.LOAD_MODEL_STATUS_ALLOCATED
        ):
            # Call ConfirmModel to wait for completion
            success = self.confirm_model_loaded(model_path, replica_uuid)
            if not success:
                raise RuntimeError(f"Failed to confirm model loading for {model_path}")

        assert response.mem_handle is not None
        return response.mem_handle.cuda_ipc_handle

    def confirm_model_loaded(self, model_path: str, replica_uuid: str) -> bool:
        logger.info(f"confirm_model_loaded: {model_path}, {replica_uuid}")
        request = store_daemon_pb2.ConfirmModelRequest(
            model_path=model_path,
            replica_uuid=replica_uuid,
            target_device_type=store_daemon_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            _ = self.stub.ConfirmModel(request)
            logger.info("Model loaded")
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error("Model not loaded")
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
    # Verification helpers
    # ------------------------------------------------------------------

    def wait_model_verification(
        self,
        model_identifier: str,
        replica_uuid: str,
        timeout_ms: int = 30000,
    ) -> store_daemon_pb2.VerificationResponse | None:
        """Block until the daemon returns a PASSED/FAILED status or timeout."""

        request = store_daemon_pb2.VerificationRequest(
            model_identifier=model_identifier,
            replica_uuid=replica_uuid,
            timeout_ms=timeout_ms,
        )

        try:
            response = self.stub.WaitModelVerification(
                request, timeout=timeout_ms / 1000 + 5
            )
            return response
        except grpc.RpcError as e:
            logger.error(f"wait_model_verification RPC failed: {e}")
            return None
