#  Copyright (c) 2025, StepCast Team.

"""gRPC client for Global Store communication.

This module provides a client interface for the Web UI backend to communicate
with the Global Store via gRPC, eliminating direct DuckDB access.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

# from typing
from typing import Callable, Optional, TypeVar

import grpc

from scstore.logger import init_logger
from scstore.proto import global_store_pb2, global_store_pb2_grpc  # noqa: E402

# Generic type for synchronous RPC return types
T = TypeVar("T")


logger = init_logger(__name__)


class WorkerInfoWrapper:
    """Wrapper for WorkerInfo with strict timestamp normalization.

    This wrapper enforces that `last_heartbeat_timestamp` provided by the server
    is a Unix timestamp in SECONDS, as specified in the proto definition.

    Rules:
    - Accept only values in [0, 4102444800] (0 or a sensible seconds-based epoch).
      4102444800 corresponds to 2100-01-01 UTC to catch millisecond inputs.
    - Value 0 means "unknown/not set".
    - Expose both the raw seconds via `last_heartbeat_timestamp` (int) and
      a convenience `last_heartbeat_datetime` (Optional[datetime]).
    - Any value outside the valid range raises ValueError.
    """

    def __init__(
        self, worker_info: global_store_pb2.ListActiveWorkersResponse.WorkerInfo
    ):
        self._worker_info = worker_info

        raw_ts = int(worker_info.last_heartbeat_timestamp)

        # Validate strictly against seconds-based range; reject millisecond-style values
        # 0 is allowed (unknown / not set)
        upper_bound_seconds = 4102444800  # 2100-01-01 UTC
        if raw_ts < 0 or raw_ts > upper_bound_seconds:
            raise ValueError(
                f"Invalid last_heartbeat_timestamp: {raw_ts}. Expected Unix seconds in [0, {upper_bound_seconds}]"
            )

        self._last_heartbeat_seconds: int = raw_ts
        self._last_heartbeat_dt: Optional[datetime]
        if raw_ts == 0:
            self._last_heartbeat_dt = None
        else:
            self._last_heartbeat_dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)

    def __getattr__(self, name: str):
        """Forward all other attributes to the underlying WorkerInfo."""
        return getattr(self._worker_info, name)

    @property
    def worker_id(self) -> str:
        return self._worker_info.worker_id

    @property
    def accepting_new_requests(self) -> bool:
        return self._worker_info.accepting_new_requests

    @property
    def mem_pool_available_size(self) -> int:
        return self._worker_info.mem_pool_available_size

    @property
    def last_heartbeat_timestamp(self) -> int:
        """Return the last heartbeat as Unix seconds (strict)."""
        return self._last_heartbeat_seconds

    @property
    def last_heartbeat_datetime(self) -> Optional[datetime]:
        """Convenience accessor for the last heartbeat as datetime (UTC)."""
        return self._last_heartbeat_dt


@dataclass
class GlobalStoreClientConfig:
    """Configuration for Global Store gRPC client."""

    host: str = "127.0.0.1"
    port: int = 50051
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    @property
    def address(self) -> str:
        """Get the full gRPC address."""
        return f"{self.host}:{self.port}"


class GlobalStoreClient:
    """Async wrapper around synchronous gRPC client for Global Store operations."""

    def __init__(self, config: GlobalStoreClientConfig) -> None:
        """Initialize the client with configuration.

        Parameters
        ----------
        config : GlobalStoreClientConfig
                Client configuration
        """
        self.config = config
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[global_store_pb2_grpc.GlobalModelStoreStub] = None
        self._connect_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def connect(self) -> None:
        """Establish connection to the Global Store."""
        async with self._connect_lock:
            if self._channel is not None:
                return

            logger.info("Connecting to Global Store at %s", self.config.address)

            def _sync_connect():
                channel = grpc.insecure_channel(self.config.address)
                stub = global_store_pb2_grpc.GlobalModelStoreStub(channel)

                # Use dedicated HealthCheck RPC for connectivity verification
                request = global_store_pb2.HealthCheckRequest()
                stub.HealthCheck(request, timeout=5.0)
                return channel, stub

            try:
                # Run synchronous connection in thread pool
                loop = asyncio.get_event_loop()
                self._channel, self._stub = await loop.run_in_executor(
                    self._executor, _sync_connect
                )
                logger.info("Successfully connected to Global Store")
            except Exception as e:
                logger.error("Failed to connect to Global Store: %s", e)
                await self.close()
                raise

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel is not None:
            # Run channel close in executor since it's synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._channel.close)
            self._channel = None
            self._stub = None

        # Shutdown the executor
        self._executor.shutdown(wait=False)

    async def ensure_connected(self) -> None:
        """Ensure the client is connected."""
        if self._channel is None:
            await self.connect()

    async def _execute_with_retry(self, sync_operation: Callable[[], T]) -> T:
        """Execute a synchronous operation with retry logic.

        This helper ensures that synchronous RPC operations executed in a
        thread pool are retried with the configured policy and that the
        original return type *T* is preserved for the caller.
        """

        last_error: Optional[grpc.RpcError] = None
        for attempt in range(self.config.max_retries):
            try:
                await self.ensure_connected()
                # Run the synchronous operation in the thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._executor, sync_operation)
            except grpc.RpcError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        "gRPC call failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.config.max_retries,
                        e,
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error(
                        "gRPC call failed after %d attempts", self.config.max_retries
                    )

        # If we reach here, all retries have failed; re-raise the last error.
        assert (
            last_error is not None
        )  # For type checker – last_error is set if we exit loop
        raise last_error

    # Worker Management Methods

    async def list_active_workers(
        self, include_unavailable: bool = False
    ) -> list[WorkerInfoWrapper]:
        """List active workers in the Global Store.

        Parameters
        ----------
        include_unavailable : bool
                Whether to include workers not accepting new requests

        Returns
        -------
        list[WorkerInfoWrapper]
                List of worker information with normalized timestamps
        """

        def _call():
            request = global_store_pb2.ListActiveWorkersRequest(
                include_unavailable=include_unavailable
            )
            response = self._stub.ListActiveWorkers(  # type: ignore[union-attr]
                request, timeout=self.config.timeout
            )
            # Wrap each worker info to normalize and validate timestamps
            return [WorkerInfoWrapper(worker) for worker in response.workers]

        return await self._execute_with_retry(_call) or []

    # Model Replica Methods

    async def list_model_replicas(
        self,
        model_id: Optional[str] = None,
        node_id: Optional[str] = None,
        memory_type: Optional[int] = None,  # Use int instead of proto enum
        device_id: Optional[int] = None,
    ) -> dict[str, list[global_store_pb2.MemoryInfo]]:
        """List model replicas with optional filters.

        Parameters
        ----------
        model_id : str, optional
                Filter by content-addressed model ID (mi2:...)
        node_id : str, optional
                Filter by node ID
        memory_type : MemoryType, optional
                Filter by memory type (GPU, RAM, DISK)
        device_id : int, optional
                Filter by device ID

        Returns
        -------
        dict[str, list[MemoryInfo]]
                Map of model_id to their replica memory info
        """

        def _call():
            request = global_store_pb2.ListModelReplicasRequest()

            if model_id is not None:
                request.model_id = model_id
            if node_id is not None:
                request.node_id = node_id
            if memory_type is not None:
                request.memory_type = memory_type  # type: ignore[assignment]
            if device_id is not None:
                request.device_id = device_id

            response = self._stub.ListModelReplicas(  # type: ignore[union-attr]
                request, timeout=self.config.timeout
            )

            # Convert to Python dict
            result = {}
            for mid, mem_list in response.model_replicas.items():
                result[mid] = list(mem_list.list)
            return result

        return await self._execute_with_retry(_call) or {}

    async def get_model_info(
        self, model_id: str
    ) -> Optional[list[global_store_pb2.MemoryInfo]]:
        """Get replicas information for a specific content-addressed model.

        Parameters
        ----------
        model_id : str
                Content-addressed model ID (mi2:...)

        Returns
        -------
        list[MemoryInfo] or None
                Replica memory info list if found
        """

        def _call():
            request = global_store_pb2.GetModelInfoByIdRequest(model_id=model_id)
            response = self._stub.GetModelInfoById(  # type: ignore[union-attr]
                request, timeout=self.config.timeout
            )

            if response.status == global_store_pb2.Status.OK:
                return list(response.replicas)
            elif response.status == global_store_pb2.Status.NOT_FOUND:
                return None
            else:
                raise RuntimeError(
                    f"Failed to get model info by id: status={response.status}"
                )

        return await self._execute_with_retry(_call)

    # Transport Methods (for monitoring)

    async def get_active_transports(self) -> list[dict]:
        """Get active transport operations.

        Note: The current proto doesn't have a ListTransports RPC,
        so this is a placeholder that would need to be implemented
        either by adding a new RPC or tracking transports locally.

        Returns
        -------
        list[dict]
                List of active transport operations
        """
        # TODO: Implement when ListTransports RPC is available
        logger.warning("get_active_transports not implemented - proto needs extension")
        return []

    # Summary Methods

    async def get_summary_stats(self) -> dict:
        """Get summary statistics for the dashboard.

        Returns
        -------
        dict
                Summary statistics including worker count, replica count, etc.
        """
        # Get workers
        workers = await self.list_active_workers(include_unavailable=True)
        active_workers = [w for w in workers if w.accepting_new_requests]

        # Get all replicas
        all_replicas = await self.list_model_replicas()

        # Calculate statistics
        total_replicas = sum(len(replicas) for replicas in all_replicas.values())
        gpu_replicas = 0
        ram_replicas = 0
        disk_replicas = 0

        for replicas in all_replicas.values():
            for replica in replicas:
                if replica.memory_type == global_store_pb2.MemoryType.GPU:
                    gpu_replicas += 1
                elif replica.memory_type == global_store_pb2.MemoryType.RAM:
                    ram_replicas += 1
                elif replica.memory_type == global_store_pb2.MemoryType.DISK:
                    disk_replicas += 1

        return {
            "total_workers": len(workers),
            "active_workers": len(active_workers),
            "total_models": len(all_replicas),
            "total_replicas": total_replicas,
            "gpu_replicas": gpu_replicas,
            "ram_replicas": ram_replicas,
            "disk_replicas": disk_replicas,
            "active_transports": 0,  # TODO: Implement when available
        }


# Singleton instance
_client: Optional[GlobalStoreClient] = None
_client_lock = asyncio.Lock()


async def get_global_store_client(config: GlobalStoreClientConfig) -> GlobalStoreClient:
    """Get or create the singleton Global Store client.

    Parameters
    ----------
    config : GlobalStoreClientConfig
            Client configuration

    Returns
    -------
    GlobalStoreClient
            The singleton client instance
    """
    global _client

    async with _client_lock:
        if _client is None:
            _client = GlobalStoreClient(config)
            await _client.connect()
        return _client


async def close_global_store_client() -> None:
    """Close the singleton Global Store client."""
    global _client

    async with _client_lock:
        if _client is not None:
            await _client.close()
            _client = None
