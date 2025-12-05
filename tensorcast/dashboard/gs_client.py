#  Copyright (c) 2025, TensorCast Team.

"""Async Global Store client used by the dashboard backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Callable

import grpc
from google.protobuf import wrappers_pb2

from tensorcast.dashboard.metrics import GRPC_COUNTER, GRPC_LATENCY
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2, memory_tier_pb2_grpc


class GlobalStoreStatusError(RuntimeError):
    """Raised when a Global Store RPC returns a non-OK status payload."""

    def __init__(self, rpc: str, status_code: int):
        self.rpc = rpc
        self.status_code = status_code
        self.status_name = _status_name(status_code)
        super().__init__(f"{rpc} returned status {self.status_name}")


def _status_name(status_code: int) -> str:
    try:
        return global_store_pb2.Status.Name(status_code)
    except ValueError:
        return f"UNKNOWN_{status_code}"


def _load_ca_cert(path: Path | None) -> bytes | None:
    if path is None:
        return None
    data = path.read_bytes()
    if not data:
        raise ValueError(f"CA certificate at {path} is empty")
    return data


@dataclass(slots=True)
class ClientConfig:
    target: str
    secure: bool
    ca_cert: Path | None
    timeout_sec: float


class GlobalStoreClient:
    """Thin async wrapper around the Global Store gRPC stub."""

    def __init__(self, config: ClientConfig):
        self._config = config
        self._channel: grpc.aio.Channel | None = None
        self._stub: global_store_pb2_grpc.GlobalStoreServiceStub | None = None
        self._memory_tier_stub: memory_tier_pb2_grpc.MemoryTierServiceStub | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialise the gRPC channel and stub if not already created."""

        if self._stub is not None:
            return

        async with self._lock:
            if self._stub is not None:
                return

            if self._config.secure:
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=_load_ca_cert(self._config.ca_cert)
                )
                channel = grpc.aio.secure_channel(self._config.target, credentials)
            else:
                channel = grpc.aio.insecure_channel(self._config.target)

            self._channel = channel
            self._stub = global_store_pb2_grpc.GlobalStoreServiceStub(channel)
            self._memory_tier_stub = memory_tier_pb2_grpc.MemoryTierServiceStub(channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            self._memory_tier_stub = None

    async def _call(
        self,
        rpc_name: str,
        method: Callable[..., Awaitable[object]],
        request: object,
        *,
        timeout: float | None = None,
    ) -> object:
        if self._stub is None:
            raise RuntimeError("GlobalStoreClient.connect() must be called before use")

        deadline = timeout or self._config.timeout_sec
        start = perf_counter()
        try:
            response = await method(request, timeout=deadline)
        except grpc.aio.AioRpcError as exc:  # noqa: PERF203 - explicit error handling
            status = exc.code().name if exc.code() is not None else "UNKNOWN"
            GRPC_COUNTER.labels(rpc=rpc_name, status=status).inc()
            GRPC_LATENCY.labels(rpc=rpc_name).observe(perf_counter() - start)
            raise

        duration = perf_counter() - start
        GRPC_COUNTER.labels(rpc=rpc_name, status="OK").inc()
        GRPC_LATENCY.labels(rpc=rpc_name).observe(duration)
        return response

    async def health_check(self) -> global_store_pb2.HealthCheckResponse:
        await self.connect()
        assert self._stub is not None
        response = await self._call(
            "HealthCheck",
            self._stub.HealthCheck,
            global_store_pb2.HealthCheckRequest(),
        )
        assert isinstance(response, global_store_pb2.HealthCheckResponse)
        return response

    async def list_active_workers(
        self, include_unavailable: bool
    ) -> global_store_pb2.ListActiveWorkersResponse:
        await self.connect()
        assert self._stub is not None
        response = await self._call(
            "ListActiveWorkers",
            self._stub.ListActiveWorkers,
            global_store_pb2.ListActiveWorkersRequest(
                include_unavailable=include_unavailable
            ),
        )
        assert isinstance(response, global_store_pb2.ListActiveWorkersResponse)
        return response

    async def list_replicas(
        self,
        *,
        artifact_id: str | None = None,
        node_id: str | None = None,
        node_address: str | None = None,
        memory_type: common_pb2.MemoryType | None = None,
        device_id: int | None = None,
        page_token: str | None = None,
        page_size: int | None = None,
    ) -> global_store_pb2.ListReplicasV2Response:
        await self.connect()
        assert self._stub is not None

        pagination = None
        if page_token is not None or page_size is not None:
            pagination = common_pb2.Pagination()
            if page_size is not None:
                pagination.page_size = page_size
            if page_token is not None:
                pagination.page_token = page_token

        request = global_store_pb2.ListReplicasV2Request()
        if artifact_id:
            request.artifact_id = artifact_id
        if node_id:
            request.node_id = node_id
        if node_address:
            request.node_address = node_address
        if memory_type is not None:
            request.memory_type = memory_type
        if device_id is not None:
            request.device_id = device_id
        if pagination is not None:
            request.pagination.CopyFrom(pagination)

        response = await self._call(
            "ListReplicasV2",
            self._stub.ListReplicasV2,
            request,
        )
        assert isinstance(response, global_store_pb2.ListReplicasV2Response)
        return response

    async def get_artifact_info(
        self,
        artifact_id: str,
        *,
        include_replicas: bool,
        include_view: bool,
        include_leaves: bool,
        space_canonical: bool,
        view_id: str | None,
        leaf_indices: list[int] | None,
    ) -> global_store_pb2.GetArtifactInfoByIdResponse:
        await self.connect()
        assert self._stub is not None

        request = global_store_pb2.GetArtifactInfoByIdRequest(artifact_id=artifact_id)
        if include_replicas:
            request.include_replicas.CopyFrom(wrappers_pb2.BoolValue(value=True))
        request.include_leaves = include_leaves
        request.include_view_meta = include_view

        if space_canonical:
            request.canonical = True
        elif view_id:
            request.view_id = view_id

        if leaf_indices:
            request.leaf_idxs.extend(leaf_indices)

        response = await self._call(
            "GetArtifactInfoById",
            self._stub.GetArtifactInfoById,
            request,
        )
        assert isinstance(response, global_store_pb2.GetArtifactInfoByIdResponse)
        if response.status != global_store_pb2.Status.STATUS_OK:
            raise GlobalStoreStatusError("GetArtifactInfoById", response.status)
        return response

    async def query_chunk_locations(
        self, artifact_id: str, chunk_indices: list[int] | None
    ) -> global_store_pb2.QueryChunkLocationsResponse:
        await self.connect()
        assert self._stub is not None

        request = global_store_pb2.QueryChunkLocationsRequest(artifact_id=artifact_id)
        if chunk_indices:
            request.chunk_indices.extend(chunk_indices)

        response = await self._call(
            "QueryChunkLocations",
            self._stub.QueryChunkLocations,
            request,
        )
        assert isinstance(response, global_store_pb2.QueryChunkLocationsResponse)
        if response.status != global_store_pb2.Status.STATUS_OK:
            raise GlobalStoreStatusError("QueryChunkLocations", response.status)
        return response

    async def list_memory_tier_statuses(
        self, node_id: str | None
    ) -> memory_tier_pb2.ListMemoryTierStatusesResponse:
        await self.connect()
        assert self._memory_tier_stub is not None

        request = memory_tier_pb2.ListMemoryTierStatusesRequest()
        if node_id:
            request.node_id = node_id

        response = await self._call(
            "ListMemoryTierStatuses",
            self._memory_tier_stub.ListMemoryTierStatuses,
            request,
        )
        assert isinstance(response, memory_tier_pb2.ListMemoryTierStatusesResponse)
        return response

    async def list_memory_tier_leases(
        self, node_id: str | None, states: list[memory_tier_pb2.LeaseState] | None
    ) -> memory_tier_pb2.ListOutstandingLeasesResponse:
        await self.connect()
        assert self._memory_tier_stub is not None

        request = memory_tier_pb2.ListOutstandingLeasesRequest()
        if node_id:
            request.node_id = node_id
        if states:
            request.states.extend(states)

        response = await self._call(
            "ListOutstandingLeases",
            self._memory_tier_stub.ListOutstandingLeases,
            request,
        )
        assert isinstance(response, memory_tier_pb2.ListOutstandingLeasesResponse)
        return response
