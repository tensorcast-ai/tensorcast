#  Copyright (c) 2025, StepCast Team.

"""REST API endpoints for Web UI."""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

# DAO imports removed - now using gRPC client directly
from scstore.global_store.webui_backend.models import (
    ApiResponse,
    GlobalMetrics,
    MemoryType,
    PaginationMeta,
)

if TYPE_CHECKING:
    from scstore.global_store.webui_backend.grpc_client import GlobalStoreClient

logger = logging.getLogger(__name__)

api_router = APIRouter()

# Module-level Query defaults to avoid B008 linting error
_memory_type_query_none = Query(None)


def _unix_timestamp_to_iso(timestamp: int) -> str:
    """Convert Unix timestamp (seconds) to ISO string format.

    Args:
        timestamp: Unix timestamp in seconds

    Returns:
        ISO format timestamp string (YYYY-MM-DDTHH:MM:SS.sssZ)
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


async def get_grpc_client(request: Request) -> "GlobalStoreClient":
    """Get gRPC client dependency."""
    webui_app = request.app.extra.get("webui_app")
    return await webui_app.get_grpc_client()


# Dependency injection
GrpcClient = Annotated["GlobalStoreClient", Depends(get_grpc_client)]


@api_router.get("/summary", response_model=ApiResponse)
async def get_summary(client: GrpcClient) -> ApiResponse:
    """Get global metrics summary."""
    # Get summary statistics from gRPC client
    stats = await client.get_summary_stats()

    # Calculate total and available memory from workers
    workers = await client.list_active_workers(include_unavailable=True)
    total_memory = sum(w.mem_pool_total_size for w in workers)
    available_memory = sum(w.mem_pool_available_size for w in workers)

    metrics = GlobalMetrics(
        total_workers=stats["total_workers"],
        active_workers=stats["active_workers"],
        total_replicas=stats["total_replicas"],
        available_replicas=stats["total_replicas"],  # TODO: Need to track availability
        total_artifacts=stats["total_artifacts"],
        active_transports=stats["active_transports"],
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
    )

    return ApiResponse(data=metrics)


@api_router.get("/workers", response_model=ApiResponse)
async def list_workers(
    client: GrpcClient,
    include_unavailable: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
) -> ApiResponse:
    """List workers with pagination."""
    # Get all workers from gRPC
    workers = await client.list_active_workers(include_unavailable=include_unavailable)

    # Convert proto messages to dict format expected by frontend
    worker_list = []
    for w in workers:
        # Get replica count for this worker
        all_replicas = await client.list_replicas()
        replica_count = 0
        for artifact_replicas in all_replicas.values():
            for replica in artifact_replicas:
                if replica.node_id == w.node_id:
                    replica_count += 1

        last_dt = w.last_heartbeat_datetime
        worker_list.append(
            {
                "worker_id": w.worker_id,
                "node_id": w.node_id,
                "node_address": w.node_address,
                "grpc_port": w.grpc_port,
                "p2p_port": w.p2p_port,
                "mem_pool_total_size": w.mem_pool_total_size,
                "mem_pool_available_size": w.mem_pool_available_size,
                "accepting_new_requests": w.accepting_new_requests,
                "last_heartbeat": last_dt.isoformat() if last_dt else None,
                "replica_count": replica_count,
                "registered_at": None,  # Not available from proto
                "updated_at": last_dt.isoformat() if last_dt else None,
            }
        )

    # Apply pagination
    total_count = len(worker_list)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_workers = worker_list[start:end]

    total_pages = (total_count + page_size - 1) // page_size
    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )

    return ApiResponse(data=paginated_workers, meta=meta.__dict__)


@api_router.get("/workers/{worker_id}", response_model=ApiResponse)
async def get_worker(worker_id: str, client: GrpcClient) -> ApiResponse:
    """Get a specific worker by ID."""
    # Get all workers and find the specific one
    workers = await client.list_active_workers(include_unavailable=True)

    for w in workers:
        if w.worker_id == worker_id:
            # Get replica count for this worker
            all_replicas = await client.list_replicas()
            replica_count = 0
            for artifact_replicas in all_replicas.values():
                for replica in artifact_replicas:
                    if replica.node_id == w.node_id:
                        replica_count += 1

            last_dt = w.last_heartbeat_datetime
            worker_data = {
                "worker_id": w.worker_id,
                "node_id": w.node_id,
                "node_address": w.node_address,
                "grpc_port": w.grpc_port,
                "p2p_port": w.p2p_port,
                "mem_pool_total_size": w.mem_pool_total_size,
                "mem_pool_available_size": w.mem_pool_available_size,
                "accepting_new_requests": w.accepting_new_requests,
                "last_heartbeat": last_dt.isoformat() if last_dt else None,
                "replica_count": replica_count,
                "registered_at": None,  # Not available from proto
                "updated_at": last_dt.isoformat() if last_dt else None,
            }
            return ApiResponse(data=worker_data)

    raise HTTPException(status_code=404, detail="Worker not found")


@api_router.get("/replicas", response_model=ApiResponse)
async def list_replicas(
    client: GrpcClient,
    artifact_id: str | None = Query(None),
    node_id: str | None = Query(None),
    memory_type: MemoryType | None = _memory_type_query_none,
    worker_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
) -> ApiResponse:
    """List replicas with filters and pagination."""
    # Convert MemoryType enum to proto value if provided
    proto_memory_type = None
    if memory_type:
        memory_type_map = {
            MemoryType.GPU: 0,
            MemoryType.RAM: 1,
            MemoryType.DISK: 2,
        }
        proto_memory_type = memory_type_map.get(memory_type)

    # ------------------------------------------------------------------
    # Fetch replicas from Global-Store.  The remote service *should* honour
    # the filter arguments but during unit-testing we use a dummy
    # ``AsyncMock`` implementation that simply returns static data and ignores
    # any request parameters.  To make the behaviour consistent (and the
    # public API robust against older server versions) we therefore apply the
    # same filtering rules again on the client side after fetching the data.
    # ------------------------------------------------------------------

    all_replicas = await client.list_replicas(
        artifact_id=artifact_id,
        node_id=node_id,
        memory_type=proto_memory_type,
    )

    # If worker_id filter is specified, we need to map it to node_id
    if worker_id:
        workers = await client.list_active_workers(include_unavailable=True)
        worker_node_id = None
        for w in workers:
            if w.worker_id == worker_id:
                worker_node_id = w.node_id
                break
        if not worker_node_id:
            # Worker not found, return empty results
            return ApiResponse(
                data=[],
                meta={
                    "page": page,
                    "page_size": page_size,
                    "total_count": 0,
                    "total_pages": 0,
                },
            )
        node_id = worker_node_id

    # Flatten replicas into a list
    replica_list = []
    replica_id_counter = 0  # Generate replica IDs since proto doesn't have them

    for artifact_id_key, replicas in all_replicas.items():
        # Apply *artifact_id* filter locally in case the server ignored it
        if artifact_id and artifact_id_key != artifact_id:
            continue

        for r in replicas:
            # Apply node_id filter if specified
            if node_id and r.node_id != node_id:
                continue

            # Apply *memory_type* filter locally – the server is expected to
            # do this, but a defensive check avoids surprises.
            if proto_memory_type is not None and r.memory_type != proto_memory_type:
                continue

            replica_list.append(
                {
                    "replica_id": f"replica-{replica_id_counter}",
                    "artifact_id": artifact_id_key,
                    "node_id": r.node_id,
                    "node_address": r.node_address,
                    "node_port": r.node_port,
                    "memory_type": ["GPU", "RAM", "DISK"][r.memory_type],
                    "memory_size": r.memory_size,
                    "device_id": r.device_id if r.memory_type == 0 else None,
                    "is_available": True,  # Assume available since it's in the list
                    "current_requests": 0,  # Not available in proto
                    "max_concurrency": 5,  # Default value
                    "worker_id": worker_id if worker_id else None,
                    "created_at": None,
                    "updated_at": None,
                }
            )
            replica_id_counter += 1

    # Apply pagination
    total_count = len(replica_list)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_replicas = replica_list[start:end]

    total_pages = (total_count + page_size - 1) // page_size
    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )

    return ApiResponse(data=paginated_replicas, meta=meta.__dict__)


@api_router.get("/replicas/{replica_id}", response_model=ApiResponse)
async def get_replica(replica_id: str, client: GrpcClient) -> ApiResponse:
    """Get a specific replica by ID."""
    # Since proto doesn't have replica IDs, we need to enumerate all replicas
    # This is a limitation of the current proto design
    all_replicas = await client.list_replicas()

    replica_id_counter = 0
    for artifact_id_key, replicas in all_replicas.items():
        for r in replicas:
            if f"replica-{replica_id_counter}" == replica_id:
                replica_data = {
                    "replica_id": replica_id,
                    "artifact_id": artifact_id_key,
                    "node_id": r.node_id,
                    "node_address": r.node_address,
                    "node_port": r.node_port,
                    "memory_type": ["GPU", "RAM", "DISK"][r.memory_type],
                    "memory_size": r.memory_size,
                    "device_id": r.device_id if r.memory_type == 0 else None,
                    "is_available": True,
                    "current_requests": 0,
                    "max_concurrency": 5,
                    "created_at": None,
                    "updated_at": None,
                }
                return ApiResponse(data=replica_data)
            replica_id_counter += 1

    raise HTTPException(status_code=404, detail="Replica not found")


@api_router.get("/artifacts", response_model=ApiResponse)
async def list_artifacts(client: GrpcClient) -> ApiResponse:
    """List all artifacts with summary statistics."""
    # Get all replicas grouped by artifact
    all_replicas = await client.list_replicas()

    # Build artifact statistics
    artifacts = []
    for artifact_id, replicas in all_replicas.items():
        gpu_count = sum(1 for r in replicas if r.memory_type == 0)  # GPU = 0
        ram_count = sum(1 for r in replicas if r.memory_type == 1)  # RAM = 1
        disk_count = sum(1 for r in replicas if r.memory_type == 2)  # DISK = 2

        total_size = sum(r.memory_size for r in replicas)
        unique_nodes = {r.node_id for r in replicas}

        artifacts.append(
            {
                "artifact_id": artifact_id,
                "replica_count": len(replicas),
                "gpu_replicas": gpu_count,
                "ram_replicas": ram_count,
                "disk_replicas": disk_count,
                "total_size": total_size,
                "node_count": len(unique_nodes),
            }
        )

    return ApiResponse(data=artifacts)


@api_router.get("/artifacts/{artifact_id}", response_model=ApiResponse)
async def get_artifact(artifact_id: str, client: GrpcClient) -> ApiResponse:
    """Get a specific artifact's summary by artifact_id."""
    # Get artifact info
    artifact_info = await client.get_artifact_info(artifact_id)
    if not artifact_info:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Get detailed replica information
    all_replicas = await client.list_replicas(artifact_id=artifact_id)
    replicas = all_replicas.get(artifact_id, [])

    gpu_count = sum(1 for r in replicas if r.memory_type == 0)
    ram_count = sum(1 for r in replicas if r.memory_type == 1)
    disk_count = sum(1 for r in replicas if r.memory_type == 2)

    total_size = sum(r.memory_size for r in replicas)
    unique_nodes = {r.node_id for r in replicas}

    # Build per-node statistics
    nodes_data = {}
    for r in replicas:
        if r.node_id not in nodes_data:
            nodes_data[r.node_id] = {
                "node_id": r.node_id,
                "replica_count": 0,
                "gpu_replicas": 0,
                "ram_replicas": 0,
                "disk_replicas": 0,
                "total_memory": 0,
            }

        nodes_data[r.node_id]["replica_count"] += 1  # type: ignore[operator]
        nodes_data[r.node_id]["total_memory"] += int(r.memory_size)  # type: ignore[operator]

        if r.memory_type == 0:
            nodes_data[r.node_id]["gpu_replicas"] += 1  # type: ignore[operator]
        elif r.memory_type == 1:
            nodes_data[r.node_id]["ram_replicas"] += 1  # type: ignore[operator]
        elif r.memory_type == 2:
            nodes_data[r.node_id]["disk_replicas"] += 1  # type: ignore[operator]

    artifact_summary = {
        "artifact_id": artifact_id,
        "replica_count": len(replicas),
        "gpu_replicas": gpu_count,
        "ram_replicas": ram_count,
        "disk_replicas": disk_count,
        "total_size": total_size,
        "node_count": len(unique_nodes),
        "nodes": list(nodes_data.values()),
    }

    return ApiResponse(data=artifact_summary)


@api_router.get("/nodes", response_model=ApiResponse)
async def list_nodes(client: GrpcClient) -> ApiResponse:
    """List all nodes with aggregated statistics."""
    # Get all replicas and workers to build node statistics
    all_replicas = await client.list_replicas()
    workers = await client.list_active_workers(include_unavailable=True)

    # Build node statistics from replicas
    nodes_data = {}

    # First, add all nodes from workers
    for w in workers:
        nodes_data[w.node_id] = {
            "node_id": w.node_id,
            "node_address": w.node_address,
            "worker_count": 0,
            "active_workers": 0,
            "replica_count": 0,
            "artifact_count": 0,
            "gpu_memory": 0,
            "ram_memory": 0,
            "disk_memory": 0,
            "total_memory": 0,
            "artifacts": set(),
        }

    # Count workers per node
    for w in workers:
        if w.node_id in nodes_data:
            nodes_data[w.node_id]["worker_count"] += 1
            if w.accepting_new_requests:
                nodes_data[w.node_id]["active_workers"] += 1

    # Add replica statistics
    for artifact_id_key, replicas in all_replicas.items():
        for r in replicas:
            if r.node_id not in nodes_data:
                # Node has replicas but no active workers
                nodes_data[r.node_id] = {
                    "node_id": r.node_id,
                    "node_address": r.node_address,
                    "worker_count": 0,
                    "active_workers": 0,
                    "replica_count": 0,
                    "artifact_count": 0,
                    "gpu_memory": 0,
                    "ram_memory": 0,
                    "disk_memory": 0,
                    "total_memory": 0,
                    "artifacts": set(),
                }

            nodes_data[r.node_id]["replica_count"] += 1
            nodes_data[r.node_id]["artifacts"].add(artifact_id_key)
            nodes_data[r.node_id]["total_memory"] += int(r.memory_size)

            if r.memory_type == 0:  # GPU
                nodes_data[r.node_id]["gpu_memory"] += int(r.memory_size)
            elif r.memory_type == 1:  # RAM
                nodes_data[r.node_id]["ram_memory"] += int(r.memory_size)
            elif r.memory_type == 2:  # DISK
                nodes_data[r.node_id]["disk_memory"] += int(r.memory_size)

    # Convert sets to counts and materialize_replica final data
    nodes_list = []
    for data in nodes_data.values():
        data["artifact_count"] = len(data["artifacts"])
        del data["artifacts"]  # Remove the set
        nodes_list.append(data)

    # Sort by node_id for consistent ordering
    nodes_list.sort(key=lambda x: str(x["node_id"]))

    return ApiResponse(data=nodes_list)


@api_router.get("/transports", response_model=ApiResponse)
async def list_transports(
    client: GrpcClient,
    status: str | None = Query(None),
    artifact_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
) -> ApiResponse:
    """List transports with filters and pagination."""
    # Note: The current proto doesn't have a ListTransports RPC
    # This is a limitation that should be addressed by extending the proto
    # For now, return empty data or mock data
    # TODO: Use client parameter when proto is extended
    _ = client  # Acknowledge unused parameter

    logger.warning("Transport listing not available via gRPC - proto needs extension")

    # Return empty results for now
    # In production, you would either:
    # 1. Add a ListTransports RPC to the proto
    # 2. Track transports locally in the Web UI
    # 3. Use a separate service for transport history

    transports = []

    # Apply filters if any transports exist
    if status:
        transports = [t for t in transports if t.get("status") == status]
    if artifact_id:
        transports = [t for t in transports if t.get("artifact_id") == artifact_id]

    # Apply pagination
    total_count = len(transports)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_transports = transports[start:end]

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )

    return ApiResponse(data=paginated_transports, meta=meta.__dict__)


@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "global_store_webui"}
