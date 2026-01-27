#  Copyright (c) 2025-2026, TensorCast Team.

"""Mock Global Store gRPC server for running the dashboard locally.

Run:
    uv run tensorcast/dashboard/mock_gs_server.py --port 50051

Then set the dashboard env:
    TENSORCAST_GS_ADDR=127.0.0.1:50051 uv run uvicorn tensorcast.dashboard.api:app --reload
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc
from tensorcast.proto.memory_tier.v1 import memory_tier_pb2, memory_tier_pb2_grpc


def _ts_now() -> Timestamp:
    ts = Timestamp()
    now = datetime.now(tz=timezone.utc)
    ts.FromDatetime(now)
    return ts


def _ts_future(seconds: int) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime.now(tz=timezone.utc) + timedelta(seconds=seconds))
    return ts


class MockGlobalStoreService(global_store_pb2_grpc.GlobalStoreServiceServicer):
    async def HealthCheck(
        self,
        request: global_store_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> global_store_pb2.HealthCheckResponse:
        return global_store_pb2.HealthCheckResponse(
            status=global_store_pb2.Status.STATUS_OK
        )

    async def ListActiveWorkers(
        self,
        request: global_store_pb2.ListActiveWorkersRequest,
        context: grpc.aio.ServicerContext,
    ) -> global_store_pb2.ListActiveWorkersResponse:
        workers = [
            global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                worker_id="worker-1",
                node_id="node-1",
                node_address="127.0.0.1",
                grpc_port=55051,
                p2p_port=56051,
                mem_pool_total_size=8 * 1024**3,
                mem_pool_available_size=6 * 1024**3,
                accepting_new_requests=True,
                last_heartbeat_ts=_ts_now(),
                state_version=1,
                status=global_store_pb2.ConnectionStatus.CONNECTION_STATUS_CONNECTED,
            ),
            global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
                worker_id="worker-2",
                node_id="node-2",
                node_address="127.0.0.2",
                grpc_port=55052,
                p2p_port=56052,
                mem_pool_total_size=16 * 1024**3,
                mem_pool_available_size=12 * 1024**3,
                accepting_new_requests=True,
                last_heartbeat_ts=_ts_now(),
                state_version=3,
                status=global_store_pb2.ConnectionStatus.CONNECTION_STATUS_CONNECTED,
            ),
        ]
        return global_store_pb2.ListActiveWorkersResponse(workers=workers)

    async def ListReplicasV2(
        self,
        request: global_store_pb2.ListReplicasV2Request,
        context: grpc.aio.ServicerContext,
    ) -> global_store_pb2.ListReplicasV2Response:
        records: list[global_store_pb2.ArtifactReplicaRecord] = []
        for idx in range(3):
            memory_info = common_pb2.MemoryInfo(
                node_id=f"node-{idx + 1}",
                node_address=f"127.0.0.{idx + 1}",
                device_id=idx,
                memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
                memory_size=128 * 1024**2,
                creation_ts=_ts_now(),
                expires_at=_ts_future(3600),
            )
            records.append(
                global_store_pb2.ArtifactReplicaRecord(
                    artifact_id=request.artifact_id or f"artifact-{idx}",
                    memory_info=memory_info,
                )
            )
        page_info = common_pb2.PageInfo(next_page_token="", total_size=3)
        return global_store_pb2.ListReplicasV2Response(
            replicas=records, page_info=page_info
        )

    async def GetArtifactInfoById(
        self,
        request: global_store_pb2.GetArtifactInfoByIdRequest,
        context: grpc.aio.ServicerContext,
    ) -> global_store_pb2.GetArtifactInfoByIdResponse:
        replicas = [
            common_pb2.MemoryInfo(
                node_id="node-1",
                node_address="127.0.0.1",
                device_id=0,
                memory_type=common_pb2.MemoryType.MEMORY_TYPE_RAM,
                memory_size=64 * 1024**2,
                creation_ts=_ts_now(),
                expires_at=_ts_future(1800),
            )
        ]
        leaves = [
            global_store_pb2.Leaf(leaf_idx=0, digest=b"deadbeef"),
            global_store_pb2.Leaf(leaf_idx=1, digest=b"cafebabe"),
        ]
        coverage = [
            global_store_pb2.PartialCoverageDetail(
                hash_space=common_pb2.HashSpaceRef(
                    byte_space=common_pb2.ByteSpaceRef(
                        kind=common_pb2.BYTE_SPACE_KIND_CANONICAL, id=""
                    ),
                    canonical_index_multihash="mh-index",
                ),
                missing_ranges=[
                    global_store_pb2.Range(off=0, len=1024),
                    global_store_pb2.Range(off=8192, len=512),
                ],
            )
        ]
        view_meta = global_store_pb2.ViewMeta(
            view_spec_json="{}",
            view_size=128 * 1024**2,
            view_data_hash="hash123",
            verified_at=_ts_now(),
        )
        descriptor = common_pb2.ArtifactDescriptor(
            artifact_id=request.artifact_id or "artifact-0",
            index_multihash="mh-index",
            data_multihash="mh-data",
            schema_version="v3",
            encoding="json",
            total_size=64 * 1024**2,
            id_kind=common_pb2.ArtifactIdKind.ARTIFACT_ID_KIND_MI2,
        )
        return global_store_pb2.GetArtifactInfoByIdResponse(
            status=global_store_pb2.Status.STATUS_OK,
            replicas=replicas,
            leaves=leaves,
            partial_coverage=coverage,
            view_meta=view_meta,
            descriptor=descriptor,
        )

    async def QueryChunkLocations(
        self,
        request: global_store_pb2.QueryChunkLocationsRequest,
        context: grpc.aio.ServicerContext,
    ) -> global_store_pb2.QueryChunkLocationsResponse:
        locations = [
            global_store_pb2.ChunkLocation(
                chunk_idx=i,
                node_id=f"node-{i % 2 + 1}",
                node_address=f"127.0.0.{i % 2 + 1}",
                p2p_port=56051 + (i % 2),
                state=global_store_pb2.ChunkState.CHUNK_STATE_HOT,
                node_load_ratio=0.2 + 0.1 * i,
                device_uuid=f"GPU-UUID-{i % 2}",
                replica=i % 3,
            )
            for i in (request.chunk_indices or [0, 1, 2])
        ]
        return global_store_pb2.QueryChunkLocationsResponse(
            status=global_store_pb2.Status.STATUS_OK,
            locations=locations,
        )


class MockMemoryTierService(memory_tier_pb2_grpc.MemoryTierServiceServicer):
    async def PublishMemoryTierStatus(
        self,
        request: memory_tier_pb2.PublishMemoryTierStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.PublishMemoryTierStatusResponse:
        return memory_tier_pb2.PublishMemoryTierStatusResponse()

    async def RequestMemoryTierLease(
        self,
        request: memory_tier_pb2.RequestMemoryTierLeaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.RequestMemoryTierLeaseResponse:
        lease = memory_tier_pb2.MemoryTierLease(
            lease_id="lease-1",
            node_id=request.node_id or "node-1",
            kind=request.kind or memory_tier_pb2.LEASE_KIND_PREEMPTIBLE,
            artifact_id=request.artifact_id or "mi2:mock",
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=4),
            chunk_ids=[0, 1, 2, 3],
            ledger_version=request.ledger_version or 1,
            bytes=request.bytes or 128 * 1024**2,
            workload_id=request.workload_id or "mock-workload",
            state=memory_tier_pb2.LEASE_STATE_PENDING,
            request_id=request.request_id or "req-1",
            issued_at_ns=request.issued_at_ns or 0,
            ack_epoch_ns=0,
            expires_at_ns=0,
        )
        return memory_tier_pb2.RequestMemoryTierLeaseResponse(lease=lease)

    async def AcknowledgeMemoryTierLease(
        self,
        request: memory_tier_pb2.AcknowledgeMemoryTierLeaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.AcknowledgeMemoryTierLeaseResponse:
        lease = memory_tier_pb2.MemoryTierLease(
            lease_id=request.lease_id or "lease-1",
            node_id=request.node_id or "node-1",
            kind=memory_tier_pb2.LEASE_KIND_PREEMPTIBLE,
            artifact_id=request.artifact_id or "mi2:mock",
            chunk_range=request.chunk_range
            or memory_tier_pb2.ChunkRange(start=0, count=4),
            chunk_ids=list(request.chunk_ids) or [0, 1, 2, 3],
            ledger_version=request.ledger_version or 1,
            bytes=request.bytes or 128 * 1024**2,
            workload_id="mock-workload",
            state=memory_tier_pb2.LEASE_STATE_ACTIVE,
            request_id=request.request_id or "req-1",
            issued_at_ns=request.ack_epoch_ns or 0,
            ack_epoch_ns=request.ack_epoch_ns or 0,
            expires_at_ns=0,
        )
        return memory_tier_pb2.AcknowledgeMemoryTierLeaseResponse(lease=lease)

    async def RevokeMemoryTierLease(
        self,
        request: memory_tier_pb2.RevokeMemoryTierLeaseRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.RevokeMemoryTierLeaseResponse:
        lease = memory_tier_pb2.MemoryTierLease(
            lease_id=request.lease_id or "lease-1",
            node_id="node-1",
            kind=memory_tier_pb2.LEASE_KIND_PREEMPTIBLE,
            artifact_id="mi2:mock",
            chunk_range=memory_tier_pb2.ChunkRange(start=0, count=4),
            chunk_ids=[0, 1, 2, 3],
            ledger_version=1,
            bytes=128 * 1024**2,
            workload_id="mock-workload",
            state=memory_tier_pb2.LEASE_STATE_REVOKING,
            request_id="req-1",
            issued_at_ns=0,
            ack_epoch_ns=0,
            expires_at_ns=0,
        )
        return memory_tier_pb2.RevokeMemoryTierLeaseResponse(lease=lease)

    async def ListOutstandingLeases(
        self,
        request: memory_tier_pb2.ListOutstandingLeasesRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.ListOutstandingLeasesResponse:
        leases = [
            memory_tier_pb2.MemoryTierLease(
                lease_id="lease-1",
                node_id=request.node_id or "node-1",
                kind=memory_tier_pb2.LEASE_KIND_PREEMPTIBLE,
                artifact_id="mi2:mock",
                chunk_range=memory_tier_pb2.ChunkRange(start=0, count=4),
                chunk_ids=[0, 1, 2, 3],
                ledger_version=1,
                bytes=128 * 1024**2,
                workload_id="mock-workload",
                state=memory_tier_pb2.LEASE_STATE_ACTIVE,
                request_id="req-1",
                issued_at_ns=0,
                ack_epoch_ns=0,
                expires_at_ns=0,
            )
        ]
        return memory_tier_pb2.ListOutstandingLeasesResponse(leases=leases)

    async def ListMemoryTierStatuses(
        self,
        request: memory_tier_pb2.ListMemoryTierStatusesRequest,
        context: grpc.aio.ServicerContext,
    ) -> memory_tier_pb2.ListMemoryTierStatusesResponse:
        statuses = [
            memory_tier_pb2.MemoryTierStatus(
                node_id="node-1",
                worker_id="worker-1",
                stable_total_bytes=8 * 1024**3,
                stable_used_bytes=2 * 1024**3,
                preemptible_total_bytes=4 * 1024**3,
                preemptible_marked_bytes=1 * 1024**3,
                faults_per_sec=0.05,
                rehydrate_p99_ns=5_000_000,
                enable_preemptible=True,
                memory_tier_config_json='{"tiers":["stable","preemptible"]}',
                epoch_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
            )
        ]
        return memory_tier_pb2.ListMemoryTierStatusesResponse(statuses=statuses)


async def _serve_async(port: int) -> None:
    server = grpc.aio.server()
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(
        MockGlobalStoreService(), server
    )
    memory_tier_pb2_grpc.add_MemoryTierServiceServicer_to_server(
        MockMemoryTierService(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Mock Global Store listening on 0.0.0.0:{port}")
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Global Store gRPC server")
    parser.add_argument("--port", type=int, default=50051, help="Listen port")
    args = parser.parse_args()
    asyncio.run(_serve_async(args.port))


if __name__ == "__main__":
    main()
