#  Copyright (c) 2025, TensorCast Team.

"""Mock Global Store gRPC server for running the dashboard locally.

Run:
    uv run tensorcast/dashboard/mock_gs_server.py --port 50051

Then set the dashboard env:
    TENSORCAST_GS_ADDR=127.0.0.1:50051 uv run uvicorn tensorcast.dashboard.api:app --reload
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc


def _ts_now() -> Timestamp:
    ts = Timestamp()
    now = datetime.now(tz=timezone.utc)
    ts.FromDatetime(now)
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
            )
        ]
        leaves = [
            global_store_pb2.Leaf(leaf_idx=0, digest=b"deadbeef"),
            global_store_pb2.Leaf(leaf_idx=1, digest=b"cafebabe"),
        ]
        coverage = [
            global_store_pb2.PartialCoverageDetail(
                space_kind=global_store_pb2.ByteSpaceKind.BYTE_SPACE_KIND_CANONICAL,
                space_id=request.artifact_id or "artifact-0",
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
        return global_store_pb2.GetArtifactInfoByIdResponse(
            status=global_store_pb2.Status.STATUS_OK,
            replicas=replicas,
            leaves=leaves,
            partial_coverage=coverage,
            view_meta=view_meta,
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


async def _serve_async(port: int) -> None:
    server = grpc.aio.server()
    global_store_pb2_grpc.add_GlobalStoreServiceServicer_to_server(
        MockGlobalStoreService(), server
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
