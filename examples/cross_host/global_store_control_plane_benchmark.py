#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import grpc

from tensorcast.global_store.config.settings import (
    GlobalStoreConfig,
    WorkerControlReducerConfig,
)
from tensorcast.global_store.launcher import start_global_store_server
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.config.v1 import global_store_config_pb2 as gsc_pb
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc


@dataclass
class OpStats:
    latencies_ms: list[float] = field(default_factory=list)
    ok: int = 0
    err: int = 0

    def record(self, latency_ms: float, ok: bool) -> None:
        self.latencies_ms.append(latency_ms)
        if ok:
            self.ok += 1
        else:
            self.err += 1


class StatsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, OpStats] = defaultdict(OpStats)

    def record(self, op: str, latency_ms: float, ok: bool) -> None:
        with self._lock:
            self._stats[op].record(latency_ms, ok)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            for op, stats in self._stats.items():
                lats = sorted(stats.latencies_ms)
                p50 = lats[int(len(lats) * 0.5)] if lats else None
                p95 = lats[int(len(lats) * 0.95)] if lats else None
                p99 = lats[int(len(lats) * 0.99)] if lats else None
                mean = statistics.mean(lats) if lats else None
                result[op] = {
                    "ok": stats.ok,
                    "err": stats.err,
                    "total": stats.ok + stats.err,
                    "latency_ms": {
                        "mean": mean,
                        "p50": p50,
                        "p95": p95,
                        "p99": p99,
                    },
                }
            return result


def _parse_prometheus_metrics(payload: str, metric_names: set[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in payload.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            metric_token, value_token = text.split(None, 1)
        except ValueError:
            continue
        metric_name = metric_token.split("{", 1)[0]
        if metric_name not in metric_names:
            continue
        try:
            value = float(value_token.strip())
        except ValueError:
            continue
        values[metric_name] = values.get(metric_name, 0.0) + value
    return values


def _scrape_metrics(url: str, metric_names: set[str]) -> dict[str, float]:
    try:
        with urlopen(url, timeout=1.5) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except (OSError, TimeoutError, URLError):
        return {}
    return _parse_prometheus_metrics(payload, metric_names)


def _series_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    p95 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))]
    return {
        "latest": float(values[-1]),
        "mean": float(statistics.mean(values)),
        "p95": float(p95),
        "max": float(max(values)),
    }


def build_reconcile_inventory(
    worker_index: int, chunk_count: int, node_address: str
) -> list[common_pb2.ReplicaInfo]:
    # Keep reconcile payload realistic but bounded.
    artifact_id = f"mi2:reconcile-artifact-{worker_index}"
    memory_info = common_pb2.MemoryInfo(
        node_id=f"node-{worker_index}",
        node_address=node_address,
        node_port=37000 + worker_index,
        memory_type=common_pb2.MEMORY_TYPE_GPU,
        device_id=0,
        memory_size=chunk_count * 4096,
    )
    memory_info.byte_space.kind = common_pb2.BYTE_SPACE_KIND_CANONICAL
    replica = common_pb2.ReplicaInfo(
        ref=common_pb2.ReplicaRef(artifact_id=artifact_id, replica_id=""),
        memory_info=memory_info,
    )
    return [replica]


def main() -> int:
    for noisy_logger in (
        "tensorcast.global_store.chunk_directory_repository",
        "tensorcast.global_store.repositories.chunk_directory_repository",
        "tensorcast.global_store.services.chunk_service",
        "tensorcast.global_store.services.recovery_service",
        "tensorcast.global_store.rpc.worker_rpc_handler",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Global Store control-plane mixed workload benchmark"
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resolver-threads", type=int, default=32)
    parser.add_argument("--resolve-qps-per-thread", type=float, default=20.0)
    parser.add_argument("--heartbeat-interval-s", type=float, default=0.5)
    parser.add_argument("--reconcile-interval-s", type=float, default=2.0)
    parser.add_argument("--chunk-sync-interval-s", type=float, default=2.0)
    parser.add_argument("--chunk-count", type=int, default=128)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--reducer-shards", type=int, default=1)
    parser.add_argument("--reducer-queue-capacity", type=int, default=2048)
    parser.add_argument("--reducer-coalesce-window-ms", type=int, default=50)
    parser.add_argument("--metrics-scrape-interval-s", type=float, default=1.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    cfg = GlobalStoreConfig(
        db_file=None,
        listen_host="127.0.0.1",
        listen_port=0,
        metrics_port=0,
        max_workers=max(1, int(args.max_workers)),
        worker_control_reducer=WorkerControlReducerConfig(
            shard_count=max(1, int(args.reducer_shards)),
            queue_capacity=max(1, int(args.reducer_queue_capacity)),
            coalesce_window_ms=max(0, int(args.reducer_coalesce_window_ms)),
        ),
    )
    pb_cfg = gsc_pb.GlobalStoreConfig()
    started = start_global_store_server(cfg, pb_cfg)
    addr = f"{started.listen_host}:{started.listen_port}"
    node_address = str(started.advertise_host)

    channel = grpc.insecure_channel(addr)
    runtime_stub = global_store_pb2_grpc.ClusterRuntimeServiceStub(channel)
    catalog_stub = global_store_pb2_grpc.ArtifactCatalogServiceStub(channel)
    metrics_url = f"http://127.0.0.1:{int(started.metrics_port)}/metrics"
    tracked_metrics = {
        "tc_grpc_server_inflight_requests",
        "tc_grpc_server_inflight_peak_requests",
        "tc_control_plane_executor_max_threads",
        "tc_control_plane_executor_live_threads",
        "tc_control_plane_executor_busy_threads",
        "tc_control_plane_executor_idle_threads",
        "tc_control_plane_executor_queue_depth",
        "tc_worker_control_reducer_queue_depth",
        "tc_worker_heartbeat_buffer_pending",
    }
    metrics_series: dict[str, list[float]] = defaultdict(list)
    last_metric_sample: dict[str, float] = {}

    stats = StatsCollector()
    worker_ids: list[str] = []
    generations: list[int] = []

    try:
        # Seed mapping
        upsert_resp = catalog_stub.UpsertKeyMapping(
            global_store_pb2.UpsertKeyMappingRequest(
                key="model:latest",
                artifact_id="mi2:seed",
            ),
            timeout=10.0,
        )
        if upsert_resp.status != global_store_pb2.Status.STATUS_OK:
            raise RuntimeError(f"UpsertKeyMapping failed: {upsert_resp}")

        # Register workers
        for i in range(args.workers):
            req = global_store_pb2.RegisterWorkerRequest(
                daemon_id=f"daemon-{i}",
                node_id=f"node-{i}",
                node_address=node_address,
                grpc_port=43000 + i,
                p2p_port=37000 + i,
                mem_pool_total_size=1 << 30,
                mem_pool_available_size=1 << 30,
                capability_flags=0,
            )
            resp = runtime_stub.RegisterWorker(req, timeout=10.0)
            if resp.status != global_store_pb2.Status.STATUS_OK:
                raise RuntimeError(f"RegisterWorker failed i={i}: {resp}")
            worker_ids.append(resp.worker_id)
            generations.append(max(1, int(resp.reconcile_generation or 1)))

        stop_event = threading.Event()
        start_ts = time.monotonic()

        def rate_sleep(target_qps: float, started_at: float, count: int) -> None:
            if target_qps <= 0:
                return
            expected_elapsed = count / target_qps
            actual_elapsed = time.monotonic() - started_at
            delay = expected_elapsed - actual_elapsed
            if delay > 0:
                time.sleep(min(delay, 0.2))

        def resolver_worker(thread_idx: int) -> None:
            req = global_store_pb2.ResolveKeyMappingRequest(key="model:latest")
            call_count = 0
            loop_start = time.monotonic()
            while not stop_event.is_set():
                t0 = time.monotonic()
                ok = True
                try:
                    resp = catalog_stub.ResolveKeyMapping(req, timeout=5.0)
                    ok = resp.status == global_store_pb2.Status.STATUS_OK
                except Exception:
                    ok = False
                stats.record(
                    "resolve_key_mapping", (time.monotonic() - t0) * 1000.0, ok
                )
                call_count += 1
                rate_sleep(float(args.resolve_qps_per_thread), loop_start, call_count)

        def heartbeat_worker(worker_index: int) -> None:
            worker_id = worker_ids[worker_index]
            interval = max(0.01, float(args.heartbeat_interval_s))
            version = 1
            while not stop_event.is_set():
                t0 = time.monotonic()
                ok = True
                try:
                    req = global_store_pb2.WorkerHeartbeatRequest(
                        worker_id=worker_id,
                        mem_pool_available_size=1 << 30,
                        accepting_new_requests=True,
                        state_version=version,
                        state_checksum="",
                    )
                    resp = runtime_stub.WorkerHeartbeat(req, timeout=5.0)
                    ok = resp.status == global_store_pb2.Status.STATUS_OK
                except Exception:
                    ok = False
                stats.record("worker_heartbeat", (time.monotonic() - t0) * 1000.0, ok)
                time.sleep(interval)

        def reconcile_worker(worker_index: int) -> None:
            worker_id = worker_ids[worker_index]
            daemon_id = f"daemon-{worker_index}"
            generation = generations[worker_index]
            request_seq = 0
            inventory = build_reconcile_inventory(
                worker_index, int(args.chunk_count), node_address
            )
            interval = max(0.05, float(args.reconcile_interval_s))
            while not stop_event.is_set():
                request_seq += 1
                t0 = time.monotonic()
                ok = True
                try:
                    req = global_store_pb2.ReconcileWorkerStateRequest(
                        worker_id=worker_id,
                        daemon_id=daemon_id,
                        generation=generation,
                        request_seq=request_seq,
                        inventory=inventory,
                        request_kind=global_store_pb2.RECONCILE_REQUEST_KIND_SNAPSHOT,
                    )
                    resp = runtime_stub.ReconcileWorkerState(req, timeout=10.0)
                    ok = resp.result_kind in {
                        global_store_pb2.RECONCILE_RESULT_KIND_APPLIED,
                        global_store_pb2.RECONCILE_RESULT_KIND_NOOP,
                        global_store_pb2.RECONCILE_RESULT_KIND_IGNORED_STALE,
                    }
                except Exception:
                    ok = False
                stats.record(
                    "reconcile_worker_state", (time.monotonic() - t0) * 1000.0, ok
                )
                time.sleep(interval)

        def chunk_sync_worker(worker_index: int) -> None:
            worker_id = worker_ids[worker_index]
            node_id = f"node-{worker_index}"
            interval = max(0.05, float(args.chunk_sync_interval_s))
            artifact_id = f"mi2:chunk-artifact-{worker_index}"
            while not stop_event.is_set():
                updates = [
                    global_store_pb2.ChunkStateUpdate(
                        artifact_id=artifact_id,
                        chunk_idx=chunk_idx,
                        state=random.choice(
                            [
                                global_store_pb2.CHUNK_STATE_HOT,
                                global_store_pb2.CHUNK_STATE_COPIED_GPU,
                            ]
                        ),
                        device_uuid="gpu-0",
                        replica=0,
                    )
                    for chunk_idx in range(int(args.chunk_count))
                ]
                t0 = time.monotonic()
                ok = True
                try:
                    req = global_store_pb2.BatchUpdateChunkStatesRequest(
                        worker_id=worker_id,
                        node_id=node_id,
                        updates=updates,
                    )
                    resp = runtime_stub.BatchUpdateChunkStates(req, timeout=15.0)
                    ok = resp.status == global_store_pb2.Status.STATUS_OK and int(
                        resp.updates_applied
                    ) == len(updates)
                except Exception:
                    ok = False
                stats.record(
                    "batch_update_chunk_states", (time.monotonic() - t0) * 1000.0, ok
                )
                time.sleep(interval)

        threads: list[threading.Thread] = [
            threading.Thread(
                target=resolver_worker,
                args=(i,),
                daemon=True,
                name=f"resolver-{i}",
            )
            for i in range(int(args.resolver_threads))
        ]
        for i in range(int(args.workers)):
            threads.extend(
                [
                    threading.Thread(
                        target=heartbeat_worker,
                        args=(i,),
                        daemon=True,
                        name=f"heartbeat-{i}",
                    ),
                    threading.Thread(
                        target=reconcile_worker,
                        args=(i,),
                        daemon=True,
                        name=f"reconcile-{i}",
                    ),
                    threading.Thread(
                        target=chunk_sync_worker,
                        args=(i,),
                        daemon=True,
                        name=f"chunk-{i}",
                    ),
                ]
            )

        for t in threads:
            t.start()

        last_print = time.monotonic()
        duration = float(args.duration_s)
        metric_interval = max(0.2, float(args.metrics_scrape_interval_s))
        next_metric_at = time.monotonic()
        while True:
            now = time.monotonic()
            if now - start_ts >= duration:
                break
            if now >= next_metric_at:
                sample = _scrape_metrics(metrics_url, tracked_metrics)
                if sample:
                    last_metric_sample = sample
                    for metric_name, metric_value in sample.items():
                        metrics_series[metric_name].append(float(metric_value))
                next_metric_at = now + metric_interval
            if now - last_print >= 5.0:
                snap = stats.snapshot()
                elapsed = now - start_ts
                print(f"[progress] elapsed={elapsed:.1f}s")
                for op, item in sorted(snap.items()):
                    total = int(item["total"])
                    rps = total / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  - {op}: total={total} ok={item['ok']} err={item['err']} rps={rps:.1f} "
                        f"p95_ms={item['latency_ms']['p95']}"
                    )
                if last_metric_sample:
                    print(
                        "  - control_plane: inflight={:.0f} inflight_peak={:.0f} busy={:.0f} idle={:.0f} queue={:.0f} reducer_queue={:.0f} heartbeat_buffer={:.0f} max_workers={:.0f}".format(
                            float(
                                last_metric_sample.get(
                                    "tc_grpc_server_inflight_requests", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_grpc_server_inflight_peak_requests", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_control_plane_executor_busy_threads", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_control_plane_executor_idle_threads", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_control_plane_executor_queue_depth", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_worker_control_reducer_queue_depth", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_worker_heartbeat_buffer_pending", 0.0
                                )
                            ),
                            float(
                                last_metric_sample.get(
                                    "tc_control_plane_executor_max_threads", 0.0
                                )
                            ),
                        )
                    )
                last_print = now
            time.sleep(0.2)

        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)

        elapsed = max(0.001, time.monotonic() - start_ts)
        snapshot = stats.snapshot()
        for item in snapshot.values():
            total = int(item["total"])
            item["rps"] = total / elapsed
        metric_summary = {
            metric_name: _series_summary(metric_values)
            for metric_name, metric_values in sorted(metrics_series.items())
        }
        metric_summary = {
            metric_name: summary
            for metric_name, summary in metric_summary.items()
            if summary is not None
        }

        result = {
            "config": {
                "duration_s": duration,
                "workers": int(args.workers),
                "resolver_threads": int(args.resolver_threads),
                "resolve_qps_per_thread": float(args.resolve_qps_per_thread),
                "heartbeat_interval_s": float(args.heartbeat_interval_s),
                "reconcile_interval_s": float(args.reconcile_interval_s),
                "chunk_sync_interval_s": float(args.chunk_sync_interval_s),
                "chunk_count": int(args.chunk_count),
                "gs_max_workers": int(args.max_workers),
                "reducer_shards": int(args.reducer_shards),
                "reducer_queue_capacity": int(args.reducer_queue_capacity),
                "reducer_coalesce_window_ms": int(args.reducer_coalesce_window_ms),
                "metrics_scrape_interval_s": float(args.metrics_scrape_interval_s),
            },
            "elapsed_s": elapsed,
            "ops": snapshot,
            "control_plane_metrics": metric_summary,
        }

        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                f.write(text)

    finally:
        started.server.stop(grace=1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
