#  Copyright (c) 2025, TensorCast Team.

"""Human and machine-readable status routines for StoreDaemon."""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any, Optional

import click
import grpc
import psutil

from tensorcast.daemon_runtime_config import load_daemon_config
from tensorcast.proto.daemon.v1 import store_daemon_pb2, store_daemon_pb2_grpc
from tensorcast.store_session_registry import (
    StoreSessionRecord,
)
from tensorcast.store_session_registry import (
    iter_records as iter_store_sessions,
)

from .filesys import read_json_locked
from .paths import session_paths


def _format_bytes(bytes_value: int) -> str:
    if bytes_value < 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"


def get_process_info(pid: int) -> dict[str, Any] | None:
    try:
        p = psutil.Process(pid)
        return {
            "pid": pid,
            "started": time.ctime(p.create_time()),
            "cpu_percent": p.cpu_percent(),
            "memory_mb": p.memory_info().rss / 1024 / 1024,
        }
    except Exception:
        return {"pid": pid}


def _display_detailed_status(
    pid: int, response: "store_daemon_pb2.GetDetailedStatusResponse"
) -> None:
    click.echo("\n" + "=" * 60)
    click.echo("StoreDaemon Status")
    click.echo("=" * 60)
    click.echo(f"PID: {pid}")
    click.echo(f"Uptime: {_format_duration(response.uptime_seconds)}")
    click.echo(f"Health: {'Healthy' if response.is_healthy else 'Unhealthy'}")
    if response.is_shutting_down:
        click.echo("Status: SHUTTING DOWN")
    click.echo(f"Registered: {'Yes' if response.is_registered else 'No'}")
    if response.worker_id:
        click.echo(f"Worker ID: {response.worker_id}")
    mp = response.memory_pool_info
    click.echo("\nMemory Pool Status")
    click.echo("=" * 60)
    click.echo(f"Total Size: {_format_bytes(mp.total_size_bytes)}")
    if mp.total_size_bytes > 0:
        percent_avail = (mp.available_bytes / mp.total_size_bytes) * 100
        click.echo(
            f"Available: {_format_bytes(mp.available_bytes)} ({percent_avail:.1f}%)"
        )
        click.echo(
            f"Allocated: {_format_bytes(mp.allocated_bytes)} ({100 - percent_avail:.1f}%)"
        )
    else:
        click.echo(f"Available: {_format_bytes(mp.available_bytes)}")
        click.echo(f"Allocated: {_format_bytes(mp.allocated_bytes)}")
    if mp.allocated_chunks_count > 0:
        click.echo(f"Chunks: {mp.allocated_chunks_count} allocated")
    if mp.chunk_size_bytes > 0:
        click.echo(f"Chunk Size: {_format_bytes(mp.chunk_size_bytes)}")
    if response.gpu_devices:
        click.echo(f"\nGPU Devices ({len(response.gpu_devices)} total)")
        click.echo("=" * 60)
        for gpu in response.gpu_devices:
            click.echo(f"\nGPU {gpu.device_id}")
            if gpu.device_uuid:
                click.echo(f"  UUID: {gpu.device_uuid}")
            click.echo(f"  Total Memory: {_format_bytes(gpu.total_memory_bytes)}")
            if gpu.total_memory_bytes > 0:
                used_percent = (gpu.used_memory_bytes / gpu.total_memory_bytes) * 100
                click.echo(
                    f"  Used: {_format_bytes(gpu.used_memory_bytes)} ({used_percent:.1f}%)"
                )
                click.echo(
                    f"  Free: {_format_bytes(gpu.free_memory_bytes)} ({100 - used_percent:.1f}%)"
                )
            else:
                click.echo(f"  Used: {_format_bytes(gpu.used_memory_bytes)}")
                click.echo(f"  Free: {_format_bytes(gpu.free_memory_bytes)}")
            if gpu.loaded_replicas:
                click.echo(f"  Replicas ({len(gpu.loaded_replicas)} loaded):")
                for artifact in gpu.loaded_replicas:
                    click.echo(
                        f"    - {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
                    )
    if response.cpu_replicas:
        click.echo(f"\nCPU Memory Replicas ({len(response.cpu_replicas)} total)")
        click.echo("=" * 60)
        for artifact in response.cpu_replicas:
            click.echo(
                f"- {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
            )
            if artifact.is_registered_for_comm:
                click.echo("  [Registered for RDMA]")
    comm = response.communication_info
    if comm.enabled:
        click.echo("\nCommunication Status")
        click.echo("=" * 60)
        click.echo("RDMA: Enabled")
        if comm.total_transfers > 0:
            click.echo(f"Total Transfers: {comm.total_transfers}")
            click.echo(
                f"Bytes Transferred: {_format_bytes(comm.total_bytes_transferred)}"
            )
            if comm.total_transfer_errors > 0:
                click.echo(f"Transfer Errors: {comm.total_transfer_errors}")
    click.echo("\nSummary")
    click.echo("=" * 60)
    click.echo(f"Total Replicas Loaded: {response.total_replicas_loaded}")
    click.echo(
        f"Total Artifact Size: {_format_bytes(response.total_artifact_size_bytes)}"
    )
    click.echo(f"Storage Path: {response.storage_path}")
    if response.num_worker_threads > 0:
        click.echo(f"Worker Threads: {response.num_worker_threads}")
    click.echo("")


def _format_timestamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _store_session_status(record: StoreSessionRecord, alive: bool) -> str:
    if record.closed_at is not None:
        return "CLOSED"
    return "ACTIVE" if alive else "STALE"


def _display_store_sessions() -> None:
    sessions = list(iter_store_sessions())
    if not sessions:
        click.echo("\nNo Store sessions recorded yet.")
        return
    click.echo("\n" + "=" * 60)
    click.echo("Store Sessions")
    click.echo("=" * 60)
    for record in sorted(sessions, key=lambda rec: rec.last_activity_at, reverse=True):
        alive = False
        with contextlib.suppress(Exception):
            alive = psutil.pid_exists(record.pid)
        click.echo(f"\nSession ID: {record.session_id}")
        click.echo(f"  Daemon Endpoint: {record.daemon_endpoint}")
        click.echo(f"  PID: {record.pid}")
        click.echo(f"  Status: {_store_session_status(record, alive)}")
        click.echo(f"  Created: {_format_timestamp(record.created_at)}")
        click.echo(f"  Last Activity: {_format_timestamp(record.last_activity_at)}")
        if record.closed_at is not None:
            click.echo(f"  Closed: {_format_timestamp(record.closed_at)}")
        click.echo(f"  Active Leases: {record.active_leases}")
        click.echo(f"  Pending Futures: {record.pending_futures}")
        if record.capabilities:
            click.echo("  Capabilities:")
            for key, value in sorted(record.capabilities.items()):
                click.echo(f"    {key}: {value}")
    click.echo("")


def check_service_status(
    *,
    session_id: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    inst = session_paths(session_id)
    try:
        data = read_json_locked(inst.pids_json)
        pid = int(data.get("processes", [{}])[0].get("pid", 0))
    except Exception:
        pid = 0
    if pid <= 0:
        click.echo(f"Daemon session {inst.id} has no running process.")
        sys.exit(1)

    try:
        if host is None or port is None:
            derived_host = None
            derived_port: int | None = None
            try:
                meta = read_json_locked(inst.meta_json)
                addr = meta.get("address")
                if addr and ":" in addr:
                    h, p = addr.split(":", 1)
                    derived_host = h
                    derived_port = int(p)
                else:
                    cfg_path = meta.get("config_path")
                    if cfg_path:
                        cfg = load_daemon_config(cfg_path)
                        if cfg.server.listen.host:
                            derived_host = cfg.server.listen.host
                        if cfg.server.listen.port:
                            derived_port = int(cfg.server.listen.port)
            except Exception:
                pass
            host = host or derived_host or "127.0.0.1"
            port = port or derived_port or 50052

        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
        response = stub.GetDetailedStatus(
            store_daemon_pb2.GetDetailedStatusRequest(), timeout=5.0
        )
        try:
            meta = read_json_locked(inst.meta_json)
            p2p = meta.get("p2p_address")
            if p2p:
                click.echo(f"P2P Address: {p2p}")
        except Exception:
            pass
        _display_detailed_status(pid, response)
        _display_store_sessions()
    except Exception as e:  # noqa: BLE001
        click.echo(f"StoreDaemon is running with PID {pid}")
        click.echo(f"  (Unable to connect to gRPC service: {e})")
        info = get_process_info(pid)
        if info and len(info) > 1:
            if "started" in info:
                click.echo(f"  Started: {info['started']}")
            if "cpu_percent" in info:
                click.echo(f"  CPU: {info['cpu_percent']}%")
            if "memory_mb" in info:
                click.echo(f"  Memory: {info['memory_mb']:.1f} MB")
        _display_store_sessions()
    sys.exit(0)
