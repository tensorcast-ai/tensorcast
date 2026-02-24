#  Copyright (c) 2025-2026, TensorCast Team.

"""Global Store configuration settings (file-based only)."""

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from google.protobuf import json_format as _pb_json
from pydantic import BaseModel, ConfigDict

from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import (
    global_store_config_pb2 as gsc_pb2,
)

# GlobalStoreConfig now leverages Pydantic for validation / immutability


class DigestWriteLimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_leaf_writes_per_request: int = 16_384
    max_proof_digests_per_request: int = 16_384
    max_total_digests_per_request: int = 32_768
    max_digest_bytes_per_request: int = 2 * 1024 * 1024


class OperationLeasePolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_ttl_ms: int = 30_000
    max_ttl_ms: int = 300_000


class OperationWriteLimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_status_update_interval_ms: int = 1_000


class RetentionPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    operations_ttl_ms: int = 86_400_000  # 24h
    assembly_proof_commitments_ttl_ms: int = 86_400_000  # 24h
    piece_proof_digests_ttl_ms: int = 86_400_000  # 24h


class GlobalStoreLimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    digest_writes: DigestWriteLimitsConfig = DigestWriteLimitsConfig()
    operation_leases: OperationLeasePolicyConfig = OperationLeasePolicyConfig()
    retention: RetentionPolicyConfig = RetentionPolicyConfig()
    operation_writes: OperationWriteLimitsConfig = OperationWriteLimitsConfig()


class WorkerControlReducerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shard_count: int = 1
    queue_capacity: int = 2048
    coalesce_window_ms: int = 50


class KeyMappingPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    alias_cache_ttl_ms: int = 1000


class GlobalStoreConfig(BaseModel):
    """Configuration for Global Store service."""

    model_config = ConfigDict(frozen=True)

    # Database settings
    # Optional path to a persistent on-disk DuckDB database. When ``None`` a
    # purely in-memory (ephemeral) database will be used. The CLI and daemon
    # will create a temporary on-disk database if persistence is required, so
    # using ``None`` is the safest default for unit-tests which should not
    # touch the real filesystem.
    db_file: Optional[Path] = None

    # Worker management settings
    heartbeat_timeout_ms: int = 30000  # 30 seconds
    cleanup_interval_ms: int = 60000  # 1 minute
    default_heartbeat_interval_ms: int = 5000  # 5 seconds
    memory_tier_snapshot_retention_ms: int = 600000  # 10 minutes
    memory_tier_snapshot_max_rows: int = 200
    memory_tier_publish_interval_ms: int = 5000  # Daemon publish hint
    worker_control_reducer: WorkerControlReducerConfig = WorkerControlReducerConfig()
    key_mapping_policy: KeyMappingPolicyConfig = KeyMappingPolicyConfig()

    # Server settings
    listen_host: str = "127.0.0.1"
    listen_port: int = 50051
    advertise_host: str | None = None
    advertise_port: int | None = None
    max_workers: int = 10

    # Performance settings
    transport_wait_retry_interval_ms: int = 200

    # Maintenance settings
    optimize_interval_ms: int = 3_600_000  # 1 hour

    # Metrics settings
    metrics_port: int = 8000
    # Cluster identity (opaque token used to prevent split-brain)
    cluster_token: Optional[str] = None
    limits: GlobalStoreLimitsConfig = GlobalStoreLimitsConfig()

    @property
    def port(self) -> int:
        """Backwards-compatible alias for listen_port."""

        return self.listen_port

    # Environment-based loader removed in final scheme

    @classmethod
    def from_file(cls, path: str) -> "GlobalStoreConfig":
        """Create config from a YAML or JSON file validated against Proto schema.

        This loader parses YAML/JSON into tensorcast.config.v1.GlobalStoreConfig
        (strict unknown-key rejection), then maps the fields into the
        Pydantic model used by the Python service.
        """
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Parse into proto (strict)
        pb = gsc_pb2.GlobalStoreConfig()
        # Normalize user-friendly enum strings (e.g., grpc, INFO) before parsing
        if isinstance(data, dict):
            normalize_enum_aliases_inplace(data, gsc_pb2.GlobalStoreConfig.DESCRIPTOR)
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)

        # Map to Pydantic
        # Database
        db_file = (
            Path(pb.database.db_file).expanduser() if pb.database.db_file else None
        )
        # Server
        listen_host = "127.0.0.1"
        listen_port = 50051
        advertise_host: str | None = None
        advertise_port: int | None = None
        if pb.server.HasField("listen"):
            listen_host = pb.server.listen.host or listen_host
            listen_port = int(pb.server.listen.port or 0) or 0
        if pb.server.HasField("advertise"):
            advertise_host = pb.server.advertise.host or None
            advertise_port = (
                int(pb.server.advertise.port) if pb.server.advertise.port > 0 else None
            )
        # Preserve default when field is absent; accept explicit 0 when provided
        server_section = data.get("server", {}) if isinstance(data, dict) else {}
        has_max_workers = isinstance(server_section, dict) and (
            "max_workers" in server_section
        )
        max_workers = int(pb.server.max_workers) if has_max_workers else 10
        metrics_port = (
            int(pb.server.metrics_port)
            if pb.server.metrics_port
            or (isinstance(server_section, dict) and "metrics_port" in server_section)
            else 8000
        )

        # Worker policy durations are in seconds+nanos
        def _dur_ms(dur) -> int:
            # google.protobuf.Duration parsed from JSON strings "Xs" yields
            # (seconds, nanos). Convert to milliseconds.
            return int(dur.seconds * 1000 + dur.nanos // 1_000_000)

        heartbeat_timeout_ms = (
            _dur_ms(pb.worker_policy.heartbeat_timeout)
            if pb.worker_policy.HasField("heartbeat_timeout")
            else 30000
        )
        cleanup_interval_ms = (
            _dur_ms(pb.worker_policy.cleanup_interval)
            if pb.worker_policy.HasField("cleanup_interval")
            else 60000
        )
        default_hb_ms = (
            _dur_ms(pb.worker_policy.default_heartbeat_interval)
            if pb.worker_policy.HasField("default_heartbeat_interval")
            else 5000
        )
        # Clamp negatives to 0
        heartbeat_timeout_ms = max(0, heartbeat_timeout_ms)
        cleanup_interval_ms = max(0, cleanup_interval_ms)
        default_hb_ms = max(0, default_hb_ms)

        snapshot_retention_ms = 600_000
        snapshot_max_rows = 200
        publish_interval_ms = 5000
        reducer_cfg = WorkerControlReducerConfig()
        key_mapping_policy = KeyMappingPolicyConfig()
        if pb.worker_policy.HasField("memory_tiers"):
            mt = pb.worker_policy.memory_tiers
            snapshot_retention_ms = (
                _dur_ms(mt.snapshot_retention)
                if mt.HasField("snapshot_retention")
                else snapshot_retention_ms
            )
            snapshot_max_rows = int(mt.snapshot_max_rows) or snapshot_max_rows
            publish_interval_ms = (
                _dur_ms(mt.publish_interval)
                if mt.HasField("publish_interval")
                else publish_interval_ms
            )
        if pb.worker_policy.HasField("control_reducer"):
            reducer_pb = pb.worker_policy.control_reducer
            reducer_cfg = WorkerControlReducerConfig(
                shard_count=max(1, int(reducer_pb.shard_count or 0))
                if int(reducer_pb.shard_count or 0) > 0
                else reducer_cfg.shard_count,
                queue_capacity=max(1, int(reducer_pb.queue_capacity or 0))
                if int(reducer_pb.queue_capacity or 0) > 0
                else reducer_cfg.queue_capacity,
                coalesce_window_ms=max(
                    0,
                    _dur_ms(reducer_pb.coalesce_window)
                    if reducer_pb.HasField("coalesce_window")
                    else reducer_cfg.coalesce_window_ms,
                ),
            )
        if pb.worker_policy.HasField("key_mapping"):
            key_mapping_pb = pb.worker_policy.key_mapping
            alias_cache_ttl_ms = (
                _dur_ms(key_mapping_pb.alias_cache_ttl)
                if key_mapping_pb.HasField("alias_cache_ttl")
                else key_mapping_policy.alias_cache_ttl_ms
            )
            key_mapping_policy = KeyMappingPolicyConfig(
                alias_cache_ttl_ms=max(0, alias_cache_ttl_ms),
            )

        # Limits
        limits = GlobalStoreLimitsConfig()
        if pb.HasField("limits"):
            pb_limits = pb.limits
            digest_writes = limits.digest_writes
            if pb_limits.HasField("digest_writes"):
                dw = pb_limits.digest_writes
                digest_writes = DigestWriteLimitsConfig(
                    max_leaf_writes_per_request=int(dw.max_leaf_writes_per_request)
                    or digest_writes.max_leaf_writes_per_request,
                    max_proof_digests_per_request=int(dw.max_proof_digests_per_request)
                    or digest_writes.max_proof_digests_per_request,
                    max_total_digests_per_request=int(dw.max_total_digests_per_request)
                    or digest_writes.max_total_digests_per_request,
                    max_digest_bytes_per_request=int(dw.max_digest_bytes_per_request)
                    or digest_writes.max_digest_bytes_per_request,
                )

            op_leases = limits.operation_leases
            if pb_limits.HasField("operation_leases"):
                ol = pb_limits.operation_leases
                op_leases = OperationLeasePolicyConfig(
                    default_ttl_ms=_dur_ms(ol.default_ttl)
                    if ol.HasField("default_ttl")
                    else op_leases.default_ttl_ms,
                    max_ttl_ms=_dur_ms(ol.max_ttl)
                    if ol.HasField("max_ttl")
                    else op_leases.max_ttl_ms,
                )

            op_writes = limits.operation_writes
            if pb_limits.HasField("operation_writes"):
                ow = pb_limits.operation_writes
                op_writes = OperationWriteLimitsConfig(
                    min_status_update_interval_ms=_dur_ms(ow.min_status_update_interval)
                    if ow.HasField("min_status_update_interval")
                    else op_writes.min_status_update_interval_ms,
                )

            retention = limits.retention
            if pb_limits.HasField("retention"):
                rp = pb_limits.retention
                retention = RetentionPolicyConfig(
                    operations_ttl_ms=_dur_ms(rp.operations_ttl)
                    if rp.HasField("operations_ttl")
                    else retention.operations_ttl_ms,
                    assembly_proof_commitments_ttl_ms=_dur_ms(
                        rp.assembly_proof_commitments_ttl
                    )
                    if rp.HasField("assembly_proof_commitments_ttl")
                    else retention.assembly_proof_commitments_ttl_ms,
                    piece_proof_digests_ttl_ms=_dur_ms(rp.piece_proof_digests_ttl)
                    if rp.HasField("piece_proof_digests_ttl")
                    else retention.piece_proof_digests_ttl_ms,
                )

            limits = GlobalStoreLimitsConfig(
                digest_writes=digest_writes,
                operation_leases=op_leases,
                retention=retention,
                operation_writes=op_writes,
            )

        return cls(
            db_file=db_file,
            heartbeat_timeout_ms=heartbeat_timeout_ms,
            cleanup_interval_ms=cleanup_interval_ms,
            default_heartbeat_interval_ms=default_hb_ms,
            memory_tier_snapshot_retention_ms=max(0, snapshot_retention_ms),
            memory_tier_snapshot_max_rows=max(0, snapshot_max_rows),
            memory_tier_publish_interval_ms=max(0, publish_interval_ms),
            worker_control_reducer=reducer_cfg,
            key_mapping_policy=key_mapping_policy,
            listen_host=listen_host or "127.0.0.1",
            listen_port=listen_port if listen_port >= 0 else 0,
            advertise_host=advertise_host,
            advertise_port=advertise_port,
            max_workers=max_workers,
            transport_wait_retry_interval_ms=200,
            optimize_interval_ms=3_600_000,
            # Metrics port: keep default unless overridden elsewhere
            metrics_port=metrics_port if metrics_port >= 0 else 0,
            cluster_token=(pb.meta.cluster_token or None)
            if pb.HasField("meta")
            else None,
            limits=limits,
        )

    @staticmethod
    def load_proto_from_file(path: str) -> gsc_pb2.GlobalStoreConfig:
        """Load and return the GlobalStoreConfig proto from YAML/JSON (strict)."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        pb = gsc_pb2.GlobalStoreConfig()
        if isinstance(data, dict):
            normalize_enum_aliases_inplace(data, gsc_pb2.GlobalStoreConfig.DESCRIPTOR)
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
        return pb


# Singleton config instance
_config: Optional[GlobalStoreConfig] = None


def get_config() -> GlobalStoreConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise RuntimeError(
            "GlobalStoreConfig not initialized. Call set_config() first."
        )
    return _config


def set_config(cfg: GlobalStoreConfig) -> None:
    global _config
    _config = cfg
