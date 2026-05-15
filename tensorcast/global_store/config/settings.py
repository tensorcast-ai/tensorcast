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


class SourceBalanceWeightsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_load_weight: float = 1.0
    worker_load_weight: float = 1.0
    recent_assignment_penalty_weight: float = 1.0
    diffusion_bonus_weight: float = 1.0


class GroupDispatchPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fairness_floor_ratio: float = 0.25
    completion_bias_weight: float = 1.0
    starvation_aging_threshold_ms: int = 5_000
    queue_scan_limit: int = 128
    dispatch_batch_limit: int = 16
    group_source_spread_weight: float = 2.0
    group_source_soft_cap_ratio: float = 1.3
    group_source_min_candidates_for_enforce: int = 3


class TransportSchedulerPolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str = "LEGACY"
    source_balance_weights: SourceBalanceWeightsConfig = SourceBalanceWeightsConfig()
    group_dispatch: GroupDispatchPolicyConfig = GroupDispatchPolicyConfig()


class GroupRealizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    default_deadline_ms: int = 30_000
    max_total_parts: int = 1024
    transaction_ttl_ms: int = 300_000
    expiration_scan_interval_ms: int = 30_000
    max_parts_per_version_set: int = 4096
    publish_authority_mode: str = "AUTO_WHEN_READY"
    max_active_transactions: int = 4096
    max_waiters_per_transaction: int = 1024
    min_wait_poll_interval_ms: int = 25
    max_wait_poll_interval_ms: int = 500
    max_member_reports_per_rpc: int = 1024
    cleanup_batch_limit: int = 256


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
    transport_scheduler: TransportSchedulerPolicyConfig = (
        TransportSchedulerPolicyConfig()
    )
    group_realization: GroupRealizationConfig = GroupRealizationConfig()

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

        worker_policy_section = (
            data.get("worker_policy", {}) if isinstance(data, dict) else {}
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
        scheduler_policy = TransportSchedulerPolicyConfig()
        group_realization = GroupRealizationConfig()
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
        if pb.worker_policy.HasField("transport_scheduler"):
            scheduler_pb = pb.worker_policy.transport_scheduler
            scheduler_section = (
                worker_policy_section.get("transport_scheduler", {})
                if isinstance(worker_policy_section, dict)
                else {}
            )
            if (
                scheduler_pb.mode
                == gsc_pb2.TransportSchedulerMode.TRANSPORT_SCHEDULER_MODE_SOURCE_BALANCE
            ):
                scheduler_mode = "SOURCE_BALANCE"
            elif (
                scheduler_pb.mode
                == gsc_pb2.TransportSchedulerMode.TRANSPORT_SCHEDULER_MODE_GROUP_DISPATCH
            ):
                scheduler_mode = "GROUP_DISPATCH"
            else:
                scheduler_mode = "LEGACY"

            source_weights = scheduler_policy.source_balance_weights
            if scheduler_pb.HasField("source_balance_weights"):
                source_pb = scheduler_pb.source_balance_weights
                source_section = (
                    scheduler_section.get("source_balance_weights", {})
                    if isinstance(scheduler_section, dict)
                    else {}
                )
                source_weights = SourceBalanceWeightsConfig(
                    replica_load_weight=float(source_pb.replica_load_weight)
                    if "replica_load_weight" in source_section
                    else source_weights.replica_load_weight,
                    worker_load_weight=float(source_pb.worker_load_weight)
                    if "worker_load_weight" in source_section
                    else source_weights.worker_load_weight,
                    recent_assignment_penalty_weight=float(
                        source_pb.recent_assignment_penalty_weight
                    )
                    if "recent_assignment_penalty_weight" in source_section
                    else source_weights.recent_assignment_penalty_weight,
                    diffusion_bonus_weight=float(source_pb.diffusion_bonus_weight)
                    if "diffusion_bonus_weight" in source_section
                    else source_weights.diffusion_bonus_weight,
                )

            group_dispatch = scheduler_policy.group_dispatch
            if scheduler_pb.HasField("group_dispatch"):
                dispatch_pb = scheduler_pb.group_dispatch
                dispatch_section = (
                    scheduler_section.get("group_dispatch", {})
                    if isinstance(scheduler_section, dict)
                    else {}
                )
                group_dispatch = GroupDispatchPolicyConfig(
                    fairness_floor_ratio=float(dispatch_pb.fairness_floor_ratio)
                    if "fairness_floor_ratio" in dispatch_section
                    else group_dispatch.fairness_floor_ratio,
                    completion_bias_weight=float(dispatch_pb.completion_bias_weight)
                    if "completion_bias_weight" in dispatch_section
                    else group_dispatch.completion_bias_weight,
                    starvation_aging_threshold_ms=(
                        _dur_ms(dispatch_pb.starvation_aging_threshold)
                        if "starvation_aging_threshold" in dispatch_section
                        else group_dispatch.starvation_aging_threshold_ms
                    ),
                    queue_scan_limit=max(
                        1,
                        int(dispatch_pb.queue_scan_limit)
                        if "queue_scan_limit" in dispatch_section
                        else group_dispatch.queue_scan_limit,
                    ),
                    dispatch_batch_limit=max(
                        1,
                        int(dispatch_pb.dispatch_batch_limit)
                        if "dispatch_batch_limit" in dispatch_section
                        else group_dispatch.dispatch_batch_limit,
                    ),
                    group_source_spread_weight=float(
                        dispatch_pb.group_source_spread_weight
                    )
                    if "group_source_spread_weight" in dispatch_section
                    else group_dispatch.group_source_spread_weight,
                    group_source_soft_cap_ratio=float(
                        dispatch_pb.group_source_soft_cap_ratio
                    )
                    if "group_source_soft_cap_ratio" in dispatch_section
                    else group_dispatch.group_source_soft_cap_ratio,
                    group_source_min_candidates_for_enforce=max(
                        1,
                        int(dispatch_pb.group_source_min_candidates_for_enforce)
                        if "group_source_min_candidates_for_enforce" in dispatch_section
                        else group_dispatch.group_source_min_candidates_for_enforce,
                    ),
                )
            scheduler_policy = TransportSchedulerPolicyConfig(
                mode=scheduler_mode,
                source_balance_weights=source_weights,
                group_dispatch=group_dispatch,
            )
        if pb.worker_policy.HasField("group_realization"):
            group_pb = pb.worker_policy.group_realization
            group_section = (
                worker_policy_section.get("group_realization", {})
                if isinstance(worker_policy_section, dict)
                else {}
            )
            if (
                group_pb.publish_authority_mode
                == gsc_pb2.GroupPublishAuthorityMode.GROUP_PUBLISH_AUTHORITY_MODE_COORDINATOR_EXPLICIT
            ):
                publish_authority_mode = "COORDINATOR_EXPLICIT"
            else:
                publish_authority_mode = "AUTO_WHEN_READY"
            group_realization = GroupRealizationConfig(
                enabled=bool(group_pb.enabled)
                if "enabled" in group_section
                else group_realization.enabled,
                default_deadline_ms=max(
                    0,
                    int(group_pb.default_deadline_ms)
                    if "default_deadline_ms" in group_section
                    else group_realization.default_deadline_ms,
                ),
                max_total_parts=max(
                    1,
                    int(group_pb.max_total_parts)
                    if "max_total_parts" in group_section
                    else group_realization.max_total_parts,
                ),
                transaction_ttl_ms=max(
                    0,
                    int(group_pb.transaction_ttl_ms)
                    if "transaction_ttl_ms" in group_section
                    else group_realization.transaction_ttl_ms,
                ),
                expiration_scan_interval_ms=max(
                    0,
                    int(group_pb.expiration_scan_interval_ms)
                    if "expiration_scan_interval_ms" in group_section
                    else group_realization.expiration_scan_interval_ms,
                ),
                max_parts_per_version_set=max(
                    1,
                    int(group_pb.max_parts_per_version_set)
                    if "max_parts_per_version_set" in group_section
                    else group_realization.max_parts_per_version_set,
                ),
                publish_authority_mode=publish_authority_mode,
                max_active_transactions=max(
                    1,
                    int(group_pb.max_active_transactions)
                    if "max_active_transactions" in group_section
                    else group_realization.max_active_transactions,
                ),
                max_waiters_per_transaction=max(
                    1,
                    int(group_pb.max_waiters_per_transaction)
                    if "max_waiters_per_transaction" in group_section
                    else group_realization.max_waiters_per_transaction,
                ),
                min_wait_poll_interval_ms=max(
                    0,
                    int(group_pb.min_wait_poll_interval_ms)
                    if "min_wait_poll_interval_ms" in group_section
                    else group_realization.min_wait_poll_interval_ms,
                ),
                max_wait_poll_interval_ms=max(
                    1,
                    int(group_pb.max_wait_poll_interval_ms)
                    if "max_wait_poll_interval_ms" in group_section
                    else group_realization.max_wait_poll_interval_ms,
                ),
                max_member_reports_per_rpc=max(
                    1,
                    int(group_pb.max_member_reports_per_rpc)
                    if "max_member_reports_per_rpc" in group_section
                    else group_realization.max_member_reports_per_rpc,
                ),
                cleanup_batch_limit=max(
                    1,
                    int(group_pb.cleanup_batch_limit)
                    if "cleanup_batch_limit" in group_section
                    else group_realization.cleanup_batch_limit,
                ),
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
            transport_scheduler=scheduler_policy,
            group_realization=group_realization,
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
